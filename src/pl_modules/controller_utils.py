import sys
from collections import OrderedDict
from os import getpid
from os.path import join, exists
from time import sleep

import gym
import numpy as np
import torch
from torchvision.transforms import transforms

from src.pl_modules.controller import Controller
from src.pl_modules.mdrnn import MDRNNModel, MDRNNCell
from src.pl_modules.vae import VaeModel


ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE = 3, 32, 256, 64, 64

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((RED_SIZE, RED_SIZE)),
        transforms.ToTensor(),
    ]
)


def rnn_adjust_parameters(state_dict) -> OrderedDict:
    state_dict = {
        k.replace("model.", ""): v for k, v in state_dict.items() if "vae" not in k
    }
    return OrderedDict({k.strip("_l0"): v for k, v in state_dict.items()})


def flatten_parameters(params):
    """Flattening parameters.
    :args params: generator of parameters (as returned by module.parameters())
    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()


def unflatten_parameters(params, example, device):
    """Unflatten parameters.
    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters
    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx : idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened


def load_parameters(params, controller):
    """Load flattened parameters into controller.
    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)


class RolloutGenerator(object):
    """Utility to generate rollouts.
    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.
    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v0 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """

    def __init__(self, vae_path, mdrnn_path, controller_path, device, time_limit):
        """Build vae, rnn, controller and environment."""

        # load VAE
        self.vae = VaeModel.load_from_checkpoint(vae_path, map_location=device)

        # load MDRNN
        rnn_state = torch.load(mdrnn_path, map_location={"cuda:0": str(device)})
        rnn_state = rnn_state["state_dict"]
        rnn_state = rnn_adjust_parameters(rnn_state)
        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict(rnn_state)

        # instantiate Controller
        self.controller = Controller(LSIZE, RSIZE, ASIZE).to(device)

        # load Controller if it was previously saved
        if exists(controller_path):
            ctrl_state = torch.load(
                controller_path, map_location={"cuda:0": str(device)}
            )
            print("Loading Controller with reward {}".format(ctrl_state["reward"]))
            self.controller.load_state_dict(ctrl_state["state_dict"])

        # instantiate environment
        self.env = gym.make("CarRacing-v0")

        self.device = device
        self.time_limit = time_limit

    def get_action_and_transition(self, obs, hidden):
        """Get action and transition.
        Encode obs to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the MDRNN and compute the controller
        corresponding action.
        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor
        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        _, latent_mu, _ = self.vae(obs)
        action = self.controller(latent_mu, hidden[0])
        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden)
        return action.squeeze().cpu().numpy(), next_hidden

    def rollout(self, params, render=False):
        """Execute a rollout and returns minus cumulative reward.
        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.
        :args params: parameters as a single 1D np array
        :returns: minus cumulative reward
        """
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)

        obs = self.env.reset()

        # This first render is required !
        self.env.render()

        hidden = [torch.zeros(1, RSIZE).to(self.device) for _ in range(2)]

        cumulative = 0
        i = 0
        while True:
            obs = transform(obs).unsqueeze(0).to(self.device)
            action, hidden = self.get_action_and_transition(obs, hidden)
            obs, reward, done, _ = self.env.step(action)

            if render:
                self.env.render()

            cumulative += reward
            if done or i > self.time_limit:
                return -cumulative
            i += 1


def slave_routine(p_queue, r_queue, e_queue, p_index, logdir, time_limit, tmp_dir):
    """Thread routine.
    Threads interact with p_queue, the parameters queue, r_queue, the result
    queue and e_queue the end queue. They pull parameters from p_queue, execute
    the corresponding rollout, then place the result in r_queue.
    Each parameter has its own unique id. Parameters are pulled as tuples
    (s_id, params) and results are pushed as (s_id, result).  The same
    parameter can appear multiple times in p_queue, displaying the same id
    each time.
    As soon as e_queue is non empty, the thread terminate.
    When multiple gpus are involved, the assigned gpu is determined by the
    process index p_index (gpu = p_index % n_gpus).
    :args p_queue: queue containing couples (s_id, parameters) to evaluate
    :args r_queue: where to place results (s_id, results)
    :args e_queue: as soon as not empty, terminate
    :args p_index: the process index
    """
    # init routine
    gpu = p_index % torch.cuda.device_count()
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    # redirect streams
    sys.stdout = open(join(tmp_dir, str(getpid()) + ".out"), "a")
    sys.stderr = open(join(tmp_dir, str(getpid()) + ".err"), "a")

    with torch.no_grad():
        r_gen = RolloutGenerator(logdir, device, time_limit)

        while e_queue.empty():
            if p_queue.empty():
                sleep(0.1)
            else:
                s_id, params = p_queue.get()
                r_queue.put((s_id, r_gen.rollout(params)))



