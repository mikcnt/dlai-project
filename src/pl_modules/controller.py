import os
from os import mkdir
from os.path import join, exists
from time import sleep
from typing import Tuple, List

import cma
import gym
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.multiprocessing import Process, Queue
from torchvision.transforms import transforms
import wandb

from src.pl_modules.controller_utils import (
    rnn_adjust_parameters,
    load_parameters,
    flatten_parameters,
    to_log_cma,
)
from src.common.utils import PROJECT_ROOT
from src.pl_modules.mdrnn import MDRNNCell
from src.pl_modules.vae import VaeModel
from src.plunder_standardize_colors import standardize_colors


class Controller(nn.Module):
    def __init__(self, latents, recurrents, actions, action_space=4):
        super().__init__()
        self.fc = nn.Linear(in_features=latents + recurrents, out_features=action_space)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=1)
        action_probs = self.softmax(self.fc(cat_in))
        return action_probs


class RolloutGenerator(object):
    """Utility to generate rollouts.
    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDN-RNN.
    """

    def __init__(self, cfg, device, time_limit, render=False):
        """Build vae, rnn, controller and environment."""
        self.cfg = cfg
        self.render = render

        # transformations
        self.transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor(),]
        )

        # load VAE
        self.vae = VaeModel.load_from_checkpoint(
            cfg.model.vae_checkpoint_path, map_location=device
        ).to(device)
        # load MDRNN
        rnn_state = torch.load(cfg.model.mdrnn_checkpoint_path, map_location=device)
        rnn_state = rnn_state["state_dict"]
        rnn_state = rnn_adjust_parameters(rnn_state)
        self.mdrnn = MDRNNCell(cfg.model.LSIZE, cfg.model.ASIZE, cfg.model.RSIZE, 5).to(
            device
        )
        self.mdrnn.load_state_dict(rnn_state)

        # instantiate Controller
        self.controller = Controller(
            cfg.model.LSIZE, cfg.model.RSIZE, cfg.model.ASIZE
        ).to(device)

        # load Controller if it was previously saved
        if exists(cfg.model.controller_checkpoint_path):
            ctrl_state = torch.load(
                cfg.model.controller_checkpoint_path, map_location=device,
            )
            print("Loading Controller with reward {}".format(ctrl_state["reward"]))
            self.controller.load_state_dict(ctrl_state["state_dict"])

        # instantiate environment
        self.env = gym.make(
            self.cfg.model.environment_name,
            use_backgrounds=False,
            restrict_themes=True,
            use_monochrome_assets=True,
            distribution_mode=self.cfg.model.distribution_mode,
            start_level=0,
            num_levels=20,
            use_sequential_levels=True,
            render_mode="human" if self.render else "rgb_array",
        )

        self.device = device
        self.time_limit = time_limit

    def get_action_and_transition(
        self, obs: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[np.ndarray, torch.Tensor]:
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
        # reconstruction observation stats
        _, latent_mu, _ = self.vae(obs)

        # compute action
        action_probs = self.controller(latent_mu, hidden[0])
        action_probs = action_probs.cpu().detach().numpy()
        choices = []
        for i in range(action_probs.shape[0]):
            choice = np.argmax(action_probs[i]) * 3
            choices.append(choice)
        action = torch.tensor(choices, device=self.device).unsqueeze(0)

        # update MDN-RNN hidden state
        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden)
        return action.squeeze().cpu().numpy(), next_hidden

    def rollout(self, params: torch.Tensor) -> int:
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
        obs = standardize_colors(obs, self.cfg.model.distribution_mode)

        # This first render is required!
        self.env.render()

        # instantiate first hidden state of the MDN-RNN
        hidden = [
            torch.zeros(1, self.cfg.model.RSIZE, device=self.device) for _ in range(2)
        ]

        cumulative = 0
        i = 0
        # generate rollout
        while True:
            obs = self.transform(obs).unsqueeze(0).to(self.device)

            # compute action and update hidden state
            action, hidden = self.get_action_and_transition(obs, hidden)

            # execute action and get reward
            obs, reward, done, _ = self.env.step(action)
            obs = standardize_colors(obs, self.cfg.model.distribution_mode)

            if self.render:
                self.env.render()

            # only return the cumulative reward
            cumulative += reward
            if done or i > self.time_limit:
                return -cumulative

            i += 1


class ControllerPipeline(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.controller = Controller(cfg.model.LSIZE, cfg.model.RSIZE, cfg.model.ASIZE)

        # main path
        self.logdir = cfg.model.controller_path
        os.makedirs(self.logdir, exist_ok=True)

        # create ctrl dir if non exitent
        self.ctrl_dir = join(self.logdir, "ctrl")
        if not exists(self.ctrl_dir):
            mkdir(self.ctrl_dir)

        # optimization algorithm variables
        self.pop_size = cfg.model.pop_size
        self.n_samples = cfg.model.n_samples
        self.target_return = cfg.model.target_return

        self.time_limit = 100000  # basically infinite. we decided not to stop a game episode based on time

        # Max number of workers
        self.max_workers = cfg.model.max_worker
        self.num_workers = min(self.max_workers, self.n_samples * self.pop_size)

        # Define queues and start workers
        self.p_queue = Queue()
        self.r_queue = Queue()
        self.e_queue = Queue()

        for p_index in range(self.num_workers):
            Process(
                target=self.slave_routine,
                args=(self.p_queue, self.r_queue, self.e_queue, p_index),
            ).start()

    def evaluate(
        self, solutions: List[np.ndarray], results: List[float], rollouts: int = 100
    ) -> Tuple[np.ndarray, float, float]:
        """Give current controller evaluation.
        Evaluation is minus the cumulated reward averaged over rollout runs.
        :args solutions: CMA set of solutions
        :args results: corresponding results
        :args rollouts: number of rollouts
        :returns: minus averaged cumulated reward
        """
        index_min = np.argmin(results)
        best_guess = solutions[index_min]
        restimates = []

        for s_id in range(rollouts):
            self.p_queue.put((s_id, best_guess))

        print("Evaluating...")
        for _ in tqdm(range(rollouts)):
            while self.r_queue.empty():
                sleep(0.1)
            restimates.append(self.r_queue.get()[1])

        return best_guess, np.mean(restimates), np.std(restimates)

    def slave_routine(
        self, p_queue: Queue, r_queue: Queue, e_queue: Queue, p_index: int
    ) -> None:
        """Thread routine.
        Threads interact with p_queue (the parameters queue), r_queue (the result
        queue) and e_queue (the end queue). They pull parameters from p_queue, execute
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
        device = torch.device(
            "cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu"
        )

        with torch.no_grad():
            r_gen = RolloutGenerator(self.cfg, device, self.time_limit)

            while e_queue.empty():
                if p_queue.empty():
                    sleep(0.1)
                else:
                    s_id, params = p_queue.get()
                    r_queue.put((s_id, r_gen.rollout(params)))

    def optimize(self) -> None:
        """Launch the CMA optimization algorithm."""
        # initialize the wandb project (for each thread)
        wandb.init(**self.cfg.logging)

        # define current best and load parameters
        cur_best = None
        ctrl_file = join(self.ctrl_dir, "best.tar")
        print("Attempting to load previous best...")
        if exists(ctrl_file):
            state = torch.load(ctrl_file, map_location={"cuda:0": "cpu"})
            cur_best = -state["reward"]

            self.controller.load_state_dict(state["state_dict"])
            print("Previous best was {}...".format(-cur_best))

        # initialize the CMA Evolution algorithm
        parameters = self.controller.parameters()
        es = cma.CMAEvolutionStrategy(
            flatten_parameters(parameters), 0.1, {"popsize": self.pop_size}
        )

        epoch = 0
        log_step = self.cfg.log_step

        # start the training
        while True:
            if cur_best is not None and -cur_best >= self.target_return:
                print("Already better than target, breaking...")
                break

            # result list
            r_list = [0] * self.pop_size

            # Generate step
            solutions = es.ask()

            # push parameters to queue
            for s_id, s in enumerate(solutions):
                for _ in range(self.n_samples):
                    self.p_queue.put((s_id, s))

            # retrieve results
            pbar = tqdm(total=self.pop_size * self.n_samples)
            for _ in range(self.pop_size * self.n_samples):
                while self.r_queue.empty():
                    sleep(0.1)
                r_s_id, r = self.r_queue.get()
                r_list[r_s_id] += r / self.n_samples
                pbar.update(1)
            pbar.close()

            # Test step
            es.tell(solutions, r_list)

            es.disp()
            to_log = to_log_cma(es)

            # Select step
            # evaluation and saving
            if epoch % log_step == log_step - 1:
                best_params, best, std_best = self.evaluate(solutions, r_list)
                to_log["evaluation_best"] = best
                to_log["evaluation_std_best"] = std_best

                print("Current evaluation: {}".format(best))
                print("cur best: ", cur_best, "best: ", best)

                if not cur_best or cur_best > best:
                    cur_best = best
                    print(
                        "Saving new best with value {}+-{}...".format(
                            -cur_best, std_best
                        )
                    )
                    load_parameters(best_params, self.controller)
                    torch.save(
                        {
                            "epoch": epoch,
                            "reward": -cur_best,
                            "state_dict": self.controller.state_dict(),
                        },
                        join(self.ctrl_dir, "best.tar"),
                    )
                if -best > self.target_return:
                    print(
                        "Terminating controller training with value {}...".format(best)
                    )
                    break

            wandb.log(data=to_log, step=es.countiter)
            epoch += 1

        es.result_pretty()
        self.e_queue.put("EOP")


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="controller")
def main(cfg: DictConfig):
    controller_pipeline = ControllerPipeline(cfg)
    controller_pipeline.optimize()


if __name__ == "__main__":
    main()
