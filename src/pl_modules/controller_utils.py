from collections import OrderedDict

import numpy as np
import torch


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


def to_log_cma(es):
    to_log = {}

    # rewards (best, worst, mean, std)
    to_log["best_performance"] = - min(es.fit.fit)
    to_log["worst_performance"] = - max(es.fit.fit)
    to_log["mean_performance"] = - np.mean(es.fit.fit)
    to_log["std_performance"] = - np.std(es.fit.fit)
    to_log["performance_mean_plus_std"] = (
        to_log["mean_performance"] + to_log["std_performance"]
    )
    to_log["performance_mean_minus_std"] = (
        to_log["mean_performance"] - to_log["std_performance"]
    )

    # iterations and number of evaluations
    to_log["iteration"] = es.countiter
    to_log["function_evals"] = es.countevals

    # misc
    to_log["axis_ratio"] = es.D.max() / es.D.min()
    to_log["sigma"] = es.sigma
    to_log["min_max"] = es.sigma * min(es.sigma_vec * es.dC ** 0.5)
    to_log["std"] = es.sigma * max(es.sigma_vec * es.dC ** 0.5)

    return to_log
