from time import sleep
from typing import Any, Dict, Sequence, Tuple, Union
import cma
import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig
from tqdm import tqdm

from src.common.utils import PROJECT_ROOT

import torch
import torch.nn as nn


class Controller(nn.Module):
    def __init__(self, latents, recurrents, actions):
        super().__init__()
        self.fc = nn.Linear(in_features=latents + recurrents, out_features=actions)

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=1)
        return self.fc(cat_in)


def evaluate(solutions, results, rollouts=100):
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
        p_queue.put((s_id, best_guess))

    print("Evaluating...")
    for _ in tqdm(range(rollouts)):
        while r_queue.empty():
            sleep(0.1)
        restimates.append(r_queue.get()[1])

    return best_guess, np.mean(restimates), np.std(restimates)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="controller")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
