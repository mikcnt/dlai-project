import glob
import os
from os import listdir
from os.path import join
from typing import Dict

import hydra
import numpy as np
import omegaconf
import torch
from omegaconf import ValueNode
from torch.utils.data import Dataset

from src.common.utils import PROJECT_ROOT
from src.plumber_standardize_colors import standardize_colors


class GameSceneDataset(Dataset):
    def __init__(
        self,
        name: ValueNode,
        path: ValueNode,
        std_colors: bool,
    ):
        super().__init__()
        self.path = path
        self.name = name
        self.std_colors = std_colors
        self.fpaths = sorted(glob.glob(os.path.join(path, "rollout_*/img_*.npz")))
        self.indices = np.arange(0, len(self.fpaths))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        npz = np.load(self.fpaths[self.indices[idx]])
        obs_np = npz["observations"]

        if self.std_colors:
            obs_np = standardize_colors(obs_np)

        obs = (
            torch.as_tensor(obs_np, dtype=torch.float32).permute(2, 0, 1)
            / 255
        )
        return obs

    def __repr__(self) -> str:
        return f"MyDataset({self.name=}, {self.path=})"


class GameEpisodeDataset(Dataset):
    def __init__(
        self,
        name: ValueNode,
        path: ValueNode,
        seq_len: int = 32,
    ):
        super().__init__()
        self.path = path
        self.name = name
        self.fpaths = sorted(glob.glob(os.path.join(path, "rollout_*/rollout_*.npz")))
        self.indices = np.arange(0, len(self.fpaths))
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # load rollout
        npz = np.load(self.fpaths[self.indices[idx]])

        # observations and next observations
        obs = npz["observations"] / 255
        roll_dim, H, W, C = obs.shape
        n_seq = (
            roll_dim // self.seq_len
            if roll_dim % self.seq_len != 0
            else roll_dim // self.seq_len - 1
        )
        end_seq = n_seq * self.seq_len

        # current observations (exclude last, it has no following observation)
        curr_obs = obs[:end_seq]
        curr_obs = curr_obs.reshape([-1, self.seq_len, H, W, C])
        curr_obs = torch.as_tensor(curr_obs, dtype=torch.float32).permute(0, 1, 4, 2, 3)

        # next observations (exclude first, it has no previous observation)
        next_obs = obs[1 : end_seq + 1]
        next_obs = next_obs.reshape([-1, self.seq_len, H, W, C])
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32).permute(0, 1, 4, 2, 3)

        # actions
        # we need to cut the actions according to the number of actions done in the current rollout
        actions = npz["actions"][:roll_dim]
        actions = actions[:end_seq].reshape([-1, self.seq_len, 1])
        actions = torch.as_tensor(actions, dtype=torch.float)

        # reward
        rewards = npz["rewards"]
        rewards = rewards[:end_seq].reshape([-1, self.seq_len])
        rewards = torch.as_tensor(rewards, dtype=torch.float)

        # terminal
        terminals = npz["terminals"]
        terminals = terminals[:end_seq].reshape([-1, self.seq_len])
        terminals = torch.as_tensor(terminals, dtype=torch.float)

        return {
            "obs": curr_obs,
            "actions": actions,
            "rewards": rewards,
            "terminals": terminals,
            "next_obs": next_obs,
        }

    def __repr__(self) -> str:
        return f"MyDataset({self.name=}, {self.path=})"


class MyDataset(Dataset):
    def __init__(self, path, name):
        super(MyDataset, self).__init__()
        self.name = name
        self.items = [join(path, item) for item in listdir(path)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return torch.load(self.items[idx])


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    dataset: GameEpisodeDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    return dataset


if __name__ == "__main__":
    main()
