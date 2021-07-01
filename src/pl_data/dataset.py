import glob
import os
from typing import Dict

import hydra
import numpy as np
import omegaconf
import torch
from omegaconf import ValueNode
from torch.utils.data import Dataset

from src.common.utils import PROJECT_ROOT


class GameSceneDataset(Dataset):
    def __init__(
        self,
        name: ValueNode,
        path: ValueNode,
    ):
        super().__init__()
        self.path = path
        self.name = name
        self.fpaths = sorted(glob.glob(os.path.join(path, "rollout_*/img_*.npz")))
        self.indices = np.arange(0, len(self.fpaths))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        npz = np.load(self.fpaths[self.indices[idx]])
        obs = (
            torch.as_tensor(npz["observations"], dtype=torch.float32).permute(2, 0, 1)
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
        npz = np.load(self.fpaths[self.indices[idx]])
        obs = npz["observations"] / 255
        roll_dim, H, W, C = obs.shape
        actions = npz["actions"][:roll_dim]
        n_seq = roll_dim // self.seq_len
        end_seq = n_seq * self.seq_len

        obs = obs[:end_seq].reshape([-1, self.seq_len, H, W, C])
        actions = actions[:end_seq].reshape([-1, self.seq_len])

        obs = torch.as_tensor(obs).permute(0, 1, 4, 2, 3)
        actions = torch.as_tensor(actions, dtype=torch.float)

        return {"observations": obs, "actions": actions}

    def __repr__(self) -> str:
        return f"MyDataset({self.name=}, {self.path=})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    dataset: GameEpisodeDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )


if __name__ == "__main__":
    main()
