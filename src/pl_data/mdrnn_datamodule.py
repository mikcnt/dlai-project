import random
from typing import Optional, Sequence

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.common.utils import PROJECT_ROOT
from src.pl_data.CustomDataLoader import CustomDataLoader


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class MdRnnDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        batch_size: DictConfig,
        drop_last: DictConfig,
    ):
        super().__init__()
        self.datasets = datasets
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        # Here you should instantiate your datasets, you may also split the train into train and validation if needed.
        if stage is None or stage == "fit":
            self.train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.val_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.val
            ]

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.test
            ]

    def train_dataloader(self) -> CustomDataLoader:
        return CustomDataLoader(
            self.train_dataset,
            batch_size=self.batch_size.train,
            shuffle=True,
            drop_last=self.drop_last.train,
        )

    def val_dataloader(self) -> Sequence[CustomDataLoader]:
        return [
            CustomDataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                drop_last=self.drop_last.val,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[CustomDataLoader]:
        return [
            CustomDataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                drop_last=self.drop_last.test,
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="mdrnn")
def main(cfg: omegaconf.DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )


if __name__ == "__main__":
    main()