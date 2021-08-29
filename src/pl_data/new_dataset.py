import os
from os.path import join
import torch
from tqdm import tqdm

from src.common.utils import get_env
from src.pl_data.CustomDataLoader import CustomDataLoader
from src.pl_data.dataset import GameEpisodeDataset

name = "."


def generate_mdrnn_data(split_part, std_colors=True):
    assert split_part in ["train", "val", "test"]

    old_path = get_env(f"YOUR_{split_part.upper()}_DATASET_PATH")
    new_path = old_path.replace(split_part, "new_" + split_part)
    os.makedirs(new_path, exist_ok=True)

    dataset = GameEpisodeDataset(name=name, path=old_path, std_colors=std_colors)
    loader = CustomDataLoader(dataset, batch_size=32, drop_last=True)

    for i, batch in enumerate(tqdm(loader)):
        batch = {k: v.squeeze() for k, v in batch.items()}
        torch.save(batch, join(new_path, f"{i}.pt"))


if __name__ == "__main__":
    for split_part in ["train", "val", "test"]:
        print(f"Generating the {split_part} MDRNN dataset...")
        generate_mdrnn_data(split_part)
        print("Done.")
