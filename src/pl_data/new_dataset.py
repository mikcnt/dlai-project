from os.path import join
import torch
from tqdm import tqdm

from src.common.utils import get_env
from src.pl_data.CustomDataLoader import CustomDataLoader
from src.pl_data.dataset import GameEpisodeDataset

name = "."
old_path = get_env("YOUR_VAL_DATASET_PATH")
new_path = old_path.replace("val", "new_val")


if __name__ == "__main__":
    dataset = GameEpisodeDataset(name=name, path=old_path)

    loader = CustomDataLoader(dataset, batch_size=32, drop_last=True)

    batch = next(iter(loader))

    count = 0
    for i, batch in enumerate(tqdm(loader)):
        count += 1
        batch = {k: v.squeeze() for k, v in batch.items()}
        torch.save(batch, join(new_path, f"{i}.pt"))
