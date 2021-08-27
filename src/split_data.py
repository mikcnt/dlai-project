from os.path import join

import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old_dir",
        type=str,
        help="Non splitted files path",
        default="../all_rollouts/",
    )
    parser.add_argument("--data_dir", type=str, help="Data path", default="../data/")

    parser.add_argument(
        "--ratio",
        help="Train, validation, test ratio",
        nargs=3,
        type=float,
        default=(0.7, 0.2, 0.1),
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for environment", default=42
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    all_files = os.listdir(args.old_dir)
    files = all_files.copy()
    n_files = len(files)
    ratio = tuple(args.ratio)

    train_n_files = int(n_files * ratio[0])
    val_n_files = int(n_files * ratio[1])
    test_n_files = n_files - train_n_files - val_n_files

    train_files = np.random.choice(files, train_n_files, replace=False)
    files = [x for x in files if x not in train_files]

    val_files = np.random.choice(files, val_n_files, replace=False)
    files = [x for x in files if x not in val_files]

    test_files = files
    print(join(args.data_dir, "train"))
    os.makedirs(join(args.data_dir, "train"), exist_ok=True)
    os.makedirs(join(args.data_dir, "val"), exist_ok=True)
    os.makedirs(join(args.data_dir, "test"), exist_ok=True)

    for file in tqdm(all_files):
        if file in train_files:
            subfolder_path = "train"
        elif file in val_files:
            subfolder_path = "val"
        else:
            subfolder_path = "test"

        old_path = join(args.old_dir, file)
        new_path = join(args.data_dir, subfolder_path, file)
        shutil.move(old_path, new_path)
