import argparse
import os
from os.path import join
import random

import gym
import numpy as np

from src.plunder_standardize_colors import standardize_colors


def generate_data(rollouts, data_dir, std_colors, distribution_mode):
    """Generates data"""
    env = gym.make(
        "procgen:procgen-plunder-v0",
        use_backgrounds=False,
        restrict_themes=True,
        use_monochrome_assets=True,
        distribution_mode=distribution_mode,
    )
    seq_len = 1000

    for i in range(rollouts):
        os.makedirs(f"{data_dir}/rollout_{i}", exist_ok=True)
        env.reset()

        # sample random actions using seed
        a_rollout = [random.randint(0, env.action_space.n - 1) for _ in range(seq_len)]

        s_rollout = []
        r_rollout = []
        d_rollout = []

        t = 0
        while True:
            action = a_rollout[t]

            s, r, done, _ = env.step(action)

            if std_colors:
                s = standardize_colors(s, distribution_mode=distribution_mode)

            # save single move frame
            np.savez(
                join(data_dir, f"rollout_{i}/img_{t}"),
                observations=np.array(s),
                rewards=np.array(r),
                actions=np.array(action),
                terminals=np.array(done),
            )
            t += 1
            s_rollout += [s]
            r_rollout += [r]
            d_rollout += [done]
            # save entire rollout
            if done:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                np.savez(
                    join(data_dir, f"rollout_{i}/rollout_{i}"),
                    observations=np.array(s_rollout),
                    rewards=np.array(r_rollout),
                    actions=np.array(a_rollout),
                    terminals=np.array(d_rollout),
                )
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=int, help="Number of rollouts", default=1000)
    parser.add_argument(
        "--dir", type=str, help="Where to place rollouts", default="../all_rollouts/"
    )
    parser.add_argument(
        "--std_colors",
        default=True,
        type=bool,
        help="Standardization of the colors in the images",
    )
    parser.add_argument(
        "--distribution_mode",
        type=str,
        help="Distribution mode of the gym environment",
        default="easy",
    )

    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    os.makedirs(args.dir, exist_ok=True)
    generate_data(args.rollouts, args.dir, args.std_colors, args.distribution_mode)
