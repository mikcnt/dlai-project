import argparse
import os
from os.path import join, exists
import random

import gym
import numpy as np


def generate_data(rollouts, data_dir):
    """Generates data"""

    env = gym.make(
        "procgen:procgen-plunder-v0",
        use_backgrounds=False,
        restrict_themes=True,
        use_monochrome_assets=True,
    )
    seq_len = 1000

    for i in range(rollouts):
        os.makedirs(f"{data_dir}/rollout_{i}", exist_ok=True)
        env.reset()

        a_rollout = [env.action_space.sample() for _ in range(seq_len)]

        s_rollout = []
        r_rollout = []
        d_rollout = []

        t = 0
        while True:
            action = a_rollout[t]

            s, r, done, _ = env.step(action)
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
    random.seed(42)
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=int, help="Number of rollouts", default=1000)
    parser.add_argument(
        "--dir", type=str, help="Where to place rollouts", default="../all_rollouts/"
    )
    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    generate_data(args.rollouts, args.dir)
