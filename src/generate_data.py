import argparse
import os
from os.path import join, exists
import random

import gym
import numpy as np


# def sample_continuous_policy(env, seq_len, dt):
#     """Sample a continuous policy.
#     Atm, action_space is supposed to be a box environment. The policy is
#     sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).
#     :args action_space: gym action space
#     :args seq_len: number of actions returned
#     :args dt: temporal discretization
#     :returns: sequence of seq_len actions
#     """
#     actions = [env.action_space.sample()]
#     print(env.action_space)
#     exit()
#     for _ in range(seq_len):
#         daction_dt = np.random.randn()
#         actions.append(
#             np.clip(
#                 actions[-1] + math.sqrt(dt) * daction_dt,
#                 env.action_space.low,
#                 env.action_space.high,
#             )
#         )
#     return actions


def generate_data(rollouts, data_dir, noise_type):  # pylint: disable=R0914
    """Generates data"""
    assert exists(data_dir), "The data directory does not exist..."

    # env = gym.make("procgen:procgen-coinrun-v0")
    env = gym.make("procgen:procgen-coinrun-v0")
    seq_len = 1000

    for i in range(rollouts):
        os.makedirs(f"{data_dir}/rollout_{i}", exist_ok=True)
        env.reset()
        # env.env.viewer.window.dispatch_events()
        if noise_type == "white":
            a_rollout = [env.action_space.sample() for _ in range(seq_len)]
        # elif noise_type == "brown":
        #     a_rollout = sample_continuous_policy(env, seq_len, 1.0 / 50)

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
            # env.env.viewer.window.dispatch_events()
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
    parser.add_argument(
        "--policy",
        type=str,
        choices=["white", "brown"],
        help="Noise type used for action sampling.",
        default="white",
    )
    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    generate_data(args.rollouts, args.dir, args.policy)
