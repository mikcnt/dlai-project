import gym
import matplotlib.pyplot as plt

"""
0: sinistra
1: sinistra
2: sinistra
3: non fare niente
4: non fare niente
5: non fare niente
6: destra
7: destra
8: destra
9: spara
"""

[0, 3, 6, 9]
action = 0

if __name__ == "__main__":
    distribution_mode = "easy"

    env = gym.make(
        "procgen:procgen-plunder-v0",
        use_backgrounds=False,
        restrict_themes=True,
        use_monochrome_assets=True,
        distribution_mode=distribution_mode,
    )

    i = 0
    while True:
        print(action)
        obs, r, done, _ = env.step(action)

        plt.imshow(obs)
        plt.show()
        if i == 10:
            break
        i += 1
