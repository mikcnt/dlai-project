import gym
import hydra
import torch
from omegaconf import DictConfig

from src.common.utils import PROJECT_ROOT
from src.pl_modules.controller import RolloutGenerator


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="controller")
def main(cfg: DictConfig):
    device = torch.device("cuda")
    rollout_generator = RolloutGenerator(
        cfg=cfg, device=device, time_limit=100000, render=True
    )
    cumulative = rollout_generator.rollout(params=None)
    print(cumulative)


if __name__ == "__main__":
    main()
