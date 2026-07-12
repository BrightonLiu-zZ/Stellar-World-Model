"""Stage 1 entrypoint: pretrain one variant-by-seed run.

Run (from repo src/ on PYTHONPATH, in the swm env):
    python -m swm.train variant=A seed=0
    python -m swm.train variant=B seed=0
    python -m swm.train variant=B seed=0 data.limit=50 data.batch_size=32 train.max_epochs=2 train.wandb.mode=disabled  # smoke
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from swm.train.loop import train


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
