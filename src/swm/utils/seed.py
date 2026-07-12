from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Pin every RNG (python, numpy, torch CPU and CUDA) to one seed.
    Makes a variant-by-seed run reproducible so an A-vs-B difference comes from lambda, not noise.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
