import torch
import random
import numpy as np


def seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
