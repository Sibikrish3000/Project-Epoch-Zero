import torch
import numpy as np
import random

def set_seed(seed=42):
    """
    Set seeds for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Physical constants
R_EARTH = 6378.137      # km
MU_EARTH = 398600.4418  # km^3/s^2
