import os
import random
import numpy as np
import torch
import logging

# For colored terminal text
from colorama import Fore, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL

logger = logging.getLogger(__name__)


def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    logger.info(f"Setting random seed to {seed} for reproducibility...")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info('Random seed set successfully')
