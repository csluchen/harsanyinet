import os
import torch
import numpy as np
import random

def setup_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False
