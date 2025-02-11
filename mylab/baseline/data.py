from torch.utils.data import Dataset
import numpy as np
import os
import torch
import torch.nn as nn
import random
import pandas as pd
from tqdm import tqdm

class CollateFunc:
    def __init__(self, args):
        pass

    def __call__(self, batch):
        # x, xn, now_y

        x = [i[0] for i in batch] # seq * dim
        market_info = [i[1] for i in batch]
        xn = [i[2] for i in batch]   
        y = [i[3] for i in batch]
        x = np.stack(x, axis = 0)
        market_info = np.stack(market_info, axis = 0)
        xn = np.stack(xn, axis = 0)
        y = np.stack(y, axis = 0)
        x = torch.FloatTensor(x)
        market_info = torch.FloatTensor(market_info)
        xn = torch.FloatTensor(xn)
        y = torch.FloatTensor(y)
        
        return x, market_info, xn, y
