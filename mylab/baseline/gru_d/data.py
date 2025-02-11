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
        print('use grud collect func')
        pass

    def __call__(self, batch):
        #combined_t, combined_data, combined_mask, xn, np.array(len(combined_t)), now_y
        # combined_data_and_mask_and_t, delta_t, xn, np.array(combined_data_and_mask_and_t.shape[1]), now_y

        x = [i[0] for i in batch]
        market_info = [i[1] for i in batch]
        xn = [i[2] for i in batch]
        mask = [i[3] for i in batch]
        delta_t_x = [i[4] for i in batch]
        delta_t_xn = [i[5] for i in batch]
        now_y = [i[6] for i in batch]

        
        x = np.stack(x, axis = 0)
        market_info = np.stack(market_info, axis = 0)
        xn = np.stack(xn, axis = 0)
        mask = np.stack(mask, axis = 0)
        delta_t_x = np.stack(delta_t_x, axis = 0)
        delta_t_xn = np.stack(delta_t_xn, axis = 0)
        y = np.stack(now_y, axis = 0)

        x = torch.FloatTensor(x)
        market_info = torch.FloatTensor(market_info)
        xn = torch.FloatTensor(xn)
        mask = torch.FloatTensor(mask)
        delta_t_x = torch.FloatTensor(delta_t_x)
        delta_t_xn = torch.FloatTensor(delta_t_xn)
        y = torch.FloatTensor(y)
        #length = torch.FloatTensor(length)
        #delta_t = torch.FloatTensor(delta_t)
        #xn_delta_t = torch.FloatTensor(xn_delta_t)

        return x, market_info, delta_t_x, delta_t_xn, mask, xn, y

        # combined_t batch*p*seq
        # combined_data batch*p*seq*dim
        # combined_mask batch*p*seq
        # delta_t batch*p*seq
        # length batch * p
        # xn batch * dim