import pandas as pd
import pickle
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
import torchcde
import random
import copy

def to_cuda_tensor(value, device, enable = False):
    if enable:
        assert device is not None
        if isinstance(value, list):
            return [torch.tensor(i).to(device) for i in value]
        else:
            return torch.tensor(value).to(device)
    else:
        return value


class IrregularUserTransactionData(Dataset):
    def __init__(self, 
                 user_data, 
                 user_data_y,
                 mean,
                 std, 
                 time_range,
                 args,
                 data = None,
                 type_ = 'train',
                 data_y_mode = 'trx_or_not',
                 to_cuda = False,
                 device = None,
                 sim_mask_gen = None, 
                 use_co_trx_sim_mask = False,
                 train_start_time = None,
                 range_len = 900,
                 ref_seq_len = 64,
                 sim_mask_end_time = None,
                 all_data_index = None,
                 all_data_array = None,
                 no_trx_xn_list = None,
                 sim_mask_array = None,
                 index2hash = None,
                 model_type = 'baseline', # mtand, node, grud
                 grud_data = False,
                 seq_len = 48,
                 get_data_by_seq_len = True,
                 market_info_array = None,
                 market_info = None,
                 ): # 'trx_direction', 'trx_or_not', 'both'
        super().__init__()

        self.to_cuda = to_cuda
        self.device = device
        self.args = args
        self.ref_seq_len = ref_seq_len
        self.seq_len = seq_len
        self.range_len = range_len
        self.ref_diff = (range_len - mean[0]) / std[0]
        self.grud_data = grud_data
        self.get_data_by_seq_len = get_data_by_seq_len

        print(self.args.model_type)

        if (all_data_index is not None) and (all_data_array is not None):
            self.use_co_trx_sim_mask = use_co_trx_sim_mask
            self.all_data_index = all_data_index
            self.all_data_array = all_data_array
            self.sim_mask_array = sim_mask_array
            self.no_trx_xn_list = no_trx_xn_list 
            self.index2hash = index2hash
            self.market_info_array = market_info_array

        else:
            
            self.mean = mean.astype('float32')
            self.std = std.astype('float32')

            self.market_info = market_info
            self.all_time_index = {}
            self.data_index = {}
            self.data_len = 0
            self.data_y_mode = data_y_mode
            self.use_co_trx_sim_mask = use_co_trx_sim_mask
            self.sim_mask_gen = sim_mask_gen
            if sim_mask_end_time is None:
                self.sim_mask_end_time = time_range[-1]
            else:
                self.sim_mask_end_time = sim_mask_end_time
    
            self.user_data = user_data
            self.user_data_y = user_data_y
    
            user_hash_dict = {}
    
            all_dim = None
    
            for j in tqdm(self.user_data):
                now_data = self.user_data[j]['data']
    
                for i in range(len(now_data)):
                    now_time = int(now_data[i][0])
                    now_data[i][0] = now_time
    
                    if all_dim is None:
                        all_dim = now_data.shape[1]

                    if (now_time >= (time_range[0] - range_len - 1)) and (now_time < time_range[1]):
                        
                        if now_time not in self.all_time_index:
                            self.all_time_index[now_time] = {}
                            self.all_time_index[now_time]['has_trx'] = False
                            self.all_time_index[now_time]['index'] = []
                        user_hash_dict[j] = 1
                        
                        if abs(self.user_data[j]['label'][i] - 0) < 0.1:
                            self.all_time_index[now_time]['index'].append([j, i, False])
                        else:
                            self.all_time_index[now_time]['index'].append([j, i, True])
                            self.all_time_index[now_time]['has_trx'] = True
    
            all_p = len(user_hash_dict)
            all_dim = all_dim
            all_time = len([1 for i in self.all_time_index if self.all_time_index[i]['has_trx'] == True])
    
            if train_start_time is not None:
                time_range[0] = train_start_time + range_len
    
            self.load_all_by_time(all_p, all_dim, all_time, time_range)
            self.normalize()

    def load_all_by_time(self, all_p, all_dim, all_time, time_range):

        print('use_grud_data', self.grud_data)
        
        self.all_data_array = np.zeros([all_p, all_time, all_dim + 2], dtype = 'float32')#D3SemiSparseArray([all_p, all_time, all_dim + 2])
        if self.market_info is not None:
            for i in self.market_info:
                market_info_dim = len(self.market_info[i])
            self.market_info_array = np.zeros([all_time, market_info_dim], dtype = 'float32')
        self.no_trx_xn_list = []
        ## 前all_dim是x，后面是mask, 再后面是没正则化的时间
        ## self.all_data_mask_array = np.zeros([all_p, all_time, all_dim], dtype = int)
        self.all_data_index = []

        if self.grud_data:
            last_ob_x_time = [0 for i in range(all_p)]
            last_ob_x_data_index = [None for i in range(all_p)]
        self.all_time_list = {}
        all_time_list = []
        hash2pindex = {}
        count_p_index = 0
        all_time_sorted = sorted(list(self.all_time_index.keys()))

        now_time_index = 0
        no_trx_index = 0
        for now_time in tqdm(all_time_sorted):

            #计算x的起点
            if not self.get_data_by_seq_len:
                start_time_index = now_time_index
                for i in range(len(all_time_list) - 1, -1, -1):
                    if all_time_list[i] >= now_time - self.range_len:
                        start_time_index = i
                    else:
                        break
            else:
                start_time_index = None

            # 有交易时，写入x
            if self.all_time_index[now_time]['has_trx']:
                self.all_time_list[now_time] = 0
                self.all_data_array[:, now_time_index, 0] = now_time
                self.all_data_array[:, now_time_index, -2] = now_time
                if self.market_info is not None:
                    self.market_info_array[now_time_index] = self.market_info[now_time]

            if self.grud_data:
                now_not_missing_p_index = []
            for hash_id, num, is_trx in self.all_time_index[now_time]['index']:
                if hash_id not in hash2pindex:
                    now_p_index = count_p_index
                    hash2pindex[hash_id] = now_p_index
                    count_p_index += 1
                else:
                    now_p_index = hash2pindex[hash_id]
                
                #如果是交易数据则写入x
                if is_trx:
                    if self.grud_data:
                        last_ob_x_data_index[now_p_index] = [hash_id, num]
                        now_not_missing_p_index.append(now_p_index)
                    self.all_data_array[now_p_index, now_time_index, :-2] = self.user_data[hash_id]['data'][num]
                    self.all_data_array[now_p_index, now_time_index, -1] = 1
                    if self.grud_data:
                        self.all_data_array[now_p_index, now_time_index, -2] = now_time - last_ob_x_time[now_p_index]
                        last_ob_x_time[now_p_index] = now_time
                else:  #如果不是交易则写入不交易记录
                    if self.grud_data:
                        self.no_trx_xn_list.append([self.user_data[hash_id]['data'][num], np.array([now_time-last_ob_x_time[now_p_index]], dtype = 'float32')])
                    else:
                        self.no_trx_xn_list.append([self.user_data[hash_id]['data'][num], None])
                
                if (now_time >= time_range[0]) and (now_time < time_range[1]):
                    data_now_y = self.user_data_y[hash_id][num]
                    if self.data_y_mode == 'trx_or_not':
                        now_y = np.array(data_now_y[2:], dtype = 'float32')
                    else:
                        now_y = np.array(data_now_y[0:2], dtype = 'float32')
                    if (not self.all_time_index[now_time]['has_trx']) and (not is_trx):
                        if not self.get_data_by_seq_len:
                            self.all_data_index.append([now_p_index, now_time_index + 1, start_time_index, no_trx_index, now_y, is_trx])
                        else:
                            start_time_index = now_time_index + 1 - self.seq_len
                            if start_time_index >= 0:
                                self.all_data_index.append([now_p_index, now_time_index + 1, start_time_index, no_trx_index, now_y, is_trx])
                    else:
                        if not self.get_data_by_seq_len:
                            self.all_data_index.append([now_p_index, now_time_index, start_time_index, no_trx_index, now_y, is_trx])
                        else:
                            start_time_index = now_time_index - self.seq_len
                            if start_time_index >= 0:
                                self.all_data_index.append([now_p_index, now_time_index, start_time_index, no_trx_index, now_y, is_trx])
                                
                if not is_trx:
                    no_trx_index += 1

            if self.grud_data:
                for missing_p_index in range(all_p):
                    if missing_p_index not in now_not_missing_p_index:
                        if last_ob_x_data_index[missing_p_index] is not None:
                            [hash_id, num] = last_ob_x_data_index[missing_p_index]
                            self.all_data_array[missing_p_index, now_time_index, :-2] = self.user_data[hash_id]['data'][num]
                            self.all_data_array[missing_p_index, now_time_index, -2] = now_time - last_ob_x_time[now_p_index]

            if self.all_time_index[now_time]['has_trx']:
                now_time_index += 1

        if self.use_co_trx_sim_mask:
            sort_hash_list = sorted(list(hash2pindex.keys()), key=lambda x: hash2pindex[x])
            self.sim_mask_array = self.sim_mask_gen.cal_sim_metrix(sort_hash_list, sort_hash_list, self.sim_mask_end_time)

        self.index2hash = {hash2pindex[hash__]: hash__ for hash__ in hash2pindex}

    def normalize(self):
        min_value = np.min(np.min(self.all_data_array, axis = 1), axis = 0)[1:-2]
        
        for index, i in enumerate(tqdm(self.all_data_array)):
            self.all_data_array[index, :, 1:-2] = np.log(i[:, 1:-2] - min_value + 1)
        for index, (xn_array, delta_t_xn) in enumerate(tqdm(self.no_trx_xn_list)):
            xn_array[1:] = np.log(xn_array[1:] - min_value + 1)
            self.no_trx_xn_list[index] = [xn_array, delta_t_xn]

        trx_array = np.concatenate([i[i[:, -1].astype(bool), :-2] for i in self.all_data_array], axis = 0)
        mean = np.mean(trx_array, axis = 0)
        std = np.std(trx_array, axis = 0)
        min_ = np.min(trx_array, axis = 0).astype('float64')
        max_ = np.max(trx_array, axis = 0).astype('float64')

        self.std = std
        
        for index, i in enumerate(tqdm(self.all_data_array)):
            self.all_data_array[index, :, 1:-2] = (i[:, 1:-2] - mean[1:]) / std[1:]
            self.all_data_array[index, :, 0] = (i[:, 0] - min_[0]) / (max_[0] - min_[0])
            
        for index, (xn_array, delta_t_xn) in enumerate(tqdm(self.no_trx_xn_list)):
            xn_array[1:] = (xn_array[1:] - mean[1:]) / std[1:]
            xn_array[0] = (xn_array[0] - min_[0]) / (max_[0] - min_[0])
            self.no_trx_xn_list[index] = [xn_array, delta_t_xn]
            
    def get_base_xn(self, is_trx, no_trx_index, now_p_index, now_time_index):
        if is_trx:
            return self.all_data_array[now_p_index, now_time_index, :-2]
        else:
            return self.no_trx_xn_list[no_trx_index][0]

    def get_grud_xn(self, is_trx, no_trx_index, now_p_index, now_time_index):
        if is_trx:
            return self.all_data_array[now_p_index, now_time_index, :-2], self.all_data_array[now_p_index, now_time_index, -2:-1]
        else:
            return self.no_trx_xn_list[no_trx_index]

    def get_base_x_and_market_info(self, is_trx, start_time_index, now_time_index):
        
        return self.all_data_array[:, start_time_index: now_time_index], self.market_info_array[start_time_index: now_time_index]
    
    def get_data_iteract(self, now_p_index, now_time_index, start_time_index, no_trx_index, now_y, is_trx):

        xn = self.get_base_xn(is_trx, no_trx_index, now_p_index, now_time_index)
        base_data_array, market_info_array = self.get_base_x_and_market_info(is_trx, start_time_index, now_time_index)
        
        mask = base_data_array[:, :, -1]
        now_y = now_y
        
        # Reordering of multiple user data and deletion of non-transacted users
        now_p_list_index = list(np.where(np.sum(mask, axis = 1) != 0)[0])
        if now_p_index in now_p_list_index:
            now_p_list_index.remove(now_p_index)
        now_p_list_index.append(now_p_index)
        multiuser_x = base_data_array[now_p_list_index]

        return multiuser_x, xn, now_y

    def get_data_grud(self, now_p_index, now_time_index, start_time_index, no_trx_index, now_y, is_trx):

        xn, delta_t_xn = self.get_grud_xn(is_trx, no_trx_index, now_p_index, now_time_index)
        base_data_array, market_info_array = self.get_base_x_and_market_info(is_trx, start_time_index, now_time_index)
        
        mask = base_data_array[now_p_index, :, -1]
        single_x = base_data_array[now_p_index, :, :-2]
        delta_t_x = base_data_array[now_p_index, :, -2]
        now_y = now_y

        return single_x, market_info_array, xn, mask, delta_t_x, delta_t_xn, now_y

    def get_data_baseline(self, now_p_index, now_time_index, start_time_index, no_trx_index, now_y, is_trx):

        xn = self.get_base_xn(is_trx, no_trx_index, now_p_index, now_time_index)
        base_data_array, market_info_array = self.get_base_x_and_market_info(is_trx, start_time_index, now_time_index)
        
        single_x = base_data_array[now_p_index, :, :-2] # seq * dim
        now_y = now_y

        return single_x, market_info_array, xn, now_y

    def get_data_by_index(self, index):
        now_p_index, now_time_index, start_time_index, no_trx_index, now_y, is_trx = self.all_data_index[index]
        if self.args.model_type == 'baseline': # mtand, node
            return self.get_data_baseline(now_p_index, now_time_index, start_time_index, no_trx_index, now_y, is_trx)
        elif self.args.model_type == 'cryptomixer':
            return self.get_data_iteract(now_p_index, now_time_index, start_time_index, no_trx_index, now_y, is_trx)
        elif self.args.model_type == 'grud':
            return self.get_data_grud(now_p_index, now_time_index, start_time_index, no_trx_index, now_y, is_trx)
        else:
            raise ValueError()

    def get_data_array(self):
        if self.use_co_trx_sim_mask:
            return self.all_data_index, self.all_data_array, self.no_trx_xn_list, self.sim_mask_array, self.index2hash, self.market_info_array
        else:
            return self.all_data_index, self.all_data_array, self.no_trx_xn_list, None, self.index2hash, self.market_info_array

    def __len__(self):
        return len(self.all_data_index)

    def __getitem__(self, index):
        return self.get_data_by_index(index)

    def get_people_num(self):
        return len(self.all_data_array)

    def get_user_hash_by_index(self, index):
        now_p_index, now_time_index, start_time_index, no_trx_index, now_y, is_trx = self.all_data_index[index]
        
        base_data_array = self.get_base_x(is_trx, start_time_index, now_time_index)
        combined_mask = base_data_array[:, :, -1]
        now_p_list_index = list(np.where(np.sum(combined_mask, axis = 1) != 0)[0])
        now_p_has_trx = False
        if now_p_index in now_p_list_index:
            now_p_list_index.remove(now_p_index)
            now_p_has_trx = True
            
        return self.index2hash[now_p_index], [self.index2hash[i] for i in now_p_list_index], now_p_has_trx

    def get_all_time_index(self):
        return sorted(list(self.all_time_list.keys()))

    def get_std(self):
        return self.std

def get_train_test_data(user_data, mean, std, start_time, end_time, args, market_info):
    split_rate = args.split_rate
    negetive_sample_rate = args.negetive_sample_rate
    split_by = args.data_split_by#'time' # 'data'
    model_type = args.model_type
    grud_data = True if args.model == 'grud' else False

    user_data_y = {}
    user_all_data_num = {}
    for i in tqdm(user_data):
        now_y = np.zeros([user_data[i]['data'].shape[0], 4])
        remain_index = []
        user_all_data_num[i] = 0
        for j in range(len(user_data[i]['data'])):
            if abs(user_data[i]['label'][j] - 0) < 0.1:
                now_y[j, 2] = 1
                if args.data_y_mode == 'trx_or_not':
                    if random.random() <= negetive_sample_rate:
                        remain_index.append(j)
            else:
                user_all_data_num[i] += 1
                now_y[j, 3] = 1
                if user_data[i]['data'][j, 1] > 0:
                    now_y[j, 0] = 1
                else:
                    now_y[j, 1] = 1
                remain_index.append(j)
        user_data_y[i] = now_y[remain_index]
        user_data[i]['data'] = user_data[i]['data'][remain_index]
        user_data[i]['label'] = user_data[i]['label'][remain_index]

    if args.use_drop_data:
        drop_list = []
        for i in tqdm(user_data):
            if (user_all_data_num[i] < args.drop_max_len):#(len(user_data[i]['data']) < args.drop_max_len):
                drop_list.append(i)
        for i in drop_list:
            del user_data[i]
            del user_data_y[i]
            del user_all_data_num[i]


    print(f'总数据量：{sum(user_all_data_num.values())}')
    print(f'用户量：{len(user_all_data_num)}')

    train_start_time = start_time

    sim_mask_gen = None
    
    if split_by == 'data':
        train_time = [-1, 1e20]
        temp_data = IrregularUserTransactionData(user_data, 
                                                     user_data_y,
                                                     mean,
                                                     std, 
                                                     train_time,
                                                     args,
                                                     data = None,
                                                     type_ = 'train',
                                                     data_y_mode = args.data_y_mode,
                                                     to_cuda = args.data_to_cuda,
                                                     device = args.device,
                                                     sim_mask_gen = sim_mask_gen, 
                                                     use_co_trx_sim_mask = args.use_co_trx_sim_mask,
                                                     train_start_time = train_start_time,
                                                     range_len = args.range_len,
                                                     seq_len = args.seq_len,
                                                     get_data_by_seq_len = args.get_data_by_seq_len,
                                                     model_type = model_type,
                                                     market_info = market_info,
                                                     grud_data = grud_data)
        std = temp_data.get_std()
        all_data_index, all_data_array, no_trx_xn_list, sim_mask_array, index2hash, market_info_array = temp_data.get_data_array()
        all_data_index_length = len(all_data_index)
        
        train_data_index = all_data_index[:int(all_data_index_length * (split_rate[0] / sum(split_rate)))]
        val_data_index = all_data_index[int(all_data_index_length * (split_rate[0] / sum(split_rate))):int(all_data_index_length * ((split_rate[0] + split_rate[1]) / sum(split_rate)))]
        test_data_index = all_data_index[int(all_data_index_length * ((split_rate[0] + split_rate[1]) / sum(split_rate))):]
        
        return [IrregularUserTransactionData(user_data, 
                                                     user_data_y,
                                                     mean,
                                                     std, 
                                                     train_time,
                                                     args,
                                                     use_co_trx_sim_mask = args.use_co_trx_sim_mask,
                                                     all_data_index = train_data_index,
                                                     all_data_array = all_data_array,
                                                     no_trx_xn_list = no_trx_xn_list,
                                                     sim_mask_array = sim_mask_array,
                                                     range_len = args.range_len,
                                                     model_type = model_type,
                                                     grud_data = grud_data,
                                                     market_info_array = market_info_array,
                                                     index2hash = index2hash),
                    IrregularUserTransactionData(user_data, 
                                                     user_data_y,
                                                     mean,
                                                     std, 
                                                     train_time,
                                                     args,
                                                     use_co_trx_sim_mask = args.use_co_trx_sim_mask,
                                                     all_data_index = val_data_index,
                                                     all_data_array = all_data_array,
                                                     no_trx_xn_list = no_trx_xn_list,
                                                     sim_mask_array = sim_mask_array,
                                                     range_len = args.range_len,
                                                     model_type = model_type,
                                                     grud_data = grud_data,
                                                     market_info_array = market_info_array,
                                                     index2hash = index2hash),
                    IrregularUserTransactionData(user_data, 
                                                     user_data_y,
                                                     mean,
                                                     std, 
                                                     train_time,
                                                     args,
                                                     use_co_trx_sim_mask = args.use_co_trx_sim_mask,
                                                     all_data_index = test_data_index,
                                                     all_data_array = all_data_array,
                                                     no_trx_xn_list = no_trx_xn_list,
                                                     sim_mask_array = sim_mask_array,
                                                     range_len = args.range_len,
                                                     model_type = model_type,
                                                     grud_data = grud_data,
                                                     market_info_array = market_info_array,
                                                     index2hash = index2hash),
                    list(np.where(std==0)[0])] 

    