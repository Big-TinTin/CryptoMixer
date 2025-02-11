import torch
import numpy as np
import random
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from tqdm import tqdm
from itertools import chain
import matplotlib
import matplotlib.pyplot as plt
import pyecharts.options as opts
from pyecharts.charts import Line


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_html_file_name(path):
    try:
        now_num = max([int(i[:-5]) for i in os.listdir(path) if i[0] != '.'])
        now_num = now_num + 1
    except:
        now_num = 1
    return str(now_num) + '.html'

def drawline_save(y_list, path, file_name, model_name_list):
    colors = ['black', 'red', 'orange', 'green', 'blue', 'purple', 'deeppink', 'yellow', 'gold', 'steelblue', 'stategrey']
    line=(Line(init_opts=opts.InitOpts(width='720px', height='400px', bg_color='white')))
    line.add_xaxis(xaxis_data=[i for i in range(max([len(j) for j in y_list]))])
    count = 0
    for y in y_list:
    
        line.add_yaxis(series_name=model_name_list[count],y_axis=y,symbol="circle",
                is_symbol_show=True,is_smooth=False,
                itemstyle_opts={"color": colors[count]},symbol_size=5)
        #plt.plot(x_list, y, marker='*', color='b', label='y2-data')
        count += 1
    
    line.set_series_opts(
            label_opts=opts.LabelOpts(is_show=False),
        )
    line.set_global_opts(title_opts=opts.TitleOpts(title="loss收敛图"),
                         xaxis_opts=opts.AxisOpts(name="epochs",name_rotate=0,
                                name_textstyle_opts={"color": "black","fontSize":15},
                                axislabel_opts={"rotate":0,"color":"blue","fontSize":12},
                                is_show = True,is_inverse = False, 
                                name_location = 'center',name_gap = 45),
                         yaxis_opts=opts.AxisOpts(name="loss", 
                                name_textstyle_opts={"color": "black","fontSize":15},
                                axislabel_opts={"rotate": 0,"color":"blue","fontSize":12},
                                is_show = True, name_location = 'center',name_gap = 35, split_number = 6))
    
    line.render(path + '/' + file_name)

class LossFigure:
    def __init__(self, path):
        self.path = path

        self.model_loss = {}
        self.now_model = None

    def add_model(self, model_name):
        self.now_model = model_name
        self.model_loss[model_name] = []

    def add_loss(self, loss_value, model_name = None):
        if model_name is None:
            model_name = self.now_model
        self.model_loss[model_name].append(loss_value)

    def draw_model(self, model_name = None):
        if model_name is None:
            model_name = self.now_model
        file_name = find_html_file_name(self.path)
        drawline_save([self.model_loss[model_name]], self.path, file_name, [model_name])
        return self.path + '/' + file_name

    def draw_all(self):
        file_name = find_html_file_name(self.path)
        model_name_list = list(self.model_loss.keys())
        y_list = [self.model_loss[i] for i in model_name_list]
        drawline_save(y_list, self.path, file_name, model_name_list)
        return self.path + '/' + file_name

class EarlyStopAndSave:
    def __init__(self, monitor, method, patient, model, save_path, model_name, printf = True, overwrite = False, distributin_shift_min = None):
        self.monitor = monitor
        self.method = method
        self.patient = patient
        
        self.model = model
        self.save_path = save_path
        self.model_name = model_name
        
        self.early_stop_metrics = None
        self.early_stop_patient = 0
        self.early_best_epoch = None
        
        self.printf = printf
        self.overwrite = overwrite

        self.distributin_shift_min = distributin_shift_min
        
        os.makedirs(save_path, exist_ok=True)
        
    def save(self, epoch):
        path = self.save_path
        
        if self.overwrite:
            if self.model_name is not None:
                path += f'/{self.model_name}.h5'
            else:
                path += f'/temp.h5'
        else:
            if self.model_name is not None:
                path += f'/{self.model_name}{epoch}.h5'
            else:
                path += f'/{epoch}.h5'
          
        torch.save(self.model.state_dict(), path)
    
    def restore(self):
        path = self.save_path
        
        if self.overwrite:
            if self.model_name is not None:
                path += f'/{self.model_name}.h5'
            else:
                path += f'/temp.h5'
        else:
            if self.model_name is not None:
                path += f'/{self.model_name}{self.early_best_epoch}.h5'
            else:
                path += f'/{self.early_best_epoch}.h5'

        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        print(f'restore from {self.early_best_epoch} epoch')

    def __call__(self, metrics, epoch):
        
        now_metric = [float(metrics[self.monitor])]
        if self.distributin_shift_min is not None:
            for i in self.distributin_shift_min:
                now_metric += [float(metrics[i])]
            now_metric = min(now_metric)
        print(now_metric)
        
        if self.early_stop_metrics is not None:
            if self.method == 'min':
                if now_metric < self.early_stop_metrics:
                    self.early_stop_metrics = now_metric
                    self.early_stop_patient = 0
                    self.early_best_epoch = epoch
                    self.save(epoch)
                else:
                    self.early_stop_patient+=1
            elif self.method == 'max':
                if now_metric > self.early_stop_metrics:
                    self.early_stop_metrics = now_metric
                    self.early_stop_patient = 0
                    self.early_best_epoch = epoch
                    self.save(epoch)
                else:
                    self.early_stop_patient+=1
                    
            if self.early_stop_patient >= self.patient:
                if self.printf:
                    print(f'训练结束, 最佳epoch为{self.early_best_epoch+1}')
                    print(f'文件名为{self.early_best_epoch}.hs')
                self.restore()
                return True
            
        else:
            self.early_stop_metrics = now_metric
            self.early_best_epoch = epoch
            self.save(epoch)
        if self.printf:
            print(f'目前最佳epoch为{self.early_best_epoch+1}')
        return False