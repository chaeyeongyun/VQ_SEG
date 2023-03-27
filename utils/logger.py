import torch 
import wandb
from typing import List
import os
import numpy as np
#TODO: 개선/ 추가
class BaseLogger():
    def __init__(self, cfg, logger_name):
        wandb.init(project=cfg.project_name,
                   name=logger_name
                #    notes="baseline",
                #    tags = ["csp+unet+cutmix"]
        )
        wandb.config.update(cfg)
        # config setting
        self.config_dict = dict()
       
        # initialize log dict
        self.log_dict = dict()
        
        self.img_dict = None
        self.table_dict = None
        self.temp_dict = None
            
    def logging(self, epoch):
        log_dict = self.log_dict.copy()
        if self.img_dict:
            log_dict = dict(self.log_dict, **self.img_dict) 
            
        if self.table_dict:
            log_dict = dict(self.log_dict, **self.table_dict) 
        
        if self.temp_dict:
            log_dict = dict(self.log_dict, **self.temp_dict)
            
        wandb.log(log_dict, step=epoch)

    def config_update(self):
        wandb.config.update(self.config_dict, allow_val_change=True)
     
    def image_update(self, image:np.ndarray, caption:str):
        img = wandb.Image(image, mode='RGB', caption=caption)
        # wandb.log({'examples':img})
        self.img_dict = {'example':img}
    
    def table_update(self, name:str, columns:List, data:List):
        table = wandb.Table(columns=columns, data=data)
        self.table_dict = {name: table}
    
    def temp_update(self, d:dict):
        self.temp_dict = d
        
class Logger(BaseLogger):
    def __init__(self, cfg, logger_name):
        wandb.init(project=cfg.project_name,
                   name=logger_name
                #    notes="baseline",
                #    tags = ["csp+unet+cutmix"]
        )
        # config setting
        self.config_dict = dict()
        for key in cfg.train.wandb_config:
            self.config_dict[key] = cfg.train[key]
        
        for i in cfg.train.wandb_metrics:
            if i in ["loss"]:
                wandb.define_metric(i, summary='min')
            if i in ["miou"]:
                wandb.define_metric(i, summary='max')
            if i in ["perplexity"]:
                wandb.define_metric(i, summary='max')
        
        # initialize log dict
        self.log_dict = dict()
        for key in cfg.train.wandb_log:
            self.log_dict[key] = None
        
        self.img_dict = None
        self.table_dict = None
        self.temp_dict = None
            
class TestLogger(BaseLogger):
    def __init__(self, cfg, logger_name):
        wandb.init(project=cfg.project_name,
                   name=logger_name
                #    notes="baseline",
                #    tags = ["csp+unet+cutmix"]
        )
         # config setting
        self.config_dict = dict()
       
        # initialize log dict
        self.log_dict = dict()
        
        self.img_dict = None
        self.table_dict = None
        self.temp_dict = None
    
    def logging(self):
        log_dict = self.log_dict.copy()
        if self.img_dict:
            log_dict = dict(self.log_dict, **self.img_dict) 
            
        if self.table_dict:
            log_dict = dict(self.log_dict, **self.table_dict) 
        
        if self.temp_dict:
            log_dict = dict(self.log_dict, **self.temp_dict)
            
        wandb.log(log_dict)

def list_to_separate_log(l:List, name):
    output = dict()
    for i, item in enumerate(l):
        output[f'{name}_{i}'] = item
    return output

def dict_to_table_log(d:List):
    columns = []
    data = []
    for key in d.keys():
        columns.append(key)
        data.append(d[key])
    return wandb.Table(data=[data], columns=columns)

def make_img_table(filename_list, img_list_1, img_list_2, columns):
    data = []
    for filename, img_1, img_2 in zip(filename_list, img_list_1, img_list_2):
        data.append([filename, wandb.Image(img_1, mode='RGB'), wandb.Image(img_2, mode='RGB')])
    return wandb.Table(data=data, columns=columns)