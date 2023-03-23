import torch 
import wandb
from typing import List
import os
import numpy as np
#TODO: 개선/ 추가
class Logger():
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
            
    def logging(self, epoch):
        if self.img_dict:
            wandb.log(dict(self.log_dict, **self.img_dict), step=epoch)
        else:
            wandb.log(self.log_dict, step=epoch)

    def config_update(self):
        wandb.config.update(self.config_dict, allow_val_change=True)
     
    def image_update(self, image:np.ndarray, caption:str):
        img = wandb.Image(image, mode='RGB', caption=caption)
        # wandb.log({'examples':img})
        self.img_dict = {'example':img}
    # def end(self, summary_dict):
    #     for key, value in summary_dict.items():
    #         wandb.run.summary[key] = value

