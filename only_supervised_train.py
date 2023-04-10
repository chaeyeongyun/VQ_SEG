import argparse
import matplotlib.pyplot as plt
import os
from itertools import cycle
from tqdm import tqdm
import time
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models
from utils.logger import Logger, list_to_separate_log
from utils.ckpoints import  save_ckpoints, load_ckpoints, save_tar
from utils.load_config import get_config_from_json
from utils.device import device_setting
from utils.processing import detach_numpy
from utils.visualize import make_example_img, save_img
from utils.seg_tools import img_to_label
from utils.lr_schedulers import WarmUpPolyLR, CosineAnnealingLR


from data.dataset import BaseDataset

from loss import make_loss
from measurement import Measurement

# 일단 no cutmix version
def train(cfg):
    if cfg.wandb_logging:
        logger_name = cfg.project_name+str(len(os.listdir(cfg.train.save_dir)))
        save_dir = os.path.join(cfg.train.save_dir, logger_name)
        os.makedirs(save_dir)
        ckpoints_dir = os.path.join(save_dir, 'ckpoints')
        os.mkdir(ckpoints_dir)
        if cfg.train.save_img:
            img_dir = os.path.join(save_dir, 'imgs')
            os.mkdir(img_dir)
        log_txt = open(os.path.join(save_dir, 'log_txt'), 'w')
    logger = Logger(cfg, logger_name) if cfg.wandb_logging else None
    
    half=cfg.train.half
    
    num_classes = cfg.num_classes
    batch_size = cfg.train.batch_size
    num_epochs = cfg.train.num_epochs
    device = device_setting(cfg.train.device)
    measurement = Measurement(num_classes)
    
    model_1 = models.networks.make_model(cfg.model).to(device)
    
    
    # initialize differently (segmentation head)
    if cfg.train.init_weights:
        models.init_weight([model_1.decoder, model_1.segmentation_head], nn.init.kaiming_normal_,
                        nn.BatchNorm2d, cfg.train.bn_eps, cfg.train.bn_momentum, 
                        mode='fan_in', nonlinearity='relu')
    
    criterion = make_loss(cfg.train.criterion, num_classes, ignore_index=255)
    
    sup_dataset = BaseDataset(os.path.join(cfg.train.data_dir, 'train'), split='labelled', resize=cfg.resize)
    
    sup_loader = DataLoader(sup_dataset, batch_size=batch_size, shuffle=True)
    
    
    if cfg.train.lr_scheduler.name == 'warmuppoly':
        lr_sched_cfg = cfg.train.lr_scheduler
        lr_scheduler = WarmUpPolyLR(cfg.train.learning_rate, lr_power=cfg.lr_sched_cfg.lr_power, 
                                    total_iters=len(sup_loader)*num_epochs,
                                    warmup_steps=len(sup_loader)*cfg.lr_sched_cfg.warmup_epoch)
    elif cfg.train.lr_scheduler.name == 'cosineannealing':
        lr_sched_cfg = cfg.train.lr_scheduler
        lr_scheduler = CosineAnnealingLR(start_lr = cfg.train.learning_rate, min_lr=lr_sched_cfg.min_lr, total_iters=len(sup_loader)*num_epochs, warmup_steps=lr_sched_cfg.warmup_steps)
    
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.999))
    
        
    
    # progress bar
    cps_loss_weight = cfg.train.cps_loss_weight
    total_commitment_loss_weight = cfg.train.total_commitment_loss_weight
    scaler = torch.cuda.amp.GradScaler(enabled=half)
    for epoch in range(num_epochs):
        trainloader = iter(sup_loader)
        crop_iou, weed_iou, back_iou = 0, 0, 0
        sum_sup_loss = 0
        sum_commitment_loss = 0
        sum_loss = 0
        sum_miou = 0
        ep_start = time.time()
        pbar =  tqdm(range(len(sup_loader)))
        for batch_idx in pbar:
            sup_dict = next(trainloader)
            l_input, l_target = sup_dict['img'], sup_dict['target']
            l_target = img_to_label(l_target, cfg.pixel_to_label)
        
            ## predict in supervised manner ##
            optimizer_1.zero_grad()
        
            l_input = l_input.to(device)
            l_target = l_target.to(device)
        
     
            with torch.cuda.amp.autocast(enabled=half):
                pred_sup_1, commitment_loss_l1, code_usage_l1 = model_1(l_input)
                if batch_idx == 0:
                    sum_code_usage = torch.zeros_like(code_usage_l1)
            
            
            with torch.cuda.amp.autocast(enabled=half):
                ## supervised loss
                sup_loss = criterion(pred_sup_1, l_target)
                commitment_loss = commitment_loss_l1
                
                ## learning rate update
                current_idx = epoch * len(sup_loader) + batch_idx
                learning_rate = lr_scheduler.get_lr(current_idx)
                # update the learning rate
                optimizer_1.param_groups[0]['lr'] = learning_rate
                
                loss = sup_loss + total_commitment_loss_weight*commitment_loss
                sum_code_usage += code_usage_l1 
                
            scaler.scale(loss).backward()
            scaler.step(optimizer_1)
            
            scaler.update()
            
            
            step_miou, iou_list = measurement.miou(measurement._make_confusion_matrix(pred_sup_1.detach().cpu().numpy(), l_target.detach().cpu().numpy()))
            sum_miou += step_miou
            sum_loss += loss.item()
            sum_sup_loss += sup_loss.item()
            sum_commitment_loss += commitment_loss.item()
            back_iou += iou_list[0]
            weed_iou += iou_list[1]
            crop_iou += iou_list[2]
            print_txt = f"[Epoch{epoch}/{cfg.train.num_epochs}][Iter{batch_idx+1}/{len(sup_loader)}] lr={learning_rate:.5f}" \
                            + f"miou={step_miou}, sup_loss_1={sup_loss.item():.4f}, commitment_loss={commitment_loss.item():.4f}, loss={loss.item():.4f}"
            pbar.set_description(print_txt, refresh=False)
            if logger != None:
                log_txt.write(print_txt)
        
        ## end epoch ## 
        code_usage = (sum_code_usage / len(sup_loader)).tolist()
        if isinstance(code_usage, float): code_usage = [code_usage]
        back_iou, weed_iou, crop_iou = back_iou / len(sup_loader), weed_iou / len(sup_loader), crop_iou / len(sup_loader)
        
        sup_loss = sum_sup_loss / len(sup_loader)
        
        commitment_loss = sum_commitment_loss / len(sup_loader)
        loss = sum_loss / len(sup_loader)
        miou = sum_miou / len(sup_loader)
        
        print_txt = f"[Epoch{epoch}]" \
                            + f"miou={miou}, sup_loss={sup_loss:.4f}, commitment_loss={commitment_loss:.4f}, loss={loss:.4f}"
        print(print_txt)
        if logger != None:
            log_txt.write(print_txt)
            params = [l_input, l_target, pred_sup_1, None, None]
            params = [detach_numpy(i) if i is not None else i for i in params ]
            example = make_example_img(*params)
            logger.image_update(example, f'{epoch}ep')
            if cfg.train.save_img:
                save_img(img_dir, f'output_{epoch}ep.png', example)
            if epoch % 10 == 0:
                torch.save(model_1.state_dict(),
                            os.path.join(ckpoints_dir, f"{epoch}ep.pth"))
            torch.save(model_1.state_dict(),
                        os.path.join(ckpoints_dir, f"last.pth"))
            # wandb logging
            for key in logger.config_dict.keys():
                logger.config_dict[key] = eval(key)
            for key in logger.log_dict.keys():
                if key=="code_usage":
                    logger.temp_update(list_to_separate_log(l=eval(key), name=key))
                else:logger.log_dict[key] = eval(key)
            
            logger.logging(epoch=epoch)
            logger.config_update()
    if logger != None: 
        log_txt.close()
        logger.finish()
    if cfg.train.save_as_tar:
        save_tar(save_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./config/only_sup_kmeans.json')
    opt = parser.parse_args()
    cfg = get_config_from_json(opt.config_path)
    # debug
    # cfg.resize=32
    # cfg.project_name = 'debug'
    # cfg.wandb_logging = False
    # cfg.train.half=False
    # cfg.resize = 256
    # train(cfg)
    cfg.train.criterion = "dice_loss"
    cfg.model.params.vq_cfg.num_embeddings = [0, 0, 512, 512, 512]
    train(cfg)
    cfg.model.params.vq_cfg.num_embeddings = [0, 0, 2048, 2048, 2048]
    train(cfg)