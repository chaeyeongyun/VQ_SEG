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
from utils.visualize import make_example_img_salient, save_img
from utils.seg_tools import img_to_label
from utils.lr_schedulers import WarmUpPolyLR, CosineAnnealingLR


from data.dataset import BaseDataset, SalientDataset

from loss import make_loss
from measurement import Measurement

def salient_masking(salient_maps, pred):
    (bg_n, bg_y, bg_x) = (salient_maps<=0.1).nonzero(as_tuple=True) # 0.3이하인 곳의 인ㅔ스
    mask = torch.ones_like(pred, dtype=torch.float)
    mask[bg_n, 1, bg_y, bg_x] = 0 # 0.3 이하인 픽셀의 weed score을 0으로
    mask[bg_n, 2, bg_y, bg_x] = 0 # 0.3 이하인 픽셀의 crop score을 0으로
    return pred * mask

def salient_max(salient_maps, pred):
    (bg_n, bg_y, bg_x) = (salient_maps<=0.3).nonzero(as_tuple=True) # 0.3이하인 곳의 인ㅔ스
    mask = torch.zeros_like(pred, dtype=torch.float)
    mask[bg_n, 0, bg_y, bg_x] = 1
    return pred + mask

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
    cfg.model.params['activation'] = nn.Softmax
    model_1 = models.networks.make_model(cfg.model).to(device)
    model_2 = models.networks.make_model(cfg.model).to(device)
    
    # initialize differently (segmentation head)
    if cfg.train.init_weights:
        models.init_weight([model_1.decoder, model_1.segmentation_head], nn.init.kaiming_normal_,
                        nn.BatchNorm2d, cfg.train.bn_eps, cfg.train.bn_momentum, 
                        mode='fan_in', nonlinearity='relu')
        models.init_weight([model_2.decoder, model_2.segmentation_head], nn.init.kaiming_normal_,
                        nn.BatchNorm2d, cfg.train.bn_eps, cfg.train.bn_momentum, 
                        mode='fan_in', nonlinearity='relu')
    
    criterion = make_loss(cfg.train.criterion, num_classes, ignore_index=255)
    
    sup_dataset = SalientDataset(os.path.join(cfg.train.data_dir, 'train'), cfg.train.salient_dir, split='labelled', resize=cfg.resize)
    unsup_dataset = SalientDataset(os.path.join(cfg.train.data_dir, 'train'), cfg.train.salient_dir, split='unlabelled', resize=cfg.resize)
    
    sup_loader = DataLoader(sup_dataset, batch_size=batch_size, shuffle=False)
    unsup_loader = DataLoader(unsup_dataset, batch_size=batch_size, shuffle=False)
    
    
    if cfg.train.lr_scheduler.name == 'warmuppoly':
        lr_sched_cfg = cfg.train.lr_scheduler
        lr_scheduler = WarmUpPolyLR(cfg.train.learning_rate, lr_power=cfg.lr_sched_cfg.lr_power, 
                                    total_iters=len(unsup_loader)*num_epochs,
                                    warmup_steps=len(unsup_loader)*cfg.lr_sched_cfg.warmup_epoch)
    elif cfg.train.lr_scheduler.name == 'cosineannealing':
        lr_sched_cfg = cfg.train.lr_scheduler
        lr_scheduler = CosineAnnealingLR(start_lr = cfg.train.learning_rate, min_lr=lr_sched_cfg.min_lr, total_iters=len(unsup_loader)*num_epochs, warmup_steps=lr_sched_cfg.warmup_steps)
    
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.999))
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.999))
        
    
    # progress bar
    cps_loss_weight = cfg.train.cps_loss_weight
    total_commitment_loss_weight = cfg.train.total_commitment_loss_weight
    scaler = torch.cuda.amp.GradScaler(enabled=half)
    for epoch in range(num_epochs):
        trainloader = iter(zip(cycle(sup_loader), unsup_loader))
        crop_iou, weed_iou, back_iou = 0, 0, 0
        sum_cps_loss, sum_sup_loss_1, sum_sup_loss_2 = 0, 0, 0
        sum_commitment_loss = 0
        sum_loss = 0
        sum_miou = 0
        ep_start = time.time()
        pbar =  tqdm(range(len(unsup_loader)))
        for batch_idx in pbar:
            sup_dict, unsup_dict = next(trainloader)
            l_input, l_target = sup_dict['img'], sup_dict['target']
            l_salient = sup_dict['salient_map']
            l_target = img_to_label(l_target, cfg.pixel_to_label)
            ul_input = unsup_dict['img']
            ul_salient = unsup_dict['salient_map']
            ## predict in supervised manner ##
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            l_input = l_input.to(device)
            l_target = l_target.to(device)
            ul_input = ul_input.to(device)
     
            with torch.cuda.amp.autocast(enabled=half):
                pred_sup_1, commitment_loss_l1, code_usage_l1 = model_1(l_input)
                pred_sup_2, commitment_loss_l2, code_usage_l2 = model_2(l_input)
                pred_sup_1, pred_sup_2 = salient_max(l_salient, pred_sup_1), salient_max(l_salient, pred_sup_2)
                ## predict in unsupervised manner ##
                pred_ul_1, commitment_loss_ul1, code_usage_ul1 = model_1(ul_input)
                pred_ul_2, commitment_loss_ul2, code_usage_ul2 = model_2(ul_input)
                pred_ul_1, pred_ul_2 = salient_max(ul_salient, pred_ul_1), salient_max(ul_salient, pred_ul_2)
                if batch_idx == 0:
                    sum_code_usage = torch.zeros_like(code_usage_l1)
            ## cps loss ##
            pred_1 = torch.cat([pred_sup_1, pred_ul_1], dim=0)
            pred_2 = torch.cat([pred_sup_2, pred_ul_2], dim=0)
            
            # pseudo label
            pseudo_1 = torch.argmax(pred_1, dim=1).long()
            pseudo_2 = torch.argmax(pred_2, dim=1).long()
            
            
            with torch.cuda.amp.autocast(enabled=half):
                ## cps loss
                cps_loss = criterion(pred_1, pseudo_2) + criterion(pred_2, pseudo_1)
                ## supervised loss
                sup_loss_1 = criterion(pred_sup_1, l_target)
                sup_loss_2 = criterion(pred_sup_2, l_target)
                sup_loss = sup_loss_1 + sup_loss_2
                
                commitment_loss = commitment_loss_l1 + commitment_loss_l2 + commitment_loss_ul1 + commitment_loss_ul2
                
                ## learning rate update
                current_idx = epoch * len(unsup_loader) + batch_idx
                learning_rate = lr_scheduler.get_lr(current_idx)
                # update the learning rate
                optimizer_1.param_groups[0]['lr'] = learning_rate
                optimizer_2.param_groups[0]['lr'] = learning_rate
                
                loss = sup_loss + cps_loss_weight*cps_loss + total_commitment_loss_weight*commitment_loss
                sum_code_usage += (code_usage_l1 + code_usage_l2 + code_usage_ul1 + code_usage_ul2) / 4 
                
            scaler.scale(loss).backward()
            scaler.step(optimizer_1)
            scaler.step(optimizer_2)
            scaler.update()
            
            
            step_miou, iou_list = measurement.miou(measurement._make_confusion_matrix(pred_sup_1.detach().cpu().numpy(), l_target.detach().cpu().numpy()))
            sum_miou += step_miou
            sum_loss += loss.item()
            sum_cps_loss += cps_loss.item()
            sum_sup_loss_1 += sup_loss_1.item()
            sum_sup_loss_2 += sup_loss_2.item()
            sum_commitment_loss += commitment_loss.item()
            back_iou += iou_list[0]
            weed_iou += iou_list[1]
            crop_iou += iou_list[2]
            print_txt = f"[Epoch{epoch}/{cfg.train.num_epochs}][Iter{batch_idx+1}/{len(unsup_loader)}] lr={learning_rate:.5f}" \
                            + f"miou={step_miou}, sup_loss_1={sup_loss_1:.4f}, sup_loss_2={sup_loss_2:.4f}, cps_loss={cps_loss:.4f}"
            pbar.set_description(print_txt, refresh=False)
            if logger != None:
                log_txt.write(print_txt)
        
        ## end epoch ## 
        code_usage = (sum_code_usage / len(unsup_loader)).tolist()
        if isinstance(code_usage, float): code_usage = [code_usage]
        back_iou, weed_iou, crop_iou = back_iou / len(unsup_loader), weed_iou / len(unsup_loader), crop_iou / len(unsup_loader)
        cps_loss = sum_cps_loss / len(unsup_loader)
        sup_loss_1 = sum_sup_loss_1 / len(unsup_loader)
        sup_loss_2 = sum_sup_loss_2 / len(unsup_loader)
        commitment_loss = sum_commitment_loss / len(unsup_loader)
        loss = sum_loss / len(unsup_loader)
        miou = sum_miou / len(unsup_loader)
        
        print_txt = f"[Epoch{epoch}]" \
                            + f"miou={miou}, sup_loss_1={sup_loss_1:.4f}, sup_loss_2={sup_loss_2:.4f}, cps_loss={cps_loss:.4f}"
        print(print_txt)
        if logger != None:
            log_txt.write(print_txt)
            params = [l_input, l_target, pred_sup_1, ul_input, pred_ul_1, l_salient, ul_salient]
            params = [detach_numpy(i) for i in params]
            example = make_example_img_salient(*params)
            logger.image_update(example, f'{epoch}ep')
            if cfg.train.save_img:
                save_img(img_dir, f'output_{epoch}ep.png', example)
            if epoch % 10 == 0:
                save_ckpoints(model_1.state_dict(),
                            model_2.state_dict(),
                            epoch,
                            batch_idx,
                            optimizer_1.state_dict(),
                            optimizer_2.state_dict(),
                            os.path.join(ckpoints_dir, f"{epoch}ep.pth"))
            save_ckpoints(model_1.state_dict(),
                        model_2.state_dict(),
                        epoch,
                        batch_idx,
                        optimizer_1.state_dict(),
                        optimizer_2.state_dict(),
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
    parser.add_argument('--config_path', default='./config/cps_vqv2_kmeans_salient.json')
    opt = parser.parse_args()
    cfg = get_config_from_json(opt.config_path)
    # debug
    # cfg.resize=32
    # cfg.project_name = 'debug'
    # cfg.wandb_logging = False
    # cfg.train.half=False
    # cfg.resize = 256
    # train(cfg)
    cfg.project_name = 'VQUnet_kmenas_salient_max'
    cfg.train.criterion = "dice_loss"
    cfg.model.params.vq_cfg.num_embeddings = [0, 0, 512, 512, 512]
    train(cfg)