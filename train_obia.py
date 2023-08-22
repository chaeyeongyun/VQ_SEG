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
from test_detailviz import test as real_test

import models
from models.networks.fcn import FCN32s
from utils.logger import Logger, list_to_separate_log
from utils.ckpoints import  save_ckpoints, load_ckpoints, save_tar
from utils.load_config import get_config_from_json
from utils.device import device_setting
from utils.processing import detach_numpy
from utils.visualize import make_example_img, save_img
from utils.seg_tools import img_to_label
from utils.lr_schedulers import WarmUpPolyLR, CosineAnnealingLR
from utils.seed import seed_everything

from data.dataset import OBIADataset

from loss import make_loss
from measurement import Measurement
from warmup_scheduler import GradualWarmupScheduler

def test(test_loader, model, measurement:Measurement, cfg):
    sum_miou = 0
    for data in tqdm(test_loader):
        input_img, mask_img, filename = data['img'], data['target'], data['filename']
        input_img = input_img.to(list(model.parameters())[0].device)
        mask_cpu = img_to_label(mask_img, cfg.pixel_to_label).cpu().numpy()
        model.eval()
        with torch.no_grad():
            pred = model(input_img)[0]
        miou, _ = measurement.miou(measurement._make_confusion_matrix(pred.detach().cpu().numpy(), mask_cpu))
        sum_miou += miou
    miou = sum_miou / len(test_loader)
    print(f'test miou : {miou}')
    model.train()
    return miou
# 일단 no cutmix version
def train(cfg):
    seed_everything()
    if cfg.wandb_logging:
        root, percent = os.path.split(cfg.train.data_dir)
        root, dataset = os.path.split(root)
        logger_name = cfg.project_name+"_"+dataset+"_"+percent+"_"+str(len(os.listdir(cfg.train.save_dir)))
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
    
    model = FCN32s(**cfg.model.params).to(device)

    criterion = make_loss(cfg.train.criterion.name, num_classes)
    
    traindataset = OBIADataset(os.path.join(cfg.train.data_dir, 'train'), batch_size=batch_size, resize=cfg.resize)
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=False)
    ##test##
    testdataset = OBIADataset(os.path.join(cfg.test.data_dir, 'test'), batch_size=1, resize=cfg.resize)
    testloader = DataLoader(testdataset, batch_size=1, shuffle=False)
    ###
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.999))
    
    if cfg.train.lr_scheduler.name == 'warmuppoly':
        lr_sched_cfg = cfg.train.lr_scheduler
        lr_scheduler = WarmUpPolyLR(cfg.train.learning_rate, lr_power=cfg.lr_sched_cfg.lr_power, 
                                    total_iters=len(trainloader)*num_epochs,
                                    warmup_steps=len(trainloader)*cfg.lr_sched_cfg.warmup_epoch)
    elif cfg.train.lr_scheduler.name == 'cosineannealing':
        lr_sched_cfg = cfg.train.lr_scheduler
        lr_scheduler = CosineAnnealingLR(start_lr = cfg.train.learning_rate, min_lr=lr_sched_cfg.min_lr, total_iters=len(trainloader)*num_epochs, warmup_steps=lr_sched_cfg.warmup_steps)
    
    # progress bar
    scaler = torch.cuda.amp.GradScaler(enabled=half)
    best_miou = 0
    for epoch in range(num_epochs):
        # trainloader = iter(trainloader)
        model.train()
        crop_iou, weed_iou, back_iou = 0, 0, 0
        sum_loss = 0
        sum_miou = 0
        trainloader_iter = iter(trainloader)
        pbar =  tqdm(range(len(trainloader)))
        for batch_idx in pbar:
            sup_dict = trainloader_iter.next()
            l_input, l_target = sup_dict['img'], sup_dict['target']
            l_target = img_to_label(l_target, cfg.pixel_to_label)
            
            ## predict in supervised manner ##
            optimizer.zero_grad()
            
            l_input = l_input.to(device)
            l_target = l_target.to(device)
            with torch.cuda.amp.autocast(enabled=half):
                pred = model(l_input)

            with torch.cuda.amp.autocast(enabled=half):
                loss = criterion(pred, l_target)
                
                # ## learning rate update
                current_idx = epoch * len(trainloader) + batch_idx
                learning_rate = lr_scheduler.get_lr(current_idx)
                # update the learning rate
                optimizer.param_groups[0]['lr'] = learning_rate
                learning_rate = optimizer.param_groups[0]['lr']
                
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            
            step_miou, iou_list = measurement.miou(measurement._make_confusion_matrix(pred.detach().cpu().numpy(), l_target.detach().cpu().numpy()))
            sum_miou += step_miou
            sum_loss += loss.item()
           
            back_iou += iou_list[0]
            weed_iou += iou_list[1]
            crop_iou += iou_list[2]
            print_txt = f"[Epoch{epoch}/{cfg.train.num_epochs}][Iter{batch_idx+1}/{len(trainloader)}] lr={learning_rate:.5f}" \
                            + f"miou={step_miou:.5f}, loss={loss.item():.4f}"
            pbar.set_description(print_txt, refresh=False)
            if logger != None:
                log_txt.write(print_txt)
        
        ## end epoch ## 
        back_iou, weed_iou, crop_iou = back_iou / len(trainloader), weed_iou / len(trainloader), crop_iou / len(trainloader)
        loss = sum_loss / len(trainloader)
        miou = sum_miou / len(trainloader)
        
        print_txt = f"[Epoch{epoch}]" \
                            + f"miou={miou}, loss={loss:.4f}"
        print(print_txt)
        test_miou = test(testloader, model, measurement, cfg)
        print(f"test_miou : {test_miou:.4f}")
        if best_miou <= test_miou:
            best_miou = test_miou
            if logger is not None:
                torch.save(model.state_dict(), os.path.join(ckpoints_dir, f"best_test_miou.pth"))
        
        if logger != None:
            log_txt.write(print_txt)
            params = [l_input, l_target, pred]
            params = [detach_numpy(i) for i in params] + [None, None]
            example = make_example_img(*params)
            logger.image_update(example, f'{epoch}ep')
            if cfg.train.save_img:
                save_img(img_dir, f'output_{epoch}ep.png', example)
            if epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(ckpoints_dir, f"{epoch}ep.pth"))
            torch.save(model.state_dict(), os.path.join(ckpoints_dir, f"last.pth"))
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
    cfg.test.weights = os.path.join(ckpoints_dir, "best_test_miou.pth")
    real_test(cfg)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./config/obia_CWFID.json')
    opt = parser.parse_args()
    cfg = get_config_from_json(opt.config_path)
    # cfg.project_name = "debug"
    # cfg.wandb_logging = False
    # cfg.train.device = -1
    # cfg.resize = 64
    # cfg.half = False
    cfg.train.wandb_log.append('test_miou')
    train(cfg)
    cfg.train.data_dir = "../data/semi_sup_data/CWFID/percent_20"
    train(cfg)
    cfg.train.data_dir = "../data/semi_sup_data/CWFID/percent_10"
    train(cfg)
    
    
    cfg.train.data_dir = "../data/semi_sup_data/IJRR2017/percent_30"
    cfg.test.data_dir =  "../data/semi_sup_data/IJRR2017/percent_30"
    cfg.train.save_dir = "../drive/MyDrive/related_work/IJRR2017"
    cfg.test.save_dir =  "../drive/MyDrive/related_work/IJRR2017"
    train(cfg)
    cfg.train.data_dir = "../data/semi_sup_data/IJRR2017/percent_20"
    cfg.test.data_dir =  "../data/semi_sup_data/IJRR2017/percent_20"
    train(cfg)
    cfg.train.data_dir = "../data/semi_sup_data/IJRR2017/percent_10"
    cfg.test.data_dir =  "../data/semi_sup_data/IJRR2017/percent_10"
    train(cfg)
    
    cfg.train.data_dir = "../data/semi_sup_data/rice_s_n_w/percent_30"
    cfg.test.data_dir =  "../data/semi_sup_data/rice_s_n_w/percent_30"
    cfg.test.save_dir =  "../drive/MyDrive/related_work/rice_s_n_w"
    cfg.train.save_dir = "../drive/MyDrive/related_work/rice_s_n_w"
    train(cfg)
    cfg.train.data_dir = "../data/semi_sup_data/rice_s_n_w/percent_20"
    cfg.test.data_dir =  "../data/semi_sup_data/rice_s_n_w/percent_20"
    train(cfg)
    cfg.train.data_dir = "../data/semi_sup_data/rice_s_n_w/percent_10"
    cfg.test.data_dir =  "../data/semi_sup_data/rice_s_n_w/percent_10"
    train(cfg)