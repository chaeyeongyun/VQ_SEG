import optuna
from optuna import Trial, visualization
import math
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
from utils.seed import seed_everything

from data.dataset import BaseDataset

from loss import make_loss
from measurement import Measurement

import matplotlib.pyplot as plt

def test(test_loader, model, measurement:Measurement, cfg):
    sum_miou = 0
    for data in tqdm(test_loader):
        input_img, mask_img, filename = data['img'], data['target'], data['filename']
        input_img = input_img.to(model.device)
        mask_cpu = img_to_label(mask_img, cfg.pixel_to_label).cpu().numpy()
        model.eval()
        with torch.no_grad():
            pred = model(input_img)[0]
        miou, _ = measurement.miou(measurement._make_confusion_matrix(pred.detach().cpu().numpy(), mask_cpu))
        sum_miou += miou
    miou = sum_miou / len(test_loader)
    print(f'test miou : {miou}')
    return miou
        
def train(trial, cfg):
    seed_everything()
    cfg.train.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
    cfg.train.total_commitment_loss_weight = trial.suggest_float('total_commitment_loss_weight', 0.5, 2)
    cfg.train.total_prototype_loss_weight = trial.suggest_float('total_prototype_loss_weight', 0.2, 5)
    cfg.train.cps_loss_weight = trial.suggest_float('cps_loss_weight', 1, 2)
    logger_name = cfg.project_name+str(len(os.listdir(cfg.train.save_dir)))
    # if cfg.wandb_logging:
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
    
    sup_dataset = BaseDataset(os.path.join(cfg.train.data_dir, 'train'), split='labelled',  batch_size=batch_size, resize=cfg.resize)
    unsup_dataset = BaseDataset(os.path.join(cfg.train.data_dir, 'train'), split='unlabelled',  batch_size=batch_size, resize=cfg.resize)
    
    sup_loader = DataLoader(sup_dataset, batch_size=batch_size, shuffle=True)
    unsup_loader = DataLoader(unsup_dataset, batch_size=batch_size, shuffle=True)
    
    ##test##
    test_dataset = BaseDataset(os.path.join(cfg.test.data_dir, 'test'), split='labelled', batch_size=1, resize=cfg.resize)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    ###
    
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
    total_prototype_loss_weight = cfg.train.total_prototype_loss_weight
    scaler = torch.cuda.amp.GradScaler(enabled=half)
    for epoch in range(num_epochs):
        trainloader = iter(zip(cycle(sup_loader), unsup_loader))
        crop_iou, weed_iou, back_iou = 0, 0, 0
        sum_cps_loss, sum_sup_loss_1, sum_sup_loss_2 = 0, 0, 0
        sum_commitment_loss = 0
        sum_prototype_loss = 0
        sum_loss = 0
        sum_miou = 0
        ep_start = time.time()
        pbar =  tqdm(range(len(unsup_loader)))
        model_1.train()
        model_2.train()
        for batch_idx in pbar:
            sup_dict, unsup_dict = next(trainloader)
            l_input, l_target = sup_dict['img'], sup_dict['target']
            l_target = img_to_label(l_target, cfg.pixel_to_label)
            ul_input = unsup_dict['img']
            ## predict in supervised manner ##
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            l_input = l_input.to(device)
            l_target = l_target.to(device)
            ul_input = ul_input.to(device)

            pseudo_1 = model_1.pseudo_label(ul_input)
            pseudo_2 = model_2.pseudo_label(ul_input)
            with torch.cuda.amp.autocast(enabled=half):
                pred_sup_1, commitment_loss_l1, code_usage_l1, prototype_loss_l1 = model_1(l_input, l_target)
                pred_sup_2, commitment_loss_l2, code_usage_l2, prototype_loss_l2 = model_2(l_input, l_target)
                ## predict in unsupervised manner ##
                pred_ul_1, commitment_loss_ul1, code_usage_ul1, prototype_loss_ul1 = model_1(ul_input, pseudo_2)
                pred_ul_2, commitment_loss_ul2, code_usage_ul2, prototype_loss_ul2 = model_2(ul_input, pseudo_1)
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
                ## commitment_loss
                commitment_loss = commitment_loss_l1 + commitment_loss_l2 + commitment_loss_ul1 + commitment_loss_ul2
                commitment_loss *= total_commitment_loss_weight
                ## prototype_loss
                prototype_loss = prototype_loss_l1 + prototype_loss_l2 + prototype_loss_ul1 + prototype_loss_ul2
                prototype_loss *= total_prototype_loss_weight
                
                ## learning rate update
                current_idx = epoch * len(unsup_loader) + batch_idx
                learning_rate = lr_scheduler.get_lr(current_idx)
                # update the learning rate
                optimizer_1.param_groups[0]['lr'] = learning_rate
                optimizer_2.param_groups[0]['lr'] = learning_rate
                
                loss = sup_loss + cps_loss_weight*cps_loss + commitment_loss + prototype_loss
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
            sum_prototype_loss += prototype_loss.item()
            back_iou += iou_list[0]
            weed_iou += iou_list[1]
            crop_iou += iou_list[2]
            print_txt = f"[Epoch{epoch}/{cfg.train.num_epochs}][Iter{batch_idx+1}/{len(unsup_loader)}] lr={learning_rate:.5f}" \
                            + f"miou={step_miou}, sup_loss_1={sup_loss_1:.4f}, prototype_loss={prototype_loss.item():.4f}, cps_loss={cps_loss.item():.4f}, commitment_loss={commitment_loss.item():.4f}"
            
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
        prototype_loss = sum_prototype_loss / len(unsup_loader)
        loss = sum_loss / len(unsup_loader)
        miou = sum_miou / len(unsup_loader)
        
        print_txt = f"[Epoch{epoch}]" \
                            + f"miou={miou}, sup_loss_1={sup_loss_1:.4f}, prototype_loss={prototype_loss:.4f}, cps_loss={cps_loss:.4f}, commitment_loss={commitment_loss:.4f}"
        print(print_txt)
        test_miou = test(test_loader, model_1, measurement, cfg)
        if epoch > 50 and test_miou < 0.5:
            log_txt.close()
            if logger is not None:
                logger.finish() 
            return test_miou
        if cfg.train.save_img:
            log_txt.write(print_txt)
            params = [l_input, l_target, pred_sup_1, ul_input, pred_ul_1]
            params = [detach_numpy(i) for i in params]
            example = make_example_img(*params)
            example = make_example_img(*params)
            save_img(img_dir, f'output_{epoch}ep.png', example)
        
        if logger != None: 
            logger.image_update(example, f'{epoch}ep')
            for key in logger.config_dict.keys():
                logger.config_dict[key] = eval(key)
            for key in logger.log_dict.keys():
                if key=="code_usage":
                    logger.temp_update(list_to_separate_log(l=eval(key), name=key))
                else:logger.log_dict[key] = eval(key)
            logger.logging(epoch=epoch)
            logger.config_update()
            
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
            
            
        # if epoch>=3 and math.isnan(prototype_loss):
        #     log_txt.close()
        #     if logger is not None:logger.finish()
        #     return test_miou
    if logger != None: 
        log_txt.close()
        logger.finish()
    if cfg.train.save_as_tar:
        save_tar(save_dir)
    return test_miou

def main(cfg):
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
    logger_name = cfg.project_name+str(len(os.listdir(cfg.train.save_dir)))
    
    try:
        study.optimize(lambda trial: train(trial, cfg))
    except KeyboardInterrupt:
        print(f'Best trial : test_miou = {study.best_trial.value()}\nparams=study.best_trial.value()')
        df = study.trials_dataframe()
        importance_graph = visualization.plot_param_importance(study)
        optim_history = visualization.plot_optimization_history(study)
        save_dir = cfg.train.save_dir
        plt.imsave(os.path.join(save_dir, 'importance_graph.png'), importance_graph)
        plt.imsave(os.path.join(save_dir, 'importance_graph.png'), optim_history)
        df.to_csv(os.path.join(save_dir, 'params.csv'), sep=',', na_rep='NaN')
    else:
        print("No Exception")
        df = study.trials_dataframe()
        importance_graph = visualization.plot_param_importance(study)
        optim_history = visualization.plot_optimization_history(study)
        save_dir = cfg.train.save_dir
        plt.imsave(os.path.join(save_dir, 'importance_graph.png'), importance_graph)
        plt.imsave(os.path.join(save_dir, 'importance_graph.png'), optim_history)
        df.to_csv(os.path.join(save_dir, 'params.csv'), sep=',', na_rep='NaN')
    
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./config/vq_pt_unet.json')
    opt = parser.parse_args()
    cfg = get_config_from_json(opt.config_path)
    cfg.train.save_dir = '../drive/MyDrive/optuna_train/CWFID'
    cfg.project_name = 'Optuna_VQPTUnet'
    # debug
    # cfg.resize=64
    # cfg.project_name = 'debug'
    cfg.wandb_logging = False
    # cfg.train.half=False
    # cfg.resize = 256
    # train(cfg)
    main(cfg)