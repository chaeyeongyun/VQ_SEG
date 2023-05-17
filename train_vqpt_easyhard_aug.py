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
from utils.lr_schedulers import get_scheduler
from utils.seed import seed_everything
from utils.train_tools import make_optim_paramgroup

from data.dataset import BaseDataset
from data.augmentations import CutMix, similarity_transform, inverse_similarity_transform

from loss import make_loss
from measurement import Measurement

def train(cfg):
    seed_everything()
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
        logger = Logger(cfg, logger_name) 
    else: logger = None
    
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
    # criterion = make_loss(cfg.train.criterion, num_classes, ignore_index=255)
    ####### experiment #######
    criterion = nn.CrossEntropyLoss(torch.Tensor([0.5, 1., 1.]).to(device))
    sup_dataset = BaseDataset(os.path.join(cfg.train.data_dir, 'train'), split='labelled',  batch_size=batch_size, resize=cfg.resize)
    unsup_dataset = BaseDataset(os.path.join(cfg.train.data_dir, 'train'), split='unlabelled',  batch_size=batch_size, resize=cfg.resize)
    
    sup_loader = DataLoader(sup_dataset, batch_size=batch_size, shuffle=True)
    unsup_loader = DataLoader(unsup_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.999))
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.999))
    decoder_lr_times = cfg.train.get("decoder_lr_times", False)
    if decoder_lr_times:
        param_list_1 = make_optim_paramgroup(model_1, lr=cfg.train.learning_rate, decoder_lr_times=decoder_lr_times)
        param_list_2 = make_optim_paramgroup(model_2, lr=cfg.train.learning_rate, decoder_lr_times=decoder_lr_times)
        optimizer_1 = torch.optim.Adam(param_list_1, betas=(0.9, 0.999))
        optimizer_2 = torch.optim.Adam(param_list_2, betas=(0.9, 0.999))
        
    else:
        optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.999))
        optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.999))
    
    lr_scheduler_1 = get_scheduler(cfg.train.lr_scheduler, len(unsup_loader), num_epochs, optimizer_1)
    lr_scheduler_2 = get_scheduler(cfg.train.lr_scheduler, len(unsup_loader), num_epochs, optimizer_2)
    
    hard_aug = CutMix(ratio=cfg.train.cutmix_ratio)
    easy_aug = similarity_transform
    # progress bar
    cps_loss_weight = cfg.train.cps_loss_weight
    total_commitment_loss_weight = cfg.train.total_commitment_loss_weight
    total_prototype_loss_weight = cfg.train.total_prototype_loss_weight
    scaler = torch.cuda.amp.GradScaler(enabled=half)
    for epoch in range(num_epochs):
        trainloader = iter(zip(cycle(sup_loader), unsup_loader))
        crop_iou, weed_iou, back_iou = 0, 0, 0
        sum_cps_loss, sum_commitment_loss, sum_prototype_loss = 0, 0, 0
        sum_sup_loss, sum_unsup_loss = 0, 0
        sum_loss = 0
        sum_miou = 0
        ep_start = time.time()
        pbar =  tqdm(range(len(unsup_loader)))
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
            # TODO: easy aug, hard aug for unlabeled data
            easy_ul, aug, angle = easy_aug(ul_input)
            hard_ul, mask = hard_aug(ul_input)
            pseudo_1 = hard_aug(inverse_similarity_transform(model_1.pseudo_label(easy_ul), aug, angle), mask=mask)[0].detach()
            pseudo_2 = hard_aug(inverse_similarity_transform(model_2.pseudo_label(easy_ul), aug, angle), mask=mask)[0].detach()
            with torch.cuda.amp.autocast(enabled=half):
                pred_sup_1, commitment_loss_l1, code_usage_l1, prototype_loss_l1 = model_1(l_input, l_target)
                pred_sup_2, commitment_loss_l2, code_usage_l2, prototype_loss_l2 = model_2(l_input, l_target)
                ## predict in unsupervised manner ##
                pred_ul_1, commitment_loss_ul1, code_usage_ul1, prototype_loss_ul1 = model_1(hard_ul, pseudo_2)
                pred_ul_2, commitment_loss_ul2, code_usage_ul2, prototype_loss_ul2 = model_2(hard_ul, pseudo_1)
                if batch_idx == 0:
                    sum_code_usage = torch.zeros_like(code_usage_l1)
        
            # pseudo label
            sup_pseudo_1 = torch.argmax(pred_sup_1, dim=1).long()
            sup_pseudo_2 = torch.argmax(pred_sup_2, dim=1).long()
            
            
            with torch.cuda.amp.autocast(enabled=half):
                ## cps loss
                sup_cps_loss = criterion(pred_sup_1, sup_pseudo_2) + criterion(pred_sup_2, sup_pseudo_1)
                unsup_cps_loss = criterion(pred_ul_1, pseudo_2) + criterion(pred_ul_2, pseudo_1)
                cps_loss = sup_cps_loss + unsup_cps_loss
                cps_loss *= cps_loss_weight
                ## supervised loss
                sup_loss = criterion(pred_sup_1, l_target) + criterion(pred_sup_2, l_target)
                
                ## commitment_loss
                sup_commit_loss = commitment_loss_l1 + commitment_loss_l2
                unsup_commit_loss =  commitment_loss_ul1 + commitment_loss_ul2
                commitment_loss =  sup_commit_loss + unsup_commit_loss
                commitment_loss *= total_commitment_loss_weight
                ## prototype_loss
                sup_prototype_loss = prototype_loss_l1 + prototype_loss_l2
                unsup_prototype_loss = prototype_loss_ul1 + prototype_loss_ul2
                prototype_loss = sup_prototype_loss + unsup_prototype_loss
                prototype_loss *= total_prototype_loss_weight
                
                ## learning rate update
                learning_rate = lr_scheduler_1.get_lr()[0]
                lr_scheduler_1.step()
                lr_scheduler_2.step()
                
                loss = sup_loss + cps_loss + commitment_loss + prototype_loss
                sum_code_usage += (code_usage_l1 + code_usage_l2 + code_usage_ul1 + code_usage_ul2) / 4 
                
            scaler.scale(loss).backward()
            scaler.step(optimizer_1)
            scaler.step(optimizer_2)
            scaler.update()
            
            
            step_miou, iou_list = measurement.miou(measurement._make_confusion_matrix(pred_sup_1.detach().cpu().numpy(), l_target.detach().cpu().numpy()))
            sum_miou += step_miou
            sum_loss += loss.item()
            sum_cps_loss += cps_loss.item()
            sum_sup_loss += (sup_cps_loss+sup_commit_loss+sup_prototype_loss+sup_loss).item()
            sum_unsup_loss += (unsup_cps_loss+unsup_commit_loss+unsup_prototype_loss).item()
            sum_commitment_loss += commitment_loss.item()
            sum_prototype_loss += prototype_loss.item()
            back_iou += iou_list[0]
            weed_iou += iou_list[1]
            crop_iou += iou_list[2]
            print_txt = f"[Epoch{epoch}/{cfg.train.num_epochs}][Iter{batch_idx+1}/{len(unsup_loader)}] lr={learning_rate:.5f}" \
                            + f"miou={step_miou}, sup_loss ={sup_loss:.4f}, prototype_loss={prototype_loss.item():.4f}, cps_loss={cps_loss.item():.4f}, commitment_loss={commitment_loss.item():.4f}"
            pbar.set_description(print_txt, refresh=False)
            if logger != None:
                log_txt.write(print_txt)
        
        ## end epoch ## 
        code_usage = (sum_code_usage / len(unsup_loader)).tolist()
        if isinstance(code_usage, float): code_usage = [code_usage]
        back_iou, weed_iou, crop_iou = back_iou / len(unsup_loader), weed_iou / len(unsup_loader), crop_iou / len(unsup_loader)
        cps_loss = sum_cps_loss / len(unsup_loader)
        sup_loss = sum_sup_loss / len(unsup_loader)
        unsup_loss = sum_unsup_loss / len(unsup_loader)
        commitment_loss = sum_commitment_loss / len(unsup_loader)
        prototype_loss = sum_prototype_loss / len(unsup_loader)
        loss = sum_loss / len(unsup_loader)
        miou = sum_miou / len(unsup_loader)
        
        print_txt = f"[Epoch{epoch}]" \
                            + f"miou={miou}, sup_loss{sup_loss:.4f}, unsup_loss{unsup_loss:.4f}, prototype_loss={prototype_loss:.4f}, cps_loss={cps_loss:.4f}, commitment_loss={commitment_loss:.4f}"
        print(print_txt)
        if logger != None:
            log_txt.write(print_txt)
            params = [l_input, l_target, pred_sup_1, hard_ul, pred_ul_1]
            params = [detach_numpy(i) for i in params]
            example = make_example_img(*params)
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
    parser.add_argument('--config_path', default='./config/vq_pt_unet_easyhard.json')
    opt = parser.parse_args()
    cfg = get_config_from_json(opt.config_path)
    # debug
    # cfg.resize=512
    # cfg.project_name = 'debug'
    # cfg.wandb_logging = False
    # cfg.train.half=False
    cfg.resize = 448
    # train(cfg)
    # cfg.train.decoder_lr_times = 10
    # train(cfg)
    cfg.train.criterion = 'cross_entropy'
    train(cfg)