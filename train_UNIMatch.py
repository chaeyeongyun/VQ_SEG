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
from models.networks.deeplabv3 import UniMatch
from utils.logger import Logger, list_to_separate_log
from utils.ckpoints import  save_ckpoints, load_ckpoints, save_tar
from utils.load_config import get_config_from_json
from utils.device import device_setting
from utils.processing import detach_numpy
from utils.visualize import make_example_img, save_img
from utils.seg_tools import img_to_label
from utils.lr_schedulers import WarmUpPolyLR, CosineAnnealingLR
from utils.seed import seed_everything
from test_detailviz import test as real_test
from data.dataset import BaseDataset
from data.augmentations import make_cutout_mask
from loss import make_loss
from measurement import Measurement

# 일단 no cutmix version
def score_mask(pred, pseudo, th=0.95):
    pred_prob = torch.softmax(pred, dim=1)
    pred_max = pred_prob.max(dim=1)[0]
    return torch.where(pred_max > th, pseudo, 255)

def cutmix(input:torch.Tensor, label:torch.Tensor, ratio:float):
    batch_size = input.shape[0]
    input_aug, label_aug = [], []
    for i in range(batch_size):
        mask = make_cutout_mask(input.shape[-2:], ratio).to(input.device)
        input_aug.append((input[i]*mask + input[(i+1)%batch_size]*(1-mask)).unsqueeze(0))
        if label is not None:
            label_aug.append((label[i]*mask + label[(i+1)%batch_size]*(1-mask)).unsqueeze(0)) 
    input_aug = torch.cat(input_aug, dim=0)
    if label is not None:
        label_aug = torch.cat(label_aug, dim=0)
    return input_aug, label_aug

class Cutmix():
    def __init__(self, ratio=0.5) :
        self.ratio = ratio
    def __call__(self, input, label):
        input_aug, label_aug = cutmix(input, label, self.ratio)
        return input_aug, label_aug


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
    return miou
# 일단 no cutmix version
def score_mask(pred, pseudo, th=0.7):
    pred_prob = torch.softmax(pred, dim=1)
    pred_max = pred_prob.max(dim=1)[0]
    return torch.where(pred_max > th, pseudo, 255)

def train(cfg):
    seed_everything()
    if cfg.wandb_logging:
        dataset, manner = cfg.train.data_dir.split("/")[-2:]
        logger_name = cfg.project_name+f"_{dataset}_{manner}_"+str(len(os.listdir(cfg.train.save_dir)))
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
    
    model = UniMatch(
        encoder_name="resnet50",
        num_classes=3,
        encoder_weights="imagenet")
    hard_aug = Cutmix()
    # initialize differently (segmentation head)
    if cfg.train.init_weights:
        models.init_weight([model.decoder, model.segmentation_head], nn.init.kaiming_normal_,
                        nn.BatchNorm2d, cfg.train.bn_eps, cfg.train.bn_momentum, 
                        mode='fan_in', nonlinearity='relu')
    loss_weight = cfg.train.criterion.get("weight", None)
    loss_weight = torch.tensor(loss_weight) if loss_weight is not None else loss_weight
    ce_loss = nn.CrossEntropyLoss(weight=loss_weight, ignore_index=255)
    
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
    
    optimizer= torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.999))
    
    # progress bar
    cps_loss_weight = 1
    scaler = torch.cuda.amp.GradScaler(enabled=half)
    confidence_threshold = 0.95
    best_miou = 0
    for epoch in range(num_epochs):
        trainloader = iter(zip(cycle(sup_loader), unsup_loader))
        crop_iou, weed_iou, back_iou = 0, 0, 0
        sum_cps_loss, sum_sup_loss_1, sum_sup_loss_2 = 0, 0, 0
        sum_loss = 0
        sum_miou = 0
        ep_start = time.time()
        pbar =  tqdm(range(len(unsup_loader)))
        model.train()
        for batch_idx in pbar:
            sup_dict, unsup_dict = next(trainloader)
            l_input, l_target = sup_dict['img'], sup_dict['target']
            l_target = img_to_label(l_target, cfg.pixel_to_label)
            ul_input = unsup_dict['img']
            
            with torch.no_grad():
                model.eval()
                pred_u = model(ul_input)[0].detach()
            ul_mix_input_1, ul_mix_pred_1 = hard_aug(ul_input, pred_u)
            ul_mix_input_2, ul_mix_pred_2 = hard_aug(ul_input, pred_u)
            model.train()
            ## predict in supervised manner ##
            optimizer.zero_grad()
            l_input = l_input.to(device)
            l_target = l_target.to(device)
            ul_input = ul_input.to(device)
            ul_mix_input_1 = ul_mix_input_1.to(device)
            ul_mix_input_2 = ul_mix_input_2.to(device)

            with torch.cuda.amp.autocast(enabled=half):
                pred_l = model(l_input)[0]
                pred_ul_1, pred_ul_fp = model(ul_input, need_fp=True)
                pred_ul_mix_1 = model(ul_mix_input_1)[0]
                pred_ul_mix_2 = model(ul_mix_input_2)[0]
           
            with torch.cuda.amp.autocast(enabled=half):
                ## supervised loss
                sup_loss = ce_loss(pred_l, l_target) 
                mix_pseudo_1 = score_mask(ul_mix_pred_1, torch.argmax(ul_mix_pred_1, dim=1))
                loss_u_1 = ce_loss(pred_ul_mix_1, mix_pseudo_1)
                mix_pseudo_2 = score_mask(ul_mix_pred_2, torch.argmax(ul_mix_pred_2, dim=1))
                loss_u_2 = ce_loss(pred_ul_mix_2, mix_pseudo_2)
                fp_pseudo = score_mask(pred_u, torch.argmax(pred_u, dim=1))
                loss_u_fp = ce_loss(pred_ul_fp, fp_pseudo)
                loss = (sup_loss + loss_u_1 * 0.25 + loss_u_2 * 0.25 + loss_u_fp * 0.5) / 2
                
                ## learning rate update
                current_idx = epoch * len(unsup_loader) + batch_idx
                learning_rate = lr_scheduler.get_lr(current_idx)
                # update the learning rate
                optimizer.param_groups[0]['lr'] = learning_rate
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            step_miou, iou_list = measurement.miou(measurement._make_confusion_matrix(pred_l.detach().cpu().numpy(), l_target.detach().cpu().numpy()))
            sum_miou += step_miou
            sum_loss += loss.item()
            back_iou += iou_list[0]
            weed_iou += iou_list[1]
            crop_iou += iou_list[2]
            print_txt = f"[Epoch{epoch}/{cfg.train.num_epochs}][Iter{batch_idx+1}/{len(unsup_loader)}] lr={learning_rate:.5f}" \
                            + f"miou={step_miou}, loss={loss.item():.4f}"
            pbar.set_description(print_txt, refresh=False)
            if logger != None:
                log_txt.write(print_txt)
        
        ## end epoch ## 
        back_iou, weed_iou, crop_iou = back_iou / len(unsup_loader), weed_iou / len(unsup_loader), crop_iou / len(unsup_loader)
        loss = sum_loss / len(unsup_loader)
        miou = sum_miou / len(unsup_loader)
        
        print_txt = f"[Epoch{epoch}]" \
                            + f"miou={miou:.4f}, loss={loss:.4f}"
        print(print_txt)
        test_miou = test(test_loader, model, measurement, cfg)
        print(f"test_miou : {test_miou:.4f}")
        if best_miou <= test_miou:
            best_miou = test_miou
            if logger is not None:
                save_ckpoints(model.state_dict(),
                            epoch,
                            batch_idx,
                            optimizer.state_dict(),
                            os.path.join(ckpoints_dir, f"best_test_miou.pth"))
        
        if logger != None:
            log_txt.write(print_txt)
            params = [l_input, l_target, pred_l, ul_input, pred_ul_1]
            params = [detach_numpy(i) for i in params]
            example = make_example_img(*params)
            logger.image_update(example, f'{epoch}ep')
            if cfg.train.save_img:
                save_img(img_dir, f'output_{epoch}ep.png', example)
            if epoch % 10 == 0:
                save_ckpoints(model.state_dict(),
                            epoch,
                            batch_idx,
                            optimizer.state_dict(),
                            os.path.join(ckpoints_dir, f"{epoch}ep.pth"))
            save_ckpoints(model.state_dict(),
                        epoch,
                        batch_idx,
                        optimizer.state_dict(),
                        os.path.join(ckpoints_dir, f"last.pth"))
            # wandb logging
            for key in logger.config_dict.keys():
                logger.config_dict[key] = eval(key)
            for key in ["loss", "learning_rate", "miou",  "crop_iou", "weed_iou", "back_iou"]:
                logger.log_dict[key] = eval(key)
            
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
    parser.add_argument('--config_path', default='./config/vqreptunet1x1.json')
    # parser.add_argument('--config_path', default='./config/vqreptunetangular.json')
    opt = parser.parse_args()
    
    for cfgfile in ['./config/vqreptunet1x1.json', "./config/vqreptunet1x1_IJRR2017.json", "./config/vqreptunet1x1_rice_s_n_w.json"]:
        for percent in ["percent_30", "percent_20", "percent_10"]:
            cfg = get_config_from_json(cfgfile)
            cfg.train.wandb_log.append('test_miou')
            root, p = os.path.split(cfg.train.data_dir) 
            cfg.train.data_dir = os.path.join(root, percent)
            cfg.model.name = "deeplabv3plus"
            for v in ["vq_cfg", "margin", "scale", "use_feature"]:
                cfg.model.params.pop(v)
            cfg.model.params.encoder_weights = "imagenet"
            cfg.project_name = "UNIMatch"
            cfg.resize=32
            cfg.project_name = 'debug'
            cfg.wandb_logging = False
            # cfg.train.device = -1
            # cfg.train.half = False
            train(cfg)
