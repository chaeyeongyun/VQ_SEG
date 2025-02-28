import argparse
import matplotlib.pyplot as plt
import os
from itertools import cycle
from tqdm import tqdm
import time
import wandb
import numpy as np

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


def to_pseudo_label(tensor:torch.Tensor):
    return torch.argmax(tensor, dim=1).long()

def cal_unsuploss_weight(pred, target, percent, pred_teacher):
    batch_size, num_class, h, w = pred.shape
    with torch.no_grad():
        # drop pixels with high entropy
        prob = torch.softmax(pred_teacher, dim=1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

        thresh = np.percentile(
            entropy[target != 255].detach().cpu().numpy().flatten(), percent
        )
        thresh_mask = entropy.ge(thresh).bool() * (target != 255).bool()

        target[thresh_mask] = 255
        weight = batch_size * h * w / torch.sum(target != 255)
    return weight
#TODO: 
def conf_ce_loss(inputs, targets,
                 conf_mask=True, threshold=0.6,
                 threshold_neg=.0, temperature_value=1):
    # target => logit, input => logit
    pass_rate = {}
    if conf_mask:
        # for negative
        targets_prob = torch.softmax(targets/temperature_value, dim=1) # temperature 적용 softmax
        
        # for positive
        targets_real_prob = torch.softmax(targets, dim=1) # 일반 softmax
        
        weight = targets_real_prob.max(1)[0] # 가장 큰 값
        total_number = len(targets_prob.flatten(0)) # 걍 일렬로 쭉 펴버려
        boundary = ["< 0.1", "0.1~0.2", "0.2~0.3",
                    "0.3~0.4", "0.4~0.5", "0.5~0.6",
                    "0.6~0.7", "0.7~0.8", "0.8~0.9",
                    "> 0.9"]

        rate = [torch.sum((torch.logical_and((i - 1) / 10 < targets_real_prob, targets_real_prob < i / 10)) == True)
                / total_number for i in range(1, 11)] # 위에 나온 boundary 범위별로 리스트 생성

        max_rate = [torch.sum((torch.logical_and((i - 1) / 10 < weight, weight < i / 10)) == True)
                    / weight.numel() for i in range(1, 11)]

        pass_rate["entire_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, rate)]
        pass_rate["max_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, max_rate)]

        mask = (weight >= threshold) # 가장 큰 값에서 threshold보다 큰 픽셀들만 1

        mask_neg = (targets_prob < threshold_neg) # threshold_neg보다 작은 값들만 1

        neg_label = torch.nn.functional.one_hot(torch.argmax(targets_prob, dim=1)).type(targets.dtype) # negative 대한 prob이었던 temperature + softmax를 onehot 라벨로 변환
        if neg_label.shape[-1] != 3: # 클래스수 안맞게 나올 경우에 대한거임
            neg_label = torch.cat((neg_label, torch.zeros([neg_label.shape[0], neg_label.shape[1],
                                                           neg_label.shape[2], 3 - neg_label.shape[-1]]).cuda()),
                                  dim=3)
        neg_label = neg_label.permute(0, 3, 1, 2)
        neg_label = 1 - neg_label
          
        if not torch.any(mask):  # True값이 하나라도 있으면
            neg_prediction_prob = torch.clamp(1-torch.softmax(inputs, dim=1), min=1e-7, max=1.)
            negative_loss_mat = -(neg_label * torch.log(neg_prediction_prob))
            zero = torch.tensor(0., dtype=torch.float, device=negative_loss_mat.device)
            loss_unsup = zero
            # return zero, pass_rate, negative_loss_mat[mask_neg].mean()
        else:
            positive_loss_mat = torch.nn.functional.cross_entropy(inputs, torch.argmax(targets, dim=1), reduction="none")
            positive_loss_mat = positive_loss_mat * weight

            neg_prediction_prob = torch.clamp(1-torch.softmax(inputs, dim=1), min=1e-7, max=1.)
            negative_loss_mat = -(neg_label * torch.log(neg_prediction_prob))
            loss_unsup = positive_loss_mat[mask].mean()
            # return positive_loss_mat[mask].mean(), pass_rate, negative_loss_mat[mask_neg].mean()
        neg_loss = negative_loss_mat[mask_neg].mean()
        # for negative learning
        if threshold_neg > .0:
            confident_reg = .5 * torch.mean(torch.softmax(inputs, dim=1) ** 2)
            loss_unsup += neg_loss
            loss_unsup += confident_reg

        return loss_unsup
    else:
        raise NotImplementedError


# 일단 no cutmix version
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
    
    # criterion = make_loss(cfg.train.criterion, num_classes, ignore_index=255)
    criterion = nn.CrossEntropyLoss()
    sup_dataset = BaseDataset(os.path.join(cfg.train.data_dir, 'train'), split='labelled',  batch_size=batch_size, resize=cfg.resize)
    unsup_dataset = BaseDataset(os.path.join(cfg.train.data_dir, 'train'), split='unlabelled',  batch_size=batch_size, resize=cfg.resize)
    
    sup_loader = DataLoader(sup_dataset, batch_size=batch_size, shuffle=True)
    unsup_loader = DataLoader(unsup_dataset, batch_size=batch_size, shuffle=True)
    
    
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
            # pred_1 = torch.cat([pred_sup_1, pred_ul_1], dim=0)
            # pred_2 = torch.cat([pred_sup_2, pred_ul_2], dim=0)
            # pseudo label
            # pseudo_1 = torch.argmax(pred_1, dim=1).long()
            # pseudo_2 = torch.argmax(pred_2, dim=1).long()
            pseudo_l_1, pseudo_l_2 = to_pseudo_label(pred_sup_1), to_pseudo_label(pred_sup_2)
            pseudo_ul_1, pseudo_ul_2 = to_pseudo_label(pred_ul_1), to_pseudo_label(pred_ul_2)
            
            with torch.cuda.amp.autocast(enabled=half):
                ## cps loss
                l_index = pred_sup_1.shape[0]
                percent_unreliable = cfg.train.unsup_loss_drop_percent * (1-epoch/num_epochs)
                drop_percent = 100 - percent_unreliable
                unreliable_weight_1 = cal_unsuploss_weight(pred_ul_2, pseudo_ul_1.clone(), drop_percent, pred_ul_1.detach())
                unreliable_weight_2 = cal_unsuploss_weight(pred_ul_1, pseudo_ul_2.clone(), drop_percent, pred_ul_2.detach())
                cps_loss_l = criterion(pred_sup_1, pseudo_l_2) + criterion(pred_sup_2, pseudo_l_1)
                cps_loss_ul = unreliable_weight_2*conf_ce_loss(pred_ul_1, pred_ul_2) + unreliable_weight_1*conf_ce_loss(pred_ul_2, pred_ul_1)
                cps_loss = cps_loss_l + cps_loss_ul
            
                ## supervised loss
                sup_loss_1 = criterion(pred_sup_1, l_target)
                sup_loss_2 = criterion(pred_sup_2, l_target)
                sup_loss = sup_loss_1 + sup_loss_2
               ## commitment_loss
                commitment_loss = commitment_loss_l1 + commitment_loss_l2 + unreliable_weight_1*commitment_loss_ul1 + unreliable_weight_2*commitment_loss_ul2
                commitment_loss *= total_commitment_loss_weight
                ## prototype_loss
                prototype_loss = prototype_loss_l1 + prototype_loss_l2 + prototype_loss_ul1*unreliable_weight_1 + prototype_loss_ul2*unreliable_weight_2
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
        if logger != None:
            log_txt.write(print_txt)
            params = [l_input, l_target, pred_sup_1, ul_input, pred_ul_1]
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
    parser.add_argument('--config_path', default='./config/vq_pt_unet.json')
    opt = parser.parse_args()
    cfg = get_config_from_json(opt.config_path)
    # debug
    # cfg.resize=512
    # cfg.project_name = 'debug'
    # cfg.wandb_logging = False
    # cfg.train.half=False
    cfg.project_name = 'VQPT+confCE'
    cfg.resize = 448
    # train(cfg)
    train(cfg)
    # cfg.model.params.pop("encoder_weights")
    # train(cfg)