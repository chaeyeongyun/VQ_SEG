from easydict import EasyDict
import torch 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from glob import glob
import os
import argparse

import copy
from utils.load_config import get_config_from_json
from utils.device import device_setting
from utils.visualize import make_test_detailed_img, save_img_list
from utils.seg_tools import img_to_label
from utils.logger import TestLogger, dict_to_table_log, make_img_table
from measurement import Measurement
import models
from utils.seed import seed_everything
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

from data.dataset import NormalizedDataset

from loss import make_loss
from measurement import Measurement

def make_filename(filename_list, insert):
    for i, filename in enumerate(filename_list):
        filename, ext = os.path.splitext(filename)
        filename_list[i] = filename+insert+ext
    return filename_list

def real_test(cfg):
    seed_everything()
    # debug
    # cfg.resize=32
    # cfg.project_name = 'debug'
    # cfg.test.weights = "../drive/MyDrive/semi_sup_train/CWFID/debug/ckpoints"
    num_classes = cfg.num_classes
    batch_size = cfg.test.batch_size
    pixel_to_label_map = cfg.pixel_to_label
    weights = cfg.test.weights
    device = device_setting(cfg.test.device)
    
    model = models.networks.make_model(cfg.model).to(device)
    # logger_name = (cfg.model.name+"_"+model.encoder.__class__.__name__+"_"+model.decoder.__class__.__name__+"_"+str(len(os.listdir(cfg.test.save_dir))))
    l =  cfg.test.weights.split('/')
    logger_name = l[l.index('ckpoints')-2]+"/"+l[l.index('ckpoints')-1]
    # make save folders
    save_dir = os.path.join(cfg.test.save_dir, l[l.index('ckpoints')-1])
    os.makedirs(save_dir, exist_ok=True)
    img_dir = os.path.join(save_dir, 'imgs')
    os.makedirs(img_dir, exist_ok=True)
    
    logger = TestLogger(cfg, logger_name) if cfg.wandb_logging else None
    test_data = NormalizedDataset(os.path.join(cfg.test.data_dir, 'test'), split='labelled', resize=cfg.resize, target_resize=False)
    testloader = DataLoader(test_data, batch_size, shuffle=False)
    f = open(os.path.join(save_dir, 'results.txt'), 'w')
    f.write(f"data_dir:{cfg.test.data_dir}, weights:{cfg.test.weights}, save_dir:{cfg.test.save_dir}")

    best_result = None
    if os.path.isfile(cfg.test.weights):
        best_result = test_loop(model, weights, num_classes, pixel_to_label_map, testloader, device)
        
    elif os.path.isdir(cfg.test.weights):
        weights_list = glob(os.path.join(cfg.test.weights, '*.pth'))
        weights_list.sort()
        best_miou = 0.
        for weights in weights_list:
            result = test_loop(model, weights, num_classes, pixel_to_label_map, testloader, device)
            if result == None: continue
            if result.metrics.test_miou >= best_miou:
                best_miou = copy.deepcopy(result.metrics.test_miou)
                best_result = copy.deepcopy(result)
    
    assert best_result != None, "weights file has some problem"
    f.write(best_result.result_txt)
    save_img_list(img_dir, make_filename(best_result.visualize.filename, "_v1"), best_result.visualize.viz_v1)
    save_img_list(img_dir, make_filename(best_result.visualize.filename, "_v2"), best_result.visualize.viz_v2)
    if logger != None:
        logger.table_dict = {
            "metrics":dict_to_table_log(best_result.metrics),
            "visualize":make_img_table(best_result.visualize.filename, best_result.visualize.viz_v1, best_result.visualize.viz_v2, columns=["filename", 'viz_v1', "viz_v2"])
            }
        logger.logging()
    print("best_result:\n"+best_result.result_txt)
    if logger is not None: 
        logger.finish()
def test_loop(model:nn.Module, weights_file:str, num_classes:int, pixel_to_label_map:dict, testloader:DataLoader, device:torch.device):
    try:
        weights = torch.load(weights_file)
        weights = weights.get('model_1', weights)
    except:
        return None
    
    model.load_state_dict(weights)
    model.eval()
    measurement = Measurement(num_classes)
    test_acc, test_miou = 0, 0
    test_precision, test_recall, test_f1score = 0, 0, 0
    iou_per_class = np.array([0]*num_classes, dtype=np.float64)
    viz_v1_list, viz_v2_list = [], []
    filename_list = []
    for data in tqdm(testloader):
        input_img, mask_img, filename = data['img'], data['target'], data['filename']
        input_img = input_img.to(device)
        mask_cpu = img_to_label(mask_img, pixel_to_label_map)
        
        with torch.no_grad():
            pred = model(input_img)
            pred = pred[0] if isinstance(pred, tuple) else pred
        
        pred = F.interpolate(pred, mask_img.shape[-2:], mode='bilinear')
        pred_numpy, mask_numpy = pred.detach().cpu().numpy(), mask_cpu.cpu().numpy()
        
        acc_pixel, batch_miou, iou_ndarray, precision, recall, f1score = measurement(pred_numpy, mask_numpy) 
        
        test_acc += acc_pixel
        test_miou += batch_miou
        iou_per_class += iou_ndarray
        
        test_precision += precision
        test_recall += recall
        test_f1score += f1score
        
        save_size = (mask_img.shape[-2]//2, mask_img.shape[-1]//2)
        input_img = F.interpolate(input_img.detach().cpu(), save_size, mode='bilinear')
        pred_cpu = F.interpolate(pred.detach().cpu(), save_size, mode='bilinear')
        mask_cpu = F.interpolate(mask_img.detach().cpu().unsqueeze(1), save_size, mode='nearest').squeeze(1)
        mask_cpu = img_to_label(mask_cpu, pixel_to_label_map)
        viz_v1, viz_v2 = make_test_detailed_img(input_img.numpy(), pred_cpu, mask_cpu, colormap = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0.5, 0.5, 0.5], [230/255, 145/255, 56/255], [1, 217/255, 102/255]]))
        viz_v1_list.append(viz_v1)
        viz_v2_list.append(viz_v2)
        filename_list.append(*filename)
        
    # test finish
    test_acc = test_acc / len(testloader)
    test_miou = test_miou / len(testloader)
    test_ious = np.round((iou_per_class / len(testloader)), 5).tolist()
    test_precision /= len(testloader)
    test_recall /= len(testloader)
    test_f1score /= len(testloader)
    
    result_txt = "load model(.pt) : %s \n Testaccuracy: %.4f, Test miou: %.4f" % (weights_file,  test_acc, test_miou)       
    result_txt += f"\niou per class {list(map(lambda x: round(x, 4), test_ious))}"
    result_txt += f"\nprecision : {test_precision:.4f}, recall : {test_recall:.4f}, f1score : {test_f1score:.4f} " 
    print(result_txt)
    return_dict = {
        "metrics":
            {
                "test_acc":test_acc,
                "test_miou":test_miou,
                "test_ious":test_ious,
                "test_precision":test_precision,
                "test_recall":test_recall,
                "test_f1score": test_f1score
            },
        "visualize":{
            "viz_v1":viz_v1_list,
            "viz_v2":viz_v2_list,
            "filename":filename_list
            },
        "result_txt":result_txt
        }
    return EasyDict(return_dict)

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
        logger_name = cfg.project_name+"_hybrid_norm_"+str(len(os.listdir(cfg.train.save_dir)))
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
    loss_weight = cfg.train.criterion.get("weight", None)
    loss_weight = torch.tensor(loss_weight) if loss_weight is not None else loss_weight
    ce_loss = nn.CrossEntropyLoss(weight=loss_weight, ignore_index=255)
    dice_loss = make_loss(cfg.train.criterion.name, num_classes, weight=loss_weight, ignore_index=255)
    
    sup_dataset = NormalizedDataset(os.path.join(cfg.train.data_dir, 'train'), split='labelled',  batch_size=batch_size, resize=cfg.resize)
    unsup_dataset = NormalizedDataset(os.path.join(cfg.train.data_dir, 'train'), split='unlabelled',  batch_size=batch_size, resize=cfg.resize)
    
    sup_loader = DataLoader(sup_dataset, batch_size=batch_size, shuffle=True)
    unsup_loader = DataLoader(unsup_dataset, batch_size=batch_size, shuffle=True)
    ##test##
    test_dataset = NormalizedDataset(os.path.join(cfg.test.data_dir, 'test'), split='labelled', batch_size=1, resize=cfg.resize)
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
    confidence_threshold = cfg.train.confidence_threshold
    best_miou = 0
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

            # pseudo_1 = model_1.pseudo_label(ul_input)
            # pseudo_2 = model_2.pseudo_label(ul_input)
            with torch.no_grad():
                model_1.eval()
                model_2.eval()
                pseudo_1 = torch.argmax(model_1(ul_input)[0], dim=1)
                pseudo_2 = torch.argmax(model_2(ul_input)[0], dim=1)
                model_1.train()
                model_2.train()
            
            percent_unreliable = cfg.train.unsup_loss_drop_percent * (1-epoch/num_epochs)
            drop_percent = 100 - percent_unreliable
            with torch.cuda.amp.autocast(enabled=half):
                # pred_sup_1, commitment_loss_l1, code_usage_l1, prototype_loss_l1 = model_1(l_input, l_target, percent=drop_percent)
                # pred_sup_2, commitment_loss_l2, code_usage_l2, prototype_loss_l2 = model_2(l_input, l_target, percent=drop_percent)
                pred_sup_1, commitment_loss_l1, code_usage_l1, prototype_loss_l1 = model_1(l_input, l_target, percent=drop_percent)
                pred_sup_2, commitment_loss_l2, code_usage_l2, prototype_loss_l2 = model_2(l_input, l_target, percent=drop_percent)
                
                ## predict in unsupervised manner ##
                # pred_ul_1, commitment_loss_ul1, code_usage_ul1, prototype_loss_ul1 = model_1(ul_input, pseudo_2, percent=drop_percent)
                # pred_ul_2, commitment_loss_ul2, code_usage_ul2, prototype_loss_ul2 = model_2(ul_input, pseudo_1, percent=drop_percent)
                pred_ul_1, commitment_loss_ul1, code_usage_ul1, prototype_loss_ul1 = model_1(ul_input, pseudo_2, percent=drop_percent)
                pred_ul_2, commitment_loss_ul2, code_usage_ul2, prototype_loss_ul2 = model_2(ul_input, pseudo_1, percent=drop_percent)
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
                ### celoss에 대서 entropy 제한 ####
                filt_entropy_1 = score_mask(pred_1, pseudo_1, th=confidence_threshold)
                filt_entropy_2 = score_mask(pred_2, pseudo_2, th=confidence_threshold)
                cps_loss = 0.5*ce_loss(pred_1, filt_entropy_2) + 0.5*ce_loss(pred_2, filt_entropy_1) + dice_loss(pred_1, filt_entropy_2) + dice_loss(pred_2, filt_entropy_1)
                ## supervised loss
                sup_loss_1 = 0.5*ce_loss(pred_sup_1, l_target) + dice_loss(pred_sup_1, l_target)
                sup_loss_2 = 0.5*ce_loss(pred_sup_2, l_target) + dice_loss(pred_sup_2, l_target)
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
                            + f"miou={step_miou}, sup_loss_1={sup_loss_1:.4f}, prototype_loss={prototype_loss.item():.4f}, cps_loss={cps_loss:.4f}"
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
                            + f"miou={miou:.4f}, sup_loss_1={sup_loss_1:.4f}, prototype_loss={prototype_loss:.4f}, cps_loss={cps_loss:.4f}, commitment_loss={commitment_loss:.4f}"
        print(print_txt)
        test_miou = test(test_loader, model_1, measurement, cfg)
        print(f"test_miou : {test_miou:.4f}")
        if best_miou <= test_miou:
            best_miou = test_miou
            if logger is not None:
                save_ckpoints(model_1.state_dict(),
                            model_2.state_dict(),
                            epoch,
                            batch_idx,
                            optimizer_1.state_dict(),
                            optimizer_2.state_dict(),
                            os.path.join(ckpoints_dir, f"best_test_miou.pth"))
        
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
    cfg.test.weights = os.path.join(ckpoints_dir, "best_test_miou.pth")
    real_test(cfg)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./config/vqreptunet1x1_IJRR2017.json')
    # parser.add_argument('--config_path', default='./config/vqreptunetangular.json')
    opt = parser.parse_args()
    cfg = get_config_from_json(opt.config_path)
    cfg.train.wandb_log.append('test_miou')
    cfg.model.params.encoder_weights = "imagenet"
    ### debug
    # cfg.resize=64
    # cfg.project_name = 'debug'
    # cfg.wandb_logging = False
    ########
    from copy import deepcopy
    orgprojectname = deepcopy(cfg.project_name)
    cfg.train.wandb_log.append('test_miou')
    for percent in ["percent_20", "percent_10"]:
        cfg.project_name = orgprojectname + percent
        root, p = os.path.split(cfg.train.data_dir)[:]
        cfg.train.data_dir = os.path.join(root, percent)
        train(cfg)
    

