import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from PIL import Image
import os
from tqdm import tqdm
import argparse

from utils.load_config import get_config_from_yaml
from utils.logger import Logger, list_to_separate_log
from utils.ckpoints import save_vqvae
from utils.device import device_setting
from utils.lr_schedulers import CosineAnnealingLR, WarmUpPolyLR
from utils.visualize import save_img, make_selfsup_example

import wandb

import models
from data.dataset import FolderDataset

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
        logger = Logger(cfg, logger_name)
        wandb.config.update(cfg)
    else : logger = None
    
    half=cfg.train.half
    batch_size = cfg.train.batch_size
    num_epochs = cfg.train.num_epochs
    device = device_setting(cfg.train.device)
    
    model = models.networks.make_model(cfg.model).to(device)
    
    dataset = FolderDataset(cfg.train.data_dir, resize=cfg.resize)
    dataloader= DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train.learning_rate), betas=(0.9, 0.999))
    
    if cfg.train.lr_scheduler.name == 'warmuppoly':
        lr_sched_cfg = cfg.train.lr_scheduler
        lr_scheduler = WarmUpPolyLR(cfg.train.learning_rate, lr_power=cfg.lr_sched_cfg.lr_power, 
                                    total_iters=len(dataloader)*num_epochs,
                                    warmup_steps=len(dataloader)*cfg.lr_sched_cfg.warmup_epoch)
    elif cfg.train.lr_scheduler.name == 'cosineannealing':
        lr_sched_cfg = cfg.train.lr_scheduler
        lr_scheduler = CosineAnnealingLR(start_lr = cfg.train.learning_rate, min_lr=lr_sched_cfg.min_lr, total_iters=len(dataloader)*num_epochs, warmup_steps=lr_sched_cfg.warmup_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=half)
    print_txt=''
    for epoch in range(num_epochs):
        model.train()
        recon_loss_sum, commitment_loss_sum, loss_sum = 0, 0, 0
        
        # pbar = tqdm(range(len(dataloader)))
        pbar = tqdm(dataloader)
        for batch_idx, input in enumerate(pbar):
            # input = next(dataloader)
            input = input.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=half):
                output, commitment_loss, code_usage = model(input)
                if batch_idx == 0:
                    sum_code_usage = torch.zeros_like(code_usage)
                target = F.interpolate(input, size=output.shape[-2:], mode='bilinear')
                recon_loss = F.mse_loss(output, target)
                loss = recon_loss + commitment_loss
            
            # # update the learning rate
            current_idx = epoch * len(dataloader) + batch_idx
            learning_rate = lr_scheduler.get_lr(current_idx)
            optimizer.param_groups[0]['lr'] = learning_rate
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            recon_loss_sum += recon_loss.item()
            commitment_loss_sum += commitment_loss.item()
            loss_sum += loss.item()
            sum_code_usage += code_usage
            print_txt = f"[Epoch{epoch}/{cfg.train.num_epochs}][Iter{batch_idx}/{len(dataloader)}] lr={learning_rate:.5f}" \
                            + f"recon_loss={recon_loss.item():.4f}, commitment_loss={commitment_loss.item():.4f}, loss={loss.item():.4f}"
            pbar.set_description(print_txt, refresh=False)
            if logger != None:
                log_txt.write(print_txt)
        
        # end epoch #
        recon_loss = recon_loss_sum/len(dataloader)
        commitment_loss = commitment_loss_sum/len(dataloader)
        loss = loss_sum/len(dataloader)
        code_usage = (sum_code_usage / len(dataloader)).tolist()
        print_txt = f"[Epoch{epoch}] recon_loss={recon_loss:.4f}, commitment_loss={commitment_loss:.4f}, loss={loss:.4f}"
        print(print_txt)
        if logger != None:
            log_txt.write(print_txt)
            for key in logger.config_dict.keys():
                logger.config_dict[key] = eval(key)
            
            for key in logger.log_dict.keys():
                if key=="code_usage":
                    logger.temp_update(list_to_separate_log(l=eval(key), name=key))
                else:
                    logger.log_dict[key] = eval(key)
                
            cat_img = make_selfsup_example(target.detach().cpu().numpy(), output.detach().cpu().numpy())
            if cfg.train.save_img: save_img(img_dir, f'output_{epoch}ep.png', cat_img)
            logger.image_update(cat_img, caption=f"{epoch}ep")
            if epoch % 10 == 0:
                save_vqvae(model, epoch, ckpoints_dir)
            save_vqvae(model, 'last', ckpoints_dir)
            # wandb logging 왜 안되지...
            logger.logging(epoch=epoch)
            logger.config_update()
            
    if logger != None: log_txt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./config/self_sup_train.yaml')
    opt = parser.parse_args()
    cfg = get_config_from_yaml(opt.config_path)
    # debug
    cfg.resize=64
    cfg.wandb_logging = True
    cfg.project_name = 'debug'
    train(cfg)