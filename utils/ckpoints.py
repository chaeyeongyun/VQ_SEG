import torch
import os
from torch import nn

#TODO: 수정 필요
def save_ckpoints(model_1, model_2, epoch, batch_idx, optimizer_1, optimizer_2, filepath):
    torch.save({'model_1':model_1,
               'model_2':model_2,
               'epoch':epoch,
               'batch_idx':batch_idx,
               'optimizer_1':optimizer_1,
               'optimizer_2':optimizer_2}, filepath)
    
def load_ckpoints(weights_path, istrain:bool):
    ckpoints = torch.load(weights_path)
    
    if istrain:
        return ckpoints['model_2'], ckpoints['epoch'], ckpoints['batch_idx'], ckpoints['optimizer_1'], ckpoints['optimizer_2']
    else:
        return ckpoints['model_1']
    
def save_vqvae(model:nn.Module, epoch, ckpoints_dir):
    torch.save(model.backbone.state_dict(), os.path.join(ckpoints_dir, f'{epoch}ep_backbone.pth'))
    torch.save(model.vq.state_dict(), os.path.join(ckpoints_dir, f'{epoch}ep_vq.pth'))
    torch.save(model.decoder.state_dict(), os.path.join(ckpoints_dir, f'{epoch}ep_decoder.pth'))