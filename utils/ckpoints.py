import torch
import os
from torch import nn
import tarfile
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
    torch.save(model.encoder.state_dict(), os.path.join(ckpoints_dir, f'{epoch}ep_encoder.pth'))
    torch.save(model.codebook.state_dict(), os.path.join(ckpoints_dir, f'{epoch}ep_codebook.pth'))
    torch.save(model.decoder.state_dict(), os.path.join(ckpoints_dir, f'{epoch}ep_decoder.pth'))
    
def save_tar(target_path):
    head, name = os.path.split(target_path)[:]
    name += '.tar.gz'
    with tarfile.open(os.path.join(head, name), 'w:gz') as t:
        t.add(target_path)