from typing import Iterable, Union
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
import numpy as np
import random
import math
from PIL import Image
def augmentation(input:torch.Tensor, label:torch.Tensor, logits:torch.Tensor, aug_cfg:dict):
    batch_size = input.shape[0]
    input_aug, label_aug, logits_aug = [], [], []
    for i in range(batch_size):
        if aug_cfg.name=='cutout':
            mask = make_cutout_mask(input.shape[-2:], aug_cfg.ratio).to(input.device)
            label[i][(1-mask).bool()] = 255 # ignore index
            input_aug.append((input[i]*mask).unsqueeze(0))
            label_aug.append(label[i].unsqueeze(0))
            logits_aug.append((logits[i]*mask).unsqueeze())
        elif aug_cfg.name == 'cutmix':
            mask = make_cutout_mask(input.shape[-2:], aug_cfg.ratio).to(input.device)
            input_aug.append((input[i]*mask + input[(i+1)%batch_size]*(1-mask)).unsqueeze(0))
            label_aug.append((label[i]*mask + label[(i+1)%batch_size]*(1-mask)).unsqueeze(0))
            logits_aug.append((logits[i]*mask + logits[(i+1)%batch_size]*(1-mask)).unsqueeze(0))
    input_aug = torch.cat(input_aug, dim=0)
    label_aug = torch.cat(label_aug, dim=0)
    logits_aug = torch.cat(logits_aug, dim=0)

    return input_aug, label_aug, logits_aug

def make_cutout_mask(img_size:Iterable[int], ratio):
    cutout_area = img_size[0]*img_size[1]*ratio
    cut_w = np.random.randint(int(img_size[1]*ratio)+1, img_size[1])
    cut_h = int(cutout_area//cut_w)
    x1, y1 = np.random.randint(0, img_size[1]-cut_w+1), random.randint(0, img_size[0]-cut_h+1)
    mask = torch.ones(tuple(img_size))
    mask[y1:y1+cut_h, x1:x1+cut_w] = 0
    return mask.long()


    
    
class CutMix():
    def __init__(self, ratio:float):
        """cutout augmentation

        Args:
            ratio (Union): the ratio of box to input size
        """
        self.ratio = ratio
        
    def _make_mask(self, img_size:Iterable[int]):
        cutout_area = img_size[0]*img_size[1]*self.ratio
        cut_w = np.random.randint(int(img_size[1]*self.ratio)+1, img_size[1])
        cut_h = int(cutout_area//cut_w)
        x1, y1 = np.random.randint(0, img_size[1]-cut_w+1), random.randint(0, img_size[0]-cut_h+1)
        mask = torch.ones(tuple(img_size))
        mask[y1:y1+cut_h, x1:x1+cut_w] = 0
        return mask.long()
        
    def __call__(self, batch:torch.Tensor, mask:torch.Tensor=None):
        """
        Args:
            batch (torch.Tensor): mini-batch of input images
        """
        batch_size = batch.shape[0]
        h, w = batch.shape[-2:]
        if mask is None:
            mask = self._make_mask((h,w), self.ratio).to(batch.device)
        mixed = [(input[i]*mask + input[(i+1)%batch_size]*(1-mask)).unsqueeze(0) for i in range(batch_size)]
        mixed = torch.cat(mixed, dim=0)
        return mixed, mask

class CutOut():
    def __init__(self, ratio:float):
        """cutout augmentation

        Args:
            ratio (Union): the ratio of box to input input size
        """
        self.ratio = ratio
        
    def _make_mask(self, img_size:Iterable[int]):
        cutout_area = img_size[0]*img_size[1]*self.ratio
        cut_w = np.random.randint(int(img_size[1]*self.ratio)+1, img_size[1])
        cut_h = int(cutout_area//cut_w)
        x1, y1 = np.random.randint(0, img_size[1]-cut_w+1), random.randint(0, img_size[0]-cut_h+1)
        mask = torch.ones(tuple(img_size))
        mask[y1:y1+cut_h, x1:x1+cut_w] = 0
        return mask.long()
    
    def __call__(self, batch:torch.Tensor, mask:torch.Tensor=None):
        """
        Args:
            batch (torch.Tensor): mini-batch of input images
        """
        batch_size = input.shape[0]
        h, w = input.shape[-2:]
        if mask is None:
            mask = self._make_mask((h,w), self.ratio).to(batch.device)# (H, W)
        aug = [(input[i]*mask).unsqueeze(0) for i in range(batch_size)]
        aug = torch.cat(aug, dim=0)
        return aug, mask


        
def random_similarity_transform(input:torch.Tensor):
    aug = random.randint(0, 8)
    angle = .0
    if aug == 1:
        input = input.flip(-1)
    elif aug == 2:
        input = input.flip(-2)
    else:
        rand_rot = transforms.RandomRotation(90)
        angle = rand_rot.get_params([.0, 90.0])
        if aug == 3:
            TF.rotate(input, angle, interpolation=Image.Resampling.BILINEAR)
    return input, aug

def inverse_similarity_transform(input:torch.Tensor, aug_num:int):
    if aug_num == 1:
        input = input.flip(-1)
    elif aug_num == 2:
        input = input.flip(-2)
    elif aug_num == 3:
        input = torch.rot90(input,dims=(-1,-2), k=3)
    elif aug_num == 4:
        input = torch.rot90(input,dims=(-1,-2), k=2)
    elif aug_num == 5:
        input = torch.rot90(input,dims=(-1,-2), k=1)
    elif aug_num == 6:
        input = torch.rot90(input, dims=(-1,-2), k=3).flip(-1)
    elif aug_num == 7:
        input = torch.rot90(input, dims=(-1,-2), k=3).flip(-2)
    return input

