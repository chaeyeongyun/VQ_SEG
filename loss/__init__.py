from torch import nn
import torch
import torch.nn.functional as F

from .dice_loss import DiceLoss, dice_loss
from .focal_loss import FocalLoss, focal_loss
from .contrastive_loss import * 


loss_dict = {'cross_entropy':nn.CrossEntropyLoss,
                 'dice_loss':DiceLoss,
                 'focal_loss':FocalLoss,
                 'nll_loss':nn.NLLLoss}

loss_func_dict = {'cross_entropy':F.cross_entropy,
                 'dice_loss':dice_loss,
                 'focal_loss':focal_loss,
                 'nll_loss':F.nll_loss}

def make_loss(loss_name:str, num_classes:int, ignore_index:int=-100, weight:torch.Tensor=None):    
    if loss_name == 'cross_entropy':
        return loss_dict[loss_name](ignore_index=ignore_index, weight=weight)
    else:
        return loss_dict[loss_name](num_classes=num_classes, ignore_index=ignore_index, weight=weight)
def make_loss_as_func(loss_name:str):
    return loss_func_dict[loss_name]
    
def compute_class_weight(num_classes, y:torch.Tensor):
    # class_sample_count = torch.unique(y, return_counts=True)[1]
    class_sample_count = torch.bincount(torch.flatten(y), minlength=num_classes)
    class_sample_prob = class_sample_count / torch.sum(class_sample_count)
    weight = 1. - class_sample_prob
    return weight