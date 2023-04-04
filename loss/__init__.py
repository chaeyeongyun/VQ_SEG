from torch import nn
from .dice_loss import DiceLoss
from .focal_loss import FocalLoss

loss_dict = {'cross_entropy':nn.CrossEntropyLoss,
                 'dice_loss':DiceLoss,
                 'focal_loss':FocalLoss}

def make_loss(loss_name:str, num_classes:int, ignore_index:int):    
    if loss_name == 'cross_entropy':
        return loss_dict[loss_name](ignore_index=ignore_index)
    else:
        return loss_dict[loss_name](num_classes=num_classes, ignore_index=ignore_index)