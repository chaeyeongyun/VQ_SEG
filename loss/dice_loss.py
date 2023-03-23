import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_coefficient(pred:torch.Tensor, target:torch.Tensor, num_classes:int):
    """calculate dice coefficient

    Args:
        pred (torch.Tensor): (N, num_classes, H, W)
        target (torch.Tensor): (N, H, W)
        num_classes (int): the number of classes
    """
    
    if num_classes == 1:
        target = target.type(pred.type())
        pred = torch.sigmoid(pred)
        # target is onehot label
    else:
        target = target.type(pred.type()) # target과 pred의 type을 같게 만들어준다.
        target = torch.eye(num_classes)[target.long()].to(pred.device) # (N, H, W, num_classes)
        target = target.permute(0, 3, 1, 2) # (N, num_classes, H, W)
        pred = F.softmax(pred, dim=1)
    
    inter = torch.sum(pred*target, dim=(2, 3)) # (N, num_classes)
    sum_sets = torch.sum(pred+target, dim=(2, 3)) # (N, num_classes)
    dice_coefficient = (2*inter / (sum_sets+1e-6)).mean(dim=0) # (num_classes)
    return dice_coefficient
        
        
def dice_loss(pred, target, num_classes, weights:tuple=None, ignore_index=None):
    if not isinstance(pred, torch.Tensor) :
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(pred)}")

    dice = dice_coefficient(pred, target, num_classes)
    if weights is not None:
        dice_loss = 1-dice
        weights = torch.Tensor(weights)
        dice_loss = dice_loss * weights
        dice_loss = dice_loss.mean()
        
    else: 
        dice = dice.mean()
        dice_loss = 1 - dice
        
    return dice_loss

class DiceLoss(nn.Module):
    def __init__(self, num_classes, weights:tuple=None, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.weights = weights
        self.ignore_index = ignore_index
    def forward(self, pred, target):
        return dice_loss(pred, target, self.num_classes, weights=self.weights, ignore_index=self.ignore_index)

 