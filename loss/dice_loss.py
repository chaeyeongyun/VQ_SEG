import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_coefficient(pred:torch.Tensor, target:torch.Tensor, num_classes:int, ignore_index:int):
    """calculate dice coefficient

    Args:
        pred (torch.Tensor): (N, num_classes, H, W)
        target (torch.Tensor): (N, H, W)
        num_classes (int): the number of classes
    """
    b, c, h, w = pred.shape[:]
    pred = pred.reshape(b, c, -1)
    target = target.reshape(b, -1)
    mask = (target!=ignore_index)
    pred = pred * torch.stack([mask]*3, dim=1)
    target = target*mask
    
    if num_classes == 1:
        target = target.type(pred.type())
        pred = torch.sigmoid(pred)
        # target is onehot label
    else:
        target = target.type(pred.type()) # target과 pred의 type을 같게 만들어준다.
        target = torch.eye(num_classes, device=pred.device)[target.long()].to(pred.device) # (N, H, W, num_classes) onehot
        # target = target.permute(0, 3, 1, 2) # (N, num_classes, 1, HxW)
        target = target.permute(0, 2, 1)
        pred = F.softmax(pred, dim=1)
    
    # inter = torch.sum(pred*target, dim=(2, 3)) # (N, num_classes)
    inter = torch.sum(pred*target, dim=(2)) # (N, num_classes)
    # sum_sets = torch.sum(pred+target, dim=(2, 3)) # (N, num_classes)
    sum_sets = torch.sum(pred+target, dim=(2)) # (N, num_classes)
    dice_coefficient = (2*inter / (sum_sets+1e-6)).mean(dim=0) # (num_classes)
    return dice_coefficient
        
        
def dice_loss(pred:torch.Tensor, target:torch.Tensor, num_classes:int=3, weight:torch.Tensor=None, ignore_index:int=-100):

    dice = dice_coefficient(pred, target, num_classes, ignore_index=ignore_index)
    # if ignore_index>=0 and ignore_index<num_classes:
    #     dice[ignore_index] = 0
    
    if weight is not None:
        weight = weight.to(pred.device)
        dice_loss = 1-dice
        dice_loss = dice_loss * weight / torch.sum(weight) 
        dice_loss = torch.sum(dice_loss) / num_classes
        # dice = dice * weight
        # dice = dice / torch.sum(weight)
        # dice_loss = dice_loss.mean()
        # dice_loss = 1-dice
    else: 
        dice = dice.mean()
        dice_loss = 1 - dice
        
    return dice_loss

class DiceLoss(nn.Module):
    def __init__(self, num_classes, weight:torch.Tensor=None, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.ignore_index = ignore_index
    def forward(self, pred, target):
        return dice_loss(pred, target, self.num_classes, weight=self.weight, ignore_index=self.ignore_index)

 