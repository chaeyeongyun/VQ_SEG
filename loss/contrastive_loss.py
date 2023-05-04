import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F

def d3tod2(x):
    return rearrange(x, 'c h w -> (h w) c')

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.04) :
        super().__init__()
        self.temperature = temperature
    def forward(self, x:torch.Tensor, label:torch.Tensor,):
        assert x.shape[0] > 2, "supervised contrasitve loss can be computed when batch size is larger than 2"
        x_b, x_c, x_h, x_w = x.shape[:]
        if label.dim() == 3: 
            label = label.unsqueeze(1)
        if label.shape[-2:] != x.shape[-2:]:
            label = F.interpolate(label.float(), size=x.shape[-2:], mode='nearest')
        feat_1, feat_2 = x[0], x[1]
        gt1, gt2 = label[0], label[1]
        feat_1, feat_2 = d3tod2(feat_1), d3tod2(feat_2) # (HW, C)
        gt1, gt2 = d3tod2(gt1), d3tod2(gt2) # (HW, 1)
        similarity = torch.matmul(feat_1, feat_2.transpose(0,1)) # (HW, C) x (C, HW) = (HW, HW)
        similarity = torch.exp(similarity / self.temperature) 
        gt1 = gt1.expand(-1, x_h*x_w) # (HW, 1) -> (HW, HW)
        gt2 = gt2.transpose(0,1).expand(x_h*x_w, -1) # (1, HW)->(HW, HW)
        positive_y, positive_x = torch.nonzero(gt1==gt2, as_tuple=True)
        loss = -torch.log(torch.sum(similarity[positive_y, positive_x]) / torch.sum(similarity)) / (x_h * x_w * x_h * x_w)
        return loss
        
        
        
        
        
            