from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from utils.seg_tools import label_to_onehot

def l1norm(t:torch.Tensor, dim):
    return F.normalize(t, p=1, dim=dim)

class PrototypeLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim, scale, margin, use_feature=False) :
        super().__init__()
        self.use_feature = use_feature
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        
        self.embedding = nn.Embedding(num_embeddings=num_classes,
                                        embedding_dim=embedding_dim)
        # uniform distribution initialization
        self.embedding.weight.data.uniform_(-1/num_classes, 1/num_classes)
        if self.use_feature:
            for p in self.embedding.parameters():
                p.requires_grad = False
                
    def forward(self, x, gt):
        gt = gt.unsqueeze(1) if gt.dim()==3 else gt
        if gt.shape != x.shape:
            gt = F.interpolate(gt.float(), x.shape[-2:], mode='nearest').long()

        x_b, x_c, x_h, x_w = x.shape[:]
        flatten_x = rearrange(x, 'b c h w -> (b h w) c') # (BHW, C)
        flatten_gt = rearrange(gt, 'b c h w -> (b h w) c') # (BHW, 1)
        
        if self.use_feature:
            temp = []
            for i in range(self.num_classes):
                ind = (flatten_gt == i).nonzero(as_tuple=True)
                temp.append(torch.mean(flatten_x[ind[0]], dim=0, keepdim=True)) # (1, C)
            temp = torch.cat(temp, dim=0) # (num_classes, C)
            self.embedding.weight.data.copy_(temp)
        
        # l1 norm
        self.embedding.weight.data = l1norm(self.embedding.weight.data, dim=-1) # (num_classes, feat_num)
        flatten_x = l1norm(flatten_x, dim=-1)
        # cosine
        # cosine = torch.einsum('n c, p c -> n p', flatten_x, self.embedding.weight) # (BHW, num_classes)
        cosine = torch.matmul(flatten_x, self.embedding.weight.transpose(0,1))
       
        x_ind = torch.arange(x_b*x_h*x_w, dtype=torch.long)
        # margin
        cosine[x_ind, flatten_gt[:,0]] = cosine[x_ind, flatten_gt[:,0]] - self.margin
        # cosine = cosine + self.margin * flatten_gt
        # scale
        cosine = self.scale * cosine
        
        positive = torch.exp(cosine[x_ind, flatten_gt[:,0]])
        # positive = torch.exp(torch.sum(cosine * flatten_gt, dim=-1)) #(BHW,)
        sum_all = torch.sum(torch.exp(cosine), dim=-1) # (BHW, )
        loss = -torch.mean(torch.log(positive / sum_all)) 
        return loss
        
        