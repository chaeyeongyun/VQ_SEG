from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

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
                                        embedding_dim=embedding_dim, )
        # uniform distribution initialization
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
    def forward(self, x, gt):
        if gt.shape != x.shape:
            gt = F.interpolate(gt, x.shape[-2:], mode='nearest')

        b, c, h, w = x.shape[:]
        flatten_x = rearrange(x, 'b c h w -> (b h w) c')
        flatten_gt = rearrange(gt, 'b c h w -> (b h w) c')
        if self.use_feature:
            for i in range(self.num_classes):
                ind = (flatten_gt == i).nonzero(as_tuple=True)[0]
                flatten_x[]
        # l1 norm
        self.embedding.weight.data = l1norm(self.embedding.weigh.data, dim=-1) # (num_classes, feat_num)
        flatten_x = l1norm(flatten_x, dim=-1)
        # cosine
        cosine = torch.einsum('n c, p c -> n p', flatten_x, self.embedding.weight) # (BHW, num_classes)
        # scale
        cosine = self.scale * cosine
        # margin
        cosine[:,flatten_gt.long()] = cosine[:, flatten_gt.long()] + self.margin
        positive = cosine[:, flatten_gt.long()] #(BHW,)
        sum_all = torch.sum(cosine, dim=-1) # (BHW, )
        loss = torch.mean(torch.log(positive / sum_all))
        return loss
        
        