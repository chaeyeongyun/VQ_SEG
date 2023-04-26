import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils.seg_tools import label_to_onehot

def l1norm(t:torch.Tensor, dim):
    return F.normalize(t, p=1, dim=dim)

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1, activation:nn.Module=nn.Identity):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = activation() if activation!=nn.Softmax else activation(dim=1)
        super().__init__(conv2d, upsampling, activation)
        
class AngularSegmentationHead(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 num_classes, 
                 embedding_dim, 
                 scale, 
                 margin, 
                 init='uniform', 
                 kernel_size=3, 
                 upsampling=2, 
                 activation=nn.Softmax2d):
        super().__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.embedding = nn.Embedding(num_embeddings=num_classes,
                                      embedding_dim=embedding_dim)
        self.activation = activation()
        if init == 'uniform':
            # uniform distribution initialization
            self.embedding.weight.data.uniform_(-1/num_classes, 1/num_classes)
        elif init == 'normal':
            self.embedding.weight.data.normal_()
        else:
            raise ValueError('init has to be in [''uniform'', ''normal'']')
        
    def forward(self, x, gt):
        x = self.conv(x)
        x = self.upsampling(x)
        x_b, x_c, x_h, x_w = x.shape[:]
        device = x.device
        flatten_x = rearrange(x, 'b c h w -> (b h w) c')
        # l1 norm
        self.embedding.weight.data = l1norm(self.embedding.weight.data, dim=-1) # (num_classes, feat_num)
        flatten_x = l1norm(flatten_x, dim=-1)
         # cosine
        # cosine = torch.einsum('n c, p c -> n p', flatten_x, self.embedding.weight) # (BHW, num_classes) slow...
        cosine = torch.matmul(flatten_x, self.embedding.weight.transpose(0,1))
        loss = torch.tensor([0.], device=device, requires_grad=self.training, dtype=torch.float32)
        if self.training and gt is not None:
            # gt = label_to_onehot(gt, self.num_classes)
            gt = gt.unsqueeze(1) if gt.dim()==3 else gt
            flatten_gt = rearrange(gt, 'b c h w -> (b h w) c') # (BHW, num_classes)
            
            x_ind = torch.arange(x_b*x_h*x_w)
            # margin
            # cosine = cosine + self.margin * flatten_gt # too slow....
            cosine[x_ind, flatten_gt[:,0]] = cosine[x_ind, flatten_gt[:,0]] + self.margin
            # scale
            cosine = self.scale * cosine
            
            positive = torch.exp(cosine[x_ind, flatten_gt[:,0]])
            # positive = torch.exp(torch.sum(cosine * flatten_gt, dim=-1)) #(BHW,)
            sum_all = torch.sum(torch.exp(cosine), dim=-1) # (BHW, )
            loss = -torch.mean(torch.log(positive / sum_all)) 
        pred = rearrange(cosine, '(b h w) p -> b h w p', b=x_b, h=x_h, w=x_w, p=self.num_classes)
        pred = rearrange(pred, 'b h w p -> b p h w')
        pred = self.activation(pred)
        return pred, loss