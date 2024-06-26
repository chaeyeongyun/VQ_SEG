import math
import numpy as np
from utils.seg_tools import score_mask
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from utils.seg_tools import label_to_onehot

def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    n, d = t.shape[:2]
    normed_codes = l2norm(t, dim=-1)
    cosine_sim = torch.einsum('i d, j d -> i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (n ** 2) - (1 / n)

def l1norm(t:torch.Tensor, dim):
    return F.normalize(t, p=1, dim=dim)

def l2norm(t, dim):
    return F.normalize(t, p = 2, dim = dim)

def batched_sample_vectors(samples, num): # kmeans에 sample_fn으로 들어감
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim = 0)], dim = 0)

def sample_vectors(sample, num):
    num_samples, device = sample.shape[0], sample.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return sample[indices]

def batched_bincount(x, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype = dtype, device = device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values) # -1 dim 을 따라서 self tensor의 index x에 values를 더해준다.
    return target

def kmeans(flatten_x, num_clusters, num_iters, use_cosine_sim=False):
    samples = torch.unsqueeze(flatten_x, dim=0) # (1, B, HxW, C)
    samples = rearrange(samples, 'h ... d -> h (...) d') #(1, BxHxW, C)
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    means = batched_sample_vectors(samples, num_clusters) # (num_codebooks, num_clusters, C)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, 'h n d -> h d n') # (num_codebook, BxHxW, num_clusters)
        else:
            dists = -torch.cdist(samples, means, p = 2) # (num_codebook, BxHxW, num_clusters)

        buckets = torch.argmax(dists, dim = -1) # (num_codebook, BxHxW) 거리 가장 먼 것들의 index들이 나옴
        bins = batched_bincount(buckets, minlength = num_clusters) # (num_codebook, num_cluster) == (B, codebook_size)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1) # zero_mask == 0인 곳을 1로 채움. 즉 bin에서 0인 곳들을 1로 채움

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype = dtype) # (num_codebooks, num_cluster, dim)
        # self tensor와 같은 device, 같은 dtype을 가지면서 size크기의 zero tensor를 반환하는 것인데 
        # 여기서는 dtype을 따로 지정해주었고 size는 (num_codebooks, num_cluster, dim)

        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d = dim), samples) # (num_codebooks, num_cluster, dim)
         # 1 dim 을 따라서 self tensor의 index repeat(buckets, 'h n -> h n d', d = dim)에 values(samples)를 더해준다.
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1') # (num_codebooks, num_cluster, dim)
        #                                  # (num_codebooks, num_cluster) -> (num_codebooks, num_cluster, 1)
        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(rearrange(zero_mask, '... -> ... 1'), # bin이 0인 곳에 대해서 
            means, # means를 적용하고
            new_means # 아니면 new_means를 적용한다.
        ) # (num_codebooks, num_cluster, dim)
    
    return means, bins

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1, activation:nn.Module=nn.Identity):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = activation() if activation==nn.Softmax2d else activation(dim=1)
        super().__init__(conv2d, upsampling, activation)
        
class AngularSegmentationHead(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 num_classes, 
                 embedding_dim, 
                 scale, 
                 margin, 
                 init='kmeans', 
                 kernel_size=3, 
                 upsampling=2, 
                 activation=nn.Softmax2d,
                 easy_margin=True):
        super().__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.embedding = nn.Embedding(num_embeddings=num_classes,
                                      embedding_dim=embedding_dim)
        self.activation = activation()
        self.init = init
        self.initted = False
        if init == 'uniform':
            # uniform distribution initialization
            self.embedding.weight.data.uniform_(-1/num_classes, 1/num_classes)
            self.initted = True
        elif init == 'normal':
            self.embedding.weight.data.normal_()
            self.initted = True
        elif init == 'kmeans':
            pass
        else:
            raise ValueError('init has to be in [''uniform'', ''normal'', ''kmeans'']')
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
    def forward(self, x, gt):
        x = self.conv(x)
        x = self.upsampling(x)
        x_b, x_c, x_h, x_w = x.shape[:]
        device = x.device
        flatten_x = rearrange(x, 'b c h w -> (b h w) c')
        # l1 norm
        self.embedding.weight.data = l1norm(self.embedding.weight.data, dim=-1) # (num_classes, feat_num)
        flatten_x = l1norm(flatten_x, dim=-1)
        if not self.initted and self.init == 'kmeans':
            self._kmeans_init(flatten_x)
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
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            # cosine[x_ind, flatten_gt[:,0]] = cosine[x_ind, flatten_gt[:,0]] - self.margin
            cosine[x_ind, flatten_gt[:,0]] = cosine[x_ind, flatten_gt[:,0]] * phi[x_ind, flatten_gt[:,0]].to(torch.float16)
            # cosine = cosine + self.margin * flatten_gt
            # scale
            cosine = self.scale * cosine
            
            positive = torch.exp(cosine[x_ind, flatten_gt[:,0]])
            # positive = torch.exp(torch.sum(cosine * flatten_gt, dim=-1)) #(BHW,)
            sum_all = torch.sum(torch.exp(cosine), dim=-1) # (BHW, )
            loss = -torch.mean(torch.log((positive / (sum_all + 1e-7)) + 1e-7)) 
        pred = rearrange(cosine, '(b h w) p -> b h w p', b=x_b, h=x_h, w=x_w, p=self.num_classes)
        pred = rearrange(pred, 'b h w p -> b p h w')
        pred = self.activation(pred)
        return pred, loss
    
    def _kmeans_init(self, flatten_x):    
        if self.initted:
            return
        
        embed, cluster_size = kmeans(
            flatten_x,
            self.num_classes,
            num_iters=10,
        )
        
        self.embedding.weight.data.copy_(embed[0])
        self.initted = True
        
class AngularSegmentationHeadv2(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 num_classes, 
                 scale, 
                 margin, 
                 init='kmeans', 
                 kernel_size=1, 
                 upsampling=2, 
                 activation=nn.Softmax2d,
                 easy_margin=True,
                 orthogonal_reg_weight=0.):
        super().__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.embedding = nn.Embedding(num_embeddings=num_classes,
                                      embedding_dim=out_channels)
        self.activation = activation()
        self.init = init
        self.initted = False
        if init == 'uniform':
            # uniform distribution initialization
            self.embedding.weight.data.uniform_(-1/num_classes, 1/num_classes)
            self.initted = True
        elif init == 'normal':
            self.embedding.weight.data.normal_()
            self.initted = True
        elif init == 'kmeans':
            pass
        else:
            raise ValueError('init has to be in [''uniform'', ''normal'', ''kmeans'']')
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
    def forward(self, x, gt, percent, entropy):
        x = self.conv(x)  
        # x = self.upsampling(x)
        x_b, x_c, x_h, x_w = x.shape[:]
        device = x.device
        flatten_x = rearrange(x, 'b c h w -> (b h w) c') # (BxHxW, C)
        if not self.initted and self.init == 'kmeans':
            self._kmeans_init(flatten_x)
        # l2 norm
        self.embedding.weight.data = l2norm(self.embedding.weight.data, dim=-1) # (num_classes, feat_num)
        flatten_x = l2norm(flatten_x, dim=-1)

        # cosine
        cosine = F.linear(flatten_x, self.embedding.weight) 
        # cosine = torch.matmul(flatten_x, self.embedding.weight.transpose(0,1))
        loss = torch.tensor([0.], device=device, requires_grad=self.training, dtype=torch.float32)
        if self.training and gt is not None:
            # gt = label_to_onehot(gt, self.num_classes)
            gt = gt.unsqueeze(1) if gt.dim()==3 else gt
            gt = F.interpolate(gt.float(), (x_h, x_w), mode="nearest").long()
            flatten_gt = rearrange(gt, 'b c h w -> (b h w) c') # (BHW, 1)
            
            x_ind = torch.arange(x_b*x_h*x_w) 
            # margin
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
            # cosine[x_ind, flatten_gt[:,0]] = cosine[x_ind, flatten_gt[:,0]] - self.margin
            cosine[x_ind, flatten_gt[:,0]] = cosine[x_ind, flatten_gt[:,0]] * phi[x_ind, flatten_gt[:,0]].to(torch.float16)
            # cosine = cosine + self.margin * flatten_gt
            # scale
            cosine = self.scale * cosine
             ## entropy filtering ##
            with torch.no_grad():
                thresh = np.percentile(entropy.detach().cpu().numpy().flatten(), percent) # scalar
                thresh_mask = torch.le(entropy, thresh)
            cosine = cosine * thresh_mask.unsqueeze(-1)
            positive = torch.exp(cosine[x_ind, flatten_gt[:,0]])
            # positive = torch.exp(torch.sum(cosine * flatten_gt, dim=-1)) #(BHW,)
            sum_all = torch.sum(torch.exp(cosine), dim=-1) # (BHW, )
            loss = -torch.mean(torch.log((positive / (sum_all + 1e-7)) + 1e-7)) 
        pred = rearrange(cosine, '(b h w) p -> b h w p', b=x_b, h=x_h, w=x_w, p=self.num_classes)
        pred = rearrange(pred, 'b h w p -> b p h w')
        pred = self.activation(pred)
        pred = self.upsampling(pred)
        # commitment_loss = torch.tensor([0.], device=device, requires_grad=self.training, dtype=torch.float32)
        if self.training:
            class_feat =self.embedding.weight.data[gt].permute(0, 4, 2, 3, 1).squeeze(-1)
            class_feat = class_feat.detach() # codebook 쪽 sg 연산
            commitment_loss = F.mse_loss(class_feat, x) # encoder update
            # commitment_loss = commitment_loss + commitloss 
            loss  = loss + commitment_loss
            if self.orthogonal_reg_weight > 0:
                codebook = self.embedding.weight
                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight
        return pred, loss
    
    def _kmeans_init(self, flatten_x):    
        if self.initted:
            return
        
        embed, cluster_size = kmeans(
            flatten_x,
            self.num_classes,
            num_iters=10,
        )
        
        self.embedding.weight.data.copy_(embed[0])
        self.initted = True

class AngularSegmentationHeadv3(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 num_classes, 
                 scale, 
                 margin, 
                 init='kmeans', 
                 kernel_size=1, 
                 upsampling=1, 
                 activation=nn.Softmax2d,
                 easy_margin=True,
                 orthogonal_reg_weight=0.):
        super().__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.orthogonal_reg_weight = orthogonal_reg_weight
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.embedding = nn.Embedding(num_embeddings=num_classes,
                                      embedding_dim=out_channels)
        self.activation = activation()
        self.init = init
        self.initted = False
        if init == 'uniform':
            # uniform distribution initialization
            self.embedding.weight.data.uniform_(-1/num_classes, 1/num_classes)
            self.initted = True
        elif init == 'normal':
            self.embedding.weight.data.normal_()
            self.initted = True
        elif init == 'kmeans':
            pass
        else:
            raise ValueError('init has to be in [''uniform'', ''normal'', ''kmeans'']')
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
    def forward(self, x, pred, split, th):
        x = self.conv(x)  
        x = self.upsampling(x)
        x_b, x_c, x_h, x_w = x.shape[:]
        device = x.device
        flatten_x = rearrange(x, 'b c h w -> (b h w) c')
        # l1 norm
        self.embedding.weight.data = l2norm(self.embedding.weight.data, dim=-1) # (num_classes, feat_num)
        flatten_x = l2norm(flatten_x, dim=-1)
        if not self.initted and self.init == 'kmeans':
            self._kmeans_init(flatten_x)
        # cosine
        # cosine = torch.einsum('n c, p c -> n p', flatten_x, self.embedding.weight) # (BHW, num_classes) slow...
        cosine = torch.matmul(flatten_x, self.embedding.weight.transpose(0,1))
        loss = torch.tensor([0.], device=device, requires_grad=self.training, dtype=torch.float32)
        result = rearrange(cosine, '(b h w) p -> b h w p', b=x_b, h=x_h, w=x_w, p=self.num_classes)
        result = rearrange(result, 'b h w p -> b p h w')
        result = self.activation(result)
        if self.training and pred is not None:
            gt = torch.argmax(pred, dim=1) if split=="unlabeled" else pred
            gt = gt.unsqueeze(1) if gt.dim()==3 else gt
            gt = F.interpolate(gt.float(), (x_h, x_w), mode="nearest").long()
            flatten_gt = rearrange(gt, 'b c h w -> (b h w) c') # (BHW, 1)
            
            x_ind = torch.arange(x_b*x_h*x_w) 
            # margin
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            # cosine[x_ind, flatten_gt[:,0]] = cosine[x_ind, flatten_gt[:,0]] - self.margin
            cosine[x_ind, flatten_gt[:,0]] = cosine[x_ind, flatten_gt[:,0]] * phi[x_ind, flatten_gt[:,0]].to(torch.float16)
            # cosine = cosine + self.margin * flatten_gt
            # scale
            cosine = self.scale * cosine
            ## TODO: ignore index
            if self.training and split=="unlabeled" and pred is not None  and th>0:
                thresh_mask = score_mask(pred, self.th)
                thresh_mask = rearrange(gt, 'b c h w -> (b h w) c') # (BHW, 1)
                cosine = cosine * thresh_mask
            positive = torch.exp(cosine[x_ind, flatten_gt[:,0]])
            # positive = torch.exp(torch.sum(cosine * flatten_gt, dim=-1)) #(BHW,)
            sum_all = torch.sum(torch.exp(cosine), dim=-1) # (BHW, )
            loss = -torch.mean(torch.log((positive / (sum_all + 1e-7)) + 1e-7)) 
        
        # commitment_loss = torch.tensor([0.], device=device, requires_grad=self.training, dtype=torch.float32)
        if self.training:
            class_feat =self.embedding.weight.data[gt].permute(0, 4, 2, 3, 1).squeeze(-1)
            class_feat = class_feat.detach() # codebook 쪽 sg 연산
            commitment_loss = F.mse_loss(class_feat, x) # encoder update
            # commitment_loss = commitment_loss + commitloss 
            loss  = loss + commitment_loss
            if self.orthogonal_reg_weight > 0:
                codebook = self.embedding.weight
                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight
        return result, loss
    
    def _kmeans_init(self, flatten_x):    
        if self.initted:
            return
        
        embed, cluster_size = kmeans(
            flatten_x,
            self.num_classes,
            num_iters=10,
        )
        
        self.embedding.weight.data.copy_(embed[0])
        self.initted = True
        