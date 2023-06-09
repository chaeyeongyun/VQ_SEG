from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from utils.seg_tools import label_to_onehot
import numpy as np
import math

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
            new_means = l2norm(new_means, dim=-1)

        means = torch.where(rearrange(zero_mask, '... -> ... 1'), # bin이 0인 곳에 대해서 
            means, # means를 적용하고
            new_means # 아니면 new_means를 적용한다.
        ) # (num_codebooks, num_cluster, dim)
    
    return means, bins

def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    n, d = t.shape[:2]
    normed_codes = l2norm(t, dim=-1)
    cosine_sim = torch.einsum('i d, j d -> i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (n ** 2) - (1 / n)

class PrototypeLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim, scale, margin, init='kmeans', use_feature=False, easy_margin=True) :
        super().__init__()
        self.use_feature = use_feature
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.init = init
        self.embedding = nn.Embedding(num_embeddings=num_classes,
                                        embedding_dim=embedding_dim)
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
        gt = gt.unsqueeze(1) if gt.dim()==3 else gt
        if gt.shape != x.shape:
            gt = F.interpolate(gt.float(), x.shape[-2:], mode='nearest').long()

        x_b, x_c, x_h, x_w = x.shape[:]
        flatten_x = rearrange(x, 'b c h w -> (b h w) c') # (BHW, C)
        flatten_gt = rearrange(gt, 'b c h w -> (b h w) c') # (BHW, 1)
        
        if not self.initted and self.init == 'kmeans':
            self._kmeans_init(flatten_x)
        
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
        # cosine = torch.mm(flatten_x, self.embedding.weight.transpose(0,1)) # (BHW, C) x (C, 3) = (BHW, 3)
        cosine = F.linear(flatten_x, self.embedding.weight)
        
        x_ind = torch.arange(x_b*x_h*x_w, dtype=torch.long)
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
        return loss
    
    def _kmeans_init(self, flatten_x):    
        if self.initted:
            return
        
        embed, cluster_size = kmeans(
            flatten_x,
            self.num_classes,
            num_iters=10,
            use_cosine_sim=False # l1norm 해줌
        )
        
        self.embedding.weight.data.copy_(embed[0])
        self.initted = True

        
class EuclideanPrototypeLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim, init='kmeans', use_feature=False) :
        super().__init__()
        self.use_feature = use_feature
        self.num_classes = num_classes
        self.init = init
        self.embedding = nn.Embedding(num_embeddings=num_classes,
                                        embedding_dim=embedding_dim)
        # uniform distribution initialization
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
          
                
    def forward(self, x, gt):
        gt = gt.unsqueeze(1) if gt.dim()==3 else gt
        if gt.shape != x.shape:
            gt = F.interpolate(gt.float(), x.shape[-2:], mode='nearest').long()

        x_b, x_c, x_h, x_w = x.shape[:]
        flatten_x = rearrange(x, 'b c h w -> (b h w) c') # (BHW, C)
        flatten_gt = rearrange(gt, 'b c h w -> (b h w) c') # (BHW, 1)
        
        if not self.initted and self.init == 'kmeans':
            self._kmeans_init(flatten_x)
        
        if self.use_feature:
            indexes = [torch.nonzero(flatten_gt==i, as_tuple=True)[0] for i in range(self.num_classes)]
            temp = []
            for i in range(self.num_classes):
                temp.append(torch.mean(flatten_x[indexes[i]], dim=0, keepdim=True)) # (1, C)
            temp = torch.cat(temp, dim=0) # (num_classes, C)
            self.embedding.weight.data.copy_(temp)
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training, dtype=torch.float32)
        if self.training:
            class_feat =self.embedding.weight.data[gt].permute(0, 4, 2, 3, 1).squeeze(-1)
            class_feat = class_feat.detach() # codebook 쪽 sg 연산
            loss = loss + F.mse_loss(class_feat, x) # encoder update
        return loss
    
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

class LearnableEuclideanPrototypeLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim, init='kmeans', use_feature=False) :
        super().__init__()
        print("learnable")
        self.use_feature = use_feature
        self.num_classes = num_classes
        self.init = init
        self.embedding = nn.Embedding(num_embeddings=num_classes,
                                        embedding_dim=embedding_dim)
        # uniform distribution initialization
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
        self.alpha = nn.Parameter(torch.tensor(-1, dtype=torch.float), requires_grad=True)
                
    def forward(self, x, gt):
        gt = gt.unsqueeze(1) if gt.dim()==3 else gt
        if gt.shape != x.shape:
            gt = F.interpolate(gt.float(), x.shape[-2:], mode='nearest').long()

        x_b, x_c, x_h, x_w = x.shape[:]
        flatten_x = rearrange(x, 'b c h w -> (b h w) c') # (BHW, C)
        flatten_gt = rearrange(gt, 'b c h w -> (b h w) c') # (BHW, 1)
        
        if not self.initted and self.init == 'kmeans':
            self._kmeans_init(flatten_x)
        
        indexes = [torch.nonzero(flatten_gt==i, as_tuple=True)[0] for i in range(self.num_classes)]
        if self.use_feature:
            temp = []
            for i in range(self.num_classes):
                temp.append(torch.mean(flatten_x[indexes[i]], dim=0, keepdim=True)) # (1, C)
            temp = torch.cat(temp, dim=0) # (num_classes, C)
            self.embedding.weight.data.copy_(temp)
        
        distance = torch.cdist(flatten_x.unsqueeze(0), self.embedding.weight.data.unsqueeze(0), p=2).squeeze(0) # (BHW, num_classes)
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training, dtype=torch.float32)
        commitment_loss = torch.tensor([0.], device=x.device, requires_grad=self.training, dtype=torch.float32)
        for i, idx in enumerate(indexes):
            loss = loss + torch.mean(distance[idx, i])
        loss = loss / self.num_classes
        
        return loss * torch.sigmoid(self.alpha)
    
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
        
class NEDPrototypeLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim, temperature=0.04, init='class_means', use_feature=False) :
        super().__init__()
        self.use_feature = use_feature
        self.num_classes = num_classes
        self.init = init
        self.temperature = temperature
        self.embedding = nn.Embedding(num_embeddings=num_classes,
                                        embedding_dim=embedding_dim)
        # uniform distribution initialization
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
        elif init == 'class_means':
            pass
        else:
            raise ValueError('init has to be in [''uniform'', ''normal'', ''kmeans'', ''class_means'']')
          
                
    def forward(self, x, gt):
        gt = gt.unsqueeze(1) if gt.dim()==3 else gt
        if gt.shape != x.shape:
            gt = F.interpolate(gt.float(), x.shape[-2:], mode='nearest').long()

        x_b, x_c, x_h, x_w = x.shape[:]
        flatten_x = rearrange(x, 'b c h w -> (b h w) c') # (BHW, C)
        flatten_gt = rearrange(gt, 'b c h w -> (b h w) c') # (BHW, 1)
        
        if not self.initted:
            if self.init == 'kmeans':
                self._kmeans_init(flatten_x)
            elif self.init == 'class_means':
                self._class_means_init(flatten_x, flatten_gt)
        
        if self.use_feature:
            indexes = [torch.nonzero(flatten_gt==i, as_tuple=True)[0] for i in range(self.num_classes)]
            temp = []
            for i in range(self.num_classes):
                temp.append(torch.mean(flatten_x[indexes[i]], dim=0, keepdim=True)) # (1, C)
            temp = torch.cat(temp, dim=0) # (num_classes, C)
            self.embedding.weight.data.copy_(temp)
        
        distance = torch.cdist(flatten_x.unsqueeze(0), self.embedding.weight.data.unsqueeze(0), p=2).squeeze(0) # (BHW, num_classes)
        # loss = torch.tensor([0.], device=x.device, requires_grad=self.training, dtype=torch.float32)
        
        x_ind = torch.arange(x_b*x_h*x_w, dtype=torch.long)
        # positive = torch.exp(distance[x_ind, flatten_gt[:,0]]/self.temperature) # (BHW, )
        # sum_all = torch.sum(torch.exp(distance/self.temperature), dim=-1) # (BHW, )
        # loss = torch.mean(positive / sum_all)
        loss = torch.softmax(distance/self.temperature, dim=-1)
        loss = -torch.mean(loss[x_ind, flatten_gt[:,0]])
        return loss
    
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
    def _class_means_init(self, flatten_x, flatten_gt):    
        if self.initted:
            return
        print('class_means_init')
        indexes = [torch.nonzero(flatten_gt==i, as_tuple=True)[0] for i in range(self.num_classes)]
        temp = []
        for i in range(self.num_classes):
            temp.append(torch.mean(flatten_x[indexes[i]], dim=0, keepdim=True)) # (1, C)
        temp = torch.cat(temp, dim=0) # (num_classes, C)
    
        self.embedding.weight.data.copy_(temp)
        self.initted = True
        

class ReliablePrototypeLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim, scale, margin, init='kmeans', use_feature=False, easy_margin=True, 
                 orthogonal_reg_weight=0) :
        super().__init__()
        self.use_feature = use_feature
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.init = init
        self.embedding = nn.Embedding(num_embeddings=num_classes,
                                        embedding_dim=embedding_dim)
        self.orthogonal_reg_weight = orthogonal_reg_weight
        
        
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
        gt = gt.unsqueeze(1) if gt.dim()==3 else gt
        if gt.shape != x.shape:
            gt = F.interpolate(gt.float(), x.shape[-2:], mode='nearest').long()

        x_b, x_c, x_h, x_w = x.shape[:]
        flatten_x = rearrange(x, 'b c h w -> (b h w) c') # (BHW, C)
        flatten_gt = rearrange(gt, 'b c h w -> (b h w) c') # (BHW, 1)
        
        if not self.initted and self.init == 'kmeans':
            self._kmeans_init(flatten_x)
        
        if self.use_feature:
            temp = []
            for i in range(self.num_classes):
                ind = (flatten_gt == i).nonzero(as_tuple=True)
                temp.append(torch.mean(flatten_x[ind[0]], dim=0, keepdim=True)) # (1, C)
            temp = torch.cat(temp, dim=0) # (num_classes, C)
            self.embedding.weight.data.copy_(temp)
        
        # l1 norm
        # self.embedding.weight.data = l1norm(self.embedding.weight.data, dim=-1) # (num_classes, feat_num)
        # flatten_x = l1norm(flatten_x, dim=-1)
        # l2 norm
        self.embedding.weight.data = l2norm(self.embedding.weight.data, dim=-1) # (num_classes, feat_num)
        flatten_x = l2norm(flatten_x, dim=-1)
        # cosine
        # cosine = torch.einsum('n c, p c -> n p', flatten_x, self.embedding.weight) # (BHW, num_classes)
        # cosine = torch.mm(flatten_x, self.embedding.weight.transpose(0,1)) # (BHW, C) x (C, 3) = (BHW, 3)
        cosine = F.linear(flatten_x, self.embedding.weight)
        
        x_ind = torch.arange(x_b*x_h*x_w, dtype=torch.long)
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
            # percent_unreliable = cfg.train.unsup_loss.drop_percent * (1-epoch/num_epochs)
            # drop_percent = 100 - percent_unreliable
            thresh_mask = torch.le(entropy, thresh)
        ###
        # cosine = cosine * torch.stack([thresh_mask]*3, dim=-1) # entropy 작은것들만 살림.
        cosine = cosine * thresh_mask.unsqueeze(-1)
        positive = torch.exp(cosine[x_ind, flatten_gt[:,0]])
        # positive = torch.exp(torch.sum(cosine * flatten_gt, dim=-1)) #(BHW,)
        sum_all = torch.sum(torch.exp(cosine), dim=-1) # (BHW, )
        loss = -torch.mean(torch.log((positive / (sum_all + 1e-7)) + 1e-7)) 
        
        if self.orthogonal_reg_weight > 0:
            codebook = self.embedding.weight
            orthogonal_reg_loss = orthogonal_loss_fn(codebook)
            loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight
        return loss
    
    def _kmeans_init(self, flatten_x):    
        if self.initted:
            return
        
        embed, cluster_size = kmeans(
            flatten_x,
            self.num_classes,
            num_iters=10,
            use_cosine_sim=False # l1norm 해줌
        )
        
        self.embedding.weight.data.copy_(embed[0])
        self.initted = True
        
class ReliableEuclideanPrototypeLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim, init='kmeans', use_feature=False,) :
        super().__init__()
        self.use_feature = use_feature
        self.num_classes = num_classes
        self.init = init
        self.embedding = nn.Embedding(num_embeddings=num_classes,
                                        embedding_dim=embedding_dim)
        
        # uniform distribution initialization
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
          
                
    def forward(self, x, gt, percent, entropy):
        gt = gt.unsqueeze(1) if gt.dim()==3 else gt
        if gt.shape != x.shape:
            gt = F.interpolate(gt.float(), x.shape[-2:], mode='nearest').long()

        x_b, x_c, x_h, x_w = x.shape[:]
        flatten_x = rearrange(x, 'b c h w -> (b h w) c') # (BHW, C)
        flatten_gt = rearrange(gt, 'b c h w -> (b h w) c') # (BHW, 1)
        
        if not self.initted and self.init == 'kmeans':
            self._kmeans_init(flatten_x)
        
        if self.use_feature:
            temp = []
            for i in range(self.num_classes):
                ind = (flatten_gt == i).nonzero(as_tuple=True)
                temp.append(torch.mean(flatten_x[ind[0]], dim=0, keepdim=True)) # (1, C)
            temp = torch.cat(temp, dim=0) # (num_classes, C)
            self.embedding.weight.data.copy_(temp)
            
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training, dtype=torch.float32)
         ## entropy filtering ##
        with torch.no_grad():
            thresh = np.percentile(entropy.detach().cpu().numpy().flatten(), percent) # scalar
            # percent_unreliable = cfg.train.unsup_loss.drop_percent * (1-epoch/num_epochs)
            # drop_percent = 100 - percent_unreliable
            thresh_mask = torch.le(entropy, thresh)
        ###
        gt_embed = self.embedding.weight[flatten_gt[:, 0]]
        flatten_x = flatten_x * thresh_mask.unsqueeze(-1)
        gt_embed = gt_embed * thresh_mask.unsqueeze(-1)
        loss = loss + F.mse_loss(flatten_x, gt_embed)
        return loss
    
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