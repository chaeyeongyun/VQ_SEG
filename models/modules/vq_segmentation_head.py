import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat


def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def sample_vectors(sample, num):
    num_samples, device = sample.shape[0], sample.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return sample[indices]

def batched_sample_vectors(samples, num): # kmeans에 sample_fn으로 들어감
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim = 0)], dim = 0)

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

class CosinesimSegHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_embeddings,
        kmeans_init,
        kmeans_iters,
        decay,
        eps,
        num_codebook
        ):
 
        super().__init__()
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.initted = False
        
        self.num_codebook = num_codebook
        self.decay = decay
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if not kmeans_init: 
            self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
            self.initted = True
        # nn.init.kaiming_uniform_(self.embedding.weight.data)
    def forward(self, x):
        '''x shape : (B, HxW, C)'''
        x = x.float()
        x_shape, dtype = x.shape, x.dtype
        flatten_x = rearrange(x, 'b ... c -> b (...) c')
        flatten_x = l2norm(flatten_x)
        if self.kmeans_init and self.training:
            self._kmeans_init(flatten_x)
        self.embedding.weight.data.copy_(l2norm(self.embedding.weight.data))
        
        b, hw, c = flatten_x.shape[:]
        flatten_x = flatten_x.contiguous().view((b*hw, c)) # (BxHxW, C)
        distance = einsum('n d, e d -> n e', flatten_x, self.embedding.weight)
        distance = distance.contiguous().view((b, hw, self.num_embeddings))
       
        embed_idx = torch.argmax(distance, dim=-1) # (N, 모든 픽셀 수)
        embed_idx_onehot = F.one_hot(embed_idx, num_classes=self.num_embeddings) # (N, 모든 픽셀 수, num_embeddings)
        quantized = torch.matmul(embed_idx_onehot.float(), self.embedding.weight)
        
        # code_usage
        codebook_cnt = torch.bincount(embed_idx.reshape(-1), minlength=self.num_embeddings)
        zero_cnt = (codebook_cnt == 0).sum()
        code_usage = 100 * (zero_cnt / self.num_embeddings) # 낮을 수록 좋음

        return quantized, distance, embed_idx, code_usage
    
    def _kmeans_init(self, flatten_x):    
        if self.initted:
            return
        
        embed, cluster_size = kmeans(
            flatten_x,
            self.num_embeddings,
            self.kmeans_iters,
            use_cosine_sim=True
        )
        
        self.embedding.weight.data.copy_(embed[0])
        self.initted = True


class EuclideanSegHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_embeddings,
        kmeans_init,
        kmeans_iters,
        decay,
        eps,
        num_codebook
        ):
 
        super().__init__()
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.initted = False
        
        self.num_codebook = num_codebook
        self.decay = decay
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if not kmeans_init: 
            self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
            self.initted = True
        # nn.init.kaiming_uniform_(self.embedding.weight.data)
    def forward(self, x):
        '''x shape : (B, C, H, W)'''
        dtype = x.dtype
        x_shape = x.shape
        x = x.float()
        flatten_x = rearrange(x, 'b ... c -> b (...) c')
        if self.kmeans_init and self.training:
            self._kmeans_init(flatten_x)
        distance = torch.cdist(flatten_x, self.embedding.weight, p=2) # (N, 모든 픽셀 수, num_embeddings)
        embed_idx = torch.argmin(distance, dim=-1) # (N, 모든 픽셀 수)
        
        embed_idx_onehot = F.one_hot(embed_idx, num_classes=self.num_embeddings) # (N, 모든 픽셀 수, num_embeddings)
        quantized = torch.matmul(embed_idx_onehot.float(), self.embedding.weight)
        
        # code_usage
        codebook_cnt = torch.bincount(embed_idx.reshape(-1), minlength=self.num_embeddings)
        zero_cnt = (codebook_cnt == 0).sum()
        code_usage = 100 * (zero_cnt / self.num_embeddings) # 낮을 수록 좋음

        return quantized, distance, embed_idx, code_usage
    
    def _kmeans_init(self, flatten_x):    
        if self.initted:
            return
        
        embed, cluster_size = kmeans(
            flatten_x,
            self.num_embeddings,
            self.kmeans_iters,
        )
        
        self.embedding.weight.data.copy_(embed[0])
        self.initted = True


class VQSegmentationHead(nn.Module):
    def __init__(
        self,
        dim,
        num_embeddings,
        embedding_dim=None,
        decay=0.8,
        eps=1e-5,
        kmeans_init=False,
        kmeans_iters=10,
        distance='euclidean',
        commitment_weight = 1,  
        num_codebook = 1,
        activation=nn.Softmax2d
        ):
        super().__init__()
        
        embedding_dim = embedding_dim if embedding_dim!=None else dim
        self.num_embeddings = num_embeddings
        # TODO: projection require
        self.eps = eps
        self.commitment_weight = commitment_weight
        self.code_distance = distance
        codebook_dict = {'euclidean':EuclideanSegHead,
                         'cosine':CosinesimSegHead,}
        codebook_class = codebook_dict[distance]
        
        self.codebook = codebook_class(
            embedding_dim=embedding_dim,
            num_embeddings = num_embeddings,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            decay = decay,
            eps = eps,
            num_codebook=num_codebook)
        self.activation = activation()
        
    def forward(self, x):
        x = x.to(torch.float32)
        x_shape, device = x.shape, x.device
        x_h, x_w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c') # (B, HxW, C)
        quantize, distance, embed_idx,  code_usage = self.codebook(x)
        loss = torch.tensor([0.], device=device, requires_grad=self.training, dtype=torch.float32)
        if self.training:
            quantize = x + (quantize - x).detach() # preserve gradients
            if self.commitment_weight > 0:
                detached_quantize = quantize.detach() # codebook 쪽 sg 연산
                commitment_loss = F.mse_loss(detached_quantize, x) # encoder update
                loss = loss + commitment_loss * self.commitment_weight
        if self.code_distance == "euclidean":
            score = rearrange(distance, 'b (h w) c -> b c h w', h=x_h, w=x_w)  
            score =  1 - (score / torch.sum(score, dim=1, keepdim=True))
        else:
            score = rearrange(distance, 'b (h w) c -> b c h w', h=x_h, w=x_w)
        score = self.activation(score)  
        quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=x_h, w=x_w)
        embed_index = rearrange(embed_idx, 'b (h w) ... -> b h w ...', h=x_h, w=x_w)
        return quantize, score, embed_index, loss, code_usage