import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class CosinesimCodebook(nn.Module):
    # TODO: 
    def __init__(self):
        super().__init__()


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_embeddings,
        kmeans_init,
        kmeans_iters,
        decay,
        eps
        ):
        super().__init__()
        self.decay = decay
        if kmeans_init:
            # TODO: 
            raise NotImplementedError
        else:
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
            # nn.init.kaiming_uniform_(self.embedding.weight.data)
    def forward(self, x):
        x = x.float()
        x_shape, dtype = x.shape, x.dtype
        flatten_x = rearrange(x, 'b ... c -> b (...) c')
        
        distance = torch.cdist(flatten_x, self.embedding.weight, p=2) # (N, 모든 픽셀 수, num_embeddings)
        embed_idx = torch.argmin(distance, dim=-1) # (N, 모든 픽셀 수)
        embed_idx_onehot = F.one_hot(embed_idx, num_classes=self.num_embeddings) # (N, 모든 픽셀 수, num_embeddings)
        quantized = torch.matmul(embed_idx_onehot.float(), self.embedding.weight)
        
        # code_usage
        codebook_cnt = torch.bincount(embed_idx.reshape(-1), minlength=self.num_embeddings)
        zero_cnt = (codebook_cnt == 0).sum()
        code_usage = (zero_cnt / self.num_embeddings) # 낮을 수록 좋음

        return quantized, embed_idx, code_usage
        
        
        



class VectorQuantizer(nn.Module):
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
        ):
        super().__init__()
        embedding_dim = embedding_dim if embedding_dim!=None else dim
        self.num_embeddings = num_embeddings
        # TODO: projection require
        self.eps = eps
        self.commitment_weight = commitment_weight
        
        codebook_dict = {'euclidean':EuclideanCodebook,
                         'cosine':CosinesimCodebook,}
        codebook_class = codebook_dict[distance]
        
        self.codebook = codebook_class(
            embedding_dim=embedding_dim,
            num_embeddings = num_embeddings,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            decay = decay,
            eps = eps)
        
    def forward(self, x):
        x = x.to(torch.float32)
        x_shape, device = x.shape, x.device
        x_h, x_w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c') # (B, HxW, C)
        quantize, embed_idx, code_usage = self.codebook(x)
        loss = torch.tensor([0.], device=device, requires_grad=self.training, dtype=torch.float32)
        if self.training:
            quantize = x + (quantize - x).detach() # preserve gradients
            if self.commitment_weight > 0:
                detached_quantize = quantize.detach() # codebook 쪽 sg 연산
                commitment_loss = F.mse_loss(detached_quantize, x) # encoder update
                loss = loss + commitment_loss * self.commitment_weight
            
        quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=x_h, w=x_w)
        embed_index = rearrange(embed_idx, 'b (h w) ... -> b h w ...', h=x_h, w=x_w)
        return quantize, embed_index, loss, code_usage