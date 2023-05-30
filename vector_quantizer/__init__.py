from .vq_img import VectorQuantizer
from torch import nn
import copy
from easydict import EasyDict
def make_vq_module(vq_cfg:EasyDict, encoder_channels, depth):
    if isinstance(vq_cfg.num_embeddings, (int)):
        codebook = nn.ModuleList([VectorQuantizer(**vq_cfg, dim=encoder_channels[i+1])for i in range(depth)])
    elif isinstance(vq_cfg.num_embeddings, list):
        assert depth==len(vq_cfg.num_embeddings), "depth and length of vq_cfg.num_embeddings must to be same number"
        lst = []
        vq_cfg = copy.deepcopy(vq_cfg)
        num_embeddings = vq_cfg.num_embeddings
        for i, num_embed in enumerate(num_embeddings):
            vq_cfg.num_embeddings = num_embed
            if num_embed == 0:
                lst.append(Identity())
            elif num_embed > 0:
                lst.append(VectorQuantizer(**vq_cfg, dim=encoder_channels[i+1]))
            else:
                raise ValueError(f"{num_embed} is not available number of embeddings")
            
        codebook = nn.ModuleList(lst)
    else:
        raise TypeError(f"{type(vq_cfg.num_embeddings)} is not available type")
    return codebook

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Identity()
    def forward(self, x):
        return self.embedding(x), None, None, None