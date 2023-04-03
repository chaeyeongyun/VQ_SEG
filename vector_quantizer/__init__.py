from .vq_img import VectorQuantizer
from torch import nn
import copy

def make_vq_module(vq_cfg, encoder_channels, depth):
    if isinstance(vq_cfg.num_embeddings, (int)):
        codebook = nn.ModuleList([VectorQuantizer(**vq_cfg, dim=encoder_channels[i+1])for i in range(depth)])
    elif isinstance(vq_cfg.num_embeddings, list):
        l = []
        for i in range(depth):
            num_embeddings = copy.deepcopy(vq_cfg.num_embeddings)
            vq_cfg.num_embeddings = num_embeddings[i]
            l.append(VectorQuantizer(**vq_cfg, dim=encoder_channels[i+1]))
        codebook = nn.ModuleList(l)
    else:
        raise TypeError(f"{type(vq_cfg.num_embeddings)} is not available type")
    return codebook