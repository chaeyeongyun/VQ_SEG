import torch
from torch import nn
import torch.nn.functional as F
from vector_quantizer.vq_img import VectorQuantizer
from models.encoders import make_encoder
from .decoder import VQVAEDecoder

class VQVAEv2(nn.Module):
    def __init__(
        self,
        encoder_name:str,
        vq_cfg:dict,
        in_channels:int=3,
        out_channels:int=3,
        hidden_channels:int=32,
        n_resblocks=2,
        n_res_channels=32,
        depth:int=5,
        stride=2):
        super().__init__()
        self.encoder = make_encoder(encoder_name, in_channels, depth)
        encoder_channels = self.encoder.out_channels()[1:]
        self.codebook = nn.ModuleList([VectorQuantizer(**vq_cfg, dim=encoder_channels[i])for i in range(depth)])
        self.decoder = VQVAEDecoder(in_channels=sum(encoder_channels), out_channels=out_channels, 
                                    hidden_channels=hidden_channels, 
                                    n_resblocks=n_resblocks, 
                                    n_res_channels=n_res_channels,
                                    stride=stride)
    
    def forward(self, x):
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        code_usage_lst = []
        for i in range(len(features)):
            quantize, _embed_index, commitment_loss, code_usage = self.codebook[i](features[i])
            features[i] = quantize
            # sum
            loss = loss + commitment_loss
            
            code_usage_lst.append(code_usage.detach().cpu())
        # mean
        loss = loss / len(features)
        size = tuple(features[0].shape[-2:])
        features = [F.interpolate(feat, size) for feat in features]
        cat_features = torch.cat(features, dim=1)
        output = self.decoder(cat_features)
        
        return output, loss, torch.tensor(code_usage_lst)
    