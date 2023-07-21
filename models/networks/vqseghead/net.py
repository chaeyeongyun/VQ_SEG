from typing import Optional
from easydict import EasyDict
import torch
import torch.nn as nn
from models.networks.unet.decoder import UnetDecoder, CCAUnetDecoder
from models.encoders import make_encoder
from models.modules.vq_segmentation_head import VQSegmentationHead
from models.modules.prototype import *
from models.modules.conv_mixer import *
from vector_quantizer import make_vq_module
class VQSegHeadNet(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        margin=0.5,
        scale=30.0,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Softmax2d,
        upsampling=2,
        pt_init="kmeans",
        seghead_distance="euclidean"
        ):
        super().__init__()
        
        self.encoder = make_encoder(encoder_name, in_channels, depth, weights=encoder_weights, padding_mode='reflect')    
        encoder_channels = self.encoder.out_channels()
        self.codebook = make_vq_module(vq_cfg, encoder_channels, depth)
        
        if decoder_channels == None:
            decoder_channels = [i//2 for i in encoder_channels[1:]] 
            decoder_channels = decoder_channels[::-1]
        self.decoder = UnetDecoder(encoder_channels,
                                   decoder_channels)
        self.segmentation_head = VQSegmentationHead(dim = decoder_channels[-1], 
                                                                                            num_embeddings=num_classes,
                                                                                            kmeans_init=pt_init=="kmeans",
                                                                                            distance=seghead_distance,
                                                                                            activation=activation)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.device = None
        
        
    def forward(self, x, code_usage_loss=False):
        if self.device is None:
            self.device = x.device
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        code_usage_lst = []
        if code_usage_loss : usage_loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        for i in range(len(features)):
            quantize, _embed_index, commitment_loss, code_usage = self.codebook[i](features[i])
            features[i] = quantize
            # sum
            if commitment_loss is not None: loss = loss + commitment_loss
            
            if code_usage is not None: 
                code_usage_lst.append(code_usage.detach().cpu())
                if code_usage_loss: 
                    usage_loss = usage_loss + code_usage
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        _, output, _, prototype_loss, seghead_code_usage = self.segmentation_head(decoder_out)
        output = self.upsampling(output)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst), prototype_loss, seghead_code_usage
    
    @torch.no_grad()
    def pseudo_label(self, x):
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        for i in range(len(features)):
            quantize, _, _, _ = self.codebook[i](features[i])
            features[i] = quantize
        
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
        return torch.argmax(output, dim=1).long()