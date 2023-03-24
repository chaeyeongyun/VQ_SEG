import torch
import torch.nn as nn
from models.networks.unet.decoder import UnetDecoder
from models.encoders import make_encoder
from models.modules.segmentation_head import SegmentationHead
from vector_quantizer.vq_img import VectorQuantizer

class VQUnet_v1(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Identity,
        upsampling=2,
        ):
        super().__init__()
        
        self.encoder = make_encoder(encoder_name, in_channels, depth)    
        encoder_channels = self.encoder.out_channels()
        self.codebook = VectorQuantizer(**vq_cfg, dim=encoder_channels[-1])
        if decoder_channels == None:
            decoder_channels = [i//2 for i in encoder_channels[1:]] 
            decoder_channels = decoder_channels[::-1]
        self.decoder = UnetDecoder(encoder_channels,
                                   decoder_channels)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1],
                                                  out_channels=num_classes,
                                                  upsampling=upsampling,
                                                  activation=activation,
                                                  kernel_size=3)
    def forward(self, x):
        features = self.encoder(x)[1:]
        quantize, _embed_index, commitment_loss, code_usage = self.codebook(features[-1]) 
        features[-1] = quantize
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
        return output, commitment_loss, code_usage

class VQUnet_v2(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Identity,
        upsampling=2,
        ):
        super().__init__()
        
        self.encoder = make_encoder(encoder_name, in_channels, depth)    
        encoder_channels = self.encoder.out_channels()
        self.codebooks = nn.ModuleList([VectorQuantizer(**vq_cfg, dim=encoder_channels[i+1])for i in range(depth)])
        
        if decoder_channels == None:
            decoder_channels = [i//2 for i in encoder_channels[1:]] 
            decoder_channels = decoder_channels[::-1]
        self.decoder = UnetDecoder(encoder_channels,
                                   decoder_channels)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1],
                                                  out_channels=num_classes,
                                                  upsampling=upsampling,
                                                  activation=activation,
                                                  kernel_size=3)
    def forward(self, x):
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebooks) : raise NotImplementedError
        
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        code_usage_lst = []
        for i in range(len(features)):
            quantize, _embed_index, commitment_loss, code_usage = self.codebooks[i](features[i])
            features[i] = quantize
            # sum
            loss = loss + commitment_loss
            
            code_usage_lst.append(code_usage.detach().cpu())
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
        return output, loss, torch.tensor(code_usage_lst)
