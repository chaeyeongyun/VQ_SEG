from typing import Optional

import torch
import torch.nn as nn
from models.networks.unet.decoder import UnetDecoder
from models.encoders import make_encoder
from models.modules.segmentation_head import SegmentationHead, AngularSegmentationHead
from models.modules.prototype import *
from models.modules.attention import make_attentions, DRSAM
from vector_quantizer import make_vq_module


class DRSAVQUnet(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Softmax2d,
        upsampling=2,
        attention=DRSAM
        ):
        super().__init__()
        
        self.encoder = make_encoder(encoder_name, in_channels, depth, weights=encoder_weights)    
        encoder_channels = self.encoder.out_channels()
   
        self.codebook = make_vq_module(vq_cfg, encoder_channels, depth)
        
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
        # flag = list(map(lambda x: x==0, vq_cfg.num_embeddings))
        flag = [False, False, True, True, False]
        self.attention = make_attentions(attention, encoder_channels[1:], flag)
        
    def forward(self, x, code_usage_loss=False):
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        code_usage_lst = []
        if code_usage_loss : usage_loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        for i in range(len(features)):
            features[i] = self.attention[i](features[i])
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
        output = self.segmentation_head(decoder_out)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst)
    
    def load_pretrained(self, encoder_weights_path, codebook_weights_path):
        print(f"... load pretrained weights ... \nencoder:{encoder_weights_path}, codebook:{codebook_weights_path}")
        encoder_weights = torch.load(encoder_weights_path)
        codebook_weights = torch.load(codebook_weights_path)
        self.encoder.load_state_dict(encoder_weights)
        self.codebook.load_state_dict(codebook_weights)
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        for param in self.codebook.parameters():
            param.requires_grad = False