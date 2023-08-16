from typing import Optional
from easydict import EasyDict
import torch
import torch.nn as nn
from models.networks.unet.decoder import UnetDecoder, CCAUnetDecoder
from models.encoders import make_encoder
from models.modules.segmentation_head import SegmentationHead, AngularSegmentationHeadv2, AngularSegmentationHeadv3
from models.modules.prototype import *
from models.modules.attention import make_attentions, DRSAM, CCA, IMDB
from models.modules.conv_mixer import *
from vector_quantizer import make_vq_module

class NoPT(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        margin=1.5,
        scale=1.,
        use_feature=False,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Identity,
        upsampling=2,
        pt_init="kmeans"
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
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], num_classes, 1, bias=False)
        self.prototype_loss = ReliablePrototypeLoss(num_classes, decoder_channels[-1], margin=margin, scale=scale, init=pt_init, use_feature=use_feature, )
        self.device = None
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        
    def forward(self, x, gt=None, code_usage_loss=False, percent=None):
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
            # if commitment_loss is not None: loss = loss + commitment_loss
            if commitment_loss: loss = loss + commitment_loss
            
            if code_usage is not None: 
                code_usage_lst.append(code_usage.detach().cpu())
                if code_usage_loss: 
                    usage_loss = usage_loss + code_usage
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
        prototype_loss = torch.tensor([0.], device=x.device, requires_grad=False)
        output = self.upsampling(output)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst), prototype_loss

class Scheme6(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        margin=1.5,
        scale=1.,
        use_feature=False,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Identity,
        upsampling=2,
        pt_init="kmeans"
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
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], num_classes, 1, bias=False)
        self.prototype_loss = PrototypeLoss(num_classes, decoder_channels[-1], margin=margin, scale=scale, init=pt_init, use_feature=use_feature, )
        self.device = None
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        
    def forward(self, x, gt=None, code_usage_loss=False, percent=None):
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
            # if commitment_loss is not None: loss = loss + commitment_loss
            if commitment_loss: loss = loss + commitment_loss
            
            if code_usage is not None: 
                code_usage_lst.append(code_usage.detach().cpu())
                if code_usage_loss: 
                    usage_loss = usage_loss + code_usage
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
        if self.training:
            with torch.no_grad():
                prob = rearrange(output, 'b c h w -> (b h w) c') 
                prob = torch.softmax(prob, dim=1)
                entropy = -torch.sum(prob*torch.log(prob+1e-10), dim=1) # (BHW, )
            prototype_loss = self.prototype_loss(decoder_out, gt)
        else: prototype_loss = None
        output = self.upsampling(output)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst), prototype_loss
