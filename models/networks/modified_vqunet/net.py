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

def __init_weight(feature, init_func:nn, norm_layer, bn_eps, bn_momentum, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init_func(m.weight, **kwargs)

        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, init_func:nn, norm_layer, bn_eps, bn_momentum, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, init_func, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, init_func, norm_layer, bn_eps, bn_momentum,
                      **kwargs)
        
class VQRePTUnet1x1(nn.Module):
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
            if commitment_loss is not None: loss = loss + commitment_loss
            
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
            prototype_loss = self.prototype_loss(decoder_out, gt, percent=percent, entropy=entropy)
        else: prototype_loss = None
        output = self.upsampling(output)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst), prototype_loss
    
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
class VQRePTUnetDouble1x1(nn.Module):
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
        self.segmentation_head = nn.Sequential(nn.Conv2d(decoder_channels[-1], decoder_channels[-1]*2, 1, bias=False), nn.Conv2d(decoder_channels[-1]*2, num_classes, 1, bias=False))
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
            if commitment_loss is not None: loss = loss + commitment_loss
            
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
            prototype_loss = self.prototype_loss(decoder_out, gt, percent=percent, entropy=entropy)
        else: prototype_loss = None
        output = self.upsampling(output)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst), prototype_loss
class VQRePTUnet1x1v2(nn.Module):
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
        self.prototype_loss = ReliablePrototypeLossv2(num_classes, decoder_channels[-1], margin=margin, scale=scale, init=pt_init, use_feature=use_feature, )
        self.device = None
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        
    def forward(self, x, gt=None, code_usage_loss=False, th=None):
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
        output = self.segmentation_head(decoder_out)
        if self.training:
            prototype_loss = self.prototype_loss(decoder_out, gt, th)
        else: prototype_loss = None
        output = self.upsampling(output)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst), prototype_loss
    
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
class VQReTemPTUnet(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        use_feature=False,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Identity,
        upsampling=2,
        pt_init="kmeans",
        t = 0.1
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
        self.prototype_loss = StableTemperaturedPrototypeLoss(num_classes, decoder_channels[-1],t=t,  init=pt_init, use_feature=use_feature, )
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
            if commitment_loss is not None: loss = loss + commitment_loss
            
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
            prototype_loss = self.prototype_loss(decoder_out, gt, percent=percent, entropy=entropy)
        else: prototype_loss = None
        output = self.upsampling(output)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst), prototype_loss
    
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
class VQReEuPTUnet(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        use_feature=False,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Identity,
        upsampling=2,
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
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1],
                                                    out_channels=num_classes,
                                                    upsampling=1,
                                                    activation=activation,
                                                    kernel_size=3)
        self.prototype_loss = ReliableEuclideanPrototypeLoss(num_classes, decoder_channels[-1], use_feature=use_feature)
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
            if commitment_loss is not None: loss = loss + commitment_loss
            
            if code_usage is not None: 
                code_usage_lst.append(code_usage.detach().cpu())
                if code_usage_loss: 
                    usage_loss = usage_loss + code_usage
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
        with torch.no_grad():
            prob = rearrange(output, 'b c h w -> (b h w) c') 
            prob = torch.softmax(prob, dim=1)
            entropy = -torch.sum(prob*torch.log(prob+1e-10), dim=1) # (BHW, )
        prototype_loss = self.prototype_loss(decoder_out, gt, percent=percent, entropy=entropy) if self.training else None
        output = self.upsampling(output)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst), prototype_loss

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

class VQRePTUnet(nn.Module):
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
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1],
                                                  out_channels=num_classes,
                                                  upsampling=1,
                                                  activation=activation,
                                                  kernel_size=3)
        self.prototype_loss = ReliablePrototypeLoss(num_classes, decoder_channels[-1], margin=margin, scale=scale, use_feature=use_feature)
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
            if commitment_loss is not None: loss = loss + commitment_loss
            
            if code_usage is not None: 
                code_usage_lst.append(code_usage.detach().cpu())
                if code_usage_loss: 
                    usage_loss = usage_loss + code_usage
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
        with torch.no_grad():
            prob = rearrange(output, 'b c h w -> (b h w) c') 
            prob = torch.softmax(prob, dim=1)
            entropy = -torch.sum(prob*torch.log(prob+1e-10), dim=1) # (BHW, )
        prototype_loss = self.prototype_loss(decoder_out, gt, percent=percent, entropy=entropy) if self.training else None
        output = self.upsampling(output)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst), prototype_loss
    
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
class VQPatchUNet(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Identity,
        upsampling=2,
        mixer_depth=3 
        ):
        super().__init__()
        
        self.encoder = make_encoder(encoder_name, in_channels, depth, weights=encoder_weights)    
        encoder_channels = self.encoder.out_channels()
        self.encoder.bn1, self.encoder.relu = nn.Identity(), nn.Identity()
        self.encoder.conv1=  ConvMixer(in_channels=in_channels, dim=encoder_channels[1], depth=mixer_depth)
        init_weight([self.encoder.conv1], nn.init.kaiming_normal_, nn.BatchNorm2d, 1e-5, 0.1, 
                        mode='fan_in', nonlinearity='relu')
       
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
    def forward(self, x, code_usage_loss=False):
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
        output = self.segmentation_head(decoder_out)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst)
    
class VQIMDBNet(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:EasyDict,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Identity,
        upsampling=2,
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
        self.imdb = IMDB(encoder_channels[-1])
        self.device = None
        
    def forward(self, x):
        if self.device is None:
            self.device = x.device
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        code_usage_lst = []
      
        features[-1] = self.imdb(features[-1])
        for i in range(len(features)):
            quantize, _embed_index, commitment_loss, code_usage = self.codebook[i](features[i])
            features[i] = quantize
            # sum
            if commitment_loss is not None: loss = loss + commitment_loss
            
            if code_usage is not None: 
                code_usage_lst.append(code_usage.detach().cpu())
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
        return output, loss, torch.tensor(code_usage_lst)

            
class VQCANetv4(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:EasyDict,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Identity,
        upsampling=2,
        cca = [True, True, False, False, False]
        ):
        assert 'ccavq'  in encoder_name, "VQCANetv4 must user ccavq type encoder"
        super().__init__()
        self.encoder = make_encoder(
                                    encoder_name, 
                                    in_channels, 
                                    depth, 
                                    weights=encoder_weights,
                                    cca=cca,
                                    vq_cfg=vq_cfg)    
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
        
        self.device = None
        
    def forward(self, x):
        if self.device is None:
            self.device = x.device
        features, commitment_loss, code_usage = self.encoder(x)
        features = features[1:]
        # mean
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
        return output, commitment_loss, code_usage
    
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
            

class VQCANetv3(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:EasyDict,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Identity,
        upsampling=2,
        cca = [True, True, False, False, False]
        ):
        super().__init__()
        self.encoder = make_encoder(
                                    encoder_name, 
                                    in_channels, 
                                    depth, 
                                    weights=encoder_weights)    
        encoder_channels = self.encoder.out_channels()
        
        self.codebook = make_vq_module(vq_cfg, encoder_channels, depth)
        
        if decoder_channels == None:
            decoder_channels = [i//2 for i in encoder_channels[1:]] 
            decoder_channels = decoder_channels[::-1]
        self.decoder = CCAUnetDecoder(encoder_channels, decoder_channels, cca=cca)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1],
                                                  out_channels=num_classes,
                                                  upsampling=upsampling,
                                                  activation=activation,
                                                  kernel_size=3)
        
        self.device = None
        
    def forward(self, x):
        if self.device is None:
            self.device = x.device
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        code_usage_lst = []
        for i in range(len(features)):
            quantize, _embed_index, commitment_loss, code_usage = self.codebook[i](features[i])
            features[i] = quantize
            # sum
            if commitment_loss is not None: loss = loss + commitment_loss
            
            if code_usage is not None: 
                code_usage_lst.append(code_usage.detach().cpu())
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
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
            
            
class VQCANetv2(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Identity,
        upsampling=2,
        cca = [False, False, False, True, True]
        ):
        super().__init__()
        assert 'cca'  in encoder_name, "VQCANetv2 must user cca type encoder"
        self.encoder = make_encoder(
                                    encoder_name, 
                                    in_channels, 
                                    depth, 
                                    weights=encoder_weights, 
                                    cca=cca)    
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
        self.device = None
        
    def forward(self, x):
        if self.device is None:
            self.device = x.device
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        code_usage_lst = []
      
        for i in range(len(features)):
            quantize, _embed_index, commitment_loss, code_usage = self.codebook[i](features[i])
            features[i] = quantize
            # sum
            if commitment_loss is not None: loss = loss + commitment_loss
            
            if code_usage is not None: 
                code_usage_lst.append(code_usage.detach().cpu())
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
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
            
class VQCANet(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:EasyDict,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Identity,
        upsampling=2,
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
        self.cca = CCA(encoder_channels[-1], encoder_channels[-1])
        self.device = None
        
    def forward(self, x):
        if self.device is None:
            self.device = x.device
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        code_usage_lst = []
      
        features[-1] = self.cca(features[-1])
        for i in range(len(features)):
            quantize, _embed_index, commitment_loss, code_usage = self.codebook[i](features[i])
            features[i] = quantize
            # sum
            if commitment_loss is not None: loss = loss + commitment_loss
            
            if code_usage is not None: 
                code_usage_lst.append(code_usage.detach().cpu())
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
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
        flag = list(map(lambda x: x==0, vq_cfg.num_embeddings))
        # flag = [False, False, True, True, False]
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

class VQRePTUnetAngular(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        margin=1.5,
        scale=1.,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Softmax2d,
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
        self.segmentation_head = AngularSegmentationHeadv2( decoder_channels[-1], decoder_channels[-1], num_classes, scale=scale, margin=margin, upsampling=1, activation=activation)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.device = None
        
        
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
            if commitment_loss is not None: loss = loss + commitment_loss
            
            if code_usage is not None: 
                code_usage_lst.append(code_usage.detach().cpu())
                if code_usage_loss: 
                    usage_loss = usage_loss + code_usage
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        entropy = None
        if self.training:
            with torch.no_grad():
                self.segmentation_head.eval()
                tempoutput = self.segmentation_head(decoder_out, None, None, None)[0]
                prob = rearrange(tempoutput, 'b c h w -> (b h w) c') 
                prob = torch.softmax(prob, dim=1)
                entropy = -torch.sum(prob*torch.log(prob+1e-10), dim=1) # (BHW, )
                self.segmentation_head.train()
        output, prototype_loss = self.segmentation_head(decoder_out, gt, percent=percent, entropy=entropy)
        output = self.upsampling(output)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst), prototype_loss
    
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

class VQRePTUnetAngularv3(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        margin=1.5,
        scale=1.,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Softmax2d,
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
        self.segmentation_head = AngularSegmentationHeadv3( decoder_channels[-1], decoder_channels[-1], num_classes, scale=scale, margin=margin, upsampling=1, activation=activation)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.device = None
        
        
    def forward(self, x, split=None, pred=None, code_usage_loss=False, th=None):
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
        output, prototype_loss = self.segmentation_head(decoder_out, pred, split=split, th=th)
        output = self.upsampling(output)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst), prototype_loss
    
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