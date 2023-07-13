from .resnet import resnet_encoders, ResNetEncoder, CCAResNetEncoder, CCAVQResNetEncoder
from .vgg import vgg_encoders, VGGEncoder
from .pretrained_settings import *
import torch.utils.model_zoo as model_zoo

import re

def make_encoder(name:str, in_channels:int=3, depth:int=5, weights=None, padding_mode='zeros', output_stride=32, **kwargs):
    if 'resnet' in name:
        if "ccavq" in name:
            name = re.sub("ccavq", "", name)
            params = resnet_encoders[name]["params"]    
            encoder = CCAVQResNetEncoder(depth=depth, **params, in_channels=in_channels, padding_mode=padding_mode, **kwargs)
        elif "cca" in name: 
            name = re.sub("cca", "", name)
            params = resnet_encoders[name]["params"]    
            encoder = CCAResNetEncoder(depth=depth, **params, in_channels=in_channels, padding_mode=padding_mode, **kwargs)
        else:
            params = resnet_encoders[name]["params"]
            encoder = ResNetEncoder(depth=depth, **params, in_channels=in_channels, padding_mode=padding_mode, **kwargs)
    if 'vgg' in name:
        params = vgg_encoders[name]["params"]
        encoder = VGGEncoder(depth=depth, **params, in_channels=in_channels, **kwargs)
    if weights is not None:
        if "imagenet" in weights:
            load_settings = pretrain_settings[name][weights]
        else:
            assert NotImplementedError('It''s not available weights option' )
        encoder.load_state_dict(model_zoo.load_url(load_settings["url"]))
    if output_stride != 32:
        encoder.make_dilated(output_stride)
    return encoder