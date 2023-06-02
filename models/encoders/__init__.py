from .resnet import resnet_encoders, ResNetEncoder, CCAResNetEncoder, CCAVQResNetEncoder
from .convnext import convnext_encoders, ConvNextEncoder
from .pretrained_settings import *
import torch.utils.model_zoo as model_zoo

import re

def make_encoder(name:str, in_channels:int=3, depth:int=5, weights=None, padding_mode='zeros', **kwargs):
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
    if 'convnext' in name:
        params = convnext_encoders[name]["params"]
        encoder = ConvNextEncoder(depth=depth, **params, **kwargs)
        
    if weights is not None:
        if weights == "imagenet":
            settings = ImageNet()
            load_settings = settings.classifier_settings[name][weights]
        elif weights == "imagenet_ssl":
            settings = ImageNet()
            load_settings = settings.self_sup_settings[name]["ssl"]
        elif weights == "imagenet_swsl":
            settings = ImageNet()
            load_settings = settings.self_sup_settings[name]["swsl"]
        else:
            assert NotImplementedError('It''s not available weights option' )
        encoder.load_state_dict(model_zoo.load_url(load_settings["url"]))

    return encoder