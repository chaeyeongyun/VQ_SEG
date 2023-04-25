from .resnet import resnet_encoders, ResNetEncoder
from .pretrained_settings import *
import torch.utils.model_zoo as model_zoo

def make_encoder(name:str, in_channels:int, depth:int=5, weights=None, **kwargs):
    if 'resnet' in name:
        params = resnet_encoders[name]["params"]
        encoder = ResNetEncoder(depth=depth, **params, in_channels=in_channels)
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