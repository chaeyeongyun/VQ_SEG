from .resnet import resnet_encoders, ResNetEncoder

def  make_encoder(name:str, in_channels:int, depth:int=5, **kwargs):
    if 'resnet' in name:
        params = resnet_encoders[name]["params"]
        encoder = ResNetEncoder(depth=depth, **params, in_channels=in_channels)

    return encoder