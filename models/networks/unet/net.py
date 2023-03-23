import torch
import torch.nn as nn
from models.networks.unet.decoder import UnetDecoder
from models.encoders import make_encoder
from models.modules.segmentation_head import SegmentationHead

class Unet(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Identity,
        upsampling=2,
        is_vq=True):
        super().__init__()
        self.encoder = make_encoder(encoder_name, in_channels, depth)
        encoder_channels = self.encoder.out_channels()
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
        features = self.encoder(x)
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
        return output

