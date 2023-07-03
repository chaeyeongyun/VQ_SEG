from .decoder import *
from torch import nn
from models.encoders import make_encoder
from models.modules.segmentation_head import SegmentationHead

class DeepLabV3(nn.Module):
    def __init__(
        self,
        encoder_name:str,
        num_classes:int,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=256,
        depth:int=5,
        activation=nn.Identity,
        upsampling=32) :
        super().__init__()
        self.encoder = make_encoder(encoder_name, in_channels, depth, weights=encoder_weights)    
        self.decoder = DeepLabV3Decoder(
            in_channels=self.encoder.out_channels()[-1],
            out_channels=decoder_channels,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=num_classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        output = self.segmentation_head(decoder_output)

        return output

class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        encoder_name:str,
        num_classes:int,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=256,
        depth:int=5,
        encoder_output_stride: int = 16,
        decoder_atrous_rates: tuple = (12, 24, 36),
        activation=nn.Identity,
        upsampling=4) -> None:
        super().__init__()
        self.encoder = make_encoder(encoder_name, in_channels, depth, weights=encoder_weights, output_stride=encoder_output_stride)    
        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels(),
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=num_classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling
        )
        
    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        output = self.segmentation_head(decoder_output)

        return output