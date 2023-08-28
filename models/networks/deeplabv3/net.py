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
        upsampling=4):
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
        input_shape = x.shape[-2:]
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        output = self.segmentation_head(decoder_output)
        if input_shape != output.shape[-2:]:
            output = F.interpolate(output, input_shape, mode='bilinear', align_corners=False)
        return output, None
    
class UniMatch(nn.Module):
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
        upsampling=4):
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
        
    def forward(self, x, need_fp=False):
        input_shape = x.shape[-2:]
        features = self.encoder(x)
        c1, c4 = features[1], features[-1]
        if need_fp:
            decoder_output = self.decoder(torch.cat((c1, nn.Dropout2d(0.5)(c1))), torch.cat((features[2], features[2])), torch.cat((features[3], features[3])), torch.cat((features[4], features[4])),
                                torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            output = self.segmentation_head(decoder_output)
            if input_shape != output.shape[-2:]:
                output = F.interpolate(output, input_shape, mode='bilinear', align_corners=False)
            out, out_fp = output.chunk(2)
            return out, out_fp
        
        decoder_output = self.decoder(*features)
        output = self.segmentation_head(decoder_output)
        if input_shape != output.shape[-2:]:
            output = F.interpolate(output, input_shape, mode='bilinear', align_corners=False)
        return output, None