from .decoder import *
from torch import nn
from models.encoders import make_encoder
from models.modules.segmentation_head import SegmentationHead
from models.modules.attention import SKA
class SemiWeedNet(nn.Module):
    def __init__(
        self,
        encoder_name:str,
        num_classes:int,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=256,
        depth:int=5,
        encoder_output_stride: int = 16,
        decoder_atrous_rates: tuple = (6, 12, 18),
        activation=nn.Identity,
        upsampling=4) -> None:
        super().__init__()
        self.encoder = make_encoder(encoder_name, in_channels, depth, weights=encoder_weights, output_stride=encoder_output_stride)    
        encoder_channels = self.encoder.out_channels()
        self.ska = SKA(encoder_channels[-1], encoder_channels[-1])
        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=encoder_channels,
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
        self.mlp = nn.Sequential(
            nn.Linear(encoder_channels[-1], encoder_channels[-1]), 
            nn.ReLU(), 
            nn.Linear(encoder_channels[-1],128)
            )
    
    def forward(self, x, issup=False):
        features = self.encoder(x)
        last_feature = features[-1]
        features[-1] = self.ska(features[-1])        
        decoder_output = self.decoder(*features)

        output = self.segmentation_head(decoder_output)
        if self.training and not issup:
            mlp_out = F.adaptive_avg_pool2d(last_feature, output_size=(1,1))
            mlp_out = torch.flatten(mlp_out, 1)
            mlp_out = self.mlp(mlp_out)
            return output, mlp_out
        
        return output, None
    
    