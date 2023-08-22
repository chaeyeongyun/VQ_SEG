from .decoder import FCNHead
import torch.nn.functional as F
from models.encoders import make_encoder
import numpy as np
import torch
import torch.nn as nn


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN32s(nn.Module):
    def __init__(
        self,
        encoder_name:str,
        num_classes:int,
        encoder_weights=None,
        in_channels=3,
    ):
        super(FCN32s, self).__init__()
        self.encoder = make_encoder(encoder_name, in_channels=in_channels, weights=encoder_weights)
        encoder_channels = self.encoder.out_channels()
        self.decoder = FCNHead(encoder_channels[-1], num_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]
        output = self.encoder(x)
        output = self.decoder(output[-1])
        if input_shape != output.shape[-2:]:
            output = F.interpolate(output, size=input_shape, mode="bilinear", align_corners=False)
        return output, None
    
    def _initialize_weights(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)


   