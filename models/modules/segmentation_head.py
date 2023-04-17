import torch.nn as nn
class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1, activation:nn.Module=nn.Identity):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = activation() if activation!=nn.Softmax else activation(dim=1)
        super().__init__(conv2d, upsampling, activation)