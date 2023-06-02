import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvMixerLayer(nn.Sequential):
    def __init__(self, dim, kernel_size=9, patch_size=2):
        super().__init__(Residual(
                                        nn.Sequential(
                                            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                                            nn.ReLU(),
                                            nn.BatchNorm2d(dim))
                                        ),
                                    nn.Conv2d(dim, dim, kernel_size=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(dim)) 
                         
class ConvMixer(nn.Module):
    def __init__(self, in_channels, dim, depth, kernel_size=9, patch_size=2):
        super().__init__()
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.BatchNorm2d(dim)
        )
        self.mix_layers = nn.Sequential(*[ConvMixerLayer(dim, kernel_size, patch_size) for _ in range(depth)])
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.mix_layers(x)
        return x
        