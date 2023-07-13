from torch import nn
class FCNHead(nn.Sequential):
    def __init__(self, in_channels, num_classes) :
        modules = []
        # fc6
        modules.append(nn.Conv2d(in_channels, 4096, 7))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Dropout2d())

        # fc7
        modules.append(nn.Conv2d(4096, 4096, 1))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Dropout2d())

        modules.append(nn.Conv2d(4096, num_classes, 1))
        modules.append( nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32,
                                          bias=False))
        super().__init__(*modules)