from models.encoders.resnet import CCAResNetEncoder
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
params = {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
        }
m = CCAResNetEncoder(**params)
print(m)