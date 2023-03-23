import torch
from models.networks import Unet
if __name__ == '__main__':
    model = Unet('resnet50', 3)
    x = torch.randn(4, 3, 512, 512)
    output = model(x)
    a=1