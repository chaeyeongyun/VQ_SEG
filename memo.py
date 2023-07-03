from models.networks.deeplabv3 import *
from easydict import EasyDict
if __name__ == "__main__":
    params = EasyDict({
            "encoder_name":"resnet50",
            "num_classes":3,
            "depth": 5,
            "encoder_weights":"imagenet_swsl",
            })
    model = DeepLabV3(**params)
    dummy = torch.randn(2, 3, 32, 32)
    output = model(dummy)
    a= 1
    