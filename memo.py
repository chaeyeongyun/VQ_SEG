from models.networks.modified_vqunet import *
from easydict import EasyDict
if __name__ == "__main__":
    params = EasyDict({
            "encoder_name":"resnet50",
            "num_classes":3,
            "depth": 5,
            "vq_cfg":{
                "num_embeddings":[0, 0, 512, 512, 512],
                "distance":"euclidean",
                "kmeans_init": True
                },
            "encoder_weights":"imagenet_swsl",
            "mixer_depth":3
            })
    model = VQPatchUNet(**params)
    dummy = torch.randn(2, 3, 32, 32)
    output = model(dummy)
    a= 1
    