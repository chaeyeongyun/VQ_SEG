import torch
from models.networks import VQUnet_v1
if __name__ == '__main__':
    model = VQUnet_v1('resnet50', 3, {
                "dim":2048,
                "num_embeddings":2048,
                "distance":"euclidean"
            })
    x = torch.randn(4, 3, 512, 512)
    output = model(x)
    a=1