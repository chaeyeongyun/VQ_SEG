import torch
from models.networks import VQUnet_v1, VQVAEv2
import tarfile
if __name__ == '__main__':
    # model = VQVAEv2('resnet50', {
    #             "num_embeddings":2048,
    #             "distance":"euclidean"
    #         })
    # x = torch.randn(4, 3, 512, 512)
    # output = model(x)
    # a=1
    with tarfile.open("mytext.tar.gz",'w:gz') as mytar:
        mytar.add('./ttt')