# from tqdm import tqdm
# import os
# from glob import glob
# import torch
# from models.networks import VQUnet_v1, VQVAEv2
# import tarfile
# from saliency_map import get_saliency_rbd
# import matplotlib.pyplot as plt
# if __name__ == '__main__':
#     # model = VQVAEv2('resnet50', {
#     #             "num_embeddings":2048,
#     #             "distance":"euclidean"
#     #         })
#     # x = torch.randn(4, 3, 512, 512)
#     # output = model(x)
#     # a=1
#     # with tarfile.open("mytext.tar.gz",'w:gz') as mytar:
#     #     mytar.add('./ttt')
#     folders  = ["/content/data/semi_sup_data/CWFID/num30/train/input", "/content/data/semi_sup_data/CWFID/num30/test/input"]
#     # folders  = ["/content/data/semi_sup_data/CWFID/num30/test/input"]
#     save = "/content/data/semi_sup_data/CWFID/salient_map_2"
#     os.makedirs(save, exist_ok=True)
#     for folder in folders:
#         imgs = glob(os.path.join(folder, "*.png"))
#         params = {"n_segments":250,
#                   "sigma_clr":5.0,
#                   "sigma_bndcon":0.4,
#                   "sigma_spa":8,
#                   "mu":0.1}
#         f = open(os.path.join(save, 'params.txt'), 'w')
#         f.write(f"{params}")
#         f.close()
#         for img in tqdm(imgs):
#             rbd = rbd = get_saliency_rbd(img, 
#                      **params).astype('uint8')
#             filename = os.path.split(img)[-1]
#             plt.imsave(os.path.join(save, filename), rbd, cmap='gray')
# class Test():
#     def __init__(self):
#         print("exec")
#         self.init = False
#     def initialize(self):
#         self.init = True
# for i in range(5):
#     print(f"i: {i}")
#     obj = Test()
    
#     obj.initialize()
from easydict import EasyDict
from models.networks.unet import VQPTUnet
import torch
model = VQPTUnet('resnet50', 3, 
                vq_cfg=EasyDict({
                "num_embeddings":[0, 0, 512, 512, 512],
                "distance":"euclidean",
                "kmeans_init": True
                }),
                )
x = torch.randn(2, 3, 256, 256)
gt = torch.randint(0, 2, (2, 1, 256, 256), requires_grad=False).float()
model(x, gt)
