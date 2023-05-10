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
# from easydict import EasyDict
# from models.networks.unet import VQPTUnet
# import torch
# model = VQPTUnet('resnet50', 3, 
#                 vq_cfg=EasyDict({
#                 "num_embeddings":[0, 0, 512, 512, 512],
#                 "distance":"euclidean",
#                 "kmeans_init": True
#                 }),
#                 )
# x = torch.randn(2, 3, 256, 256)
# gt = torch.randint(0, 2, (2, 1, 256, 256), requires_grad=False).float()
# # model(x, gt)
# from torch import nn
# import torch
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# window_size = [3, 3]
# num_heads = 10
#  # define a parameter table of relative position bias
# relative_position_bias_table = nn.Parameter(
# torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

# # get pair-wise relative position index for each token inside the window
# coords_h = torch.arange(window_size[0])
# coords_w = torch.arange(window_size[1])
# coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
# coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
# relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
# relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
# relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
# relative_coords[:, :, 1] += window_size[1] - 1
# relative_coords[:, :, 0] *= 2 * window_size[1] - 1
# relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
# trunc_normal_(relative_position_bias_table, std=.02)
# a=1
import torch
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

 # calculate attention mask for SW-MSA
H, W = 8, 8
window_size = 4
shift_size = 2
img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
h_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
w_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
cnt = 0
for h in h_slices:
    for w in w_slices:
        img_mask[:, h, w, :] = cnt
        cnt += 1

mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
mask_windows = mask_windows.view(-1, window_size * window_size)
attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
z=1