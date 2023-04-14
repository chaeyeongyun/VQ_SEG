from tqdm import tqdm
import os
from glob import glob
import torch
from models.networks import VQUnet_v1, VQVAEv2
import tarfile
from saliency_map import get_saliency_rbd
import matplotlib.pyplot as plt
if __name__ == '__main__':
    # model = VQVAEv2('resnet50', {
    #             "num_embeddings":2048,
    #             "distance":"euclidean"
    #         })
    # x = torch.randn(4, 3, 512, 512)
    # output = model(x)
    # a=1
    # with tarfile.open("mytext.tar.gz",'w:gz') as mytar:
    #     mytar.add('./ttt')
    folders  = ["/content/data/semi_sup_data/CWFID/num30/train/input", "/content/data/semi_sup_data/CWFID/num30/test/input"]
    save = "/content/data/semi_sup_data/CWFID/salient_map"
    for folder in folders:
        imgs = glob(os.path.join(folder, "*.png"))
        for img in tqdm(imgs):
            rbd = get_saliency_rbd(img).astype('uint8')
            filename = os.path.split(img)[-1]
            plt.imsave(os.path.join(save, filename), rbd, cmap='gray')