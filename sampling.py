from tqdm import tqdm
from PIL import Image
import os
import os.path as osp
from glob import glob
from shutil import copy, rmtree, copytree
import random
import re

def bonirob_sampling(percent, data_root="/content/data/cropweed_total/IJRR2017/seg", save_root="/content/data/semi_sup_data/IJRR2017"):
    total = 400
    save_root = osp.join(save_root, f"percent_{percent}")
    save_input = osp.join(save_root, "train", "input")
    save_target = osp.join(save_root, "train", "target")
    rmtree(save_root, ignore_errors=True)
    # os.makedirs(save_input, exist_ok=True)
    os.makedirs(save_target, exist_ok=True)
    dataset = osp.split(osp.split(data_root)[0])[1]
    assert dataset == "IJRR2017", "this function is available to BoniRob dataset"
    
    images = glob(osp.join(data_root, "train","input",  "*.png"))
    indexes = random.sample(range(len(images)), int(total*percent/100))
    for index in tqdm(indexes):
        image = images[index]
        filename = osp.split(image)[1]
        target = osp.join(osp.split(osp.split(image)[0])[0], "target", filename)
        copy(target, osp.join(save_target, filename))
    copytree(f"/content/data/semi_sup_data/{dataset}/num30/test", osp.join(save_root, "test"))
    copytree(f"/content/data/semi_sup_data/{dataset}/num30/train/input", save_input)
    
def sampling(data_root, save_root, total, percent):
    save_root = osp.join(save_root, f"percent_{percent}")
    save_input = osp.join(save_root, "train", "input")
    save_target = osp.join(save_root, "train", "target")
    rmtree(save_root, ignore_errors=True)
    # os.makedirs(save_input, exist_ok=True)
    os.makedirs(save_target, exist_ok=True)
    dataset = osp.split(osp.split(data_root)[0])[1]
    assert dataset in ["CWFID", "rice_s_n_w"], "this function is available to CWFID and rics_s_n_w dataset"
    if  dataset == "CWFID":
        images = glob(osp.join(data_root, "train","input",  "*_image.png"))
    if  dataset == "rice_s_n_w":
        aug = glob(osp.join(data_root, "train","input",  "*.png"))
        images = []
        for a in aug:
            filename = osp.split(a)[1]
            if re.fullmatch("image_[0-9]+.png", filename):
                images.append(a)

    indexes = random.sample(range(len(images)), int(total*percent/100))
    for index in tqdm(indexes):
        org_image = images[index]
        filename = osp.split(org_image)[1]
        all_images = glob(osp.join(data_root, "train", "input", osp.splitext(filename)[0]+"*.png"))
        for im in all_images:
            filename = osp.split(im)[1] 
            target = osp.join(osp.split(osp.split(im)[0])[0], "target", filename)
            # copy(im, osp.join(save_input, filename))
            copy(target, osp.join(save_target, filename))
    copytree(f"/content/data/semi_sup_data/{dataset}/num30/test", osp.join(save_root, "test"))
    copytree(f"/content/data/semi_sup_data/{dataset}/num30/train/input", save_input)
        
# if __name__ == "__main__":
    # sampling(data_root="/content/data/cropweed_total/CWFID/seg", save_root="/content/data/semi_sup_data/CWFID", total=50, percent=30)
    # sampling(data_root="/content/data/cropweed_total/CWFID/seg", save_root="/content/data/semi_sup_data/CWFID", total=50, percent=20)
    # sampling(data_root="/content/data/cropweed_total/CWFID/seg", save_root="/content/data/semi_sup_data/CWFID", total=50, percent=10)
    # sampling(data_root="/content/data/cropweed_total/rice_s_n_w/seg", save_root="/content/data/semi_sup_data/rice_s_n_w", total=180, percent=30)
    # sampling(data_root="/content/data/cropweed_total/rice_s_n_w/seg", save_root="/content/data/semi_sup_data/rice_s_n_w", total=180, percent=20)
    # sampling(data_root="/content/data/cropweed_total/rice_s_n_w/seg", save_root="/content/data/semi_sup_data/rice_s_n_w", total=180, percent=10)
    # bonirob_sampling(30)
    # bonirob_sampling(20)
    # bonirob_sampling(10)