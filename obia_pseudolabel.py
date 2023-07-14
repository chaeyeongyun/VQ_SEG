import os
import os.path as osp
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from fast_slic import Slic
from PIL import Image
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def img_to_label(target_img, pixel_to_label_dict:dict):
    pixels = list(pixel_to_label_dict.keys())
    output = target_img
    for pixel in pixels:
        output = np.where(output==int(pixel), pixel_to_label_dict[pixel], output)
    return output.astype(int)

def make_feature(image, image_gray, gt_image):
    label = np.zeros((image.shape[0], image.shape[1], 1))
    slic = Slic(num_components=1600, compactness=0.5)
    assignment = slic.iterate(image) 
    clusters = np.unique(assignment).tolist()
    ## LBP ###
    METHOD = 'uniform'
    # settings for LBP
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image_gray, n_points, radius, METHOD)
    l = []
    train_y = []
    for c in clusters:
        y, x = np.where(assignment==c) 
        color = image[y,x,:].mean(axis=0)
        lbp_c = np.bincount(lbp[y, x].astype(int), minlength=int(lbp.max())+1)
        l.append(np.concatenate((color, lbp_c)))
        train_y.append(np.bincount(gt_image[y, x], minlength=3).argmax())
    train_y = np.array(train_y)
    featarray = np.stack(l)
    return featarray, train_y, assignment

def main(image_root="/content/data/semi_sup_data/CWFID/num30/train",save_path = "/content/data/semi_sup_data/CWFID/num30/train/OBIA2"):
    target_filenames = os.listdir(osp.join(image_root, "target"))
    os.makedirs(save_path, exist_ok=True)
    feat_list = []
    y_list = []
    # i=0
    for filename in tqdm(target_filenames):
        ## debug
        # i += 1
        # if i==5:
        #     break
        ##
        gt = osp.join(image_root, "target", filename)
        img = osp.join(image_root, "input", filename)
        image =Image.open(img).convert("RGB")
        image_gray = np.array(image.convert("L"))
        image = np.array(image)
        gt_image  = np.array(Image.open(gt).convert("L"))
        gt_image = img_to_label(gt_image, {0:0, 128:1, 255:2})
        featarray, train_y, _ =  make_feature(image, image_gray, gt_image)
        feat_list.append(featarray)
        y_list.append(train_y)

    features = np.concatenate(feat_list, axis=0)
    Y_train = np.concatenate(y_list, axis=0)
    model = RandomForestClassifier()
    model.fit(features, Y_train)
    ## TODO: unlabeled data labeling
    image_filenames = os.listdir(osp.join(image_root, "input"))
    unlabeled_filenames = list(set(image_filenames) - set(target_filenames))
    # i = 0
    for filename in tqdm(unlabeled_filenames):
        ## debug
        # i += 1
        # if i==5:
        #     break
        ##
        img = osp.join(image_root, "input", filename)
        image =Image.open(img).convert("RGB")
        image_gray = np.array(image.convert("L"))
        image = np.array(image)
        featarray, train_y, assignment =  make_feature(image, image_gray, gt_image)
        pred = model.predict(featarray)
        pseudo_label = pred[assignment]
        pseudo_label = img_to_label(pseudo_label, {0:0, 1:128, 2:255})
        plt.imsave(osp.join(save_path, filename), pseudo_label, cmap="gray")
       

if __name__ == "__main__":
    main()
    
    