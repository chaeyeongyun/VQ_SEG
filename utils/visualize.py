from typing import List
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


def gray_to_rgb(img:np.array):
    if img.ndim==3:
        result = np.stack([img]*3, axis=1)
    elif img.ndim==4:
        result = np.concatenate([img]*3, axis=1)
    else:
        raise NotImplementedError("this function is implemented for only dimension 3 and 4")
    return result

def pred_to_colormap(pred:np.ndarray, colormap:np.ndarray):
    pred_label = np.argmax(pred, axis=1) # (N, H, W)
    show = colormap[pred_label] # (N, H, W, 3)
    return show

def pred_to_detailed_colormap(pred:np.ndarray, target:np.ndarray, colormap:np.ndarray):
    labels = np.unique(target).tolist()
    num_classes = len(labels)
    pred_label = np.argmax(pred, axis=1) # (N, H, W)
    for label in labels:
        pred_label[(pred_label == label) & (target != label)] = label + num_classes
    # crop TP:2,red FP 5,yellow
    # weed TP:1,blue FP 4,orange
    # BG TP:0,black FP 3, gray
    # https://www.color-hex.com/color/e69138
    if num_classes == 3:
        colormap = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0.5, 0.5, 0.5], [230/255, 145/255, 56/255], [1, 217/255, 102/255]]) # (3,3)->(6,3)
    else:
        raise NotImplementedError
    show = colormap[pred_label] # (N, H, W, 3)
    return show

def target_to_colormap(target:np.ndarray, colormap:np.ndarray):
    show = colormap[target] # (N, H, W, 3)
    return show

def to_4d_shape(array:np.ndarray):
    if array.ndim == 3:
        return np.expand_dims(array, axis=1)
    else:
        raise NotImplementedError

def batch_to_grid(array:np.ndarray, transpose=True):
    array = array.transpose(0, 2, 3, 1) if transpose else array
    cat_img = np.squeeze(np.concatenate(np.split(array, len(array), axis=0), axis=1), axis=0)
    return cat_img

def mix_input_pred(input:np.ndarray, pred:np.ndarray, alpha=0.4):
    mix = input * (1-alpha) + pred * alpha
    mix = np.clip(mix, 0, 1)
    return mix
    
def make_example_img(l_input:np.ndarray, target:np.ndarray, pred:np.ndarray, ul_input:np.ndarray, ul_pred:np.ndarray, colormap=np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]]), resize_factor=0.5):
    l_input = batch_to_grid(l_input)
    target = target_to_colormap(target, colormap=colormap)
    target = batch_to_grid(target, transpose=False)
    pred = batch_to_grid(pred_to_colormap(pred, colormap=colormap), transpose=False)
    l_cat = np.concatenate((l_input, target, pred), axis=1)
    h, w, c = l_cat.shape[:]
    if ul_input is None and ul_pred is None:
        return l_cat
    
    ul_input = batch_to_grid(ul_input)
    ul_pred = batch_to_grid(pred_to_colormap(ul_pred, colormap=colormap), transpose=False)
    ul_mix = mix_input_pred(ul_input, ul_pred)
    interval = np.ones((h, 20, c), dtype=np.float64)
    cat_img = np.concatenate((l_cat, interval, ul_mix), axis=1)
    if resize_factor is not None:
        cat_img = cv2.resize(cat_img, dsize=(0,0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
    return cat_img

def make_example_img_salient(l_input:np.ndarray, target:np.ndarray, pred:np.ndarray, ul_input:np.ndarray, ul_pred:np.ndarray, l_salient:np.ndarray, ul_salient:np.ndarray, colormap=np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]]), resize_factor=0.5):
    l_salient, ul_salient = gray_to_rgb(l_salient), gray_to_rgb(ul_salient)
    
    l_input = batch_to_grid(l_input)
    target = target_to_colormap(target, colormap=colormap)
    target = batch_to_grid(target, transpose=False)
    pred = batch_to_grid(pred_to_colormap(pred, colormap=colormap), transpose=False)
    l_salient = batch_to_grid(l_salient)
    l_cat = np.concatenate((l_input, target, pred, l_salient), axis=1)
    h, w, c = l_cat.shape[:]
    if ul_input is None and ul_pred is None:
        return l_cat
    
    ul_input = batch_to_grid(ul_input)
    ul_pred = batch_to_grid(pred_to_colormap(ul_pred, colormap=colormap), transpose=False)
    ul_salient = batch_to_grid(ul_salient)
    ul_mix = mix_input_pred(ul_input, ul_pred)
    interval = np.ones((h, 20, c), dtype=np.float64)

    cat_img = np.concatenate((l_cat, interval, ul_mix, ul_salient), axis=1)
    if resize_factor is not None:
        cat_img = cv2.resize(cat_img, dsize=(0,0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
    return cat_img


def make_example_img_salient_loss(l_input:np.ndarray, target:np.ndarray, pred:np.ndarray, ul_input:np.ndarray, ul_pred:np.ndarray, 
                                  l_salient:np.ndarray, ul_salient:np.ndarray, salient_pred_l:np.ndarray, salient_pred_ul:np.ndarray,
                                  colormap=np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]]), resize_factor=0.5):

    l_salient, ul_salient = gray_to_rgb(l_salient), gray_to_rgb(ul_salient)
    salient_pred_l, salient_pred_ul = gray_to_rgb(salient_pred_l), gray_to_rgb(salient_pred_ul)

    l_input = batch_to_grid(l_input)
    target = target_to_colormap(target, colormap=colormap)
    target = batch_to_grid(target, transpose=False)
    pred = batch_to_grid(pred_to_colormap(pred, colormap=colormap), transpose=False)
    l_salient, salient_pred_l = batch_to_grid(l_salient), batch_to_grid(salient_pred_l)
    l_cat = np.concatenate((l_input, target, pred, l_salient, salient_pred_l), axis=1)
    h, w, c = l_cat.shape[:]
    if ul_input is None and ul_pred is None:
        return l_cat
    
    ul_input = batch_to_grid(ul_input)
    ul_pred = batch_to_grid(pred_to_colormap(ul_pred, colormap=colormap), transpose=False)
    ul_salient, salient_pred_ul = batch_to_grid(ul_salient), batch_to_grid(salient_pred_ul)
    ul_mix = mix_input_pred(ul_input, ul_pred)
    interval = np.ones((h, 20, c), dtype=np.float64)

    cat_img = np.concatenate((l_cat, interval, ul_mix, ul_salient, salient_pred_ul), axis=1)
    if resize_factor is not None:
        cat_img = cv2.resize(cat_img, dsize=(0,0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
    return cat_img

def make_example_img_slic(l_input:np.ndarray, target:np.ndarray, pred:np.ndarray, ul_input:np.ndarray, ul_pred:np.ndarray, l_slic:np.ndarray, ul_slic:np.ndarray, colormap=np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]]), resize_factor=0.5):
    l_input = batch_to_grid(l_input)
    target = target_to_colormap(target, colormap=colormap)
    target = batch_to_grid(target, transpose=False)
    pred = batch_to_grid(pred_to_colormap(pred, colormap=colormap), transpose=False)
    l_slic = batch_to_grid(l_slic)
    l_cat = np.concatenate((l_input, target, pred), axis=1)
    h, w, c = l_cat.shape[:]
    if ul_input is None and ul_pred is None:
        return l_cat
    
    ul_input = batch_to_grid(ul_input)
    ul_pred = batch_to_grid(pred_to_colormap(ul_pred, colormap=colormap), transpose=False)
    ul_slic = batch_to_grid(ul_slic)
    ul_mix = mix_input_pred(ul_input, ul_pred)
    interval = np.ones((h, 20, c), dtype=np.float64)
    cat_img = np.concatenate((l_cat, interval, ul_mix, ul_slic), axis=1)
    if resize_factor is not None:
        cat_img = cv2.resize(cat_img, dsize=(0,0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
    return cat_img

def make_test_img(input_array:np.ndarray, pred_array:np.ndarray, target_array:np.ndarray):
    input_array = input_array[0] if input_array.ndim==4 else input_array
    pred_array = pred_array[0] if pred_array.ndim==4 else pred_array
    target_array = target_array[0] if target_array.ndim==4 else target_array
    input_array, pred_array, target_array = [i.transpose(1, 2, 0) for i in [input_array, pred_array, target_array]]
    zeros = np.zeros(input_array.shape[0], 20, 3)
    img = np.concatenate((input_array, zeros, pred_array, zeros, target_array), axis=1)
    return img
    
def save_img(img_dir:str, filename:str, img:np.ndarray):
    plt.imsave(os.path.join(img_dir, filename), img)

def save_img_list(img_dir, filename_list:List[str], img_list:List[np.ndarray]):
    for img, filename in zip(img_list, filename_list):
        plt.imsave(os.path.join(img_dir, filename), img)
        
def make_selfsup_example(target:np.ndarray, recon:np.ndarray):
    cat_img = np.concatenate((target.transpose(0, 2, 3, 1), recon.transpose(0, 2, 3, 1)), axis=2)
    cat_img = np.squeeze(np.concatenate(np.split(cat_img, len(cat_img), axis=0), axis=1), axis=0)
    return cat_img

def make_test_img(input, pred, target, colormap:np.ndarray=np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]])):
    input = batch_to_grid(input)
    pred = batch_to_grid(pred_to_colormap(pred, colormap=colormap), transpose=False)
    target = batch_to_grid(target_to_colormap(target, colormap=colormap), transpose=False)
    viz_v1 = np.concatenate((input, target, pred), axis=1)
    viz_v2 = mix_input_pred(input, pred)
    # name, ext = os.path.splitext(filename)[:]
    # save_img(img_dir, name+'_v1'+ext, viz_v1)
    # save_img(img_dir, name+'_v2'+ext, viz_v2)
    return viz_v1, viz_v2
    
def make_test_detailed_img(input, pred, target, colormap:np.ndarray=np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]])):
    input = batch_to_grid(input)
    pred = batch_to_grid(pred_to_detailed_colormap(pred, target, colormap=colormap), transpose=False)
    target = batch_to_grid(target_to_colormap(target, colormap=colormap), transpose=False)
    viz_v1 = np.concatenate((input, target, pred), axis=1)
    viz_v2 = mix_input_pred(input, pred)
    return viz_v1, viz_v2
    