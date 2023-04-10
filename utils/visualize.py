from typing import List
import matplotlib.pyplot as plt
import numpy as np
import os

def pred_to_colormap(pred:np.ndarray, colormap:np.ndarray):
    pred_label = np.argmax(pred, axis=1) # (N, H, W)
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
    
def make_example_img(l_input:np.ndarray, target:np.ndarray, pred:np.ndarray, ul_input:np.ndarray, ul_pred:np.ndarray, colormap=np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]])):
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

    return cat_img

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
    
    
    
    