from easydict import EasyDict
import torch 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from glob import glob
import os
import argparse


from data.dataset import BaseDataset
from utils.load_config import get_config_from_json
from utils.device import device_setting
from utils.visualize import make_test_img, save_img_list
from utils.seg_tools import img_to_label
from utils.logger import TestLogger, dict_to_table_log, make_img_table
from measurement import Measurement
import models
from utils.crf import DenseCRF

def make_filename(filename_list, insert):
    for i, filename in enumerate(filename_list):
        filename, ext = os.path.splitext(filename)
        filename_list[i] = filename+insert+ext
    return filename_list

def test(cfg):
    # debug
    # cfg.resize=32
    # cfg.project_name = 'debug'
    # cfg.test.weights = "../drive/MyDrive/semi_sup_train/CWFID/debug/ckpoints"
    num_classes = cfg.num_classes
    batch_size = cfg.test.batch_size
    pixel_to_label_map = cfg.pixel_to_label
    weights = cfg.test.weights
    device = device_setting(cfg.test.device)
    
    model = models.networks.make_model(cfg.model).to(device)
    logger_name = (cfg.model.name+"_"+model.encoder.__class__.__name__+"_"+model.decoder.__class__.__name__+"_"+str(len(os.listdir(cfg.test.save_dir))))
    # make save folders
    save_dir = os.path.join(cfg.test.save_dir, logger_name)
    os.makedirs(save_dir)
    img_dir = os.path.join(save_dir, 'imgs')
    os.mkdir(img_dir)
    
    logger = TestLogger(cfg, logger_name) if cfg.wandb_logging else None
    test_data = BaseDataset(os.path.join(cfg.test.data_dir, 'test'), split='labelled', resize=cfg.resize, target_resize=False)
    testloader = DataLoader(test_data, batch_size, shuffle=False)
    f = open(os.path.join(save_dir, 'results.txt'), 'w')
    f.write(f"data_dir:{cfg.test.data_dir}, weights:{cfg.test.weights}, save_dir:{cfg.test.save_dir}")

    if os.path.isfile(cfg.test.weights):
        best_result = test_loop(model, weights, num_classes, pixel_to_label_map, testloader, device)
    elif os.path.isdir(cfg.test.weights):
        weights_list = glob(os.path.join(cfg.test.weights, '*.pth'))
        weights_list.sort()
        best_miou = 0.
        best_result = None
        for weights in weights_list:
            result = test_loop(model, weights, num_classes, pixel_to_label_map, testloader, device)
            if result == None: continue
            if result.metrics.test_miou >= best_miou:
                best_miou = result.metrics.test_miou
                best_result = result
    
    assert best_result != None, "weights file has some problem"
    f.write(best_result.result_txt)
    save_img_list(img_dir, make_filename(best_result.visualize.filename, "_v1"), best_result.visualize.viz_v1)
    save_img_list(img_dir, make_filename(best_result.visualize.filename, "_v2"), best_result.visualize.viz_v2)
    if logger != None:
        logger.table_dict = {
            "metrics":dict_to_table_log(best_result.metrics),
            "visualize":make_img_table(best_result.visualize.filename, best_result.visualize.viz_v1, best_result.visualize.viz_v2, columns=["filename", 'viz_v1', "viz_v2"])
            }
        logger.logging()
    print("best_result:\n"+best_result.result_txt)
    if logger is not None: 
        logger.finish()
def test_loop(model:nn.Module, weights_file:str, num_classes:int, pixel_to_label_map:dict, testloader:DataLoader, device:torch.device):
    try:
        weights = torch.load(weights_file)
        weights = weights.get('model_1', weights)
    except:
        return None
    model.load_state_dict(weights)
    model.eval()
    measurement = Measurement(num_classes)
    test_acc, test_miou = 0, 0
    test_precision, test_recall, test_f1score = 0, 0, 0
    iou_per_class = np.array([0]*num_classes, dtype=np.float64)
    viz_v1_list, viz_v2_list = [], []
    filename_list = []
    for data in tqdm(testloader):
        input_img, mask_img, filename = data['img'], data['target'], data['filename']
        input_img = input_img.to(device)
        mask_cpu = img_to_label(mask_img, pixel_to_label_map)
        
        with torch.no_grad():
            pred = model(input_img)[0]
        
        crf = DenseCRF()
        pred = crf(input_img[0].detach().cpu(), pred[0].detach().cpu()) # cpu
        pred = F.interpolate(torch.tensor(np.expand_dims(pred, 0)), mask_img.shape[-2:], mode='bilinear')
        mask_numpy = mask_cpu.cpu().numpy()
        acc_pixel, batch_miou, iou_ndarray, precision, recall, f1score = measurement(pred.cpu().numpy(), mask_numpy) 
        
        test_acc += acc_pixel
        test_miou += batch_miou
        iou_per_class += iou_ndarray
        
        test_precision += precision
        test_recall += recall
        test_f1score += f1score
        
        save_size = (mask_img.shape[-2]//2, mask_img.shape[-1]//2)
        input_img = F.interpolate(input_img.detach().cpu(), save_size, mode='bilinear')
        pred_cpu = F.interpolate(pred.detach().cpu(), save_size, mode='bilinear')
        mask_cpu = F.interpolate(mask_img.detach().cpu().unsqueeze(1), save_size, mode='nearest').squeeze(1)
        mask_cpu = img_to_label(mask_cpu, pixel_to_label_map)
        viz_v1, viz_v2 = make_test_img(input_img.numpy(), pred_cpu, mask_cpu,
                        colormap = np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]]))
        viz_v1_list.append(viz_v1)
        viz_v2_list.append(viz_v2)
        filename_list.append(*filename)
        
    # test finish
    test_acc = test_acc / len(testloader)
    test_miou = test_miou / len(testloader)
    test_ious = np.round((iou_per_class / len(testloader)), 5).tolist()
    test_precision /= len(testloader)
    test_recall /= len(testloader)
    test_f1score /= len(testloader)
    
    result_txt = "load model(.pt) : %s \n Testaccuracy: %.8f, Test miou: %.8f" % (weights_file,  test_acc, test_miou)       
    result_txt += f"\niou per class {test_ious}"
    result_txt += f"\nprecision : {test_precision}, recall : {test_recall}, f1score : {test_f1score} " 
    print(result_txt)
    return_dict = {
        "metrics":
            {
                "test_acc":test_acc,
                "test_miou":test_miou,
                "test_ious":test_ious,
                "test_precision":test_precision,
                "test_recall":test_recall,
                "test_f1score": test_f1score
            },
        "visualize":{
            "viz_v1":viz_v1_list,
            "viz_v2":viz_v2_list,
            "filename":filename_list
            },
        "result_txt":result_txt
        }
    return EasyDict(return_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./config/cps_vqv2.json')
    opt = parser.parse_args()
    cfg = get_config_from_json(opt.config_path)
    
    
    # test(cfg)
    # cfg = get_config_from_json('./config/only_sup_kmeans.json')
    # w_l = ["../drive/MyDrive/only_sup_train/CWFID/only_sup_kmeans132/ckpoints", 
    #        "../drive/MyDrive/only_sup_train/CWFID/only_sup_kmeans133/ckpoints",]
    # num_embeddings_l = [[0, 0, 512, 512, 512], [0, 0, 2048, 2048, 2048]]
    # for w, ne in zip(w_l, num_embeddings_l):
    #     cfg.test.weights = w
    #     cfg.model.params.vq_cfg.num_embeddings = ne
    #     # cfg.wandb_logging=False
    #     test(cfg)
    # w_l = ["../drive/MyDrive/semi_sup_train/CWFID/VQUnet_v2102/ckpoints", 
    #        "../drive/MyDrive/semi_sup_train/CWFID/VQUnet_v2103/ckpoints",
    cfg = get_config_from_json('./config/vq_pt_unet.json')
    cfg.resize = 448
    w_l = ["../drive/MyDrive/semi_sup_train/CWFID/VQPT+CRF70/ckpoints/best_test_miou.pth"]
    
    for w in w_l:
        # debug
        # cfg.resize=32
        # cfg.project_name = 'debug'
        # cfg.wandb_logging = False
        cfg.test.weights = w
        test(cfg)
        
    # cfg = get_config_from_json("./config/cps_vqv1.json")
    # w_l = ["../drive/MyDrive/semi_sup_train/CWFID/VQUnet_v186/ckpoints",
    #        "../drive/MyDrive/semi_sup_train/CWFID/VQUnet_v16/ckpoints"]
    # resize_l = [256, 512]
    # num_embeddings_l = [512, 2048]
    # for i, (w, ne) in enumerate(zip(w_l, num_embeddings_l)):
    #     cfg.test.weights = w
    #     cfg.resize = resize_l[i]
    #     cfg.model.params.vq_cfg.num_embeddings = ne
    #     test(cfg)
    
    
    # cfg = get_config_from_json("./config/cps_vqv2_pretrained.json")
    # vqv2_selfsup_folds = [
    #     "../drive/MyDrive/self_supervised/CWFID/VQUNetv2_selfsupervision13",
    #     "../drive/MyDrive/self_supervised/CWFID/VQUNetv2_selfsupervision14",
    #     "../drive/MyDrive/self_supervised/CWFID/VQUNetv2_selfsupervision23",
    #     "../drive/MyDrive/self_supervised/CWFID/VQUNetv2_selfsupervision24"
    # ]
    
    
    # ckpoint_fold = "../drive/MyDrive/semi_sup_train/CWFID/VQUnet_v2_base_selfsup"
    # start = 94
    # resize_l = [256, 256, 256, 256]
    # num_embeds_l = [2048, 2048, 512, 512]
    # epochs = [200, 400, 600, 800]
    # for i, selfsup in enumerate(vqv2_selfsup_folds):
    #     cfg.resize = resize_l[i]
    #     cfg.model.params.vq_cfg.num_embeddings = num_embeds_l[i]
    #     for epoch in epochs:
    #         cfg.test.weights = os.path.join(ckpoint_fold+str(start), 'ckpoints') 
    #         test(cfg)
    #         start += 1