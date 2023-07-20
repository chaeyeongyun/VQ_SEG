import torch

def img_to_label(target_img:torch.Tensor, pixel_to_label_dict:dict):
    pixels = list(pixel_to_label_dict.keys())
    output = target_img
    for pixel in pixels:
        output = torch.where(output==int(pixel), pixel_to_label_dict[pixel], output)
    return output.long()

def label_to_onehot(target:torch.Tensor, num_classes:int, eps:float=1e-6):
    """onehot encoding for 1 channel labelmap

    Args:
        target (torch.Tensor): shape (N, 1, H, W) or (N, H, W) have label values
        num_classes (int): the number of classes
    """
    if target.dim()==3:
        target = target.unsqueeze(1)
    onehot = torch.zeros((target.shape[0], num_classes, target.shape[2], target.shape[3]), dtype=torch.float64, device=target.device)
    onehot = onehot.scatter_(1, target.long(), 1.0) + eps
    return onehot

def score_mask(pred, th=0.7):
    pred_prob = torch.softmax(pred, dim=1)
    pred_max = pred_prob.max(dim=1)[0]
    return torch.where(pred_max > th, 1, 0).unsqueeze(1) # (N, 1, H, W)