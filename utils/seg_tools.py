import torch

def img_to_label(target_img:torch.Tensor, pixel_to_label_dict:dict):
    pixels = list(pixel_to_label_dict.keys())
    output = target_img
    for pixel in pixels:
        output = torch.where(output==int(pixel), pixel_to_label_dict[pixel], output)
    return output.long()

def label_to_onehot(target:torch.Tensor, num_classes:int):
    """onehot encoding for 1 channel labelmap

    Args:
        target (torch.Tensor): shape (N, 1, H, W) have label values
        num_classes (int): the number of classes
    """
    onehot = torch.zeros((target.shape[0], num_classes, target.shape[1], target.shape[2]), dtype=torch.float64, device=target.device)
    for c in range(num_classes):
        onehot[:, c, :, :] = (target==c)
    return onehot