import torch

def detach_numpy(tensor:torch.Tensor):
    return tensor.detach().cpu().numpy()