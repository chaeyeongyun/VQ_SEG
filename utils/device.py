import torch
from typing import Union

def device_setting(device:Union[int, str]):
    if device in ['-1', -1, 'cpu']:
        return torch.device('cpu')
    elif device == 'cuda':
        return torch.device('cuda')
    else:
        return torch.device('cuda:'+str(device))