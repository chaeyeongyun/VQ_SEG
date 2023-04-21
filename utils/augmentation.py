import random
import torch

def random_similarity_transform(input:torch.Tensor, exclude_opt:list=[]):
    opt = list(set(range(8)) - set(exclude_opt))
    aug = random.randint(0, len(opt))
    if aug == 1:
        input = input.flip(-1)
    elif aug == 2:
        input = input.flip(-2)
    elif aug == 3:
        input = torch.rot90(input,dims=(-1,-2))
    elif aug == 4:
        input = torch.rot90(input,dims=(-1,-2), k=2)
    elif aug == 5:
        input = torch.rot90(input,dims=(-1,-2), k=3)
    elif aug == 6:
        input = torch.rot90(input.flip(-1),dims=(-1,-2))
    elif aug == 7:
        input = torch.rot90(input.flip(-2),dims=(-1,-2))
    return input, aug

def inverse_similarity_transform(input:torch.Tensor, aug_num:int):
    if aug_num == 1:
        input = input.flip(-1)
    elif aug_num == 2:
        input = input.flip(-2)
    elif aug_num == 3:
        input = torch.rot90(input,dims=(-1,-2), k=3)
    elif aug_num == 4:
        input = torch.rot90(input,dims=(-1,-2), k=2)
    elif aug_num == 5:
        input = torch.rot90(input,dims=(-1,-2), k=1)
    elif aug_num == 6:
        input = torch.rot90(input, dims=(-1,-2), k=3).flip(-1)
    elif aug_num == 7:
        input = torch.rot90(input, dims=(-1,-2), k=3).flip(-2)
    return input