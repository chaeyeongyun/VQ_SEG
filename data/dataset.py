import os
import random
import numpy as np
from PIL import Image
from glob import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch.nn.functional as F


class BaseDataset(Dataset):
    def __init__(self, data_dir, split, resize=None):
        super().__init__()
        if type(resize)==int:
            self.resize = (resize, resize)
        elif type(resize) in [tuple, list]:
            self.resize = resize
        elif resize==None:
            self.resize = None
        else:
            raise ValueError(f"It's invalid type of resize {type(resize)}")
        
        self.img_dir = os.path.join(data_dir, 'input')
        if split == 'labelled':
            self.filenames = os.listdir(os.path.join(data_dir, 'target'))
            self.target_dir = os.path.join(data_dir, 'target')
        elif split == 'unlabelled':
            self.filenames = list(set(os.listdir(os.path.join(data_dir, 'input'))) - set(os.listdir(os.path.join(data_dir, 'target'))))
            self.target_dir = None
        else:
            raise ValueError(f"split has to be labelled or unlabelled")
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(os.path.join(self.img_dir, filename)).convert('RGB')
        # img = TF.to_tensor(Image.open(os.path.join(self.img_dir, filename)).convert('RGB'))
        if self.target_dir != None:
            target = Image.open(os.path.join(self.target_dir, filename)).convert('L')
        else:
            target = None
        
        if self.resize != None:
            img = img.resize(self.resize, resample=Image.Resampling.BILINEAR)
            target = target.resize(self.resize, resample=Image.Resampling.NEAREST) if target != None else None
        
        img = TF.to_tensor(img)
        target = torch.from_numpy(np.array(target)) if target != None else None
        if target == None:
            return {'filename':filename, 'img':img}
        return {'filename':filename, 'img':img, 'target':target}
        

class FolderDataset(Dataset):
    def __init__(self, data_dir, resize):
        self.images = glob(os.path.join(data_dir, '*.png'))
        self.resize  = (resize, resize) if isinstance(resize, int) else resize
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = img.resize(self.resize, resample=Image.Resampling.BILINEAR)
        img = TF.to_tensor(img)
        return img