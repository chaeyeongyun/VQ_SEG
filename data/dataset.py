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
    def __init__(self, data_dir, split, resize=None, target_resize=True):
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
        self.target_resize = target_resize
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
        if self.target_dir is not None:
            target = Image.open(os.path.join(self.target_dir, filename)).convert('L')
        else:
            target = None
        
        if self.resize is not None:
            img = img.resize(self.resize, resample=Image.Resampling.BILINEAR)
            if self.target_resize and target is not None:
                target = target.resize(self.resize, resample=Image.Resampling.NEAREST)
        
        img = TF.to_tensor(img)
        target = torch.from_numpy(np.array(target)) if target is not None else None
        if target is None:
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


class SalientDataset(Dataset):
    def __init__(self, data_dir, salient_dir, split, resize=None, target_resize=True):
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
        self.target_resize = target_resize
        if split == 'labelled':
            self.filenames = os.listdir(os.path.join(data_dir, 'target'))
            self.target_dir = os.path.join(data_dir, 'target')
        elif split == 'unlabelled':
            self.filenames = list(set(os.listdir(os.path.join(data_dir, 'input'))) - set(os.listdir(os.path.join(data_dir, 'target'))))
            self.target_dir = None
        else:
            raise ValueError(f"split has to be labelled or unlabelled")
        
        self.salient_dir = salient_dir
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(os.path.join(self.img_dir, filename)).convert('RGB')
        # img = TF.to_tensor(Image.open(os.path.join(self.img_dir, filename)).convert('RGB'))
        if self.target_dir is not None:
            target = Image.open(os.path.join(self.target_dir, filename)).convert('L')
        else:
            target = None
        salient_map = Image.open(os.path.join(self.salient_dir, filename)).convert('L')
        if self.resize is not None:
            img = img.resize(self.resize, resample=Image.Resampling.BILINEAR)
            salient_map = salient_map.resize(self.resize, resample=Image.Resampling.NEAREST)
            if self.target_resize and target is not None:
                target = target.resize(self.resize, resample=Image.Resampling.NEAREST)
        
        img = TF.to_tensor(img)
        salient_map = torch.from_numpy(np.array(salient_map)/255).to(torch.float)
        target = torch.from_numpy(np.array(target)) if target is not None else None
        if target is None:
            return {'filename':filename, 'img':img, 'salient_map':salient_map}
        return {'filename':filename, 'img':img, 'target':target, 'salient_map':salient_map}