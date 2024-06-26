from typing import List
import torch
from torch import nn
import numpy as np
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from easydict import EasyDict
from ..modules.attention import CCA
from vector_quantizer import make_vq_module

resnet_encoders = {
    "resnet18": {
       "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [2, 2, 2, 2],
        },
    },
    "resnet34": {
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet50": {
         "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet101": {
       "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
        },
    },
    "resnet152": {
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 8, 36, 3],
        },
    },
    "resnext50_32x4d": {
       "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x4d": {
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x8d": {
       "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 8,
        },
    },
    "resnext101_32x16d": {
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 16,
        },
    },
    "resnext101_32x32d": {
       "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 32,
        },
    },
    "resnext101_32x48d": {
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 48,
        },
    },
}

def replace_strides_with_dilation(module, dilation_rate):
    """Patch Conv2d modules replacing strides with dilation"""
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)

            # Kostyl for EfficientNet
            if hasattr(mod, "static_padding"):
                mod.static_padding = nn.Identity()
                
class ResNetEncoder(ResNet):
    '''for unet'''
    def __init__(self, out_channels, depth=5, in_channels=3, padding_mode='zeros', **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = in_channels
        if self._in_channels != 3:
            self.conv1 = nn.Conv2d(self._in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False, padding_mode=padding_mode)
        else:
            self.conv1 = nn.Conv2d(self._in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False, padding_mode=padding_mode)
        del self.fc
        del self.avgpool
        if padding_mode != 'zeros':
            assert padding_mode in ['reflect', 'replicate', 'circular'], f"padding_mode {padding_mode} is not available"
            self._change_padding_mode(padding_mode)
            
    def _change_padding_mode(self, padding_mode):
        for name, child in self.named_children():
            if isinstance(child, nn.Conv2d):
                self._modules[name].padding_mode = padding_mode
            elif isinstance(child, nn.Sequential):
                for sname, schild in child.named_children():
                    if isinstance(schild, nn.Conv2d):
                        # print(name,sname)
                        self._modules[name]._modules[sname].padding_mode = padding_mode
                    if isinstance(schild, Bottleneck):
                        # print(name,sname)
                        for bname, bchild in schild.named_children():
                            if isinstance(bchild, nn.Conv2d):
                                # print(name,sname, bname)
                                self._modules[name]._modules[sname]._modules[bname].padding_mode = padding_mode

    def get_stages(self):
        '''
        input size (2, 3, 512, 512)일 때
        output
        
        [
        torch.Size([2, 3, 512, 512])
        torch.Size([2, 64, 256, 256])
        torch.Size([2, 256, 128, 128])
        torch.Size([2, 512, 64, 64])
        torch.Size([2, 1024, 32, 32])
        torch.Size([2, 2048, 16, 16])
        ]
        '''
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)
    
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]
    def make_dilated(self, output_stride):
    
        if output_stride == 16:
            stage_list = [
                5,
            ]
            dilation_list = [
                2,
            ]

        elif output_stride == 8:
            stage_list = [4, 5]
            dilation_list = [2, 4]

        else:
            raise ValueError("Output stride should be 16 or 8, got {}.".format(output_stride))

        self._output_stride = output_stride

        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )

class CCAResNetEncoder(ResNet):
    '''for unet'''
    def __init__(self, out_channels, depth=5, in_channels=3, padding_mode='zeros', cca:List=[False, False, True, True, True], **kwargs):
        super().__init__(**kwargs)
        assert len(cca) == depth, "the length of cca list must be same with depth"
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = in_channels
        if self._in_channels != 3:
            self.conv1 = nn.Conv2d(self._in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False, padding_mode=padding_mode)
        else:
            self.conv1 = nn.Conv2d(self._in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False, padding_mode=padding_mode)
        del self.fc
        del self.avgpool
        if padding_mode != 'zeros':
            assert padding_mode in ['reflect', 'replicate', 'circular'], f"padding_mode {padding_mode} is not available"
            self._change_padding_mode(padding_mode)
        cca_in_channels = out_channels[1:]
        self.cca_list = nn.ModuleList([CCA(in_channels = cca_in_channels[i], out_channels=cca_in_channels[i]) if cca[i] else nn.Identity() \
                                                            for i in range(depth)])
        
            
    def _change_padding_mode(self, padding_mode):
        for name, child in self.named_children():
            if isinstance(child, nn.Conv2d):
                self._modules[name].padding_mode = padding_mode
            elif isinstance(child, nn.Sequential):
                for sname, schild in child.named_children():
                    if isinstance(schild, nn.Conv2d):
                        # print(name,sname)
                        self._modules[name]._modules[sname].padding_mode = padding_mode
                    if isinstance(schild, Bottleneck):
                        # print(name,sname)
                        for bname, bchild in schild.named_children():
                            if isinstance(bchild, nn.Conv2d):
                                # print(name,sname, bname)
                                self._modules[name]._modules[sname]._modules[bname].padding_mode = padding_mode

    def get_stages(self):
        '''
        input size (2, 3, 512, 512)일 때
        output
        
        [
        torch.Size([2, 3, 512, 512])
        torch.Size([2, 64, 256, 256])
        torch.Size([2, 256, 128, 128])
        torch.Size([2, 512, 64, 64])
        torch.Size([2, 1024, 32, 32])
        torch.Size([2, 2048, 16, 16])
        ]
        '''
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu, self.cca_list[0]),
            nn.Sequential(self.maxpool, self.layer1, self.cca_list[1]),
            nn.Sequential(self.layer2, self.cca_list[2]),
            nn.Sequential(self.layer3, self.cca_list[3]),
            nn.Sequential(self.layer4, self.cca_list[4]),
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, strict=False, **kwargs)
    
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]
    
class CCAVQResNetEncoder(ResNet):
    '''for unet'''
    def __init__(
        self, 
        out_channels, 
        depth=5, 
        in_channels=3, 
        padding_mode='zeros', 
        cca:List=[False, False, True, True, True], 
        vq_cfg=EasyDict({
                "num_embeddings":[0, 0, 512, 512, 512],
                "distance":"euclidean",
                "kmeans_init": True
                }),
        **kwargs):
        super().__init__(**kwargs)
        assert len(cca) == depth, "the length of cca list must be same with depth"
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = in_channels
        if self._in_channels != 3:
            self.conv1 = nn.Conv2d(self._in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False, padding_mode=padding_mode)
        else:
            self.conv1 = nn.Conv2d(self._in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False, padding_mode=padding_mode)
        del self.fc
        del self.avgpool
        if padding_mode != 'zeros':
            assert padding_mode in ['reflect', 'replicate', 'circular'], f"padding_mode {padding_mode} is not available"
            self._change_padding_mode(padding_mode)
        cca_in_channels = out_channels[1:]
        self.cca_list = nn.ModuleList([CCA(in_channels = cca_in_channels[i], out_channels=cca_in_channels[i]) if cca[i] else nn.Identity() \
                                                            for i in range(depth)])
        self.codebook = make_vq_module(vq_cfg, out_channels, depth)
            
    def _change_padding_mode(self, padding_mode):
        for name, child in self.named_children():
            if isinstance(child, nn.Conv2d):
                self._modules[name].padding_mode = padding_mode
            elif isinstance(child, nn.Sequential):
                for sname, schild in child.named_children():
                    if isinstance(schild, nn.Conv2d):
                        # print(name,sname)
                        self._modules[name]._modules[sname].padding_mode = padding_mode
                    if isinstance(schild, Bottleneck):
                        # print(name,sname)
                        for bname, bchild in schild.named_children():
                            if isinstance(bchild, nn.Conv2d):
                                # print(name,sname, bname)
                                self._modules[name]._modules[sname]._modules[bname].padding_mode = padding_mode

    def get_stages(self):
        '''
        input size (2, 3, 512, 512)일 때
        output
        
        [
        torch.Size([2, 3, 512, 512])
        torch.Size([2, 64, 256, 256])
        torch.Size([2, 256, 128, 128])
        torch.Size([2, 512, 64, 64])
        torch.Size([2, 1024, 32, 32])
        torch.Size([2, 2048, 16, 16])
        ]
        '''
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu, self.cca_list[0]),
            nn.Sequential(self.maxpool, self.layer1, self.cca_list[1]),
            nn.Sequential(self.layer2, self.cca_list[2]),
            nn.Sequential(self.layer3, self.cca_list[3]),
            nn.Sequential(self.layer4, self.cca_list[4]),
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        code_usage_lst = []
        features.append(x)
        for i in range(1, self._depth + 1):
            x = stages[i](x)
            x, _, commitment_loss, code_usage  = self.codebook[i-1](x)
            features.append(x)
            if commitment_loss is not None: 
                loss = loss + commitment_loss
            if code_usage is not None: 
                code_usage_lst.append(code_usage.item())
        loss = loss / self._depth
        return features, loss, torch.tensor(code_usage_lst)

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, strict=False, **kwargs)
    
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]