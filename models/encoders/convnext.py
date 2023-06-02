# https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
from typing import Any, Callable, List, Optional
from torch import nn
from torchvision.models.convnext import CNBlockConfig, ConvNeXt

convnext_encoders = {
    "convnext_tiny":{
        "params":{
            "out_channels":[3, 96, 192, 384, 768, 768],
            "block_setting" :[
                CNBlockConfig(96, 192, 3),
                CNBlockConfig(192, 384, 3),
                CNBlockConfig(384, 768, 9),
                CNBlockConfig(768, None, 3),
            ],
            "stochastic_depth_prob":0.1
            }
    },
    "convnext_small":{
        "params":{
             "out_channels":[3, 96, 192, 384, 768, 768],
            "block_setting":[
                CNBlockConfig(96, 192, 3),
                CNBlockConfig(192, 384, 3),
                CNBlockConfig(384, 768, 27),
                CNBlockConfig(768, None, 3),
             ],
            "stochastic_depth_prob":0.4
        }
    },
    "convnext_base":{
        "params":{
             "out_channels":[3, 128, 256, 512, 1024, 1024],
            "block_setting":[
                CNBlockConfig(128, 256, 3),
                CNBlockConfig(256, 512, 3),
                CNBlockConfig(512, 1024, 27),
                CNBlockConfig(1024, None, 3),
            ],
            "stochastic_depth_prob":0.5
        }
    },
    "convnext_large":{
        "params":{
            "out_channels":[3, 192, 384, 768, 1536, 1536],
            "block_setting":[
                CNBlockConfig(192, 384, 3),
                CNBlockConfig(384, 768, 3),
                CNBlockConfig(768, 1536, 27),
                CNBlockConfig(1536, None, 3),
            ],
            "stochastic_depth_prob":0.5
        }
    }
}
class ConvNextEncoder(ConvNeXt):
    def __init__(self, out_channels,  depth=5, **kwargs):
        super().__init__( **kwargs)
        self._depth = depth
        self._out_channels = out_channels
        del self.classifier
        del self.avgpool
    def get_stages(self):
        self.features = nn.ModuleList(self.features)
        return [
            nn.Identity(), 
            self.features[:2],
            self.features[2:4],
            self.features[4:6],
            self.features[6:]
        ]
    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features
    
    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("classifier.0.weight", None)
        state_dict.pop("classifier.0.bias", None)
        state_dict.pop("classifier.2.weight", None)
        state_dict.pop("classifier.2.bias", None)
        super().load_state_dict(state_dict, **kwargs)
        
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]