import torch.nn as nn
from torchvision.models.vgg import VGG
from torchvision.models.vgg import make_layers
from pretrainedmodels.models.torchvision_models import pretrained_settings

# fmt: off
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
# fmt: on


class VGGEncoder(VGG):
    def __init__(self, out_channels, config, batch_norm=False, depth=5, in_channels=3, **kwargs):
        super().__init__(make_layers(config, batch_norm=batch_norm), **kwargs)
        self._out_channels = out_channels
        if in_channels != 3:
            self.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self._depth = depth
        self._in_channels = 3
        del self.classifier

    def get_stages(self):
        stages = []
        stage_modules = []
        for module in self.features:
            if isinstance(module, nn.MaxPool2d):
                stages.append(nn.Sequential(*stage_modules))
                stage_modules = []
            stage_modules.append(module)
        stages.append(nn.Sequential(*stage_modules))
        return stages

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        keys = list(state_dict.keys())
        for k in keys:
            if k.startswith("classifier"):
                state_dict.pop(k, None)
        super().load_state_dict(state_dict, **kwargs)
    
    def out_channels(self):
        return self._out_channels


vgg_encoders = {
    "vgg11": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg11"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["A"],
            "batch_norm": False,
        },
    },
    "vgg11_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg11_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["A"],
            "batch_norm": True,
        },
    },
    "vgg13": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg13"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["B"],
            "batch_norm": False,
        },
    },
    "vgg13_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg13_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["B"],
            "batch_norm": True,
        },
    },
    "vgg16": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg16"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["D"],
            "batch_norm": False,
        },
    },
    "vgg16_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg16_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["D"],
            "batch_norm": True,
        },
    },
    "vgg19": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg19"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["E"],
            "batch_norm": False,
        },
    },
    "vgg19_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg19_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["E"],
            "batch_norm": True,
        },
    },
}
