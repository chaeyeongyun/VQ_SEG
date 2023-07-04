from copy import deepcopy

from pretrainedmodels.models.torchvision_models import pretrained_settings

pretrain_settings = deepcopy(pretrained_settings)
additional = {
    "resnet18": {"imagenet_ssl": {'url':"https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth"},  # noqa
                         "imagenet_swsl": {"url":"https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth"}},  # noqa,
    "resnet50": {"imagenet_ssl": {"url":"https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth"},  # noqa
                        "imagenet_swsl": {"url":"https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth"}},  # noqa,
    "convnext_tiny":{"imagenet":{"url":"https://download.pytorch.org/models/convnext_tiny-983f1562.pth"}},
    "convnext_small":{"imagenet":{"url":"https://download.pytorch.org/models/convnext_small-0c510722.pth"}},
    "convnext_base":{"imagenet":{"url":"https://download.pytorch.org/models/convnext_base-6075fbad.pth"}},
    "convnext_large":{"imagenet":{"url":"https://download.pytorch.org/models/convnext_large-ea097f82.pth"}}
}
for key in additional:
    if key in pretrain_settings:
        pretrain_settings[key].update(additional[key])
    else:
        pretrain_settings.update({key:additional[key]})

# class ImageNet():
#     def __init__(self):
#         self.classifier_settings = deepcopy(pretrained_settings)
#         self.classifier_settings.update({
#             "convnext_tiny":{"imagenet":{"url":"https://download.pytorch.org/models/convnext_tiny-983f1562.pth"}},
#             "convnext_small":{"imagenet":{"url":"https://download.pytorch.org/models/convnext_small-0c510722.pth"}},
#             "convnext_base":{"imagenet":{"url":"https://download.pytorch.org/models/convnext_base-6075fbad.pth"},}
#             "convnext_large":{"imagenet":{"url":"https://download.pytorch.org/models/convnext_large-ea097f82.pth"},}
#         })
#         self.self_sup_settings = {"resnet18": {
#                                     "imagenet_ssl": {'url':"https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth"},  # noqa
#                                     "imagenet_swsl": {"url":"https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth"},  # noqa
#                             },
#                             "resnet50": {
#                                 "imagenet_ssl": {"url":"https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth"},  # noqa
#                                 "imagenet_swsl": {"url":"https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth"},  # noqa
#                             }}