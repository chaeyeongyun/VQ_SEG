from copy import deepcopy

from pretrainedmodels.models.torchvision_models import pretrained_settings

class ImageNet():
    def __init__(self):
        self.classifier_settings = deepcopy(pretrained_settings)

        self.self_sup_settings = {"resnet18": {
                                    "ssl": {'url':"https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth"},  # noqa
                                    "swsl": {"url":"https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth"},  # noqa
                            },
                            "resnet50": {
                                "ssl": {"url":"https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth"},  # noqa
                                "swsl": {"url":"https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth"},  # noqa
                            }}