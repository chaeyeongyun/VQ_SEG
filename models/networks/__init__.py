from .unet.net import *
from .vqvaev2.net import *
network_dict = {
    "vqunet_v1":VQUnet_v1,
    "vqunet_v2":VQUnet_v2,
    "vqvaev2":VQVAEv2,
    "vqptunet":VQPTUnet,
    "vqeuptunet":VQEuPTUnet,
    "vqnedptunet":VQNEDPTUnet,
    "vqashunet":VQASHUnet,
    "vqatunet": VQATUnet,
    "VQUnetwithSalientloss":VQUnetwithSalientloss
}

def make_model(model_cfg:dict):
    name = model_cfg.name
    model = network_dict[name](**model_cfg.params)
    return model
    