from .unet.net import VQUnet_v1, VQUnet_v2
from .vqvaev2.net import VQVAEv2
network_dict = {
    "vqunet_v1":VQUnet_v1,
    "vqunet_v2":VQUnet_v2,
    "vqvaev2":VQVAEv2
}

def make_model(model_cfg:dict):
    name = model_cfg.name
    model = network_dict[name](**model_cfg.params)
    return model
    