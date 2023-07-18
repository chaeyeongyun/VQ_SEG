from .unet.net import *
from .deeplabv3 import *
from .vqvaev2.net import *
from .modified_vqunet import *
from .semiweednet import *
from .fcn import *
network_dict = {
    "deeplabv3":DeepLabV3,
    "deeplabv3plus":DeepLabV3Plus,
    "unetoriginal":UnetOriginal,
    "unet":Unet,
    "vqunet_v1":VQUnet_v1,
    "vqunet_v2":VQUnet_v2,
    "vqvaev2":VQVAEv2,
    "vqptunet":VQPTUnet,
    "vqeuptunet":VQEuPTUnet,
    "vqnedptunet":VQNEDPTUnet,
    "vqashunet":VQASHUnet,
    "vqatunet": VQATUnet,
    'supconvqunet':SupConVQUnet,
    "VQUnetwithSalientloss":VQUnetwithSalientloss,
    "drsavqunet":DRSAVQUnet,
    "vqashunetv2":VQASHUnetv2,
    "vqcanet":VQCANet, 
     "vqcanetv2":VQCANetv2,
     "vqcanetv3":VQCANetv3,
     "vqcanetv4":VQCANetv4,
     "vqimdbnet":VQIMDBNet,
     "vqpatchunet":VQPatchUNet,
     "vqreptunet":VQRePTUnet,
     "vqreeuptunet":VQReEuPTUnet,
      "vqreptunet1x1":VQRePTUnet1x1,
      "vqretemptunet":VQReTemPTUnet,
      "vqreptunetangular":VQRePTUnetAngular,
      "semiweednet":SemiWeedNet,
      "fcn32s":FCN32s
}

def make_model(model_cfg:dict):
    name = model_cfg.name
    model = network_dict[name](**model_cfg.params)
    return model
    