
import models
from utils.load_config import get_config_from_json
cfg = get_config_from_json("./config/vqreptunet1x1.json")
model_1 = models.networks.make_model(cfg.model)
a=1