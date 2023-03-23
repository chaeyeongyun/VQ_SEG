from easydict import EasyDict
import json
import yaml

def get_config_from_json(jsonfile):
    with open(jsonfile, 'r') as f:
        try:
           config_dict =  json.load(f) 
           config = EasyDict(config_dict)
           return config
        except ValueError:
           print("INVALID JSON file format.. Please provide a good json file")
           exit(-1)

def get_config_from_yaml(yamlfile):
    with open(yamlfile, 'r') as f:
        try:
           config_dict =  yaml.load(f, Loader=yaml.SafeLoader) 
           config = EasyDict(config_dict)
           return config
        except ValueError:
           print("INVALID yaml file format.. Please provide a good json file")
           exit(-1)