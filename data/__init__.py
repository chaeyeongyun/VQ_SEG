from .augmentations import CutMix
aug_dict = {
    "cutmix":CutMix
}
def make_aug(aug_cfg):
    aug_name = aug_cfg.pop("name")
    return aug_dict[aug_name](**aug_cfg)