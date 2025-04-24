from .build import build_loader as _build_loader
from .losses import FocalCrossEntropyCombo as focal_ce

def build_loader(config, simmim=False, is_pretrain=False):
    if not simmim:
        return _build_loader(config)

