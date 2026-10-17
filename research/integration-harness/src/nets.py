"""Trunk / head split and FiLM injection points for EEGNet and ShallowFBCSPNet (loops B, C)."""

from torch import nn

FILM_AFTER = {"eegnet": "bnorm_1", "shallow": "bnorm"}
_CHANNELS = {"eegnet": 16, "shallow": 40}


def split_trunk_head(model):
    children = list(model.named_children())
    names = [n for n, _ in children]
    i = names.index("final_layer")
    return nn.Sequential(*[m for _, m in children[:i]]), children[i][1]


def feature_channels(arch):
    return _CHANNELS[arch]
