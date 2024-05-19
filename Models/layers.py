import torch
from torch.nn import ReLU, LeakyReLU, Identity, InstanceNorm1d, BatchNorm1d, GroupNorm, LayerNorm

def get_activation_layer(activate_type):
    if activate_type == "ReLU":
        activation = ReLU()
    elif activate_type == "Leaky ReLU":
        activation = LeakyReLU()
    else:
        raise Exception("unknown activation method")

    return activation

def get_normalization_layer(normalize_type, dim, ttype=torch.float32):
    if normalize_type == "Identity":
        normalization = Identity()
    elif normalize_type == "Instance Norm":
        normalization = InstanceNorm1d(num_features=dim, dtype=ttype)
    elif normalize_type == "Batch Norm":
        normalization = BatchNorm1d(dim, dtype=ttype)
    elif normalize_type == "Group Norm":
        normalization = GroupNorm(num_groups=4, num_channels=dim, dtype=ttype)
    elif normalize_type == "Layer Norm":
        normalization = LayerNorm(normalized_shape=dim, dtype=ttype)
    else:
        raise Exception("unknown normalization method")

    return normalization