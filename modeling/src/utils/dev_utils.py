import torch
from torch import nn


def hook_fn(m, i, o):
    if i[0].shape[-1] == 412:
        print("*****START*****")
    if i[0].shape != o.shape or True:
        print(type(m), "\n", i[0].shape, "\n", o.shape)


def add_shape_hook(input: torch.Tensor, network: nn.Module):
    for _, layer in network._modules.items():
        if hasattr(layer, "children") and len(list(layer.children())) > 0:
            add_shape_hook(input, layer)
        else:
            layer.register_forward_hook(hook_fn)
