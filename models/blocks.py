"""CNN blocks used to build the models."""

import torch
from torch import nn
from typing import Type

from torch.nn.modules import activation


def get_activation(name: str) -> Type[nn.Module]:
    """Selects specified activation function.

    Args:
        name (str): Activation name.
        inplace (bool, optional): Whether to use inplace ops or not. Defaults to True.

    Returns:
        Type[nn.Module]: Activation class.
    """
    assert name in ["relu", "swish", "mish"]

    actn = None
    if name == "relu":
        actn = nn.ReLU
    elif name == "swish":
        actn = nn.SiLU
    elif name == "mish":
        actn = nn.Mish

    return actn


def dropout(drop_prob: float):
    return nn.Dropout2d(drop_prob) if drop_prob > 0 else nn.Identity()

class ConvBlock(nn.Module):
    """Conv blocks used in Lai et al. 18."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, use_bn: bool, actn_type: str = "relu"):
        super().__init__()

        activation = get_activation(actn_type)

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            activation(inplace=True),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            activation(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        )

    def forward(self, input):
        return self.layers(input)


class MLP(nn.Module):
    """MLP classifier head."""

    def __init__(self, in_features: int, layers: list, drop_prb: float, use_norm: bool, actn_type: str = "relu", num_classes: int = 14):
        super().__init__()
        
        activation = get_activation(actn_type)
        
        self.fc = []
        for i in range(len(layers)):
            self.fc.extend([
                nn.LayerNorm(in_features if not i else layers[i - 1]) if use_norm else nn.Identity(),
                nn.Linear(in_features if not i else layers[i - 1], layers[i]),
                activation(inplace=True),
                dropout(drop_prb)
            ])
        self.fc = nn.Sequential(*self.fc)

        self.head = nn.Sequential(
            nn.LayerNorm(layers[-1]) if use_norm else nn.Identity(),
            nn.Linear(layers[-1], num_classes)
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.head(x)
        return x
        