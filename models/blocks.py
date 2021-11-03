"""CNN blocks used to build the models."""

import torch
from torch import nn

supported_blocks = ["base_block", "bn_block"]

class BaseBlock(nn.Module):
    """The architecture used in Lai et al. 2018.
    """
    
    def __init__(self, C_in, C_out, kernel, stride, padding):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_out, C_out, kernel, stride, padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2))
        )

    def forward(self, input):
        return self.layers(input)


class BNBlock(nn.Module):
    """Block with Batch Normalization.
    """
    def __init__(self, C_in, C_out, kernel, stride, padding):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel, stride, padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(C_out),
            nn.Conv2d(C_out, C_out, kernel, stride, padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(C_out)
        )

    def forward(self, input):
        return self.layers(input)

class MLP(nn.Module):
    def __init__(self, in_features: int, layers: list, drop_prb: float, num_classes: int):
        super().__init__()
        self.mlp = []
        for i in range(len(layers)):
            self.mlp.extend([
                nn.Linear(in_features if i == 0 else layers[i - 1], layers[i]),
                nn.Dropout2d(drop_prb) if drop_prb else nn.Identity(),
                nn.ReLU(inplace=True)
            ])
        self.mlp.append(nn.Linear(layers[-1], num_classes))
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, input):
        return self.mlp(input)


def get_block(block_type: str):
    if block_type == "base_block":
        return BaseBlock

    elif block_type == "bn_block":
        return BNBlock

    else:
        raise ValueError(f"Unsupported block_type: {block_type}. Must be one of {supported_blocks}")