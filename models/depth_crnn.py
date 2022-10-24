import torch
from torch import nn
from models.blocks import ConvBlock, MLP
from einops import rearrange
from typing import Tuple


class DepthCRNN(nn.Module):
    def __init__(self, conv_blocks: list = [8, 16, 32], res_in : Tuple[int, int] = (50, 50), T : int = 32, num_classes : int = 14,
        drop_prb : float = 0.5, mlp_layers: list = [128], lstm_units: int = 128, lstm_layers: int = 2, use_bilstm: bool = True,
        actn_type: str = "swish", use_bn: bool = True, use_ln: bool = False, **kwargs) -> None:
        super().__init__()

        self.depth_cnn = nn.Sequential(*[ConvBlock(1 if not i else conv_blocks[i - 1], conv_blocks[i], 3, 1, 1, use_bn, actn_type) for i in range(len(conv_blocks))])

        with torch.no_grad():
            in_features = self.depth_cnn(torch.randn(1, 1, res_in[0], res_in[1])).numel()

        self.depth_lstm = nn.LSTM(in_features, lstm_units, num_layers=lstm_layers, dropout=drop_prb, bidirectional=use_bilstm, batch_first=True)
        self.depth_mlp = MLP(T * lstm_units * (2 if use_bilstm else 1), mlp_layers, drop_prb, use_ln, actn_type, num_classes)
        

    def forward(self, x):
        _, x_dpt = x
        t = x_dpt.shape[1]

        # depth images
        x_dpt = rearrange(x_dpt, "b t c h w -> (b t) c h w")
        x_dpt = self.depth_cnn(x_dpt)
        x_dpt = rearrange(x_dpt, "(b t) c h w -> b t (c h w)", t=t)
        x_dpt = self.depth_lstm(x_dpt)[0]

        x_dpt = rearrange(x_dpt, "b t f -> b (t f)")
        x_dpt = self.depth_mlp(x_dpt)
        
        return x_dpt