import torch
from torch import nn
from models.blocks import ConvBlock, MLP
from einops import rearrange
from typing import Tuple


class FeatureFusionNet(nn.Module):

    def __init__(self, conv_blocks: list = [8, 16, 32], res_in : Tuple[int, int] = (50, 50), T : int = 32, D : int = 2, num_classes : int = 14,
        drop_prb : float = 0.5, mlp_layers: list = [128], lstm_units: int = 128, lstm_layers: int = 2, use_bilstm: bool = True,
        actn_type: str = "relu", use_bn: bool = True, use_ln: bool = True, **kwargs) -> None:
        super().__init__()

        conv_blocks.insert(0, 1)
        self.depth_cnn = nn.Sequential(*[ConvBlock(conv_blocks[i - 1], conv_blocks[i], 3, 1, 1, use_bn, actn_type) for i in range(1, len(conv_blocks))])

        with torch.no_grad():
            in_features = self.depth_cnn(torch.randn(1, 1, res_in[0], res_in[1])).numel()

        self.depth_lstm = nn.LSTM(in_features, lstm_units, num_layers=lstm_layers, dropout=drop_prb, bidirectional=use_bilstm, batch_first=True)
        self.joint_lstm = nn.LSTM(22 * D, lstm_units, num_layers=lstm_layers, dropout=drop_prb, bidirectional=use_bilstm, batch_first=True)
        
        f_in = 2 * T * lstm_units * (2 if use_bilstm else 1)
        self.mlp = MLP(f_in, mlp_layers, drop_prb, use_ln, actn_type, num_classes)

    def forward(self, x):
        x_jnt, x_dpt = x
        t = x_dpt.shape[1]

        # depth images
        x_dpt = rearrange(x_dpt, "b (t c) h w -> (b t) c h w", t=t)
        x_dpt = self.depth_cnn(x_dpt)
        x_dpt = rearrange(x_dpt, "(b t) c h w -> b t (c h w)", t=t)
        x_dpt = self.depth_lstm(x_dpt)[0]

        # 2d joints
        x_jnt = self.joint_lstm(x_jnt)[0]

        # multimodal fusion
        x_fused = torch.cat((x_dpt, x_jnt), dim=1)
        x_fused = rearrange(x_fused, "b t f -> b (t f)")
        x_fused = self.mlp(x_fused)
        
        return x_fused


class ScoreFusionNet(nn.Module):

    def __init__(self, conv_blocks: list = [8, 16, 32], res_in : Tuple[int, int] = (50, 50), T : int = 32, D : int = 2, num_classes : int = 14,
        drop_prb : float = 0.5, mlp_layers: list = [128], lstm_units: int = 128, lstm_layers: int = 2, use_bilstm: bool = True,
        actn_type: str = "relu", use_bn: bool = True, use_ln: bool = True, **kwargs) -> None:
        super().__init__()

        conv_blocks.insert(0, 1)
        self.depth_cnn = nn.Sequential(*[ConvBlock(conv_blocks[i - 1], conv_blocks[i], 3, 1, 1, use_bn, actn_type) for i in range(1, len(conv_blocks))])

        with torch.no_grad():
            in_features = self.depth_cnn(torch.randn(1, 1, res_in[0], res_in[1])).numel()

        self.depth_lstm = nn.LSTM(in_features, lstm_units, num_layers=lstm_layers, dropout=drop_prb, bidirectional=use_bilstm, batch_first=True)
        self.joint_lstm = nn.LSTM(22 * D, lstm_units, num_layers=lstm_layers, dropout=drop_prb, bidirectional=use_bilstm, batch_first=True)
        
        self.depth_mlp = MLP(T * lstm_units * (2 if use_bilstm else 1), mlp_layers, drop_prb, use_ln, actn_type, num_classes)
        self.joint_mlp = MLP(T * lstm_units * (2 if use_bilstm else 1), mlp_layers, drop_prb, use_ln, actn_type, num_classes)

    def forward(self, x):
        x_jnt, x_dpt = x
        t = x_dpt.shape[1]

        # depth images
        x_dpt = rearrange(x_dpt, "b (t c) h w -> (b t) c h w", t=t)
        x_dpt = self.depth_cnn(x_dpt)
        x_dpt = rearrange(x_dpt, "(b t) c h w -> b t (c h w)", t=t)
        x_dpt = self.depth_lstm(x_dpt)[0]
        x_dpt = rearrange(x_dpt, "b t f -> b (t f)")
        x_dpt = self.depth_mlp(x_dpt)

        # 2d joints
        x_jnt = self.joint_lstm(x_jnt)[0]
        x_jnt = rearrange(x_jnt, "b t f -> b (t f)")
        x_jnt = self.joint_mlp(x_jnt)
        
        return 0.5 * (x_dpt + x_jnt)