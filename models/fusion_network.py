import torch
from torch import nn
from models.blocks import ConvBlock, MLP
from einops import rearrange
from typing import Tuple


class FeatureFusionNet(nn.Module):
    def __init__(self, conv_blocks: list = [8, 16, 32], res_in : Tuple[int, int] = (50, 50), T : int = 32, D : int = 2, num_classes : int = 14,
        drop_prb : float = 0.5, mlp_layers: list = [128], lstm_units: int = 128, lstm_layers: int = 2, use_bilstm: bool = True,
        actn_type: str = "swish", use_bn: bool = True, use_ln: bool = False, **kwargs) -> None:
        super().__init__()

        self.depth_cnn = nn.Sequential(*[ConvBlock(1 if not i else conv_blocks[i - 1], conv_blocks[i], 3, 1, 1, use_bn, actn_type) for i in range(len(conv_blocks))])

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
        x_dpt = rearrange(x_dpt, "b t c h w -> (b t) c h w")
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
        actn_type: str = "swish", use_bn: bool = True, use_ln: bool = False, **kwargs) -> None:
        super().__init__()

        self.depth_cnn = nn.Sequential(*[ConvBlock(1 if not i else conv_blocks[i - 1], conv_blocks[i], 3, 1, 1, use_bn, actn_type) for i in range(len(conv_blocks))])

        with torch.no_grad():
            in_features = self.depth_cnn(torch.randn(1, 1, res_in[0], res_in[1])).numel()

        self.depth_lstm = nn.LSTM(in_features, lstm_units, num_layers=lstm_layers, dropout=drop_prb, bidirectional=use_bilstm, batch_first=True)
        self.joint_lstm = nn.LSTM(22 * D, lstm_units, num_layers=lstm_layers, dropout=drop_prb, bidirectional=use_bilstm, batch_first=True)
        
        self.depth_mlp = MLP(T * lstm_units * (2 if use_bilstm else 1), mlp_layers, drop_prb, use_ln, actn_type, num_classes)
        self.joint_mlp = MLP(T * lstm_units * (2 if use_bilstm else 1), mlp_layers, drop_prb, use_ln, actn_type, num_classes)

    def forward(self, x):
        x_jnt, x_dpt = x
        
        # 2d joints
        x_jnt = self.joint_lstm(x_jnt)[0]

        # depth images
        x_dpt = rearrange(x_dpt, "b t c h w -> (b t) c h w")
        x_dpt = self.depth_cnn(x_dpt)
        x_dpt = rearrange(x_dpt, "(b t) c h w -> b t (c h w)", t=x_jnt.shape[1])
        x_dpt = self.depth_lstm(x_dpt)[0]

        # score fusion
        x_dpt, x_jnt = rearrange([x_dpt, x_jnt], "i b t f -> i b (t f)")
        x_dpt = self.depth_mlp(x_dpt)
        x_jnt = self.joint_mlp(x_jnt)
        
        return 0.5 * (x_dpt + x_jnt)


def model_from_name(name, num_classes):
    assert name in ["gvar_feature_fusion", "gvar_score_fusion", "vanilla_feature_fusion", "vanilla_score_fusion"]

    if name == "gvar_feature_fusion":
        model = FeatureFusionNet(num_classes=num_classes)

    elif name == "gvar_score_fusion":
        model = ScoreFusionNet(num_classes=num_classes)

    elif name == "vanilla_feature_fusion":
        model = FeatureFusionNet(res_in=(227,227), num_classes=num_classes, drop_prb=0.0, mlp_layers=[256,512,256], lstm_units=256,
                    use_bilstm=False, actn_type="relu", use_bn=False)
                    
    elif name == "vanilla_score_fusion":
        model = ScoreFusionNet(res_in=(227,227), num_classes=num_classes, drop_prb=0.0, mlp_layers=[256,512,256], lstm_units=256,
                    use_bilstm=False, actn_type="relu", use_bn=False)

    return model