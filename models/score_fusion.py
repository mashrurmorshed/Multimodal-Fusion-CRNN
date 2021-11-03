import torch
from torch import nn
from models.blocks import get_block, MLP
from typing import Tuple


class ScoreFusionNet(nn.Module):
    """Score level fusion."""

    def __init__(
        self,
        block_type : str = "base_block",
        res_in : Tuple[int, int] = (50, 50),
        T : int = 32,
        D : int = 2,
        num_classes : int = 14,
        drop_prb : float = 0.5,
        mlp_layers: list = [256, 512, 256],
        lstm_layers: int = 2,
        lstm_units: int = 128,
        use_bilstm: bool = True,
        **kwargs
    ) -> None:
    
        """[summary]

        Args:
            block_type (str, optional): [description]. Defaults to "base_block".
            res_in (Tuple[int, int], optional): [description]. Defaults to (50, 50).
            T (int, optional): [description]. Defaults to 32.
            D (int, optional): [description]. Defaults to 2.
            num_classes (int, optional): [description]. Defaults to 14.
            drop_prb (float, optional): [description]. Defaults to 0.5.
        """


        super().__init__()

        block = get_block(block_type)

        self.blocks = nn.Sequential(
            block(C_in=1, C_out=8, kernel=3, stride=1, padding=1),
            block(C_in=8, C_out=16, kernel=3, stride=1, padding=1),
            block(C_in=16, C_out=32, kernel=3, stride=1, padding=1)
        )

        with torch.no_grad():
            n_features = self.blocks(torch.randn(1, 1, res_in[0], res_in[1])).numel()

        self.image_lstm = nn.LSTM(n_features, lstm_units, num_layers=lstm_layers, dropout=drop_prb, bidirectional=use_bilstm, batch_first=True)
        self.joint_lstm = nn.LSTM(22 * D, lstm_units, num_layers=lstm_layers, dropout=drop_prb, bidirectional=use_bilstm, batch_first=True)

        self.mlp_image = MLP(
            in_features=T * ((2 * lstm_units) if use_bilstm else lstm_units),
            layers=mlp_layers,
            drop_prb=drop_prb,
            num_classes=num_classes
        )

        self.mlp_joint = MLP(
            in_features=T * ((2 * lstm_units) if use_bilstm else lstm_units),
            layers=mlp_layers,
            drop_prb=drop_prb,
            num_classes=num_classes
        )

    def forward(self, input):
        x_joint, x_im = input
        batch, T = x_im.shape[:2]

        # Images
        x_im = self.blocks(x_im.view(-1, 1, *x_im.shape[-2:]))
        x_im, _ = self.image_lstm(x_im.view((batch, T, -1)))
        x_im = self.mlp_image(x_im.reshape(batch, -1))

        # Joints
        x_joint, _ = self.joint_lstm(x_joint)
        x_joint = self.mlp_joint(x_joint.reshape(batch, -1))

        # score level fusion (avg)
        return (x_im + x_joint) / 2