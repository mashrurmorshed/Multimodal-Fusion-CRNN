import torch
from torch import nn
from models.blocks import get_block, MLP
from typing import Tuple


class FeatureFusionNet(nn.Module):
    """Feature-level fusion model."""

    def __init__(
        self,
        block_type : str = "base_block",
        conv_blocks: list = [8, 16, 32],
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

        self.blocks = nn.Sequential(*[
            block(1 if i == 0 else conv_blocks[i - 1], conv_blocks[i], 3, 1, 1) for i in range(len(conv_blocks))
        ])

        with torch.no_grad():
            n_features = self.blocks(torch.randn(1, 1, res_in[0], res_in[1])).numel()

        self.image_lstm = nn.LSTM(n_features, lstm_units, num_layers=lstm_layers, dropout=drop_prb, bidirectional=use_bilstm, batch_first=True)
        self.joint_lstm = nn.LSTM(22 * D, lstm_units, num_layers=lstm_layers, dropout=drop_prb, bidirectional=use_bilstm, batch_first=True)
        
        self.mlp = MLP(
            in_features=2 * T * ((2 * lstm_units) if use_bilstm else lstm_units),
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

        # Joints
        x_joint, _ = self.joint_lstm(x_joint)

        out = torch.cat((x_im, x_joint), dim=1).reshape(batch, -1)
        out = self.mlp(out)

        return out