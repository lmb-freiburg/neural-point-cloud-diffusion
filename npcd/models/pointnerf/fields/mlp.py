from typing import List, Optional

import torch
from torch import Tensor
from torch_knnquery import VoxelGrid

from .field import Field
from npcd.utils.model import define_mlp
from npcd.utils.positional_encoder import PositionalEncoder1D


class MLP(Field):
    def __init__(
        self,
        in_dim: int,
        voxel_grid: Optional[VoxelGrid],
        aggregator: dict,
        feat_freqs: int,
        dir_freqs: int,
        channel_layers: List[int],
        shape_layers: List[int],
        activation: str = "ReLU",
        layer_norm: bool = False,
        nerf: bool = True,
        use_dir: bool = True,
        aggregate_shape: bool = False
    ) -> None:
        super(MLP, self).__init__(in_dim, voxel_grid, aggregator, nerf, use_dir, aggregate_shape)
        self.feat_enc = PositionalEncoder1D(feat_freqs) if feat_freqs > 0 else None
        self.dir_enc = PositionalEncoder1D(dir_freqs) if self.use_dir and dir_freqs > 0 else None
        hid_dim = self.feat_enc.d_out(self.hid_dim) if self.feat_enc is not None else self.hid_dim
        channel_hid_dim = hid_dim
        if self.use_dir:
            channel_hid_dim = channel_hid_dim + (self.dir_enc.d_out(3) if self.dir_enc is not None else 3)
        self.channel_net = define_mlp(channel_layers, channel_hid_dim, d_out=3, act=activation, layer_norm=layer_norm)
        self.shape_net = define_mlp(shape_layers, hid_dim, d_out=1, act=activation, layer_norm=layer_norm)

    def get_shape(
        self, 
        feat: Tensor,
    ) -> Tensor:
        """
        Arguments:
            feat: [num_valid_pts, hid_dim]
        Return:
            shape: [num_valid_pts, 1]
        """
        if self.feat_enc is not None:
            feat = self.feat_enc(feat)
        shape = self.shape_net(feat)
        return shape

    def get_channels(
        self,
        feat: Tensor, 
        ray_dir: Optional[Tensor],
    ) -> Tensor:
        """
        Arguments:
            feat: [num_valid_pts, hid_dim]
            ray_dir: [num_valid_pts, 3]
        Return:
            channels: [num_valid_pts, 3]
        """
        if self.feat_enc is not None:
            feat = self.feat_enc(feat)
        if self.use_dir:
            if self.dir_enc is not None:
                ray_dir = self.dir_enc(ray_dir)
            feat = torch.cat((feat, ray_dir), dim=-1)
        channels = self.channel_net(feat)
        return channels