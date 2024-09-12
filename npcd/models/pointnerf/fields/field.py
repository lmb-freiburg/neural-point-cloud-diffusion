from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch_knnquery import VoxelGrid

from . import aggregators


class Field(Module, ABC):
    def __init__(
        self, 
        in_dim: int, 
        voxel_grid: Optional[VoxelGrid], 
        aggregator: dict, 
        nerf: bool = True, 
        use_dir: bool = True,
        aggregate_shape: bool = False
    ) -> None:
        super(Field, self).__init__()
        self.aggregator: aggregators.Aggregator = getattr(aggregators, aggregator.network)(in_dim, voxel_grid, **aggregator.kwargs)
        self.hid_dim = self.aggregator.out_dim
        self.nerf = nerf
        self.use_dir = use_dir
        self.aggregate_shape = aggregate_shape
        if self.nerf:
            self.shape_act = lambda x: F.softplus(x - 1)   # activation bias of -1 makes things initialize better
        else:
            self.shape_act = lambda x: x
    
    @abstractmethod
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
        raise NotImplementedError()

    @abstractmethod
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
            channels: [num_valid_pts, out_dim]
        """
        raise NotImplementedError()

    def scatter_shape(self, mask: Tensor, shape: Tensor) -> Tensor:
        if self.nerf:
            res = torch.zeros_like(mask, dtype=torch.float)
        else:
            res = torch.full_like(mask, fill_value=self.aggregator.scaled_r, dtype=torch.float)
        res.masked_scatter_(mask, shape)
        return res

    def scatter(self, mask: Tensor, source: Tensor) -> Tensor:
        val_dim = source.shape[-1]
        out_shape = mask.shape[:-1] + (val_dim,)
        res = torch.zeros(out_shape, device=source.device)
        res.masked_scatter_(mask.expand(*out_shape), source)
        return res

    def forward(
        self, 
        x: Tensor,
        ray_dir: Optional[Tensor],
        kp_pos: Tensor, 
        kp_feat: Tensor, 
        sample: bool,
        return_channels: bool = True,
        return_mask: bool = False,
        return_pts: bool = False,
        return_grad: bool = False,
        return_kp_weights: bool = False
    ) -> Dict[str, Tensor]:
        """
        Arguments:
            x: [B, num_tar, num_rays, num_shading_pts, 3]
            ray_dir: [B, num_tar, num_rays, 1, 3]
            kp_pos: [B, num_kp, 3]
            kp_feat: [B, num_kp, kp_dim]
        Return:
            Dictionary with:
            shape: [B, num_tar, num_sample_rays, max_shading_pts, 1]
            ray_sample_mask: [B, num_tar, num_rays]
            optional channels: [num_valid_pts, 3]  #  [B, num_tar, num_sample_rays, max_shading_pts, 3]
            optional mask: [B, num_tar, num_sample_rays, max_shading_pts, 1]
            optional pts: [B, num_tar, num_sample_rays, max_shading_pts, 3]
            optional grad: [B, num_tar, num_sample_rays, max_shading_pts, 3]
            optional kp_weights: Dictionary with
                shading_idx: [num_valid_pairs]
                kp_idx: [num_valid_pairs]
                weights: [num_valid_pairs]
        """
        B, num_tar = x.shape[:2]
        agg_res = self.aggregator.get_local_feat(x, kp_pos, kp_feat, sample)
        agg_kwargs = {k: agg_res[k] for k in ("weights", "shading_idx", "num_valid_pts")}
        if self.aggregate_shape:
            shape = self.get_shape(agg_res["local_feat"]) # [num_valid_pairs, 1]
            shape = self.shape_act(shape)
            if return_channels:
                local_feat = torch.cat((shape, agg_res["local_feat"]), dim=-1)    # [num_valid_pairs, hid_dim+1]
                # [num_valid_pts, out_dim+1]
                feat = self.aggregator.aggregate_local_feat(local_feat, **agg_kwargs)
                shape = feat[:, :1]
                feat = feat[:, 1:]
            else:
                # [num_valid_pts, 1]
                shape = self.aggregator.aggregate_local_feat(shape, **agg_kwargs)
        else:
            # [num_valid_pts, out_dim+1]
            feat = self.aggregator.aggregate_local_feat(agg_res["local_feat"], **agg_kwargs)
            shape = self.get_shape(feat)
            shape = self.shape_act(shape)
        
        res = {
            "shape": self.scatter_shape(agg_res["mask"], shape), 
            "ray_sample_mask": agg_res["ray_mask"],
        }
        if return_channels:
            if ray_dir is not None:
                if sample:
                    ray_dir = torch.masked_select(ray_dir, res["ray_sample_mask"][..., None, None]).view(B, num_tar, -1, 1, 3)
                ray_dir = torch.masked_select(ray_dir, agg_res["mask"]).view(-1, 3)  # [num_sample_pts, 3]
            channels = self.get_channels(feat, ray_dir)
            channels = torch.sigmoid(channels)
            res["channels"] = channels
        if return_pts:
            res["pts"] = self.scatter(agg_res["mask"], agg_res["shading_pts"])
        if return_grad:
            d_output = torch.ones_like(shape, requires_grad=False)
            grad = torch.autograd.grad(outputs=shape, inputs=agg_res["shading_pts"], grad_outputs=d_output, create_graph=True, retain_graph=True)[0]
            res["grad"] = self.scatter(agg_res["mask"], grad)
        if return_kp_weights:
            res["kp_weights"] = {k: agg_res[k] for k in ("shading_idx", "kp_idx", "weights")}
        if return_mask:
            res["mask"] = agg_res["mask"]
        return res

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
