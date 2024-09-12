from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from easydict import EasyDict as edict
import torch
from torch import Tensor
import torch.nn as nn

from ..fields.field import Field
from .ray_sampler import RaySampler
from . import math_utils


class Renderer(nn.Module, ABC):
    def __init__(
        self,
        field: Field,
        cube_scale: float,
        depth_resolution: int,
        ray_limits: Optional[Tuple[float, float]] = None,
        ray_subsamples: int = 0,
        disparity_space_sampling: bool = False,
        white_back: bool = False
    ):
        super().__init__()
        self.field = field
        self.cube_scale = cube_scale
        self.depth_resolution = depth_resolution
        self.ray_limits = ray_limits
        self.ray_subsamples = ray_subsamples
        self.disparity_space_sampling = disparity_space_sampling
        self.white_back = white_back
        self.ray_sampler = RaySampler()
        self.randomize_depth_samples = False

    def get_ray_limits(self, ray_origins: Tensor, ray_directions: Tensor):
        batch_size, num_tar, num_rays = ray_origins.shape[:3]
        if self.ray_limits is None:
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_size=self.cube_scale)
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_end[is_ray_valid].max()
        else:
            ray_start = torch.tensor([self.ray_limits[0]], device=ray_origins.device).expand(batch_size, num_tar, num_rays, 1)
            ray_end = torch.tensor([self.ray_limits[1]], device=ray_origins.device).expand(batch_size, num_tar, num_rays, 1)
        return ray_start, ray_end

    def sample(self, ray_start, ray_end):
        """
        Return depths of approximately uniformly spaced samples along rays.
        Arguments:
            ray_start: [B, num_rays, 1]
            ray_end: [B, num_rays, 1]
        Returns:
            depths: [B, num_rays, num_samples, 1]
        """
        N, M, _ = ray_start.shape
        if self.disparity_space_sampling:
            depths = torch.linspace(0,
                                    1,
                                    self.depth_resolution,
                                    device=ray_start.device).reshape(1, 1, self.depth_resolution, 1).expand(N, M, 1, 1)
            depth_delta = 1/(self.depth_resolution - 1)
            depths += torch.rand_like(depths) * depth_delta
            depths = 1./(1./ray_start * (1. - depths) + 1./ray_end * depths)
        else:
            if type(ray_start) == torch.Tensor:
                depths = math_utils.linspace(ray_start, ray_end, self.depth_resolution).permute(1,2,0,3).contiguous()
                depth_delta = (ray_end - ray_start) / (self.depth_resolution - 1)
            else:
                depths = torch.linspace(ray_start, ray_end, self.depth_resolution, device=ray_start.device).reshape(1, 1, self.depth_resolution, 1).expand(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(self.depth_resolution - 1)
        if self.randomize_depth_samples:
            depth_delta = (ray_end - ray_start)/(self.depth_resolution - 1)
            depths += torch.rand_like(depths) * depth_delta.unsqueeze(-1)
        return depths

    @staticmethod
    def get_ray_idx(mask: Tensor):
        """
        Arguments:
            mask: [B, num_tar, num_sample_rays, max_shading_pts, 1]
        Returns:
            idx: [num_valid_pts]
        """
        mask = mask.view(-1, mask.shape[-2])
        idx = torch.arange(mask.shape[0], device=mask.device).unsqueeze(-1).expand(*mask.shape)
        return idx[mask]

    @staticmethod
    def get_ray_samples(ray: Tensor, ray_sample_mask: Tensor):
        return torch.masked_select(ray, ray_sample_mask.view(ray.shape[:3] + (1,) * (ray.ndim - 3))).view(ray.shape[:2] + (-1,) + ray.shape[3:])

    @staticmethod
    def get_depths_from_shading_pts(pts: Tensor, mask: Tensor, ray_sample_mask: Optional[Tensor], ray_o: Tensor, ray_d: Tensor, ray_end: Tensor):
        # Already checked: depths are sorted because pts remain sorted according to depth
        if torch.numel(pts) == 0:
            return torch.zeros_like(mask, dtype=torch.float)
        if ray_sample_mask is not None:
            ray_o = Renderer.get_ray_samples(ray_o, ray_sample_mask)
            ray_d = Renderer.get_ray_samples(ray_d, ray_sample_mask)
            ray_end = Renderer.get_ray_samples(ray_end, ray_sample_mask)
        max_shading_pts = pts.shape[-2]
        depths = torch.nanmean((pts - ray_o) / ray_d, dim=-1, keepdim=True)     # [B, num_tar, num_rays, max_shading_pts, 1]
        depths[~mask] = -torch.inf
        depths = torch.cummax(depths, dim=-2).values
        invalid_mask = depths == -torch.inf
        depths[invalid_mask] = ray_end[..., None, :].expand(-1, -1, -1, max_shading_pts, -1)[invalid_mask]
        return depths

    @abstractmethod
    def get_alpha(
        shape: Tensor,
        depths: Tensor,
        **kwargs
    ):
        raise NotImplementedError()

    def ray_march(
        self,
        shape: Tensor, 
        depths: Tensor, 
        channels: Optional[Tensor],
        kp_weights: Optional[Dict[str, Tensor]],
        valid_mask: Optional[Tensor],
        **get_alpha_kwargs
    ):
        """
        Arguments:
            shape: [B, num_tar, num_sample_rays, max_shading_pts, 1]
            depths: [B, num_tar, num_sample_rays, max_shading_pts, 1]
            optional channels: [num_valid_pts, out_dim]
            optional kp_weights: Dictionary with
                shading_idx: [num_valid_pairs]
                kp_idx: [num_valid_pairs]
                weights: [num_valid_pairs]
            optional valid_mask: [B, num_tar, num_sample_rays, max_shading_pts, 1]
        Returns:
            mask: [B, num_tar, num_sample_rays, 1]
            depth: [B, num_tar, num_sample_rays, 1]
            optional channels: [B, num_tar, num_sample_rays, 3]
            optional kp_weights: [B, num_tar, num_sample_rays, num_kp]
                with num_kp = max {idx[kp] for kp with at least one weight > 0}
        """
        alpha = self.get_alpha(shape, depths, **get_alpha_kwargs)
        alpha_shifted = torch.cat([torch.ones_like(alpha[..., :1, :]), 1-alpha + 1e-10], dim=-2)
        weights = alpha * torch.cumprod(alpha_shifted, dim=-2)[..., :-1, :]

        weight_total = weights.sum(-2)
        composite_depth = torch.sum(weights * depths, dim=-2) / weight_total

        # clip the composite to min/max range of depths
        composite_depth = torch.nan_to_num(composite_depth, float('inf'))
        if torch.numel(composite_depth) > 0:
            composite_depth = torch.clamp(composite_depth, torch.min(depths), torch.max(depths))

        res = {
            "mask": weight_total,
            "depth": composite_depth
        }
        if channels is not None or kp_weights is not None:
            assert valid_mask is not None, "Cannot render channels or point_weights without valid_mask"
            B, num_tar, num_sample_rays = valid_mask.shape[:3]
            total_rays = B * num_tar * num_sample_rays
            ray_idx = self.get_ray_idx(valid_mask)  # [num_valid_pts]
            valid_weights = weights[valid_mask]     # [num_valid_pts]
            if channels is not None:
                weighted_channels = valid_weights.unsqueeze(1) * channels
                num_channels = channels.shape[-1]
                composite_channels = torch.zeros((total_rays, num_channels), device=channels.device)
                composite_channels.index_add_(0, ray_idx, weighted_channels)
                composite_channels = composite_channels.view(B, num_tar, num_sample_rays, num_channels)
                if self.white_back:
                    composite_channels = composite_channels + 1 - weight_total
                res["channels"] = composite_channels
            if kp_weights is not None:
                weighted_kp_weights = valid_weights[kp_weights["shading_idx"]] * kp_weights["weights"]
                num_kp = kp_weights["kp_idx"].max() + 1
                out_idx = num_kp * ray_idx[kp_weights["shading_idx"]] + kp_weights["kp_idx"]
                composite_kp_weights = torch.zeros(total_rays * num_kp, device=out_idx.device)
                composite_kp_weights.index_add_(0, out_idx, weighted_kp_weights)
                composite_kp_weights = composite_kp_weights.view(B, num_tar, num_sample_rays, num_kp)
                res["kp_weights"] = composite_kp_weights
        return res

    @abstractmethod
    def render(
        self, 
        kp_pos: Tensor, 
        kp_feat: Tensor, 
        ray_origins: Tensor, 
        ray_directions: Tensor, 
        ray_start: Tensor, 
        ray_end: Tensor, 
        sample: bool,
        return_channels: bool = True,
        return_kp_weights: bool = False
    ):
        raise NotImplementedError()

    def forward(
        self,
        kp_pos: Tensor,
        kp_feat: Tensor,
        extr: Tensor,
        intr: Tensor,
        resolution: int,
        sample: bool, 
        return_channels: bool = True, 
        return_kp_weights: bool = False
    ) -> edict:
        """
        Returns: EasyDict with
            mask: [B, num_tar, num_sample_rays, 1]
            depth: [B, num_tar, num_sample_rays, 1]
            ray_idx: [B, num_tar, num_sample_rays, 1] or None
            optional channels: [B, num_tar, num_sample_rays, 3]
            optional kp_weights: [B, num_tar, num_sample_rays, num_kp]
            NeUSRenderer: 
                grad: [B, num_tar, num_sample_rays, max_shading_pts, 3]
                valid_pts_mask: [B, num_tar, num_sample_rays, max_shading_pts, 1]
        """
        B, num_tar = extr.shape[:2]
        n_kp = kp_pos.shape[1]
        extr = extr.flatten(0, 1)
        intr = intr.flatten(0, 1)

        ray_origins, ray_directions = self.ray_sampler(extr, intr, resolution)    # [B*num_tar, num_rays, 3], [B*num_tar, num_rays, 3]
        num_instances, num_rays = ray_origins.shape[:2]

        if self.ray_subsamples and sample:
            ray_idx = torch.randperm(num_rays, device=ray_origins.device)[:self.ray_subsamples, None].expand(num_instances, -1, -1)
            # ray_idx = torch.randint(0, num_rays, (num_instances, self.ray_subsamples, 1), device=ray_origins.device)
            ray_origins = ray_origins.gather(dim=1, index=ray_idx.expand(-1, -1, 3))
            ray_directions = ray_directions.gather(dim=1, index=ray_idx.expand(-1, -1, 3))
            num_rays = self.ray_subsamples
            ray_idx = ray_idx.view(B, num_tar, num_rays, 1)

        ray_origins = ray_origins.view(B, num_tar, -1, 3)        # [B, num_tar, num_rays, 3]
        ray_directions = ray_directions.view(B, num_tar, -1, 3)  # [B, num_tar, num_rays, 3]

        ray_start, ray_end = self.get_ray_limits(ray_origins, ray_directions)

        out = self.render(
            kp_pos, 
            kp_feat, 
            ray_origins, 
            ray_directions, 
            ray_start, 
            ray_end, 
            sample, 
            return_channels,
            return_kp_weights
        )

        if return_kp_weights:
            n_missing_kp = n_kp-out["kp_weights"].shape[-1]
            if n_missing_kp > 0:
                out["kp_weights"] = torch.cat((out["kp_weights"], torch.zeros_like(out["kp_weights"][..., :n_missing_kp])), dim=-1)

        ray_sample_mask = out.pop("ray_sample_mask")    # [B, num_tar*num_rays] or None
        if sample:
            if not self.ray_subsamples:
                ray_idx = torch.arange(num_rays, device=ray_sample_mask.device)[:, None].expand(B, num_tar, -1, 1)
            ray_idx = torch.masked_select(ray_idx, ray_sample_mask[..., None]).view(B, num_tar, -1, 1)
            out["ray_idx"] = ray_idx
        return edict(out)
