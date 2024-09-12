from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from ..fields.field import Field
from .renderer import Renderer


class VolumeRenderer(Renderer):
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
        super(VolumeRenderer, self).__init__(field, cube_scale, depth_resolution, ray_limits, ray_subsamples, disparity_space_sampling, white_back)

    def get_alpha(
        self, 
        densities: Tensor, 
        depths: Tensor, 
    ):
        """
        Arguments:
            densities: [B, num_tar, num_sample_rays, max_shading_pts, 1]
            depths: [B, num_tar, num_sample_rays, max_shading_pts, 1]
        Returns:
            alpha: [B, num_tar, num_sample_rays, max_shading_pts, 1]
        """
        deltas = torch.cat((depths[..., 1:, :] - depths[..., :-1, :], torch.zeros_like(depths[..., :1, :])), dim=-2)

        density_delta = densities * deltas
        alpha = 1 - torch.exp(-density_delta)
        return alpha

    def render(
        self, 
        kp_pos: Tensor, 
        kp_feat: Tensor, 
        rays_o: Tensor, 
        rays_d: Tensor, 
        ray_start: Tensor, 
        ray_end: Tensor, 
        sample: bool, 
        return_channels: bool = True,
        return_kp_weights: bool = False
    ):
        """
        Returns: Dictionary with
            mask: [B, num_tar, num_sample_rays, 1]
            depth: [B, num_tar, num_sample_rays, 1]
            ray_sample_mask: [B, num_tar, num_rays]
            optional channels: [B, num_tar, num_sample_rays, 3]
            optional kp_weights: [B, num_tar, num_sample_rays, num_kp]
                with num_kp = max {idx[kp] for kp with at least one weight > 0}
        """
        B, num_tar, num_rays = rays_o.shape[:3]
        depths = self.sample(ray_start.flatten(0, 1), ray_end.flatten(0, 1))         # [B*num_tar, num_rays, samples_per_ray, 1]
        depths = depths.view(B, num_tar, num_rays, -1, 1)

        # Add sample dimension
        rays_o = rays_o.unsqueeze(-2)
        rays_d = rays_d.unsqueeze(-2)

        sample_coordinates = rays_o + depths * rays_d  # [B, num_tar, num_rays, samples_per_ray, 3]
        out = self.field(
            sample_coordinates, 
            rays_d, 
            kp_pos, 
            kp_feat, 
            sample=sample, 
            return_channels=return_channels, 
            return_mask=True, 
            return_pts=True,
            return_kp_weights=return_kp_weights
        )
        # [B, num_tar, num_sample_rays, max_shading_pts, 1]
        depths = self.get_depths_from_shading_pts(out["pts"], out["mask"], out["ray_sample_mask"], rays_o, rays_d, ray_end)
        res = self.ray_march(
            out["shape"], 
            depths, 
            out.get("channels", None),
            out.get("kp_weights", None),
            out.get("mask", None)
        )
        res["ray_sample_mask"] = out["ray_sample_mask"]
        return res
