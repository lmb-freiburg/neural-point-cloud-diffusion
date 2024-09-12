from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch_knnquery import VoxelGrid

from npcd.utils.util import shifted_cumsum


class Aggregator(Module, ABC):
    def __init__(self, in_dim: int, voxel_grid: Optional[VoxelGrid], k: int, r: float, max_shading_pts: int, ray_subsamples: int, out_dim: int) -> None:
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.voxel_grid = voxel_grid
        assert k > 0, "k for kNN has to be greater than zero"
        self.k = k
        self.r = r
        self.scaled_r = self.r if voxel_grid is None else self.r * max(self.voxel_grid.vsize_tup)
        self.max_shading_pts = max_shading_pts
        self.ray_subsamples = ray_subsamples
        self.out_dim = out_dim

    def query_keypoints(self, x: Tensor, kp_pos: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        parameters:
        x: [B, num_tar, num_rays, num_shading_pts, 3]
        kp_pos: [B, num_kp, 3]

        returns:
        neighbor_idx: [num_valid_pts, k] (containing -1 for invalid neighbors)
        shading_pts: [num_valid_pts, 3]
        mask: [B, num_tar, num_rays, max_shading_pts, 1]
        """
        B, num_tar, num_orig_rays, num_shading_pts = x.shape[:4]
        num_kp = kp_pos.shape[1]
        device = x.device

        x = x.flatten(1, 2)                                                                         # [B, num_tar*num_rays, num_shading_pts, 3]
        num_rays = num_tar * num_orig_rays
        if self.voxel_grid is None:
            # Implementation without voxel_grid actually finds the first max_shading_pts valid points along each ray, 
            # i.e., points with at least one neighbor within the given radius
            dist = torch.cdist(x.reshape(B, -1, 3), kp_pos).view(B, num_rays, num_shading_pts, -1)  # [B, num_rays, num_shading_pts, num_kp]
            topk = torch.topk(dist, self.k, dim=-1, largest=False, sorted=False)

            valid_neighbor_mask = topk.values < self.r                                              # [B, num_rays, num_shading_pts, k]
            neighbor_idx = topk.indices + num_kp * torch.arange(B, device=device).view(-1, 1, 1, 1)
            neighbor_idx[~valid_neighbor_mask] = -1
            valid_pts_mask = valid_neighbor_mask.any(dim=-1, keepdim=True)                          # [B, num_rays, num_shading_pts, 1]
            pts_cumsum = torch.cumsum(valid_pts_mask, dim=-2)
            valid_pts_mask = torch.logical_and(valid_pts_mask, pts_cumsum <= self.max_shading_pts)

            neighbor_idx = torch.masked_select(neighbor_idx, valid_pts_mask).view(-1, self.k)       # [num_valid_pts, k]
            shading_pts = torch.masked_select(x, valid_pts_mask).view(-1, 3)                        # [num_valid_pts, 3]
            num_valid_pts = valid_pts_mask.sum(-2, keepdim=True)
            mask = torch.arange(self.max_shading_pts, device=device).view(1, 1, -1, 1) < num_valid_pts  # [B, num_rays, max_shading_pts, 1]
        else:
            # Voxel grid implementation masks points within occupied voxels first,
            # i.e., voxels with at least one voxel within its kernel that contains at least one point.
            # However, masked points are not necessarily valid, i.e., they do not have to have a neighbor within the radius.
            sample_idx, sample_loc, ray_mask = self.voxel_grid.query(x, self.k, self.r, self.max_shading_pts)
            sample_idx = sample_idx.to(dtype=torch.int64)
            ray_mask = ray_mask.bool()
            valid_neighbor_mask = sample_idx >= 0
            valid_pts_mask = valid_neighbor_mask.any(dim=-1, keepdim=True)                          # [num_valid_rays, max_shading_pts, 1]

            mask = torch.zeros((B, num_rays, self.max_shading_pts, 1), dtype=torch.bool, device=device)
            mask.masked_scatter_(ray_mask[..., None, None], valid_pts_mask)

            neighbor_idx = torch.masked_select(sample_idx, valid_pts_mask).view(-1, self.k)         # [num_valid_pts, k]
            shading_pts = torch.masked_select(sample_loc, valid_pts_mask).view(-1, 3)               # [num_valid_pts, 3]

        shading_pts.requires_grad_(x.requires_grad)
        return neighbor_idx, shading_pts, mask.view(B, num_tar, num_orig_rays, -1, 1)

    def subsample_valid_rays(self, neighbor_idx: Tensor, shading_pts: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Arguments:
            neighbor_idx: [num_valid_pts, k]
            shading_pts: [num_valid_pts, 3]
            mask: [B, num_tar, num_rays, max_shading_pts, 1]
        Returns:
            neighbor_idx_samples: [num_sampled_pts, k]
            shading_pts_samples: [num_sampled_pts, 3]
            sampled_mask: [B, num_tar, num_samples, max_shading_pts, 1]
            ray_sample_mask: [B, num_tar, num_rays]
        """
        B, num_tar = mask.shape[:2]
        device = mask.device
        mask = mask.flatten(0, 1)                                       # [B * num_tar, num_rays, max_shading_pts, 1]
        valid_ray_mask = torch.any(mask[..., 0], dim=-1)                # [B * num_tar, num_rays]
        valid_ray_idx = torch.nonzero(valid_ray_mask)                   # [total_valid_rays, 2]
        # Shuffle within instances
        perm = torch.randperm(valid_ray_idx.size(0))                    # [total_valid_rays]
        valid_ray_idx = valid_ray_idx[perm, :]
        sorted_batch = torch.argsort(valid_ray_idx[:, 0])
        valid_ray_idx = valid_ray_idx[sorted_batch, 1]                  # [total_valid_rays]
        # Get extraction idx
        num_valid_rays = valid_ray_mask.sum(-1)                         # [B * num_tar]
        num_samples = min(num_valid_rays.min(), self.ray_subsamples)
        batch_start = shifted_cumsum(num_valid_rays, dim=0, shift=1)
        sample_idx_idx = (                                              # [B * num_tar * num_samples]
            torch.arange(num_samples, device=device).view(1, -1) \
                + batch_start.view(-1, 1)
        ).view(-1)
        # Apply extraction idx
        ray_sample_idx = valid_ray_idx[sample_idx_idx].view(B*num_tar, num_samples)
        ray_sample_mask = torch.zeros_like(valid_ray_mask)
        ray_sample_mask.scatter_(1, ray_sample_idx, True)
        sample_pts_mask = torch.masked_select(ray_sample_mask[..., None, None], mask)

        neighbor_idx_samples = neighbor_idx[sample_pts_mask, :]
        shading_pts_samples = shading_pts[sample_pts_mask, :]
        sampled_mask = torch.masked_select(mask, ray_sample_mask[..., None, None])\
            .view(B, num_tar, num_samples, self.max_shading_pts, 1)
        ray_sample_mask = ray_sample_mask.view(B, num_tar, -1)
        return neighbor_idx_samples, shading_pts_samples, sampled_mask, ray_sample_mask
        
    @staticmethod
    def get_keypoint_data(
        neighbor_idx: Tensor,
        mask: Tensor,
        kp_pos: Optional[Tensor] = None,
        kp_feat: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        k = neighbor_idx.shape[-1]
        res = {}
        data = []
        if kp_pos is not None:
            data.append(kp_pos)
        if kp_feat is not None:
            data.append(kp_feat)
        data = torch.cat(data, dim=-1)
        val_dim = data.shape[-1]
        selected = torch.index_select(data.view(-1, val_dim), 0, neighbor_idx.view(-1)).view(-1, k, val_dim)
        masked = torch.masked_select(selected, mask[..., None]).view(-1, val_dim)
        if kp_pos is not None:
            res["pos"] = masked[:, :3]
            masked = masked[:, 3:]
        if kp_feat is not None:
            res["feat"] = masked
        return res

    @staticmethod
    def mask_to_batch_ray_idx(valid_neighbor_mask: Tensor):
        """
        Arguments:
            valid_neighbor_mask: [num_valid_pts, k]
        Returns:
            batch_ray_idx: [num_valid_pairs]
        """
        num_valid_pts = valid_neighbor_mask.shape[0]
        source = torch.arange(num_valid_pts, device=valid_neighbor_mask.device).unsqueeze(1)
        return torch.masked_select(source, valid_neighbor_mask)

    @abstractmethod
    def get_local_feat(
        self, 
        x: Tensor, 
        kp_pos: Tensor, 
        kp_feat: Tensor, 
        sample: bool
    ) -> dict:
        """
        Arguments:
            x: [B, num_tar, num_rays, num_shading_pts, 3]
            kp_pos: [B, num_kp, 3]
            kp_feat: [B, num_kp, in_dim]
        Return: Dictionary with
            local_feat: [num_valid_pairs, out_dim]
            shading_pts: [num_sample_pts, 3]
            mask: [B, num_tar, num_sample_rays, max_shading_pts, 1]
            ray_mask: [B, num_tar, num_rays]
            shading_idx: [num_valid_pairs]
            kp_idx: [num_valid_pairs]
            weights: [num_valid_pairs]
            num_valid_pts (int)
        """
        raise NotImplementedError()

    @abstractmethod
    def aggregate_local_feat(
        self,
        local_feat: Tensor,
        weights: Tensor,
        shading_idx: Tensor,
        num_valid_pts: int
    ) -> Tensor:
        """
        Arguments:
            local_feat: [num_valid_pairs, out_dim]
            weights: [num_valid_pairs]
            shading_idx: [num_valid_pairs]
            num_valid_pts (int)
        Return:
            feat: [num_valid_pts, out_dim]
        """
        raise NotImplementedError()