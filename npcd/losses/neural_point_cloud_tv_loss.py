import torch
import torch.nn as nn

from npcd.utils import logging


class NeuralPointCloudTVLoss(nn.Module):
    def __init__(self, model, weight=1, verbose=True):

        super().__init__()

        self.model = model
        self.weight = weight
        self.verbose = verbose

        if self.verbose:
            logging.info(f"Initializing {self.name} loss.")

        if self.verbose:
            logging.info(f"\tWeight: {self.weight}")
            logging.info(f"Finished initializing {self.name} loss.")
            logging.info()

    @property
    def name(self):
        name = type(self).__name__
        return name

    def forward(self, sample, pred, aux, iteration):

        sub_losses = {}
        pointwise_losses = {}
        
        feats = aux['feats']
        coords = aux['coords']
        
        B, num_points = coords.shape[:2]
        total_num_points = B * num_points
        coords = coords.detach()
        device = coords.device
        x = coords.view(-1, 1, num_points, 1, 3)    # [B, 1, num_points, 1, 3]
        # VoxelGrid looses keypoints sometimes such that num_valid_points does not have to be equal to B*num_points
        # [num_valid_points, k], _, [B, 1, num_points, 50, 1]
        neighbor_idx, _, valid_point_mask = self.model.pointnerf.field.aggregator.query_keypoints(x, coords)
        valid_point_mask = valid_point_mask[..., 0, :].view(B, num_points, 1)
        # pad neighbor_idx to [B*num_points, k]
        padded_neighbor_idx = torch.full((B, num_points, neighbor_idx.shape[-1]), fill_value=-1, dtype=torch.long, device=device)
        padded_neighbor_idx[..., 0] = torch.arange(num_points, device=device).unsqueeze(0)
        padded_neighbor_idx.masked_scatter_(valid_point_mask, neighbor_idx)
        neighbor_idx = padded_neighbor_idx
        # Remove origin points as neighbors, if there are any other neighbors
        origin_idx = torch.arange(num_points, device=device)[None, :, None]
        identity_mask = neighbor_idx == origin_idx
        valid_neighbor_mask = neighbor_idx >= 0
        enough_neighbor_mask = valid_neighbor_mask.int().sum(dim=-1, keepdim=True) > 1
        delete_mask = torch.logical_and(identity_mask, enough_neighbor_mask)
        neighbor_idx[delete_mask] = -1
        neighbor_idx = neighbor_idx.flatten(0, 1)
        # Continue with modified neighbor_idx
        valid_neighbor_mask = neighbor_idx >= 0
        neighbor_idx[~valid_neighbor_mask] = 0   # for valid indices during index select
        idx = self.model.pointnerf.field.aggregator.mask_to_batch_ray_idx(valid_neighbor_mask)  # [num_valid_pairs]
        origin_pos = coords.flatten(0, 1)[idx]  # [num_valid_pairs, 3]
        neighbors = self.model.pointnerf.field.aggregator.get_keypoint_data(neighbor_idx, valid_neighbor_mask, coords, feats)
        # Compute neighbor weights as normalized inverse Euclidean distances
        weights = 1 / (torch.linalg.norm(neighbors["pos"] - origin_pos, dim=-1) + 1.e-5)   # [num_valid_pairs]
        norm = torch.zeros(total_num_points, device=device)
        norm.index_add_(0, idx, weights)
        # Compute weighted total variation
        origin_feat = feats.flatten(0, 1)[idx]    # [num_valid_pairs, feat_dim]
        feat_dist = torch.linalg.norm(neighbors["feat"] - origin_feat, ord=1, dim=-1) # [num_valid_pairs]
        weighted = weights * feat_dist
        tv = torch.zeros(total_num_points, device=device)
        tv.index_add_(0, idx, weighted)
        tv = tv.view(B, num_points)
        tv = tv * self.weight
        
        pointwise_losses['00_neural_point_cloud_tv'] = tv
        tv = tv.mean()
        sub_losses['00_neural_point_cloud_tv'] = tv
        total_loss = tv
        
        return total_loss, sub_losses, pointwise_losses
