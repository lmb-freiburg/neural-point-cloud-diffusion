import torch.nn as nn

from .pointnerf.pointnerf import PointNeRF
from .diffusion.diffusion_model import DiffusionModel


class NPCD(nn.Module):
    def __init__(self, n_obj, coords_dim, feats_dim, num_points, use_view_dir, width, layers, heads, use_flash_attn=False, pointnerf_only=False, freeze_pointnerf=False):
        super().__init__()
        
        self.pointnerf = PointNeRF(n_obj, feats_dim, num_points, use_view_dir)
        self.freeze_pointnerf = freeze_pointnerf
        if self.freeze_pointnerf:
            for param in self.pointnerf.parameters():
                param.requires_grad = False
            self.pointnerf.eval()
        
        if not pointnerf_only:
            self.diffusion = DiffusionModel(coords_dim, feats_dim, num_points, width, layers, heads, use_flash_attn)
            
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_pointnerf:
            self.pointnerf.eval()
        return self
