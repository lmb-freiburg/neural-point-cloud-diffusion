import torch
import torch.nn as nn

from npcd.utils import to_torch, get_torch_model_device, split_num

from .diffusion_processes.gaussian_diffusion import GaussianDiffusion
from .denoisers.transformer import NPCDTransformer


class UnitGaussianNormalization(nn.Module):
    def __init__(self, dim, scale_per_axis=False, clip_per_axis=False):
        super().__init__()
        self.dim = dim
        self.scale_per_axis = scale_per_axis
        self.clip_per_axis = clip_per_axis
        self.register_buffer('min', torch.zeros(dim if clip_per_axis else 1))
        self.register_buffer('max', torch.zeros(dim if clip_per_axis else 1))
        self.register_buffer('shift', torch.zeros(dim))
        self.register_buffer('scale', torch.ones(dim if scale_per_axis else 1))
        
    def set_from_all_data(self, data):
        data = to_torch(data, self.shift.device).detach()
        # data shape: (dim, num_data_points)
        assert data.shape[0] == self.dim
        data = data.reshape(self.dim, -1)
        
        self.shift[:] = data.mean(axis=1)
        self.scale[:] = data.std(axis=1 if self.scale_per_axis else [0, 1])
            
        # compute min and max on normalized data:
        data = (data - self.shift[:, None]) / self.scale[:, None]
            
        if self.clip_per_axis:
            self.min, _ = data.min(axis=1)
            self.max, _ = data.max(axis=1)
        else:
            self.min[:] = data.min()
            self.max[:] = data.max()

    def forward(self, x):
        if self.training:
            return (x - self.shift[None, :, None]) / self.scale[None, :, None]
        else:
            return x * self.scale[None, :, None] + self.shift[None, :, None]
    
    
class MinusOneToOneNormalization(nn.Module):
    def __init__(self, dim, scale_per_axis=False, clip_per_axis=False):
        super().__init__()
        self.dim = dim
        self.scale_per_axis = scale_per_axis
        self.clip_per_axis = clip_per_axis
        self.register_buffer('min', torch.zeros(dim if clip_per_axis else 1))
        self.register_buffer('max', torch.zeros(dim if clip_per_axis else 1))
        self.register_buffer('shift', torch.zeros(dim))
        self.register_buffer('scale', torch.ones(dim if scale_per_axis else 1))
        
    def set_from_all_data(self, data):
        data = to_torch(data, self.shift.device).detach()
        # data shape: (dim, num_data_points)
        assert data.shape[0] == self.dim
        data = data.reshape(self.dim, -1)
        
        self.shift = (data.min(axis=1)[0] + data.max(axis=1)[0]) / 2.
        scale = (data.max(axis=1)[0] - data.min(axis=1)[0]) / 2.
        if not self.scale_per_axis:
            self.scale[:] = scale.max()
        else:
            self.scale = scale
            
        # compute min and max on normalized data:
        data = (data - self.shift[:, None]) / self.scale[:, None]
            
        if self.clip_per_axis:
            self.min, _ = data.min(axis=1)
            self.max, _ = data.max(axis=1)
        else:
            self.min[:] = data.min()  # should be -1
            self.max[:] = data.max()  # should be 1
    
    def forward(self, x):
        if self.training:
            return (x - self.shift[None, :, None]) / self.scale[None, :, None]
        else:
            return x * self.scale[None, :, None] + self.shift[None, :, None]


class DiffusionModel(nn.Module):
    def __init__(self, coords_dim, feats_dim, num_points, width, layers, heads, use_flash_attn):
        super().__init__()
        self.coords_dim = coords_dim
        self.feats_dim = feats_dim
        self.num_points = num_points
        self.diffusion_process = GaussianDiffusion()
        self.denoiser = NPCDTransformer(coords_dim=coords_dim, feats_dim=feats_dim, width=width, layers=layers, heads=heads, use_flash_attn=use_flash_attn)
        self.coords_normalization = UnitGaussianNormalization(dim=coords_dim)
        self.feats_normalization = MinusOneToOneNormalization(dim=feats_dim)
        
    def compute_loss(self, coords, feats):
        coords = self.coords_normalization(coords)
        feats = self.feats_normalization(feats)
        
        N, _, _ = coords.shape
        t = torch.randint(0, self.diffusion_process.num_timesteps, size=(N,), device=coords.device)
        
        return self.diffusion_process.p_losses(denoise_fn=self.denoiser, coords_start=coords, feats_start=feats, t=t)

    @torch.no_grad()
    def generate(self, num, batch_size=8, progress=True):
        assert not self.training, 'Model must be in eval mode for generation'
        
        device = get_torch_model_device(self)
        coords_out = []
        feats_out = []
        for batch_size in split_num(num, batch_size):
            coords_start = torch.randn(batch_size, self.coords_dim, self.num_points, device=device)
            feats_start = torch.randn(batch_size, self.feats_dim, self.num_points, device=device)
            
            coords_ts, _, feats_ts, _ = self.diffusion_process.p_sample_loop_trajectory(denoise_fn=self.denoiser, coords_start=coords_start, feats_start=feats_start, 
                                                                                        coords_clip_range=(self.coords_normalization.min, self.coords_normalization.max),
                                                                                        feats_clip_range=(self.feats_normalization.min, self.feats_normalization.max),
                                                                                        progress=progress)
            coords = coords_ts[-1]
            feats = feats_ts[-1]
            
            coords = self.coords_normalization(coords)
            feats = self.feats_normalization(feats)
            
            coords = list(coords.unbind())
            feats = list(feats.unbind())
            coords_out += coords
            feats_out += feats
        return coords_out, feats_out
