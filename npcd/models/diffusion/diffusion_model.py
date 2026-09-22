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

    def normalize(self, x):
        return (x - self.shift[None, :, None]) / self.scale[None, :, None]

    def denormalize(self, x):
        return x * self.scale[None, :, None] + self.shift[None, :, None]

    def forward(self, x):
        if self.training:
            return self.normalize(x)
        else:
            return self.denormalize(x)


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

    def normalize(self, x):
        return (x - self.shift[None, :, None]) / self.scale[None, :, None]

    def denormalize(self, x):
        return x * self.scale[None, :, None] + self.shift[None, :, None]

    def forward(self, x):
        if self.training:
            return self.normalize(x)
        else:
            return self.denormalize(x)


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
        """Generate `num` objects, in batches of at most `batch_size`.

        Arguments:
            num: number of objects to generate
            batch_size: maximum number of objects that are generated in parallel
            progress: if True, show a progress bar for the reverse diffusion of each batch
        Return:
            coords: [num_objects, 3, num_points]
            feats: [num_objects, feats_dim, num_points]
        """
        assert not self.training, 'Model must be in eval mode for generation'

        device = get_torch_model_device(self)
        coords_out = []
        feats_out = []
        for cur_batch_size in split_num(num, batch_size):
            coords_start = torch.randn(cur_batch_size, self.coords_dim, self.num_points, device=device)
            feats_start = torch.randn(cur_batch_size, self.feats_dim, self.num_points, device=device)

            coords, feats = self.diffusion_process.p_sample_loop(denoise_fn=self.denoiser, coords_start=coords_start, feats_start=feats_start,
                                                                 coords_clip_range=(self.coords_normalization.min, self.coords_normalization.max),
                                                                 feats_clip_range=(self.feats_normalization.min, self.feats_normalization.max),
                                                                 progress=progress)

            coords = self.coords_normalization.denormalize(coords)
            feats = self.feats_normalization.denormalize(feats)

            coords_out.append(coords)
            feats_out.append(feats)
        return torch.cat(coords_out), torch.cat(feats_out)

    @torch.no_grad()
    def generate_trajectory(self, num, batch_size=8, progress=True):
        """Generate `num` objects, in batches of at most `batch_size` and return the full trajectory of the reverse diffusion process.

        Arguments:
            num: number of objects to generate
            batch_size: maximum number of objects that are generated in parallel
            progress: if True, show a progress bar for the reverse diffusion of each batch
        Return:
            coords: [num_objects, num_timesteps+1, 3, num_points]
            coords_recon: [num_objects, num_timesteps, 3, num_points]
            feats: [num_objects, num_timesteps+1, feats_dim, num_points]
            feats_recon: [num_objects, num_timesteps, feats_dim, num_points]
        """
        assert not self.training, 'Model must be in eval mode for generation'

        device = get_torch_model_device(self)
        coords_out = []
        feats_out = []
        coords_recon_out = []
        feats_recon_out = []
        for cur_batch_size in split_num(num, batch_size):
            coords_start = torch.randn(cur_batch_size, self.coords_dim, self.num_points, device=device)
            feats_start = torch.randn(cur_batch_size, self.feats_dim, self.num_points, device=device)

            coords_ts, coords_recon, feats_ts, feats_recon = \
                self.diffusion_process.p_sample_loop_trajectory(denoise_fn=self.denoiser, coords_start=coords_start, feats_start=feats_start,
                                                                coords_clip_range=(self.coords_normalization.min, self.coords_normalization.max),
                                                                feats_clip_range=(self.feats_normalization.min, self.feats_normalization.max),
                                                                progress=progress)

            coords = torch.stack([self.coords_normalization.denormalize(coords_t) for coords_t in coords_ts], dim=1)
            feats = torch.stack([self.feats_normalization.denormalize(feats_t) for feats_t in feats_ts], dim=1)
            coords_recon = torch.stack([self.coords_normalization.denormalize(coords_recon_t) for coords_recon_t in coords_recon], dim=1)
            feats_recon = torch.stack([self.feats_normalization.denormalize(feats_recon_t) for feats_recon_t in feats_recon], dim=1)

            coords_out.append(coords)
            feats_out.append(feats)
            coords_recon_out.append(coords_recon)
            feats_recon_out.append(feats_recon)
        return torch.cat(coords_out), torch.cat(coords_recon_out), torch.cat(feats_out), torch.cat(feats_recon_out)

    @torch.no_grad()
    def resample_appearance(self, coords_in, num, batch_size=8, N_rev=0, N_repaint=0, N_resample=0, progress=True):  # Algorithm 1 in the supplementary material of the paper
        # coords_in shape: [1, 3, num_points]
        assert not self.training, 'Model must be in eval mode for generation'
        assert coords_in.shape[0] == 1, 'Batch size of input coords must be 1 for generation from an existing object'

        coords_in = self.coords_normalization.normalize(coords_in)
        coords_out = []
        feats_out = []

        for current_batch_size in split_num(num, batch_size):
            coords_in_batch = torch.cat([coords_in] * current_batch_size)  # can not be done before for loop because last iteration might have smaller batch size in case num is not divisible by batch_size

            coords_ts, _, feats_ts, _ = \
                self.diffusion_process.p_sample_loop_from_existing_traj(denoise_fn=self.denoiser, 
                                                                        coords_0=coords_in_batch, feats_0=None,
                                                                        num_points=self.num_points, feats_dim=self.feats_dim,
                                                                        num_coord_noise_steps=N_rev, num_feat_noise_steps=1000,
                                                                        num_repaint_coords=0, num_repaint_feats=N_resample,
                                                                        num_repaint_steps_coords=0, num_repaint_steps_feats=N_repaint,
                                                                        resample_qsample_noise=False,
                                                                        coords_clip_range=(self.coords_normalization.min, self.coords_normalization.max),
                                                                        feats_clip_range=(self.feats_normalization.min, self.feats_normalization.max),
                                                                        progress=progress)
            coords = coords_ts[-1]
            feats = feats_ts[-1]
            coords = self.coords_normalization.denormalize(coords)
            feats = self.feats_normalization.denormalize(feats)
            coords_out.append(coords)
            feats_out.append(feats)
        return torch.cat(coords_out), torch.cat(feats_out)

    @torch.no_grad()
    def resample_shape(self, feats_in, num, batch_size=8, N_rev=0, N_repaint=0, N_resample=0, progress=True):
        # feats_in shape: [1, feats_dim, num_points]
        assert not self.training, 'Model must be in eval mode for generation'
        assert feats_in.shape[0] == 1, 'Batch size of input feats must be 1 for generation from an existing object'

        feats_in = self.feats_normalization.normalize(feats_in)
        coords_out = []
        feats_out = []

        for current_batch_size in split_num(num, batch_size):
            feats_in_batch = torch.cat([feats_in] * current_batch_size)  # can not be done before for loop because last iteration might have smaller batch size in case num is not divisible by batch_size

            coords_ts, _, feats_ts, _ = \
                self.diffusion_process.p_sample_loop_from_existing_traj(denoise_fn=self.denoiser,
                                                                        coords_0=None, feats_0=feats_in_batch,
                                                                        num_points=self.num_points, coords_dim=self.coords_dim,
                                                                        num_coord_noise_steps=1000, num_feat_noise_steps=N_rev,
                                                                        num_repaint_coords=N_resample, num_repaint_feats=0,
                                                                        num_repaint_steps_coords=N_repaint, num_repaint_steps_feats=0,
                                                                        resample_qsample_noise=False,
                                                                        coords_clip_range=(self.coords_normalization.min, self.coords_normalization.max),
                                                                        feats_clip_range=(self.feats_normalization.min, self.feats_normalization.max),
                                                                        progress=progress)
            coords = coords_ts[-1]
            feats = feats_ts[-1]
            coords = self.coords_normalization.denormalize(coords)
            feats = self.feats_normalization.denormalize(feats)
            coords_out.append(coords)
            feats_out.append(feats)
        return torch.cat(coords_out), torch.cat(feats_out)
