from easydict import EasyDict as edict
import torch
import torch.nn as nn

from torch_knnquery import VoxelGrid

from . import embeddings, fields, renderers


class PointNeRF(nn.Module):
    def __init__(self, n_obj: int, feats_dim: int, num_points: int, use_view_dir: bool):
        super().__init__()
        
        opt = _get_pointnerf_options()
        opt.model.field.kwargs.use_dir = use_view_dir
        opt.model.kp.feat_dim = feats_dim
        opt.model.kp.num = num_points
        self.opt = opt
        
        self.voxel_grid = VoxelGrid(**opt.model.voxel_grid)
        
        self.feats = getattr(embeddings, opt.model.embedding.type)(opt.model.kp.num, opt.model.kp.feat_dim, n_obj, **opt.model.embedding.kwargs)
        self.coords = embeddings.Embedding(opt.model.kp.num, 3, n_obj, **opt.model.embedding.kwargs)
        self.coords.freeze(True)
        
        kp_feat_dim = opt.model.kp.feat_dim
        self.field = getattr(fields, opt.model.field.network)(kp_feat_dim, self.voxel_grid, opt.model.field.aggregator, **opt.model.field.kwargs, nerf=opt.model.field.nerf)
        self.renderer = getattr(renderers, opt.model.renderer.network)(self.field, **opt.model.renderer.kwargs)            
    
    def train(self, mode=True):
        super().train(mode)
        self.renderer.randomize_depth_samples = mode
        return self
    
    @torch.no_grad()
    def set_all_coords(self, coords):
        coords = coords.reshape(coords.shape[0], -1)
        self.coords.get_emb().weight.copy_(coords)
        # note: this does not overwrite the grad of the coords embedding
        # due to the torch.no_grad() context manager
        
    def get_all_coords(self):
        coords = self.coords.get_emb().weight
        coords = coords.reshape(coords.shape[0], self.opt.model.kp.num, 3)
        return coords
    
    def get_all_feats(self):
        feats = self.feats.get_emb().weight
        if self.opt.model.embedding.type == "VariationalEmbedding":
            feats = feats.reshape(feats.shape[0], self.opt.model.kp.num, 2*self.opt.model.kp.feat_dim)
            feats = feats[:, :, :self.opt.model.kp.feat_dim]
        else:
            feats = feats.reshape(feats.shape[0], self.opt.model.kp.num, self.opt.model.kp.feat_dim)
        return feats

    def forward(self,
                obj_idx: torch.Tensor,
                intrinsics: torch.Tensor,
                extrinsics: torch.Tensor, 
                sample_rays: bool):      
        B = len(obj_idx)
        
        feats = self.feats(idx=obj_idx)  # [B, num_points, feat_dim]
        coords = self.coords(idx=obj_idx)  # [B, num_points, 3]
        num_points = self.opt.model.kp.num

        self.voxel_grid.set_pointset(
            coords.detach(), 
            torch.full(
                (B,), 
                fill_value=num_points, 
                device=coords.device, 
                dtype=torch.int
            )
        )
        
        if hasattr(self.feats, "get_mean_log_var_std"):
            feats_mean, feats_log_var, feats_std = self.feats.get_mean_log_var_std(idx=obj_idx)
            aux = {
                "coords": coords,
                "feats": feats_mean,
                "feats_mean": feats_mean,
                "feats_log_var": feats_log_var,
                "feats_std": feats_std
            }
        else:
            aux = {"coords": coords, "feats": feats}
            
        pred = self.renderer(
            coords,  # [B, num_points, 3]
            feats,  # [B, num_points, feat_dim]
            extrinsics,  # [B, num_views, 4, 4]
            intrinsics,  # [B, num_views, 3, 3]
            resolution=self.opt.sizes.default_resolution,  
            sample=sample_rays, 
            return_channels=True
        )
        
        # pred is a dict with keys: 'mask', 'depth', 'channels', 'ray_idx'
        # pred['mask'] is [B, num_views, num_rays, 1]
        # pred['depth'] is [B, num_views, num_rays, 1]
        # pred['channels'] contains the rendered colors and is [B, num_views, num_rays, 3]
        # pred['ray_idx'] is [B, num_views, num_rays, 1]
        
        return pred, aux
    
    def render(self, coords, feats, extrinsics, intrinsics, resolution=128, max_shading_points=None, sample_rays=False):
        
        B = len(coords)
        num_points = self.opt.model.kp.num
        
        if max_shading_points is not None:
            prev_max_shading_points = self.field.aggregator.max_shading_pts
            self.field.aggregator.max_shading_pts = max_shading_points

        self.voxel_grid.set_pointset(
            coords.detach(), 
            torch.full(
                (B,), 
                fill_value=num_points, 
                device=coords.device, 
                dtype=torch.int
            )
        )
        
        pred = self.renderer(coords, feats, extrinsics, intrinsics, resolution, sample_rays)
        
        if max_shading_points is not None:
            self.field.aggregator.max_shading_pts = prev_max_shading_points
        
        return pred


def _get_pointnerf_options():
    opt = edict()
    opt.model = edict()
    
    opt.model.kp = edict()
    opt.model.kp.num = 512  # num_points
    opt.model.kp.feat_dim = 32
    
    opt.model.embedding = edict()
    opt.model.embedding.type = "VariationalEmbedding"
    opt.model.embedding.kwargs = edict()
    opt.model.embedding.kwargs.gpu = True
    
    opt.model.voxel_grid = edict()
    opt.model.voxel_grid.voxel_size = (0.04, 0.04, 0.04)
    opt.model.voxel_grid.voxel_scale = (2, 2, 2)
    opt.model.voxel_grid.kernel_size = (3, 3, 3)
    opt.model.voxel_grid.max_points_per_voxel = 4
    opt.model.voxel_grid.max_occ_voxels_per_example = 5000
    opt.model.voxel_grid.ranges = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    
    opt.model.field = edict()
    opt.model.field.network = "MLP"
    opt.model.field.nerf = True
    opt.model.field.kwargs = edict()
    opt.model.field.kwargs.feat_freqs = 0
    opt.model.field.kwargs.dir_freqs = 8
    opt.model.field.kwargs.channel_layers = [256, 256, 256, 256]
    opt.model.field.kwargs.shape_layers = [256]
    opt.model.field.kwargs.activation = "LeakyReLU"
    opt.model.field.kwargs.layer_norm = False
    opt.model.field.kwargs.use_dir = False
    
    opt.model.field.aggregator = edict()
    opt.model.field.aggregator.network = "MLP"
    opt.model.field.aggregator.kwargs = edict()
    opt.model.field.aggregator.kwargs.k = 8
    opt.model.field.aggregator.kwargs.r = 2
    opt.model.field.aggregator.kwargs.max_shading_pts = 50
    opt.model.field.aggregator.kwargs.ray_subsamples = 128
    opt.model.field.aggregator.kwargs.n_freqs = 10
    opt.model.field.aggregator.kwargs.freq_mult = 1
    opt.model.field.aggregator.kwargs.out_dim = 256
    opt.model.field.aggregator.kwargs.layers = [256, 256, 256, 256]
    opt.model.field.aggregator.kwargs.activation = "LeakyReLU"
    opt.model.field.aggregator.kwargs.layer_norm = False
    
    opt.model.renderer = edict()
    opt.model.renderer.network = "VolumeRenderer"
    opt.model.renderer.kwargs = edict()
    opt.model.renderer.kwargs.depth_resolution = 128
    opt.model.renderer.kwargs.disparity_space_sampling = False
    opt.model.renderer.kwargs.white_back = True
    opt.model.renderer.kwargs.cube_scale = 1.0
    opt.model.renderer.kwargs.ray_subsamples = 112
    opt.model.renderer.kwargs.ray_limits = None
    
    opt.sizes = edict()
    opt.sizes.default_resolution = 128
    
    return opt
