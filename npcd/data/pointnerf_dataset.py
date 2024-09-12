import torch
import numpy as np

from .dataset import Dataset, Sample
    
    
class PointNeRFSample(Sample):
    def __init__(self, coords, feats):
        self.coords = coords
        self.feats = feats

    def load(self, root):
        out_dict = {
            "coords": self.coords,  # [coords_dim, num_points]
            "feats": self.feats,  # [feats_dim, num_points]
        }

        return out_dict
    
    
class PointNeRFDataset(Dataset):
    def __init__(self, pointnerf, **kwargs):
        super().__init__(root=None, pointnerf=pointnerf, **kwargs)
        
    def _init_samples(self, pointnerf):
        all_coords = pointnerf.get_all_coords().detach().cpu().numpy().transpose(0, 2, 1)  # [num_objects, coords_dim, num_points]
        all_feats = pointnerf.get_all_feats().detach().cpu().numpy().transpose(0, 2, 1)  # [num_objects, feats_dim, num_points]
        for coords, feats in zip(all_coords, all_feats):
            self.samples.append(PointNeRFSample(coords, feats))
            
    def get_all_coords(self):
        all_coords = np.concatenate([sample.coords for sample in self.samples], axis=-1)  # [coords_dim, num_objects*num_points]
        return all_coords

    def get_all_feats(self):
        all_feats = np.concatenate([sample.feats for sample in self.samples], axis=-1)  # [feats_dim, num_objects*num_points]
        return all_feats
