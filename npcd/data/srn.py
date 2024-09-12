import os
import os.path as osp
from random import shuffle
import queue
import threading

import PIL
import numpy as np
import tqdm
import torchvision.transforms.functional as torchvision_F
import torch
from torch.utils.data import default_collate

from npcd.utils import chunks
from .dataset import Dataset, Sample
from .registry import register_dataset


class SRNSample(Sample):

    def __init__(self, category, obj_idx, obj_name, images, intrinsics, extrinsics, view_indices):
        self.category = category
        self.obj_idx = obj_idx
        self.obj_name = obj_name
        self.images = images
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.view_indices = view_indices

    def load(self, root):
                
        out_dict = {
            "obj_idx": self.obj_idx,
            "obj_name": self.obj_name,
            "images": self.images,
            "intrinsics": self.intrinsics,
            "extrinsics": self.extrinsics,
            "view_indices": self.view_indices,
        }

        return out_dict


class SRNTrain(Dataset):
    def __init__(self, root, sample_list, views_per_sample=50, image_size=128, num_points=512, **kwargs):
        super().__init__(root=root, sample_list=sample_list, views_per_sample=views_per_sample, image_size=image_size, num_points=num_points, **kwargs)
        
    def _init_samples(self, sample_list, views_per_sample=50, image_size=128, num_points=512):
        view_indices = list(range(50))  # training samples in SRN always have 50 views
        
        # Populate object_views dictionary and views set:
        # object_views maps for each SRN object (indexed by: category, model_id, running_index) to a list of available view indices
        # views is a set of all available views from all objects
        self.object_views = {}
        self.views = set()
        for c, m, i in sample_list:
            self.object_views[c, m, i] = []
            for view in view_indices:
                self.object_views[c, m, i].append(view)
                self.views.add((c, m, i, view))

        # Construct samples for training:
        # For now each sample is constructed only in form of its indices (category, model_id, running_index, view_indices)
        # Construction works by simply splitting the views per object into chunks of size views_per_sample
        sample_indices = []
        assert 50 % views_per_sample == 0
        for (c, m, i), views in self.object_views.items():
            shuffle(views)
            sample_indices += [(c, m, i, v) for v in chunks(views, views_per_sample)]
            
        # Preload images, cameras and point clouds
        self.image_size = image_size
        self.num_points = num_points
        self.pcs = self.preload_threading(self.get_pointcloud, sample_list, data_str="point clouds")
        self.images = dict(zip(self.views, self.preload_threading(self.get_image, self.views, data_str="images")))
        self.cameras = dict(zip(self.views, self.preload_threading(self.get_camera, self.views, data_str="cameras")))
        
        # Construct samples with actual data and not only indices:
        for sample_idx in sample_indices:
            c, m, i, vs = sample_idx
            views = default_collate([self.get_view(c, m, i, v) for v in vs])  # contains img, extr, intr, view_idx for each view in the sample
            images = views["img"]  # num_views, 3, image_size, image_size
            intrinsics = views["intr"]  # num_views, 3, 3
            extrinsics = views["extr"]  # num_views, 4, 4
            view_indices = views["view_idx"]  # num_views
            obj_idx = i  # int
            obj_name = m  # str
            
            sample = SRNSample(category=c,
                               obj_idx=obj_idx,
                               obj_name=obj_name,
                               images=images,
                               intrinsics=intrinsics,
                               extrinsics=extrinsics,
                               view_indices=view_indices,)
            
            self.samples.append(sample)

    def preload_worker(self, data_list, load_func, q, lock, idx_tqdm):
        while True:
            i, idx = q.get()
            data_list[i] = load_func(idx)
            with lock:
                idx_tqdm.update()
            q.task_done()

    def preload_threading(self, load_func, idx_list, data_str="images"):
        data_list = [None]*len(idx_list)
        q = queue.Queue(maxsize=len(idx_list))
        idx_tqdm = tqdm.tqdm(range(len(idx_list)), desc="preloading {}".format(data_str), leave=False)
        for el in enumerate(idx_list):
            q.put(el)
        lock = threading.Lock()
        for ti in range(4):
            t = threading.Thread(target=self.preload_worker, args=(data_list, load_func, q, lock, idx_tqdm), daemon=True)
            t.start()
        q.join()
        idx_tqdm.close()
        assert(all(map(lambda x: x is not None, data_list)))
        return data_list

    def get_image(self, idx) -> dict:
        c, m, _, v = idx
        path = "{0}/{1}/{2}".format(self.root, c, m)
        return self.load_image(path, v, self.image_size)

    @staticmethod
    def load_image(path, fname, image_size):
        image_fname = "{0}/rgb/{1:06d}.png".format(path, fname)
        image = PIL.Image.open(image_fname).convert("RGB")
        image = image.resize((image_size, image_size))
        image = torchvision_F.to_tensor(image)
        return image
    
    def get_camera(self, idx) -> dict:
        c, m, _, v = idx
        path = "{0}/{1}/{2}".format(self.root, c, m)
        return self.load_camera(path, v, self.image_size)

    @staticmethod
    def load_camera(path, fname, image_size):
        pose_fname = "{0}/pose/{1:06d}.txt".format(path, fname)
        pose = np.loadtxt(pose_fname)
        cam2world = torch.from_numpy(pose).float().view(4, 4)
        # Invert pose (world2cam expected)
        world2cam = cam2world.clone()
        world2cam[:3, :3] = cam2world[:3, :3].transpose(-1, -2)
        world2cam[:3, 3:] = - torch.matmul(world2cam[:3, :3], cam2world[:3, 3:])
        intr = SRNTrain.parse_intrinsics(image_size, "{0}/intrinsics.txt".format(path))
        return world2cam, intr

    @staticmethod
    def parse_intrinsics(image_size, file_path):
        with open(file_path, 'r') as file:
            f, cx, cy, _ = map(float, file.readline().split())
            next(file)
            next(file)
            height, width = map(float, file.readline().split())
        assert height == width, f"Found non-square camera intrinsics in {file_path}"
        cx = cx / width * image_size
        cy = cy / height * image_size
        f = f / height * image_size
        return torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    def get_pointcloud(self, idx) -> dict:
        c, m, _ = idx
        path = "{0}/{1}/{2}".format(self.root, c, m)
        return self.load_pointcloud(path)

    def load_pointcloud(self, path):
        k_pc_fname = f"{path}/pointcloud3_{self.num_points}.npz"
        if os.path.isfile(k_pc_fname):
            npz = np.load(k_pc_fname)
            dpc = dict(
                points=torch.from_numpy(npz["points"]).float(),
                normals=torch.from_numpy(npz["normals"]).float(),
            )
        else:
            from pytorch3d.ops import sample_farthest_points
            pc_fname = "{0}/pointcloud3.npz".format(path)
            npz = np.load(pc_fname)
            points=torch.from_numpy(npz["points"]).float()
            normals=torch.from_numpy(npz["normals"]).float()
            points, idx = sample_farthest_points(points.unsqueeze(0), K=self.num_points)
            points = points.squeeze(0)
            idx = idx.squeeze(0)
            normals = normals[idx]
            np.savez(k_pc_fname, points=points.numpy(), normals=normals.numpy())
            dpc = dict(
                points=points,
                normals=normals
            )
        return dpc

    def get_view(self, c, m, i, v):
        view = {}
        path = "{0}/{1}/{2}".format(self.root, c, m)
        view["img"] = self.images[c, m, i, v] if hasattr(self, "images") else self.load_image(path, v, self.image_size)
        view["extr"], view["intr"] = self.cameras[c, m, i, v] if hasattr(self, "cameras") else self.load_camera(path, v, self.image_size)
        view["view_idx"] = v
        return view
            
    def get_all_coords(self):
        coords = [self.pcs[i]["points"] for i in range(len(self.pcs))]
        coords = torch.stack(coords, dim=0)  # num_samples, num_points, 3
        return coords


@register_dataset
class SRNCarsTrain(SRNTrain):

    base_dataset = 'srn'
    split = 'cars_train'
    
    def __init__(self, root=None, **kwargs):
        root = root if root is not None else self._get_path("srn", "root")
    
        sample_list = []
        sample_lists_path = osp.join(osp.dirname(osp.realpath(__file__)), 'sample_lists')
        split_path = osp.join(sample_lists_path, 'srn_cars_train.list')
        blacklist_path = osp.join(sample_lists_path, 'srn_cars_blacklist.list')
        blacklist = set(open(blacklist_path).read().splitlines())
        i = 0
        for shapenet_id in open(split_path).read().splitlines():
            if shapenet_id not in blacklist:
                sample_list.append(("cars", shapenet_id, i))
                i += 1
                
        super().__init__(root=root, sample_list=sample_list, **kwargs)
