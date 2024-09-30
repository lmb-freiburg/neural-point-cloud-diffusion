import os
import os.path as osp
import math
from typing import Optional
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd

from npcd.utils import chunks, writer, logging, to_torch, get_torch_model_device, unflatten_pred, FIDKID
from npcd.utils.vis import vis


class DiffusionEvaluation:
    def __init__(self,
                 num_samples: int,
                 poses_path: str,
                 intrinsics_path: str,
                 inception_pkl_path: str,
                 out_dir: Optional[str] = None,
                 generate_batch_size: int = 16,
                 render_batch_size: int = 1,
                 inception_batch_size: int = 32,
                 verbose: bool = True
                 ):

        self.verbose = verbose

        if self.verbose:
            logging.info(f"Initializing evaluation {self.name}.")

        self.out_dir = out_dir
        if self.out_dir is not None:
            self.quantitatives_dir = osp.join(self.out_dir)
            self.qualitatives_dir = osp.join(self.out_dir, "qualitative")
            self.results_file = osp.join(self.quantitatives_dir, "results.pickle")
            self.log_file_path = osp.join(self.out_dir, "log.txt")
            os.makedirs(self.out_dir, exist_ok=True)
            os.makedirs(self.quantitatives_dir, exist_ok=True)
            os.makedirs(self.qualitatives_dir, exist_ok=True)
            logging.add_log_file(self.log_file_path, flush_line=True)
        else:
            self.quantitatives_dir = None
            self.qualitatives_dir = None
            self.results_file = None
            self.log_file_path = None

        self.num_samples = num_samples
        self.poses = np.load(poses_path)
        self.intrinsics = np.load(intrinsics_path)
        self.num_images = num_samples * len(self.poses)
        self.inception_pkl_path = inception_pkl_path
        
        self.generate_batch_size = generate_batch_size
        self.render_batch_size = render_batch_size
        self.inception_batch_size = inception_batch_size

        # will be set/used in __call__:
        self.model = None
        self.fidkid = None
        self.eval_name = None
        self.finished_iterations = None
        self.sample_indices = None
        self.qualitative_indices = None
        self.cur_sample_idx = 0
        self.results = None

        if self.verbose:
            logging.info(self)
            logging.info(f"Finished initializing evaluation {self.name}.")
            logging.info()
            
    def __del__(self):
        if self.log_file_path is not None and logging is not None:
            logging.remove_log_file(self.log_file_path)

    @property
    def name(self):
        return type(self).__name__

    def __str__(self):
        ret = f"{self.name} with settings:"
        ret += f"\n\tNumber of samples: {self.num_samples}"
        ret += f"\n\tNumber of poses: {len(self.poses)}"
        ret += f"\n\tInception pickle path: {self.inception_pkl_path}"
        ret += f"\n\tGenerate batch size: {self.generate_batch_size}"
        ret += f"\n\tRender batch size: {self.render_batch_size}"
        ret += f"\n\tInception batch size: {self.inception_batch_size}"
        if self.out_dir is not None:
            ret += f"\n\tOutput directory: {self.out_dir}"
        else:
            ret += "\n\tOutput directory: None. Results will not be written to disk!"
        return ret

    @torch.no_grad()
    def __call__(self,
                 model,
                 num_qualitatives: int = 10,
                 eval_name: Optional[str] = None,
                 finished_iterations: Optional[int] = None,
                 **_):
        if self.results_file is not None and osp.exists(self.results_file):
            logging.info(f"Skipping evaluation {self.name} because it is already finished.")
            results = pd.read_pickle(self.results_file)
            return results
        
        self._init_evaluation(model=model, num_qualitatives=num_qualitatives,
                              eval_name=eval_name, finished_iterations=finished_iterations)
        self._evaluate()
        self._output_results()
        self._reset_evaluation()

    def _init_evaluation(self,
                         model,
                         num_qualitatives=10,
                         eval_name=None,
                         finished_iterations=None,):
        self.model = model
        self.model.eval()

        self.fidkid = FIDKID(num_images=self.num_images,
                             inception_pkl=self.inception_pkl_path,
                             inception_args={'type': 'StyleGAN', 'inception_path': 'data/inception-2015-12-05.pt'},
                             bgr2rgb=False)
        self.fidkid.prepare()

        self.eval_name = eval_name
        self.finished_iterations = finished_iterations
        self._init_qualitative_indices(num_qualitatives=num_qualitatives)

    def _init_qualitative_indices(self, num_qualitatives=None):
        sample_indices = list(range(self.num_samples))
        if num_qualitatives is None:
            self.qualitative_indices = []
        else:
            if num_qualitatives < 0:
                self.qualitative_indices = sample_indices
            elif num_qualitatives == 0:
                self.qualitative_indices = []
            else:
                step_size = len(sample_indices) / num_qualitatives  # <=1
                self.qualitative_indices = list(
                    set([sample_indices[int(i * step_size)] for i in range(num_qualitatives)]))

    def _evaluate(self):
        num_chunks = math.ceil(self.num_samples / self.generate_batch_size)
        
        for sample_indices in tqdm(chunks(list(range(self.num_samples)), self.generate_batch_size), total=num_chunks):

            num_to_generate = len(sample_indices)
            coords_batch, feats_batch = self.model.diffusion.generate(num=num_to_generate, batch_size=num_to_generate, progress=False)

            for sample_idx, coords, feats in zip(sample_indices, coords_batch, feats_batch):
                self.cur_sample_idx = sample_idx
                should_qualitative = (self.cur_sample_idx in self.qualitative_indices) and (self.out_dir is not None)
                
                coords = coords.permute(1, 0)[None].contiguous()  # format required by PointNeRF
                feats = feats.permute(1, 0)[None].contiguous()  # format required by PointNeRF
                
                all_images = []

                for pose_indices in chunks(list(range(len(self.poses))), self.render_batch_size):
                    pose_batch = self.poses[pose_indices]
                    pose_batch = to_torch(pose_batch, device=get_torch_model_device(self.model.pointnerf)).float()[None]
                    intrinsics_batch = self.intrinsics[pose_indices]
                    intrinsics_batch = to_torch(intrinsics_batch, device=get_torch_model_device(self.model.pointnerf)).float()[None]
                    
                    output = self.model.pointnerf.render(coords, feats, pose_batch, intrinsics_batch, resolution=128)
                    images = unflatten_pred(output.channels)[0].cpu().numpy()
                    
                    images = np.clip(images, 0, 1.)
                    images = np.round(images * 255) / 255
                    images = [image for image in images]
                    all_images += images

                all_images = np.stack(all_images)  # (num_poses, 3, H, W)
                all_images = torch.from_numpy(all_images).float()
                all_images_chunks = torch.split(all_images, self.inception_batch_size)
                for all_images_chunk in all_images_chunks:
                    self.fidkid.feed(all_images_chunk * 2 - 1, 'fakes')

                if should_qualitative:
                    self._log_qualitatives(all_images)

        self.fidkid.summary()
        self.results = pd.Series(self.fidkid._result_dict, name="metric")
        return self.results
    
    def _reset_evaluation(self):
        self.model = None
        self.fidkid = None
        self.eval_name = None
        self.finished_iterations = None
        self.sample_indices = None
        self.qualitative_indices = None
        self.cur_sample_idx = 0
        self.results = None

    def _log_qualitatives(self, images):
        for image_idx, image in enumerate(images):
            out_path = osp.join(self.qualitatives_dir, f'sample{self.cur_sample_idx:04d}-image{image_idx:04d}.png')
            vis(image, image_range_text_off=True, clipping=True, lower_clipping_thresh=0, upper_clipping_thresh=1).save(out_path)

    def _output_results(self):
        results = self.results
        
        # Print results:
        if self.verbose:
            logging.info()
            logging.info("Results:")
            logging.info(results)
            
        # Log results:
        log_name = f"eval/diffusion/unconditional_generation"
        log_name = log_name + f"/{self.eval_name}" if self.eval_name is not None else log_name
        writer.put_scalar_dict(name=log_name, scalar=results.to_dict(), step=self.finished_iterations)
        writer.write_out_storage()

        # Write results to disk:
        if self.out_dir is not None:
            if self.verbose:
                logging.info(f"Writing results to {self.out_dir}.")
            
            results.to_pickle(self.results_file)
            results.to_csv(osp.join(self.quantitatives_dir, "results.csv"))
