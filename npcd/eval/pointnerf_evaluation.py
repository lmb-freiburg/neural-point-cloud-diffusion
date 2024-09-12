import os
import os.path as osp
import time
from copy import deepcopy
from typing import Optional, Sequence, Union

import torch
import skimage.metrics
import numpy as np
import pandas as pd

from npcd.utils import to_torch, get_torch_model_device, chunks, unflatten_pred, writer, logging
from npcd.utils.vis import vis


class PointNeRFEvaluation:
    def __init__(self,
                 out_dir: Optional[str] = None,
                 eval_batch_size: int = 8,
                 verbose: bool = True
                 ):

        self.verbose = verbose

        if self.verbose:
            logging.info(f"Initializing evaluation {self.name}.")

        self.out_dir = out_dir
        if self.out_dir is not None:
            self.quantitatives_dir = osp.join(self.out_dir)
            self.sample_results_dir = osp.join(self.quantitatives_dir, "per_sample")
            self.qualitatives_dir = osp.join(self.out_dir, "qualitative")
            self.results_file = osp.join(self.quantitatives_dir, ".results_df.pickle")
            self.log_file_path = osp.join(self.out_dir, "log.txt")
            os.makedirs(self.out_dir, exist_ok=True)
            os.makedirs(self.quantitatives_dir, exist_ok=True)
            os.makedirs(self.sample_results_dir, exist_ok=True)
            os.makedirs(self.qualitatives_dir, exist_ok=True)
            logging.add_log_file(self.log_file_path, flush_line=True)
        else:
            self.quantitatives_dir = None
            self.sample_results_dir = None
            self.qualitatives_dir = None
            self.results_file = None
            self.log_file_path = None
            
        self.eval_batch_size = eval_batch_size

        # will be set/used in __call__:
        self.dataset = None
        self.dataloader = None
        self.model = None
        self.eval_name = None
        self.finished_iterations = None
        self.sample_indices = None
        self.qualitative_indices = None
        self.burn_in_samples = None
        self.cur_sample_num = 0
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
        ret += f"\n\tEvaluation batch size: {self.eval_batch_size}"
        if self.out_dir is not None:
            ret += f"\n\tOutput directory: {self.out_dir}"
        else:
            ret += "\n\tOutput directory: None. Results will not be written to disk!"
        return ret

    @torch.no_grad()
    def __call__(self,
                 dataset,
                 model,
                 samples: Optional[Union[int, Sequence[int]]] = None,
                 qualitatives: Union[int, Sequence[int]] = 10,
                 burn_in_samples: int = 3,
                 eval_name: Optional[str] = None,
                 finished_iterations: Optional[int] = None,
                 **_):
        if self.results_file is not None and osp.exists(self.results_file):
            logging.info(f"Skipping evaluation {self.name} because it is already finished.")
            results = pd.read_pickle(self.results_file)
            return results
        
        self._init_evaluation(dataset=dataset, model=model, samples=samples, qualitatives=qualitatives,
                              burn_in_samples=burn_in_samples, eval_name=eval_name, finished_iterations=finished_iterations)
        self._evaluate()
        self._output_results()
        self._reset_evaluation()

    def _init_evaluation(self,
                         dataset,
                         model,
                         samples=None,
                         qualitatives=10,
                         burn_in_samples=3,
                         eval_name=None,
                         finished_iterations=None,):
        self.dataset = dataset
        self.model = model
        self.model.eval()
        self.eval_name = eval_name
        self.finished_iterations = finished_iterations
        self._init_sample_indices(samples=samples)
        self._init_qualitative_indices(qualitatives=qualitatives)
        self._init_results()
        self.burn_in_samples = burn_in_samples
        self.dataloader = self.dataset.get_loader(batch_size=1, indices=self.sample_indices, num_workers=8)

    def _init_sample_indices(self, samples):
        if isinstance(samples, list):
            self.sample_indices = samples
            if self.verbose:
                logging.info(f"Evaluating samples with indices: {self.sample_indices}.")
        elif isinstance(samples, int) and samples > 0:
            step_size = len(self.dataset) / samples  # <=1
            self.sample_indices = [int(i*step_size) for i in range(samples)]
            if self.verbose:
                logging.info(f"Evaluating samples with indices: {self.sample_indices}.")
        else:
            self.sample_indices = list(range(len(self.dataset)))

    def _init_qualitative_indices(self, qualitatives=None):
        if qualitatives is None:
            self.qualitative_indices = []
        elif isinstance(qualitatives, list):
            self.qualitative_indices = qualitatives
        elif isinstance(qualitatives, int):
            if qualitatives < 0:
                self.qualitative_indices = self.sample_indices
            elif qualitatives == 0:
                self.qualitative_indices = []
            else:
                step_size = len(self.sample_indices) / qualitatives  # <=1
                self.qualitative_indices = list(
                    set([self.sample_indices[int(i * step_size)] for i in range(qualitatives)]))

    def _evaluate(self):
        for sample_num, (sample_idx, sample) in enumerate(zip(self.sample_indices, self.dataloader)):
            self.cur_sample_num = sample_num
            self.cur_sample_idx = sample_idx

            if self.verbose:
                logging.info(f"Processing sample {self.cur_sample_num+1} / {len(self.sample_indices)} "
                             f"(index: {self.cur_sample_idx})")

            should_qualitative = (self.cur_sample_idx in self.qualitative_indices) and (self.out_dir is not None)

            sample = to_torch(sample, device=get_torch_model_device(self.model))
            
            num_views = sample['view_indices'].shape[1]
            for batch_indices in chunks(list(range(num_views)), self.eval_batch_size):
                batch = deepcopy(sample)
                batch['images'] = batch['images'][:, batch_indices]
                batch['extrinsics'] = batch['extrinsics'][:, batch_indices]
                batch['intrinsics'] = batch['intrinsics'][:, batch_indices]
                batch['view_indices'] = batch['view_indices'][:, batch_indices]
                self._reset_memory_stats()

                # run model:
                pred, runtimes, gpu_mem = self._run_model(batch)

                # compute and log metrics:
                metrics = self._compute_metrics(sample=batch, pred=pred)
                for metrics_per_view in metrics.values():
                    metrics_per_view.update(runtimes)
                    metrics_per_view.update(gpu_mem)
                self._log_metrics(metrics)

                # compute and log qualitatives:
                if should_qualitative:
                    qualitatives = self._compute_qualitatives(sample=batch, pred=pred)
                    self._log_qualitatives(qualitatives)

        return self.results
    
    def _reset_evaluation(self):
        self.dataset = None
        self.dataloader = None
        self.model = None
        self.eval_name = None
        self.finished_iterations = None
        self.sample_indices = None
        self.qualitative_indices = None
        self.burn_in_samples = None
        self.cur_sample_idx = 0
        self.cur_sample_num = 0
        self.results = None

    def _init_results(self):
        self.results = pd.DataFrame()
        self.results.index.name = 'view_idx'
        self.results.columns.name = 'metric'
        self.results = pd.concat({0: self.results}, axis=0, names=["sample_idx"])

    def _reset_memory_stats(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

    def _run_model(self, sample):
        sample_inputs, _ = self._inputs_and_gt_from_sample(sample)
        torch.cuda.synchronize()
        start_model = time.time()
        pred, _ = self.model.pointnerf(**sample_inputs, sample_rays=False)
        torch.cuda.synchronize()
        end_model = time.time()

        # only allow time and memory measurements if batch size is 1:
        time_and_mem_valid = (self.eval_batch_size == 1) and (self.cur_sample_num >= self.burn_in_samples)
        
        runtime_model_in_sec = end_model - start_model if time_and_mem_valid else np.nan
        runtimes = {'runtime_model_in_sec': runtime_model_in_sec,
                    'runtime_model_in_msec': 1000 * runtime_model_in_sec,}

        gpu_mem_alloc = int(torch.cuda.max_memory_allocated() / 1024 / 1024) if time_and_mem_valid else np.nan
        gpu_mem_reserved = int(torch.cuda.max_memory_reserved() / 1024 / 1024) if time_and_mem_valid else np.nan
        gpu_mem = {'gpu_mem_alloc_in_mib': gpu_mem_alloc, 'gpu_mem_alloc_in_mib': gpu_mem_reserved}

        return pred, runtimes, gpu_mem
        
    def _inputs_and_gt_from_sample(self, sample):
        input_keys = ["obj_idx", "intrinsics", "extrinsics"]
        sample_inputs = {key: val for key, val in sample.items() if key in input_keys}
        sample_gt = {key: val for key, val in sample.items() if key not in input_keys}
        return sample_inputs, sample_gt

    def _compute_metrics(self, sample, pred):
        metrics = {}
        
        gt_imgs = sample['images'][0].cpu().numpy()  # num_views, 3, H, W
        
        pred_imgs = unflatten_pred(pred.channels.contiguous()[0])  # num_views, 3, H, W
        pred_imgs = pred_imgs.cpu().numpy()
        
        view_indices = sample['view_indices'][0].cpu().numpy()
        
        for gt_img, pred_img, view_idx in zip(gt_imgs, pred_imgs, view_indices):
            assert gt_img.shape == pred_img.shape
            psnr = float(skimage.metrics.peak_signal_noise_ratio(pred_img, gt_img, data_range=1))
            metrics[view_idx] = {'psnr': psnr}

        return metrics

    def _log_metrics(self, metrics):
        for view_idx, metrics_per_view in metrics.items():
            for metric, val in metrics_per_view.items():
                self.results.loc[(self.cur_sample_idx, view_idx), metric] = val
                
        self.results.sort_index(inplace=True)

    def _compute_qualitatives(self, sample, pred):
        qualitatives = {}
        
        gt_imgs = sample['images'][0].cpu().numpy()  # num_views, 3, H, W
        
        pred_imgs = unflatten_pred(pred.channels.contiguous()[0])  # num_views, 3, H, W
        pred_imgs = pred_imgs.cpu().numpy()
        
        view_indices = sample['view_indices'][0].cpu().numpy()
        
        for gt_img, pred_img, view_idx in zip(gt_imgs, pred_imgs, view_indices):
            qualitatives[view_idx] = {'gt_image': gt_img, 'pred_image': pred_img}

        return qualitatives

    def _log_qualitatives(self, qualitatives):
        for view_idx, qualitatives_per_view in qualitatives.items():
            for qualitative_name, qualitative in qualitatives_per_view.items():
                out_path = osp.join(self.qualitatives_dir, f'{self.cur_sample_idx:07d}-{view_idx:07d}-{qualitative_name}.png')
                vis(qualitative, image_range_text_off=True, clipping=True, lower_clipping_thresh=0, upper_clipping_thresh=1).save(out_path)

    def _output_results(self):
        results = self.results.mean()
        
        # Print results:
        if self.verbose:
            logging.info()
            logging.info("Results:")
            logging.info(results)
            
        # Log results:
        log_name = f"eval/pointnerf/{self.dataset.name}"
        log_name = log_name + f"/{self.eval_name}" if self.eval_name is not None else log_name
        writer.put_scalar_dict(name=log_name, scalar=results.to_dict(), step=self.finished_iterations)
        writer.write_out_storage()

        # Write results to disk:
        if self.out_dir is not None:
            if self.verbose:
                logging.info(f"Writing results to {self.out_dir}.")

            self.results.to_pickle(osp.join(self.sample_results_dir, "results.pickle"))
            self.results.to_csv(osp.join(self.sample_results_dir, "results.csv"))
            
            results.to_pickle(osp.join(self.quantitatives_dir, "results.pickle"))
            results.to_csv(osp.join(self.quantitatives_dir, "results.csv"))
