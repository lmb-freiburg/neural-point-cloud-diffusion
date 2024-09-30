import os
import os.path as osp
from typing import Optional
import time

import torch
import skimage.metrics

from npcd.utils import TrainStateSaver, WeightsOnlySaver, to_torch, get_torch_model_device, count_torch_model_parameters, unflatten_pred, writer, logging


class PointNeRFTraining:
    def __init__(self,
                 out_dir: str,
                 model: torch.nn.Module,
                 dataset: torch.utils.data.Dataset,
                 loss: torch.nn.Module,
                 batch_size: int,
                 base_learning_rate: float,
                 max_epochs: int,
                 grad_clip_max_norm: Optional[float] = None,
                 num_workers: Optional[int] = 8,
                 print_interval: Optional[int] = 100,
                 log_loss_interval: Optional[int] = 100,
                 log_interval: Optional[int] = 5000,
                 save_checkpoint_interval_min: Optional[int] = 20,
                 log_full_batch: Optional[bool] = False,
                 verbose: bool = True,):


        self.verbose = verbose

        self.out_dir = out_dir
        self._init_dirs()

        if self.verbose:
            logging.info(f"Initializing {self.name}.")

        self.model = model
        self.dataset = dataset
        self.loss = loss
        self.batch_size = batch_size
        self._setup_optimizer(base_learning_rate)
        self._setup_scheduler()
        self.grad_clip_max_norm = grad_clip_max_norm
        self.dataloader = self.dataset.get_loader(batch_size=batch_size, shuffle=True, pin_memory=True,
                                                  num_workers=num_workers, drop_last=True)

        self.max_epochs = max_epochs
        self.max_iterations = len(self.dataloader) * self.max_epochs
        writer.set_max_iterations(self.max_iterations)
        self.finished_iterations = 0
        self.saver_all = None
        self.saver_weights_only = None
        self._start_iteration = None
        self._setup_savers()
        self._restore_weights()
        self._start_iteration = self.finished_iterations

        self.log_full_batch = log_full_batch
        self.print_interval = print_interval
        self.log_interval = log_interval
        self.log_loss_interval = log_loss_interval
        self.save_checkpoint_interval_min = save_checkpoint_interval_min

        if self.verbose:
            logging.info(self)
            logging.info(f"Finished initializing {self.name}.")
            logging.info()

    @property
    def name(self):
        return type(self).__name__

    def __str__(self):
        ret = f"{self.name} with settings:"
        ret += f"\n\tOutput directory: {self.out_dir}"
        ret += f"\n\tModel parameter count: {count_torch_model_parameters(self.model)}"
        ret += f"\n\tDataset: {self.dataset.name}"
        ret += f"\n\tDataset size: {len(self.dataset)}"
        ret += f"\n\tGrad clip max norm: {self.grad_clip_max_norm}"
        ret += f"\n\tBatch size: {self.batch_size}"
        ret += f"\n\tFinished iterations: {self.finished_iterations}"
        ret += f"\n\tMax epochs: {self.max_epochs}"
        ret += f"\n\tMax iterations: {self.max_iterations}"
        
        return ret

    def _init_dirs(self):
        self.artifacts_dir = osp.join(self.out_dir, "artifacts")
        self.checkpoints_dir = osp.join(self.out_dir, "checkpoints")
        self.weights_only_checkpoints_dir = osp.join(self.out_dir, "weights_only_checkpoints_dir")
        self.checkpoints_name = "pointnerf_training"
        self.weights_only_checkpoints_name = "pointnerf"

        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.weights_only_checkpoints_dir, exist_ok=True)
        
    def _setup_optimizer(self, base_learning_rate):
        self.optimizer = torch.optim.Adam(self.model.pointnerf.parameters(), lr=base_learning_rate)
        
    def _setup_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1.0)  # Constant learning rate

    def __call__(self):
        """Run training."""

        should_continue = lambda: self.finished_iterations < self.max_iterations

        if not should_continue():
            logging.info("Training already finished.")
            return

        if self.verbose:
            logging.info(f"Starting training {self.name}.")

        self.model.pointnerf.set_all_coords(self.dataset.get_all_coords())
        self.model.train()

        should_print = lambda: self.finished_iterations % self.print_interval == 0
        should_log = lambda: self.finished_iterations % self.log_interval == 0
        should_log_loss = lambda: self.finished_iterations % self.log_loss_interval == 0

        steps_since_print = 0
        start_print_interval = time.time()
        last_checkpoint_time = time.time()

        while should_continue():
            for iter_in_epoch, sample in enumerate(self.dataloader):
                with writer.TimeWriter(name="00_overview/train_sec_per_iter", step=self.finished_iterations, write=should_log_loss(), avg_over_steps=True, update_eta=True):
                    self.optimizer.zero_grad()

                    sample = to_torch(sample, device=get_torch_model_device(self.model))
                    sample_inputs, _ = self._inputs_and_gt_from_sample(sample)

                    with writer.TimeWriter(name="00_overview/forward_pass_sec", step=self.finished_iterations, write=should_log_loss(), avg_over_steps=False):
                        pred, aux = self.model.pointnerf(**sample_inputs, sample_rays=True)

                    loss, sub_losses, pointwise_losses = self.loss(sample=sample, pred=pred, aux=aux, iteration=self.finished_iterations)
                    loss.backward()
                    if self.grad_clip_max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.pointnerf.parameters(), self.grad_clip_max_norm)

                    self.optimizer.step()
                    self.scheduler.step()

                steps_since_print += 1
                if should_print():
                    end_print_interval = time.time()
                    time_per_iteration = (end_print_interval - start_print_interval) / steps_since_print
                    logging.info(f"Iteration {self.finished_iterations}/{self.max_iterations} - "
                                 f"{time_per_iteration:1.4f} sec per iteration - loss: {loss:1.5f}")
                    start_print_interval = time.time()
                    steps_since_print = 0

                if should_log():
                    self._log_all(sample, loss, sub_losses, pointwise_losses)
                elif should_log_loss():
                    self._log_loss(loss, sub_losses, pointwise_losses)

                self.finished_iterations += 1

                if self._start_iteration < self.finished_iterations < self.max_iterations:
                    if time.time() - last_checkpoint_time > 60 * self.save_checkpoint_interval_min:
                        self._save_all()
                        last_checkpoint_time = time.time()
                        
                writer.write_out_storage()

                if not should_continue():
                    break

        self._write_checkpoints()

        if self.verbose:
            logging.info(f"Finished training {self.name}.")

    def _setup_savers(self):
        max_checkpoints_to_keep = 3
        self.saver_all = TrainStateSaver(model=self.model, 
                                         optim=self.optimizer,
                                         scheduler=self.scheduler,
                                         base_path=self.checkpoints_dir,
                                         base_name=self.checkpoints_name,
                                         max_to_keep=max_checkpoints_to_keep)

        self.saver_weights_only = WeightsOnlySaver(model=self.model,
                                                   base_path=self.weights_only_checkpoints_dir,
                                                   base_name=self.weights_only_checkpoints_name)

    def _restore_weights(self):
        all_checkpoints = sorted(self.saver_all.get_checkpoints(include_iteration=True))
        if len(all_checkpoints) > 0:
            
            logging.info("Existing checkpoints:")
            for step, checkpoint in all_checkpoints:
                logging.info(f"\t{step}: {checkpoint}")
                
            newest_checkpoint = all_checkpoints[-1][1]
            logging.info(f"Restoring training state from checkpoint {newest_checkpoint}.")
            self.saver_all.load(full_path=newest_checkpoint)
            self.finished_iterations = all_checkpoints[-1][0]

    def _write_checkpoints(self):
        if self.finished_iterations > self._start_iteration:
            self._save_all()

            if self.finished_iterations >= self.max_iterations:
                self._save_weights_only()

    def _save_all(self):
        save_path = self.saver_all.save(iteration=self.finished_iterations)
        logging.info(f"Saved training state checkpoint to {save_path}.")

    def _save_weights_only(self):
        save_path = self.saver_weights_only.save(iteration=self.finished_iterations)
        logging.info(f"Saved weights-only checkpoint to {save_path}.")
        
    def _inputs_and_gt_from_sample(self, sample):
        input_keys = ["obj_idx", "intrinsics", "extrinsics"]
        sample_inputs = {key: val for key, val in sample.items() if key in input_keys}
        sample_gt = {key: val for key, val in sample.items() if key not in input_keys}
        return sample_inputs, sample_gt

    def _log_all(self, sample, loss, sub_losses, pointwise_losses):
        self._log_pred(sample)
        self._log_loss(loss, sub_losses, pointwise_losses, scalars_only=False)
        self._log_optim()

    @torch.no_grad()
    def _log_pred(self, sample):
        base_name = '01_pred'
        
        # We have to run model again, as in the original forward pass, the 
        # prediction is only computed for a subsample of all rays.
        # We run the model again in eval mode and only for a single 
        # sample with a single view.
        
        self.model.eval()
        
        # reduce input data to batch size 1:
        sample['images'] = sample['images'][0:1]
        sample['extrinsics'] = sample['extrinsics'][0:1]
        sample['intrinsics'] = sample['intrinsics'][0:1]
        sample['view_indices'] = sample['view_indices'][0:1]
        sample['obj_idx'] = sample['obj_idx'][0:1]
        sample['obj_name'] = sample['obj_name'][0:1]

        # reduce input data to 1 view only:
        sample['images'] = sample['images'][:, 0:1]
        sample['extrinsics'] = sample['extrinsics'][:, 0:1]
        sample['intrinsics'] = sample['intrinsics'][:, 0:1]
        sample['view_indices'] = sample['view_indices'][:, 0:1]

        gt_img = sample['images'][0, 0]  # 3, H, W
        
        # apply model:
        sample_inputs, _ = self._inputs_and_gt_from_sample(sample)
        pred, aux = self.model.pointnerf(**sample_inputs, sample_rays=False)
        pred_img = unflatten_pred(pred.channels.contiguous()[0, 0])
        pred_depth = unflatten_pred(pred.depth.contiguous()[0, 0])
        pred_mask = unflatten_pred(pred.mask.contiguous()[0, 0])

        # log gt and pred image:
        writer.put_tensor(name=f"{base_name}/00_gt_image", tensor=gt_img, step=self.finished_iterations)
        writer.put_tensor(name=f"{base_name}/01_pred_image", tensor=pred_img, step=self.finished_iterations, 
                          clipping=True, lower_clipping_thresh=0, upper_clipping_thresh=1)
        
        # log pred depth and mask:
        writer.put_tensor(name=f"{base_name}/02_pred_depth", tensor=pred_depth, step=self.finished_iterations)
        writer.put_tensor(name=f"{base_name}/03_pred_mask", tensor=pred_mask, step=self.finished_iterations)
        
        # log psnr:
        pred_img = pred_img.detach().cpu().numpy()
        gt_img = gt_img.detach().cpu().numpy()
        psnr = float(skimage.metrics.peak_signal_noise_ratio(pred_img, gt_img, data_range=1))
        writer.put_scalar(name=f"{base_name}/02_psnr", scalar=psnr, step=self.finished_iterations)
        
        # log neural point cloud feature statistics:
        if "feats_mean" in aux or "feats" in aux:
            feats = aux["feats_mean"] if "feats_mean" in aux else aux["feats"]
            feats_stats = {
                '00_min': feats.min(),
                '01_max': feats.max(),
                '02_mean': feats.mean(),
            }
            writer.put_scalar_dict(f"{base_name}/03_stats/00_feats", feats_stats, self.finished_iterations)
            
        if "feats_std" in aux:
            feats_std = aux["feats_std"]
            feats_std_stats = {
                '00_min': feats_std.min(),
                '01_max': feats_std.max(),
                '02_mean': feats_std.mean(),
            }
            writer.put_scalar_dict(f"{base_name}/03_stats/01_feats_std", feats_std_stats, self.finished_iterations)
        
        self.model.train()

    def _log_loss(self, loss, sub_losses, pointwise_losses, scalars_only=True):
        base_name = '02_loss'
        writer.put_scalar(name="00_overview/00_loss", scalar=loss, step=self.finished_iterations)
        writer.put_scalar(name=f"{base_name}/00_total_loss", scalar=loss, step=self.finished_iterations)

        writer.put_scalar_dict(name=f"{base_name}/01_sub", scalar=sub_losses, step=self.finished_iterations)
        
        if not scalars_only:
            writer.put_tensor_dict(name=f"{base_name}/02_qual", tensor=pointwise_losses, step=self.finished_iterations)

    def _log_optim(self):
        optimizer = self.optimizer

        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group["lr"]
            writer.put_scalar(name=f"00_overview/lr/group_{i}", scalar=lr, step=self.finished_iterations)
