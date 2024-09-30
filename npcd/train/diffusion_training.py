import os
import os.path as osp
from typing import Optional, Sequence, Tuple
import time

import torch

from npcd.utils import TrainStateSaver, WeightsOnlySaver, EMAHandler, to_torch, get_torch_model_device, count_torch_model_parameters, unflatten_pred, writer, logging


class DiffusionTraining:
    def __init__(self,
                 out_dir: str,
                 model: torch.nn.Module,
                 dataset: torch.utils.data.Dataset,
                 batch_size: int,
                 base_learning_rate: float,
                 weight_decay: float,
                 max_iterations: int,
                 use_ema: bool = False,
                 ema_params: Optional[Sequence[Tuple[float, float, float, bool]]] = None,
                 dtype: str = 'float16',
                 grad_clip_max_norm: Optional[float] = None,
                 num_workers: int = 8,
                 print_interval: int = 100,
                 log_scalars_interval: int = 100,
                 log_interval: int = 5000,
                 save_checkpoint_interval_min: int = 20,
                 log_full_batch: bool = False,
                 verbose: bool = True,
                 **_, ):


        self.verbose = verbose

        self.out_dir = out_dir
        self._init_dirs()

        if self.verbose:
            logging.info(f"Initializing {self.name}.")

        self.model = model
        self.dataset = dataset
        self.model.diffusion.coords_normalization.set_from_all_data(self.dataset.get_all_coords())
        self.model.diffusion.feats_normalization.set_from_all_data(self.dataset.get_all_feats())
        
        self._setup_optimizer(base_learning_rate, weight_decay)
        self._setup_scheduler()
        self.batch_size = batch_size

        self.max_iterations = max_iterations
        writer.set_max_iterations(self.max_iterations)
        self.finished_iterations = 0
        
        self.use_ema = use_ema
        if self.use_ema:
            assert ema_params is not None, "ema_params must be provided if use_ema is True."
            self.ema_handler = EMAHandler(self.model, ema_params=ema_params)
            
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.autocast_ctx = torch.cuda.amp.autocast(dtype=ptdtype, enabled=(dtype != 'float32'))
        self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
        
        self.grad_clip_max_norm = grad_clip_max_norm
        self.dataloader = self.dataset.get_loader(batch_size=batch_size, shuffle=True, pin_memory=True,
                                                  num_workers=num_workers, drop_last=True)
        
        self.saver_all = None
        self.saver_weights_only = None
        self.savers_ema_weights_only = []
        self._start_iteration = None
        self._setup_savers()
        self._restore_weights()
        self._start_iteration = self.finished_iterations
        
        self.print_interval = print_interval
        self.log_interval = log_interval
        self.log_scalars_interval = log_scalars_interval
        self.save_checkpoint_interval_min = save_checkpoint_interval_min
        self.log_full_batch = log_full_batch

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
        ret += f"\n\tDataloader size: {len(self.dataloader)}"
        ret += f"\n\tBatch size: {self.batch_size}"
        ret += f"\n\tFinished iterations: {self.finished_iterations}"
        ret += f"\n\tMax iterations: {self.max_iterations}"
        return ret

    def _init_dirs(self):
        self.artifacts_dir = osp.join(self.out_dir, "artifacts")
        self.checkpoints_dir = osp.join(self.out_dir, "checkpoints")
        self.weights_only_checkpoints_dir = osp.join(self.out_dir, "weights_only_checkpoints_dir")
        self.checkpoints_name = "diffusion_training"
        self.weights_only_checkpoints_name = "npcd"

        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.weights_only_checkpoints_dir, exist_ok=True)
        
    def _setup_optimizer(self, base_learning_rate, weight_decay):
        self.optimizer = torch.optim.AdamW(self.model.diffusion.parameters(), lr=base_learning_rate, weight_decay=weight_decay)
        
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

        self.model.train()

        should_print = lambda: self.finished_iterations % self.print_interval == 0
        should_log = lambda: self.finished_iterations % self.log_interval == 0
        should_log_scalars = lambda: self.finished_iterations % self.log_scalars_interval == 0

        steps_since_print = 0
        start_print_interval = time.time()
        last_checkpoint_time = time.time()

        while should_continue():
            for iter_in_epoch, sample in enumerate(self.dataloader):
                with writer.TimeWriter(name="00_overview/train_sec_per_iter", step=self.finished_iterations, write=should_log_scalars(), avg_over_steps=True, update_eta=True):
                    
                    self.optimizer.zero_grad()

                    sample = to_torch(sample, device=get_torch_model_device(self.model))
                    coords = sample['coords']  # [N, coords_dim, num_points]
                    feats = sample['feats']  # [N, feats_dim, num_points]
            
                    with self.autocast_ctx:
                        loss, sub_losses, pointwise_losses = self.model.diffusion.compute_loss(coords=coords, feats=feats)

                    self.scaler.scale(loss).backward()
                     
                    if self.scaler.is_enabled():
                        inv_scale = 1. / self.scaler.get_scale()
                        if sub_losses is not None:
                            sub_losses = {k: v * inv_scale for k, v in sub_losses.items()}
                        if pointwise_losses is not None:
                            pointwise_losses = {k: v * inv_scale for k, v in pointwise_losses.items()}
                    
                    if self.grad_clip_max_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.diffusion.parameters(), self.grad_clip_max_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                        
                    if self.use_ema:
                        self.ema_handler.update_ema()

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
                elif should_log_scalars():
                    self._log_scalars(sample, loss, sub_losses, pointwise_losses)

                self.finished_iterations += 1

                if self._start_iteration < self.finished_iterations < self.max_iterations:
                    if time.time() - last_checkpoint_time > 60 * self.save_checkpoint_interval_min:
                        self._save_all()
                        last_checkpoint_time = time.time()
                            
                if (self._start_iteration < self.finished_iterations< self.max_iterations) and (self.finished_iterations % 200000 == 0):
                    self._save_weights_only()  # Save weights every 200k iterations
                        
                writer.write_out_storage()

                if not should_continue():
                    break

        self._write_checkpoints()

        if self.verbose:
            logging.info(f"Finished training {self.name}.")

    def _setup_savers(self):
        max_checkpoints_to_keep = 3
        self.saver_all = TrainStateSaver(model=self.model.diffusion, 
                                         optim=self.optimizer,
                                         scheduler=self.scheduler,
                                         ema_handler=self.ema_handler if self.use_ema else None,
                                         base_path=self.checkpoints_dir,
                                         base_name=self.checkpoints_name,
                                         max_to_keep=max_checkpoints_to_keep)

        self.saver_weights_only = WeightsOnlySaver(model=self.model,
                                                   base_path=self.weights_only_checkpoints_dir,
                                                   base_name=self.weights_only_checkpoints_name)
        
        if self.use_ema:
            for ema_param_string, ema_model in zip(self.ema_handler.ema_param_strings, self.ema_handler.ema_models):
                saver = WeightsOnlySaver(model=ema_model,
                                         base_path=self.weights_only_checkpoints_dir,
                                         base_name=f"{self.weights_only_checkpoints_name}-ema_{ema_param_string}")
                self.savers_ema_weights_only.append(saver)

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
                
        if self.use_ema:
            self._save_ema_weights_only()
        
    def _save_ema_weights_only(self):
        for saver in self.savers_ema_weights_only:
            save_path = saver.save(iteration=self.finished_iterations)
            print(f"Saved EMA weights-only checkpoint to {save_path}.")

    def _log_all(self, sample, loss, sub_losses, pointwise_losses):
        self._log_in_data(sample)
        self._log_loss(loss, sub_losses, pointwise_losses)
        self._log_optim()
        
    def _log_scalars(self, sample, loss, sub_losses, pointwise_losses):
        self._log_in_data(sample)
        self._log_loss(loss, sub_losses, pointwise_losses)
        self._log_optim(scalars_only=True)
        
    def _log_in_data(self, sample):
        base_name = '01_in_data'
        coords = sample['coords']
        feats = sample['feats']
        
        writer.put_scalar(name=f"{base_name}/00_coords_mean", scalar=coords.mean(), step=self.finished_iterations)
        writer.put_scalar(name=f"{base_name}/01_coords_std", scalar=coords.std(), step=self.finished_iterations)
        writer.put_scalar(name=f"{base_name}/02_coords_min", scalar=coords.min(), step=self.finished_iterations)
        writer.put_scalar(name=f"{base_name}/03_coords_max", scalar=coords.max(), step=self.finished_iterations)
        
        writer.put_scalar(name=f"{base_name}/04_feats_mean", scalar=feats.mean(), step=self.finished_iterations)
        writer.put_scalar(name=f"{base_name}/05_feats_std", scalar=feats.std(), step=self.finished_iterations)
        writer.put_scalar(name=f"{base_name}/06_feats_min", scalar=feats.min(), step=self.finished_iterations)
        writer.put_scalar(name=f"{base_name}/07_feats_max", scalar=feats.max(), step=self.finished_iterations)
        
    def _log_loss(self, loss, sub_losses, pointwise_losses):
        base_name = '03_loss'
        writer.put_scalar(name="00_overview/00_loss", scalar=loss, step=self.finished_iterations)
        writer.put_scalar(name=f"{base_name}/00_total_loss", scalar=loss, step=self.finished_iterations)
        
        writer.put_scalar_dict(name=f"{base_name}/01_sub", scalar=sub_losses, step=self.finished_iterations)
    
    def _log_optim(self, scalars_only=False):
        base_name = "04_optim"
        optimizer = self.optimizer

        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group["lr"]
            writer.put_scalar(name=f"00_overview/lr/group_{i}", scalar=lr, step=self.finished_iterations)
            writer.put_scalar(name=f"{base_name}/00_lr/group_{i}", scalar=lr, step=self.finished_iterations)
                    
        if self.use_ema:
            for ema_param_string, ema_scheduler in zip(self.ema_handler.ema_param_strings, self.ema_handler.ema_schedulers):
                cur_decay = ema_scheduler.get_value()
                writer.put_scalar(name=f"{base_name}/01_ema_decay/{ema_param_string}", scalar=cur_decay, step=self.finished_iterations)
            
        if self.scaler.is_enabled():
            writer.put_scalar(name=f"{base_name}/02_scaler_scale", scalar=self.scaler.get_scale(), step=self.finished_iterations)
