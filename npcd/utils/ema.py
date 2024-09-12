import copy
import torch
from torch import nn


class EmaScheduler:
    """Implements an EMA warmup using an inverse decay schedule.
    If inv_gamma=1 and power=1, implements a simple average. 
    inv_gamma=1, power=0.6667 reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps.
    inv_gamma=1, power=0.75 reaches decay factor 0.999 at 10K steps, 0.9999 at 215.4k steps.
    inv_gamma=1, power=0.8 reaches decay factor 0.999 at 5600 steps, 0.9999 at 100k steps.
    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
        max_value (float): The maximum EMA decay rate. Default: 1.
        start_at (int): The epoch to start averaging at. Default: 0.
        last_epoch (int): The index of last epoch. Default: 0.
    """

    def __init__(self, inv_gamma=1., power=1., min_value=0., max_value=1., start_at=0, last_epoch=0):
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.start_at = start_at
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the class as a :class:`dict`."""
        return dict(self.__dict__.items())

    def load_state_dict(self, state_dict):
        """Loads the class's state.
        Args:
            state_dict (dict): scaler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_value(self):
        """Gets the current EMA decay rate."""
        epoch = max(0, self.last_epoch - self.start_at)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power
        return 0. if epoch < 0 else min(self.max_value, max(self.min_value, value))

    def step(self):
        """Updates the step count."""
        self.last_epoch += 1


def ema_param_to_string(ema_param):
        ret = f'power{float(ema_param[0])}min{float(ema_param[1])}max{float(ema_param[2])}buffers{int(ema_param[3])}'
        ret = ret.replace('.', '_')
        return ret


def ema_param_from_string(ema_param_string):
        ema_param = ema_param_string.split('power')[1]
        ema_param = ema_param.split('min')
        power = float(ema_param[0].replace('_', '.'))
        ema_param = ema_param[1].split('max')
        min_value = float(ema_param[0].replace('_', '.'))
        ema_param = ema_param[1].split('buffers')
        max_value = float(ema_param[0].replace('_', '.'))
        buffers = ema_param[1] == '1'
        return power, min_value, max_value, buffers


class EMAHandler:
    def __init__(self, model, ema_params):
        self.running_model = model
        
        self.ema_params = ema_params  # [(power, min_value, max_value, ema_on_buffers), ...]
        self.ema_schedulers = []
        self.ema_models = []
        for ema_param in self.ema_params:
            power, min_value, max_value, _ = ema_param
            ema_scheduler = EmaScheduler(power=power, min_value=min_value, max_value=max_value)
            ema_model = copy.deepcopy(model)
            ema_model.requires_grad_(False)
            self.ema_models.append(ema_model)
            self.ema_schedulers.append(ema_scheduler)
            
    @property
    def ema_param_strings(self):
        return [ema_param_to_string(ema_param) for ema_param in self.ema_params]

    @torch.no_grad()
    def copy_params_from_model_to_ema(self):
        running_params = dict(self.running_model.named_parameters())
        
        for ema_model in self.ema_models:
            ema_model_params = dict(ema_model.named_parameters())
            assert running_params.keys() == ema_model_params.keys()
        
            for name, running_param in running_params.items():
                ema_model_params[name].copy_(running_param.data)
            
    @torch.no_grad()
    def copy_params_from_ema_to_model(self, ema_model_idx):
        ema_model = self.ema_models[ema_model_idx]
        
        running_params = dict(self.running_model.named_parameters())
        ema_model_params = dict(ema_model.named_parameters())
        assert running_params.keys() == ema_model_params.keys()
        
        for name, ema_model_param in ema_model_params.items():
            running_params[name].copy_(ema_model_param)
            # note: this should not overwrite the grad of the running model
            # due to the torch.no_grad() context manager
       
    @torch.no_grad()
    def update_ema(self):
        running_params = dict(self.running_model.named_parameters())
        
        for ema_model, ema_scheduler, ema_param in zip(self.ema_models, self.ema_schedulers, self.ema_params):
            decay = ema_scheduler.get_value()
            ema_on_buffers = ema_param[-1]
            
            ema_model_params = dict(ema_model.named_parameters())
            assert running_params.keys() == ema_model_params.keys()
            
            for name, running_param in running_params.items():
                ema_model_params[name].lerp_(running_param, 1 - decay)
                
            running_buffers = dict(self.running_model.named_buffers())
            ema_buffers = dict(ema_model.named_buffers())
            assert running_buffers.keys() == ema_buffers.keys()
            
            if ema_on_buffers:
                for name, running_buffer in running_buffers.items():
                    ema_buffers[name].lerp_(running_buffer, 1 - decay)
            else:
                for name, running_buffer in running_buffers.items():
                    ema_buffers[name].copy_(running_buffer)
                
            ema_scheduler.step()
