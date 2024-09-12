import random
import re
import collections
import os

import torch
from torch._six import string_classes
import numpy as np

from npcd.utils import logging


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def split_num(num, split_size):
    return [split_size]*(num//split_size) + ([num%split_size] if num % split_size != 0  else [])
    

def shifted_cumsum(t, dim=0, shift=1):
    slicing = dim * (slice(None),)
    return torch.cumsum(
        torch.cat(
            (
                torch.zeros_like(t[slicing + (slice(None, shift),)]),
                t[slicing + (slice(None, -shift),)]
            ),
            dim=dim
        ),
        dim=dim
    )


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
    
    
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))
    
    
def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
    
def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def set_seed(seed=None, use_gpu=True):
    if seed is None:
        seed = random.randint(1, 10000)
    logging.info(f"Setting random seed to: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cp(a, b, verbose=True, followLinks=False):                                                                                                                                                
    os.system('cp -r %s %s "%s" "%s"' % ('-v' if verbose else '', '-L' if followLinks else '', a, b))


def to_torch(data, device=None):
    # adapted from torch.utils.data._utils.collate.default_convert
    np_str_obj_array_pattern = re.compile(r'[SaUO]')

    if data is None:
        return None

    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data, device=device)
    elif isinstance(data, collections.abc.Mapping):
        try:
            return elem_type({key: to_torch(data[key], device=device) for key in data})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`...
            return {key: to_torch(data[key], device=device) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(to_torch(d, device=device) for d in data))
    elif isinstance(data, tuple):
        return [to_torch(d, device=device) for d in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        try:
            return elem_type([to_torch(d, device=device) for d in data])
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [to_torch(d, device=device) for d in data]
    else:
        return data
    
    
def to_numpy(data):
    if data is None:
        return None

    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        return data
    elif isinstance(data, collections.abc.Mapping):
        try:
            return elem_type({key: to_numpy(data[key]) for key in data})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: to_numpy(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(to_numpy(d) for d in data))
    elif isinstance(data, tuple):
        return [to_numpy(d) for d in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        try:
            return elem_type([to_numpy(d) for d in data])
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [to_numpy(d) for d in data]
    else:
        return data


def get_torch_model_device(model):
    # make sure that all parameters are on the same device:
    it = iter(model.parameters())
    is_cuda = next(it).is_cuda
    device = next(it).device
    if not all((elem.device == device) for elem in it):
        raise RuntimeError('All model parameters need to be on the same device')
    return device


def count_torch_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def subsample_gt(gt_map, ray_idx):
    samples = gt_map.flatten(-2, -1).transpose(-1, -2)
    if ray_idx is not None:
        samples = samples.expand(*ray_idx.shape[:-2], *samples.shape[-2:])
        samples = samples.gather(
            dim=-2, 
            index=ray_idx.expand(*ray_idx.shape[:-1], samples.shape[-1])
        )
    return samples


def unflatten_pred(pred):
    res = pred.transpose(-1, -2)
    m = res.shape[-1]
    side = round(m ** (0.5))
    return res.reshape(*res.shape[:-1], side, side)
