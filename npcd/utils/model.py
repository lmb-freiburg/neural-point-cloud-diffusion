import errno
from glob import glob
import os
import os.path as osp
from typing import Optional

from torch import nn


def makedirs(path: str):
    r"""Recursive directory creation function."""
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e

def restore_checkpoint(model, load_name, resume, **kwargs):
    load_name = os.path.join(load_name, "checkpoints", "last.ckpt")
    return model.load_from_checkpoint(checkpoint_path=load_name, strict=resume, **kwargs)

def define_mlp(dims, d_in, d_out: Optional[int] = None, act: str = "ReLU", layer_norm: bool = True):
    act = getattr(nn, act)
    cur_dim = d_in
    modules = []
    for dim in dims:
        l = nn.Linear(cur_dim, dim)
        modules.append(l)
        if layer_norm:
            modules.append(nn.LayerNorm(dim, elementwise_affine=False))
        modules.append(act(inplace=True))
        cur_dim = dim
    if d_out is not None:
        modules.append(nn.Linear(cur_dim, d_out))
    mlp = nn.Sequential(*modules)
    return mlp

def clean_up_checkpoints(dir):
    for clean_up in glob(dir + "/checkpoints/*.*"):
        if not clean_up.endswith("last.ckpt"):   
            os.remove(clean_up)