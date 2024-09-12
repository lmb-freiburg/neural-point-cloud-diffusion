import os.path as osp
import abc
import time

import torch
import pytoml

from npcd.utils import logging


class Sample(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def load(self, root):
        return


class Dataset(torch.utils.data.Dataset, metaclass=abc.ABCMeta):

    def __init__(self, root, verbose=True, **kwargs):

        self.verbose = verbose

        self.root = None
        self._init_root(root)

        if self.verbose:
            if self.root is not None:
                logging.info(f"Initializing dataset {self.name} from {self.root}")
            else:
                logging.info(f"Initializing dataset {self.name}")

        self.samples = []
        self._init_samples(**kwargs)

        if self.verbose:
            logging.info(f"\tNumber of samples: {len(self)}")
            logging.info(f"Finished initializing dataset {self.name}.")
            logging.info()

    @property
    def name(self):
        return type(self).__name__

    def _init_root(self, root):
        if isinstance(root, str):
            self.root = root
        elif isinstance(root, list):
            self.root = [path for path in root if osp.isdir(path)][0]

    @abc.abstractmethod
    def _init_samples(self):
        return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        sample_dict = sample.load(root=self.root)
        return sample_dict

    def _get_paths(self):
        return get_paths()

    def _get_path(self, *keys):
        return get_path(*keys)
    
    def get_loader(self, batch_size=1, shuffle=False, num_workers=0, collate_fn=None, pin_memory=False,
                   drop_last=False, worker_init_fn=None, indices=None):
        dataset = torch.utils.data.Subset(self, indices) if indices is not None else self
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                                                 num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last,
                                                 worker_init_fn=worker_init_fn)
        return dataloader


def get_paths():
    rmvd_paths_file = osp.join(osp.dirname(osp.realpath(__file__)), 'paths.toml')
    home_paths_file = osp.join(osp.expanduser('~'), 'rmvd_data_paths.toml')
    
    if osp.exists(rmvd_paths_file):
        paths_file = rmvd_paths_file
    elif osp.exists(home_paths_file):
        paths_file = home_paths_file
    else:
        raise FileNotFoundError("No paths.toml file found. Please create a paths.toml file as specified in the "
                               "rmvd/data/README.md file.")
    
    with open(paths_file, 'r') as paths_file:
        return pytoml.load(paths_file)


def get_path(*keys):
    paths = get_paths()
    path = None

    for idx, key in enumerate(keys):
        if key in paths:
            if key in paths and (isinstance(paths[key], str) or isinstance(paths[key], list)) and idx == len(keys) - 1:
                path = paths[key]
            else:
                paths = paths[key]

    return path
