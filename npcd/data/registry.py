_datasets = {}


def register_dataset(dataset_cls):
    """Register a dataset."""
    dataset_name = dataset_cls.__name__
    _datasets[dataset_name] = dataset_cls
    return dataset_cls


def list_datasets():
    """List all available datasets."""
    datasets = list(sorted(_datasets.keys()))
    return datasets


def has_dataset(name):
    """Check if dataset exists."""
    return name in _datasets


def get_dataset(name):
    """Get dataset entrypoint by name."""
    assert has_dataset(name), f'The requested dataset "{name}" does not exist. Available datasets are: {" ".join(list_datasets())}'
    return _datasets[name]
