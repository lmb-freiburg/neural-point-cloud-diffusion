from .registry import get_dataset


def create_dataset(name):
    dataset_cls = get_dataset(name=name)
    dataset = dataset_cls()
    return dataset
