from src.datasets.dataset import Dataset
from src.datasets.multiclaim.multiclaim_dataset import MultiClaimDataset
from src.datasets.multiclaim_extended.multiclaim_extended_dataset import MultiClaimExtendedDataset


def dataset_factory(name, **kwargs) -> Dataset:
    dataset = {
        'multiclaim': MultiClaimDataset,
        'multiclaim_extended': MultiClaimExtendedDataset,
    }[name](**kwargs)
    return dataset