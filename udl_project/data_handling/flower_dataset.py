from collections.abc import Callable
from pathlib import Path
from typing import Any

import kagglehub
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets


class SubsetFlowerWrapper(Dataset):
    def __init__(self, subset: Subset, transform: Callable | None = None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Inspired by: torchvision.datasets.folder: DatasetFolder.

        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample, target = self.subset[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        """Return the number of samples in the subset."""
        return len(self.subset)

    @property
    def classes(self):
        return self.subset.dataset.classes


class FlowerDataset:
    def __init__(self, train_test_split: float):
        _data_directory = self._fetch_data()

        _train_test_dataset = datasets.ImageFolder(_data_directory / "train")

        _train_size = int(train_test_split * len(_train_test_dataset))
        _test_size = len(_train_test_dataset) - _train_size

        self.train_data, self.test_data = random_split(
            _train_test_dataset, [_train_size, _test_size]
        )

    def _fetch_data(self) -> Path:
        data_directory = Path(kagglehub.dataset_download("lara311/flowers-five-classes"))
        print(f"Data directory: {data_directory}")
        return data_directory

    def get_train_subset(self, transform: Callable | None = None) -> SubsetFlowerWrapper:
        return SubsetFlowerWrapper(self.train_data, transform=transform)

    def get_test_subset(self, transform: Callable | None = None) -> SubsetFlowerWrapper:
        return SubsetFlowerWrapper(self.test_data, transform=transform)

    def get_validation_dataset(self, transform: Callable | None = None) -> SubsetFlowerWrapper:
        raise NotImplementedError("Dataset only has a train folder")
