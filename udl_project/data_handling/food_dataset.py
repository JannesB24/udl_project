from collections.abc import Callable
from pathlib import Path

import kagglehub
from torch.utils.data import Dataset
from torchvision import datasets


class FoodDataset:
    def __init__(self):
        self._data_directory = self._fetch_data()

    def _fetch_data(self) -> Path:
        data_directory = Path(kagglehub.dataset_download("msarmi9/food101tiny"))
        print(f"Data directory: {data_directory}")
        return data_directory

    def get_train_subset(self, transform: Callable | None = None) -> Dataset:
        return datasets.ImageFolder(
            self._data_directory / "data/food-101-tiny/train", transform=transform
        )

    def get_test_subset(self, transform: Callable | None = None) -> Dataset:
        return datasets.ImageFolder(
            self._data_directory / "data/food-101-tiny/valid", transform=transform
        )

    def get_validation_dataset(self, transform: Callable | None = None) -> None:
        raise NotImplementedError("Dataset only has a train folder")
