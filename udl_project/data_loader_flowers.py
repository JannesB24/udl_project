import os
from pathlib import Path
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
import logging

# TODO: possibly relocate to config file or make them accessible for other modules!?
BATCH_SIZE = 32
IMAGE_DIM = (64, 64)
NUM_WORKERS = os.cpu_count()


class DataLoaderFlowers:
    def __init__(
        self,
        dataset: datasets.ImageFolder,
        train_size: int,
        test_size: int,
        batch_size: int,
        num_workers: int,
    ) -> None:
        self.train_data, self.test_data = random_split(dataset, [train_size, test_size])
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_test_dataloader(self) -> DataLoader:
        """Creates a DataLoader for the test dataset.
        Returns:
            DataLoader: DataLoader for the test dataset.
        """
        logging.info(
            f"Creating test data loader with batch size {self.batch_size} and {self.num_workers} workers."
        )

        test_dataloader = DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        return test_dataloader

    def get_train_dataloader(self) -> DataLoader:
        """Creates a DataLoader for the training dataset.
        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        logging.debug(
            f"Creating train data loader with batch size {self.batch_size} and {self.num_workers} workers."
        )

        train_dataloader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        return train_dataloader

    @staticmethod
    def create_dataloader(
        data_directory: Path = Path("data/train"),
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        image_dim: tuple = IMAGE_DIM,
    ) -> "DataLoaderFlowers":
        """Creates an instance of the DataLoaderFlowers class.

        Args:
            data_directory (Path): Path to the directory containing the image dataset.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses to use for data loading.
            image_dim (tuple): Dimensions to which images will be resized (square).

        Returns:
            DataLoaderFlowers: DataLoaderFlowers instance
        """
        simple_transform = transforms.Compose(
            [
                transforms.Resize(image_dim),
                transforms.ToTensor(),
            ]
        )
        dataset = datasets.ImageFolder(data_directory, transform=simple_transform)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        return DataLoaderFlowers(
            dataset=dataset,
            train_size=train_size,
            test_size=test_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )
