import logging
import os
from pathlib import Path

import kagglehub
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

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
        self.num_classes = len(dataset.classes)

    def get_test_dataloader(self) -> DataLoader:
        """Creates a DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for the test dataset.
        """
        logging.info(
            f"Creating test data loader with batch size {self.batch_size} and {self.num_workers} workers."
        )

        return DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def get_train_dataloader(self) -> DataLoader:
        """Creates a DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        logging.debug(
            f"Creating train data loader with batch size {self.batch_size} and {self.num_workers} workers."
        )

        return DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    @staticmethod
    def create_dataloader(
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
        # Download latest version
        data_directory = Path(kagglehub.dataset_download("lara311/flowers-five-classes"))
        print(f"Data directory: {data_directory}")

        simple_transform = transforms.Compose(
            [
                transforms.Resize(image_dim),
                transforms.ToTensor(),
            ]
        )
        dataset = datasets.ImageFolder(data_directory / "train", transform=simple_transform)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        return DataLoaderFlowers(
            dataset=dataset,
            train_size=train_size,
            test_size=test_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )
