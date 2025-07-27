import os
from pathlib import Path
import kagglehub
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
import logging

# TODO: possibly relocate to config file or make them accessible for other modules!?
BATCH_SIZE = 32
IMAGE_DIM = 64
NUM_WORKERS = os.cpu_count()


class CustomDataLoader:
    def __init__(
        self,
        train_dataset: datasets.ImageFolder,
        test_dataset: datasets.ImageFolder,
        batch_size: int,
        num_workers: int,
    ) -> None:
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = len(train_dataset.classes)

    def get_test_dataloader(self) -> DataLoader:
        """Creates a DataLoader for the test dataset.
        Returns:
            DataLoader: DataLoader for the test dataset.
        """
        logging.info(
            f"Creating test data loader with batch size {self.batch_size} and {self.num_workers} workers."
        )

        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
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
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_dataloader

    @staticmethod
    def create_dataloader(
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        image_dim: int = IMAGE_DIM,
    ) -> "CustomDataLoader":
        """Creates an instance of the CustomDataLoader class.

        Args:
            data_directory (Path): Path to the directory containing the image dataset.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses to use for data loading.
            image_dim (tuple): Dimensions to which images will be resized (square).

        Returns:
            CustomDataLoader: CustomDataLoader instance
        """
        # Download latest version
        data_directory = Path(kagglehub.dataset_download("lara311/flowers-five-classes"))
        # data_directory = Path(kagglehub.dataset_download("msarmi9/food101tiny"))
        print(f"Data directory: {data_directory}")

        simple_transform = transforms.Compose(
            [
                transforms.Resize(image_dim),
                transforms.ToTensor(),
            ]
        )
        train_dataset = datasets.ImageFolder(data_directory / "train", transform=simple_transform)
        test_dataset = datasets.ImageFolder(data_directory / "valid", transform=simple_transform)
        return CustomDataLoader(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
