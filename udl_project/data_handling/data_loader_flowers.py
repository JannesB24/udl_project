import logging
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import kagglehub
from udl_project.data_handling.flower_dataset import FlowerDataset
from torchvision import datasets
# TODO: possibly relocate to config file or make them accessible for other modules!?
BATCH_SIZE = 32
IMAGE_DIM = (224, 224)
NUM_WORKERS = os.cpu_count()


class DataLoaderFlowers:
    def __init__(
        self,
        train_data: Dataset,
        test_data: Dataset,
        batch_size: int,
        num_workers: int,
    ) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = len(train_data.classes)

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
        flower_data_source: FlowerDataset,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        image_dim: tuple = IMAGE_DIM,
        augment_data: bool = False,
    ) -> "DataLoaderFlowers":
        """Creates an instance of the DataLoaderFlowers class.

        Args:
            flower_data_source (FlowerDataset): Source dataset containing flower images and labels.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses to use for data loading.
            image_dim (tuple): Dimensions to which images will be resized (square).
            augment_data (bool): Whether to apply data augmentation or not.

        Returns:
            DataLoaderFlowers: DataLoaderFlowers instance
        """

        # Download latest version
        data_directory = Path(kagglehub.dataset_download("lara311/flowers-five-classes"))
        print(f"Data directory: {data_directory}")

        train_transform = v2.Compose(
            [
                v2.RandomResizedCrop(image_dim, scale=(0.8, 1.0), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(degrees=45),  # +- 45 degrees
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        non_train_transform = v2.Compose(
            [
                v2.Resize(image_dim, antialias=True),
                v2.CenterCrop(image_dim),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        if augment_data:
            train_data = flower_data_source.get_train_subset(train_transform)
        else:
            train_data = flower_data_source.get_train_subset(non_train_transform)

        test_data = flower_data_source.get_test_subset(non_train_transform)

        # Create a DataLoaderFlowers instance using the split datasets
        loader = DataLoaderFlowers(
            train_data=train_data,
            test_data=test_data,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        return loader

       
    @staticmethod
    def create_dataloader_Pathchoice(
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        image_dim: tuple = IMAGE_DIM,
        kagglePath: str = "",
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
        if kagglePath == "":
            print("No dataset")
            return
        # Download latest version
        # data_directory = Path(kagglehub.dataset_download("lara311/flowers-five-classes"))
        # data_directory = Path(kagglehub.dataset_download("msarmi9/food101tiny"))
        data_directory = Path(kagglehub.dataset_download(kagglePath))
        print(f"Data directory: {data_directory}")

        simple_transform = v2.Compose(
            [
                v2.Resize(image_dim),
                v2.ToTensor(),
            ]
        )
        dataset = datasets.ImageFolder(data_directory / "train", transform=simple_transform)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        print(f"set size: {len(dataset.classes)}")

        return DataLoaderFlowers(
            dataset=dataset,
            train_size=train_size,
            test_size=test_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )



