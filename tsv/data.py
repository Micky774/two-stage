from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms as _transforms
import lightning as L
import numpy as np
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/dev/shm/CIFAR10",
        batch_size: int = 32,
        num_workers=1,
        labels_path=None,
        transforms=None,
        shuffle=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.estimated_labels = np.load(labels_path) if labels_path else None
        self.ground_truth_labels = None
        self.shuffle = shuffle
        if self.estimated_labels is not None:
            print(f"Using estimated labels loaded from {labels_path}")
        if transforms is not None:
            self.transform = transforms
            print(f"Using custom transforms: {transforms}")
        else:
            self.transform = _transforms.Compose(
                [
                    _transforms.ToTensor(),
                    # transforms.Normalize((0.1307,), (0.3081,))
                ]
            )

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_data = CIFAR10(
                self.data_dir,
                train=True,
                transform=self.transform,
                download=True,
            )
            self.ground_truth_labels = self.train_data.targets
            if self.estimated_labels is not None:
                self.train_data.targets = self.estimated_labels
            self.count = len(self.train_data)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_data = CIFAR10(
                self.data_dir,
                train=False,
                transform=self.transform,
                download=True,
            )
            self.count = len(self.test_data)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=5,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=5,
        )


class FMNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/dev/shm/FMNIST",
        batch_size: int = 32,
        num_workers=1,
        labels_path=None,
        transforms=None,
        shuffle=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.estimated_labels = np.load(labels_path) if labels_path else None
        self.ground_truth_labels = None
        self.shuffle = shuffle
        if self.estimated_labels is not None:
            print(f"Using estimated labels loaded from {labels_path}")
        if transforms is not None:
            self.transform = transforms
            print(f"Using custom transforms: {transforms}")
        else:
            self.transform = _transforms.Compose(
                [
                    _transforms.ToTensor(),
                    # transforms.Normalize((0.1307,), (0.3081,))
                ]
            )

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_data = FashionMNIST(
                self.data_dir,
                train=True,
                transform=self.transform,
                download=True,
            )
            self.ground_truth_labels = self.train_data.targets
            if self.estimated_labels is not None:
                self.train_data.targets = self.estimated_labels
            self.count = len(self.train_data)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_data = FashionMNIST(
                self.data_dir,
                train=False,
                transform=self.transform,
                download=True,
            )
            self.count = len(self.test_data)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=5,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=5,
        )


class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "MNIST",
        batch_size: int = 32,
        num_workers=4,
        labels_path=None,
        transforms=None,
        shuffle=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.estimated_labels = np.load(labels_path) if labels_path else None
        self.ground_truth_labels = None
        self.shuffle = shuffle
        if self.estimated_labels is not None:
            print(f"Using estimated labels loaded from {labels_path}")
        if transforms is not None:
            self.transform = transforms
            print(f"Using custom transforms: {transforms}")
        else:
            self.transform = _transforms.Compose(
                [
                    _transforms.ToTensor(),
                    # transforms.Normalize((0.1307,), (0.3081,))
                ]
            )

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_data = MNIST(
                self.data_dir,
                train=True,
                transform=self.transform,
                download=True,
            )
            if self.estimated_labels is not None:
                self.train_data.targets = self.estimated_labels
            self.count = len(self.train_data)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_data = MNIST(
                self.data_dir,
                train=False,
                transform=self.transform,
                download=True,
            )
            self.count = len(self.test_data)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=True,
            prefetch_factor=5,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=True,
            prefetch_factor=5,
        )
