from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms as _transforms
import lightning as L
import numpy as np
from torchvision.datasets import CIFAR10
import os
import matplotlib.pyplot as plt


class LatentDataset(Dataset):
    def __init__(self, data_path, labels_path=None):
        if data_path is None:
            raise ValueError("data_path must be provided")
        self.data_path = data_path
        print(f"Loading data from {data_path}")
        self.data = np.load(os.path.join(data_path, "embeddings.npy"))
        self.labels = np.load(os.path.join(data_path, "labels.npy"))
        print(f"Length of dataset: {self.data.shape[0]}")
        fig, axes = plt.subplots(1, 1, figsize=(20, 20))
        axes.scatter(
            self.data[:, 0],
            self.data[:, 1],
            c=self.labels,
            s=45 / np.sqrt(len(self.labels)),
            cmap="tab10",
        )
        fig.savefig(os.path.join(data_path, "embedding.png"))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class LatentDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers=1,
        labels_path=None,
        transforms=None,
        shuffle=True,
    ):
        super().__init__()
        self.data_path = data_path
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
        self.train_data = LatentDataset(self.data_path, labels_path=None)

        self.count = len(self.train_data)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=True,
            prefetch_factor=5,
        )


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
        persistent_workers=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.estimated_labels = np.load(labels_path) if labels_path else None
        self.ground_truth_labels = None
        self.shuffle = shuffle
        self.persistent_workers = persistent_workers
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
            persistent_workers=self.persistent_workers,
            pin_memory=True,
            prefetch_factor=5,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            persistent_workers=self.persistent_workers,
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
