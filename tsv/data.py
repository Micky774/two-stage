from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms as _transforms
import lightning as L
import numpy as np
from torchvision.datasets import CIFAR10
import os
import matplotlib.pyplot as plt
import torch


def get_data_module(
    dataset_name,
    batch_size,
    num_workers,
    labels_path=None,
    transforms=None,
    data_dir=None,
    shuffle=True,
    embedding_path=None,
):
    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        transforms=transforms,
        shuffle=shuffle,
        labels_path=labels_path,
    )
    if data_dir is not None:
        kwargs["data_dir"] = data_dir
    if dataset_name in ("chest-mnist", "oct-mnist", "blood-mnist"):
        kwargs["dataset_name"] = dataset_name
    if dataset_name == "latent":
        kwargs["data_path"] = embedding_path
    datamodule_cls = {
        "mnist": MNISTDataModule,
        "fmnist": FMNISTDataModule,
        "cifar10": CIFAR10DataModule,
        "latent": LatentDataModule,
        "chest-mnist": MedMNISTDataModule,
        "oct-mnist": MedMNISTDataModule,
        "blood-mnist": MedMNISTDataModule,
    }[dataset_name]
    return datamodule_cls(**kwargs)


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


class MedMNISTDataset(Dataset):
    def __init__(self, data_path, dataset_name, transform=None):
        if data_path is None:
            raise ValueError("data_path must be provided")
        self.data_path = data_path
        print(f"Loading data from {data_path}")
        self.full_data = np.load(data_path)
        self.labels = self.full_data["train_labels"]
        self.data = self.full_data["train_images"]
        self.data = self.data.astype(np.float32) / 255.0

        if dataset_name == "chest-mnist":
            self.data = np.expand_dims(self.data, axis=1)
            self.labels = self.labels[:, 0]
        elif dataset_name == "oct-mnist":
            self.data = np.expand_dims(self.data, axis=1)
        elif dataset_name == "blood-mnist":
            self.data = np.moveaxis(self.data, source=-1, destination=1)

        self.data = torch.tensor(self.data)
        self.labels = torch.tensor(self.labels)
        print(self.data.shape)
        del self.full_data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MedMNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "local_artifacts/",
        dataset_name="chestmnist",
        batch_size: int = 32,
        num_workers=1,
        labels_path=None,
        transforms=None,
        shuffle=True,
    ):
        super().__init__()
        self.data_path = os.path.join(data_dir, dataset_name + ".npz")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.dataset_name = dataset_name
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
            self.train_data = MedMNISTDataset(
                self.data_path,
                transform=self.transform,
                dataset_name=self.dataset_name,
            )
            self.count = len(self.train_data)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_data = MedMNISTDataset(
                self.data_path,
                transform=self.transform,
                dataset_name=self.dataset_name,
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
