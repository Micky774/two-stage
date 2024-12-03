from argparse import ArgumentParser
import torch
import lightning as L
from torchinfo import summary
from .two_stage_vae_model import Resnet
from lightning.pytorch.loggers import TensorBoardLogger
from .util import kaiming_init
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from lightning.pytorch.callbacks import StochasticWeightAveraging
import os
from pathlib import Path

parser = ArgumentParser()

parser.add_argument("--fast-dev", action="store_true")
parser.add_argument("--n-epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=200)
parser.add_argument("--load", type=str, default="")
parser.add_argument("--log-dir", type=str, default="")
args = parser.parse_args()


LOGDIR = args.log_dir if args.log_dir else "logs"
MODEL_NAME = "resnet_mnist"


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "MNIST", batch_size: int = 32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(
                self.data_dir,
                train=True,
                transform=self.transform,
                download=True,
            )
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir,
                train=False,
                transform=self.transform,
                download=True,
            )

        if stage == "predict":
            self.mnist_predict = MNIST(
                self.data_dir,
                train=False,
                transform=self.transform,
                download=True,
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.mnist_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )


load_path = os.path.join(LOGDIR, MODEL_NAME, f"version_{args.load}", "checkpoints")
chkpt_path = None
if os.path.exists(load_path):
    chkpt_path = os.path.join(load_path, os.listdir(load_path)[0])
    model = Resnet.load_from_checkpoint(chkpt_path)
else:
    model = Resnet(
        torch.randn(size=(1, 1, 28, 28)),
        num_encoder_scales=3,
        use_cross_entropy_loss=False,
    )
    kaiming_init(model)
summary(model, input_size=(1, 1, 28, 28))
logger = TensorBoardLogger(LOGDIR, name=MODEL_NAME)

# # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer(
    max_epochs=args.n_epochs,
    fast_dev_run=args.fast_dev,
    logger=logger,
    log_every_n_steps=10,
    gradient_clip_val=1,
    # detect_anomaly=True,
    callbacks=[
        StochasticWeightAveraging(swa_epoch_start=150, swa_lrs=5e-4, device="cuda")
    ],
)


trainer.fit(
    model=model,
    datamodule=MNISTDataModule("MNIST", batch_size=args.batch_size, num_workers=31),
    ckpt_path=chkpt_path,
)
