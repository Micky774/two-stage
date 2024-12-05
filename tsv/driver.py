from argparse import ArgumentParser
import torch
import lightning as L
from torchinfo import summary
from .two_stage_vae_model import Resnet, Simple
from lightning.pytorch.loggers import TensorBoardLogger
from .util import kaiming_init
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from lightning.pytorch.callbacks import (
    StochasticWeightAveraging,
    BatchSizeFinder,
    ModelSummary,
    LearningRateFinder,
)
import os
from pathlib import Path
from .nvp import NVP

parser = ArgumentParser()

parser.add_argument("--fast-dev", action="store_true")
parser.add_argument("--max-epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=200)
parser.add_argument("--load", type=str, default="")
parser.add_argument("--log-dir", type=str, default="")
parser.add_argument("--latent-dim", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--fc-dim", type=int, default=128)
parser.add_argument("--decoder-channels", type=int, default=32)
parser.add_argument("--decoder-scales", type=int, default=3)
parser.add_argument("--encoder-scales", type=int, default=3)
parser.add_argument("--blocks-per-scale", type=int, default=1)
parser.add_argument("--fc-scales", type=int, default=1)

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
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]
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
    print(f"Loading from {chkpt_path}")
    model = Resnet.load_from_checkpoint(chkpt_path)
else:
    model = Simple(
        torch.randn(size=(1, 1, 28, 28)),
        num_encoder_scales=args.encoder_scales,
        use_cross_entropy_loss=False,
        latent_dim=args.latent_dim,
        lr=args.lr,
        fc_dim=args.fc_dim,
        block_per_scale=args.blocks_per_scale,
        num_decoder_scales=args.decoder_scales,
        decoder_channels=args.decoder_channels,
        num_fc_scales=args.fc_scales,
    )
    kaiming_init(model)
# model = NVP(28, 2, 10, 1, 1, 1)
summary(model, input_size=(1, 1, 28, 28))
logger = TensorBoardLogger(LOGDIR, name=MODEL_NAME, log_graph=True)
trainer = L.Trainer(
    max_epochs=-1,
    # accelerator="cpu",
    # fast_dev_run=args.fast_dev,
    logger=logger,
    gradient_clip_val=1,
    # gradient_clip_algorithm="value",
    log_every_n_steps=5,
    # strategy='ddp_find_unused_parameters_true',
    # overfit_batches=1,
    # detect_anomaly=True,
    callbacks=[
        # BatchSizeFinder(mode="binsearch", init_val=1000),
        # LearningRateFinder(min_lr=1e-6, max_lr=1e-1, num_training_steps=40),
        # ModelSummary(max_depth=5),
        # StochasticWeightAveraging(swa_epoch_start=6000, swa_lrs=5e-3, device="cuda")
    ],
)


trainer.fit(
    model=model,
    datamodule=MNISTDataModule("MNIST", batch_size=args.batch_size, num_workers=31),
    ckpt_path=chkpt_path,
)
