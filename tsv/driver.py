from argparse import ArgumentParser
import torch
import lightning as L

# from torchinfo import summary
from lightning.pytorch.loggers import TensorBoardLogger
from .data import get_data_module
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    StochasticWeightAveraging,
    ModelSummary,
    Callback,
)
from .util import kaiming_init
import os
from .natvamp import get_model_cls, get_encoder_cls, get_decoder_cls
import re
import numpy as np


parser = ArgumentParser()

parser.add_argument("--fast-dev", action="store_true")
parser.add_argument("--max-epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=32)
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
parser.add_argument("--encoder-channels", type=int, default=32)
parser.add_argument("--model", type=str, default="nvp")
parser.add_argument("--version", type=str, default=None)
parser.add_argument("--devices", type=str, default="auto")
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--base-channels", type=int, default=16)
parser.add_argument("--sbd-channels", type=int, default=16)
parser.add_argument("--labels-path", type=str, default="")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--num-pseudos", type=int, default=10)
parser.add_argument("--kaiming", action="store_true")
parser.add_argument("--eta", type=float, default=0.05)
parser.add_argument("--diversity-scale", type=float, default=1)
parser.add_argument("--beta", type=float, default=1)
parser.add_argument("--sbd-kernel-size", type=int, default=3)
parser.add_argument("--alpha", type=float, default=1)
parser.add_argument("--load-epoch", type=int, default=-1)
parser.add_argument("--anneal-epochs", type=int, default=10)
parser.add_argument("--num-workers", type=int, default=1)
parser.add_argument("--constant-lr", action="store_true")
parser.add_argument("--delta", type=float, default=1)
parser.add_argument("--zeta", type=float, default=1)
parser.add_argument("--random-seed", type=int, default=42)
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--decoder-blocks", type=int, default=1)
parser.add_argument("--decoder-layers-per-block", type=int, default=4)
parser.add_argument("--one-cycle", action="store_true")
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--encoder-depth-per-block", type=int, default=2)
parser.add_argument("--intermediate-channels", type=int, default=32)
parser.add_argument("--encoder-block-channels", type=int, default=32)
parser.add_argument("--sample-size", type=int, default=1)
parser.add_argument("--freeze-pseudos", type=int, default=0)
parser.add_argument("--freeze-encoder", type=int, default=0)
parser.add_argument("--train-pseudos", action="store_true")
parser.add_argument("--pseudo-lr", type=float, default=None)
parser.add_argument("--one-cycle-warmup", type=float, default=0.3)
parser.add_argument("--decoder", type=str, default="sbd", choices=["sbd", "dense"])
parser.add_argument("--grad-clip-val", type=float, default=10)
parser.add_argument("--weight-decay", type=float, default=4e-4)
parser.add_argument("--max-batch-size", type=int, default=256)
parser.add_argument("--load-from-pt", type=str, default=None)
parser.add_argument("--disable-gamma", action="store_true")
parser.add_argument("--omega", type=float, default=0)
parser.add_argument("--sbd-depth", type=int, default=None)
parser.add_argument("--embedding-path", type=str, default=None)
parser.add_argument("--init-pseudos", action="store_true")
parser.add_argument("--patience", type=int, default=10)
parser.add_argument(
    "--grad-clip-alg", type=str, default="norm", choices=["norm", "value"]
)

parser.add_argument(
    "--scheduler",
    type=str,
    default="one-cycle",
    choices=["cosine", "one-cycle", "constant", "plateau"],
)
parser.add_argument(
    "--encoder",
    type=str,
    default="basic",
    choices=["resnet", "basic", "pretrained-resnet", "dense"],
)

args = parser.parse_args()

seed_everything(args.random_seed, workers=True)

LOGDIR = args.log_dir if args.log_dir else "logs"

SAMPLE_INPUT = {
    "mnist": torch.zeros((args.batch_size, 1, 28, 28)),
    "fmnist": torch.zeros((args.batch_size, 1, 28, 28)),
    "cifar10": torch.zeros((args.batch_size, 3, 32, 32)),
    "latent": torch.zeros((args.batch_size, args.latent_dim)),
    "chest-mnist": torch.zeros((args.batch_size, 1, 28, 28)),
    "oct-mnist": torch.zeros((args.batch_size, 1, 28, 28)),
    "blood-mnist": torch.zeros((args.batch_size, 3, 28, 28)),
}[args.dataset]

# if args.model == "resnet-50":
#     SAMPLE_INPUT = torch.zeros((args.batch_size, 3, 224, 224))


class BatchSizeScheduler(Callback):
    def __init__(self, func, max_batch_size=50, min_batch_size=0.5):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.func = func

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = pl_module.current_epoch
        pl_module.batch_size = self.func(
            epoch, self.min_batch_size, self.max_batch_size
        )
        return


class BetaScheduler(Callback):
    def __init__(self, T=50, R=0.5):
        super().__init__()
        self.T = T
        self.R = R

    def on_train_start(self, trainer, pl_module):
        self.beta = pl_module.beta

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = pl_module.current_epoch
        warmup_period = self.T * self.R
        if epoch < warmup_period:
            pl_module.beta = 0
        elif epoch < self.T:
            pl_module.beta = self.beta * (
                (epoch - warmup_period) / (self.T - warmup_period)
            )


def __make_cls_kwargs():
    cls_kwargs = dict(
        sample_input=SAMPLE_INPUT,
        lsdim=args.latent_dim,
        beta=args.beta,
        delta=args.delta,
        use_labels=args.labels_path != "",
        lr=args.lr,
        anneal_epochs=args.anneal_epochs,
        constant_lr=args.constant_lr,
        one_cycle=args.one_cycle,
        scheduler=args.scheduler,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        one_cycle_warmup=args.one_cycle_warmup,
        enable_gamma=not args.disable_gamma,
        patience=args.patience,
    )
    if args.decoder == "dense":
        decoder_kwargs = dict(
            layer_spec=[128] * 5,
        )
    if args.decoder == "sbd":
        decoder_kwargs = dict(
            input_shape=SAMPLE_INPUT.shape,
            lsdim=args.latent_dim,
            kernel_size=args.sbd_kernel_size,
            channels_per_layer=args.sbd_channels,
            num_blocks=args.decoder_blocks,
            layers_per_block=args.decoder_layers_per_block,
        )
    if args.encoder == "resnet":
        encoder_kwargs = dict(
            sample_input_shape=SAMPLE_INPUT.shape,
            blocks_per_scale=args.blocks_per_scale,
            num_encoder_scales=args.encoder_scales,
            base_channels=args.base_channels,
            encoder_depth_per_block=args.encoder_depth_per_block,
            encoder_block_channels=args.encoder_block_channels,
            intermediate_channels=args.intermediate_channels,
        )
    elif args.encoder == "basic":
        encoder_kwargs = dict(
            input_shape=SAMPLE_INPUT.shape,
        )
    if any((name in args.model for name in ("nvp", "lsv"))):
        cls_kwargs.update(
            dict(
                num_pseudos=args.num_pseudos,
                eta=args.eta,
                sample_size=args.sample_size,
                pseudo_lr=args.pseudo_lr,
                train_pseudos=args.train_pseudos,
            )
        )
    if any((name in args.model for name in ("nvpw", "lsv"))):
        cls_kwargs.update(
            dict(
                alpha=args.alpha,
                omega=args.omega,
            )
        )
    if "nvpwb" in args.model:
        cls_kwargs.update(
            dict(
                sbd_depth=args.sbd_depth,
            )
        )
    if args.encoder == "dense":
        encoder_kwargs = dict(
            layer_spec=[128] * 6,
        )
    cls_kwargs["encoder_kwargs"] = encoder_kwargs
    cls_kwargs["decoder_kwargs"] = decoder_kwargs
    cls_kwargs["encoder_cls"] = get_encoder_cls(args.encoder)
    cls_kwargs["decoder_cls"] = get_decoder_cls(args.decoder)
    return cls_kwargs


def _load(_cls, load_path, load_from_pt=None, load_epoch=-1, cls_kwargs=None):
    if load_from_pt is not None:
        load_path = os.path.join(load_path, load_from_pt)
        print(f"Loading from {load_path}")
        return torch.load(load_path), None

    load_path = os.path.join(load_path, "checkpoints")
    if not os.path.exists(load_path):
        raise ValueError(f"Checkpoint at {load_path} not found.")
    print(f"Checking for checkpoints at {load_path}")
    checkpoints = os.listdir(load_path)
    splits = [re.split("[=-]", s) for s in checkpoints]
    epoch_counts = [int(s[1]) for s in splits]
    if load_epoch != -1:
        idx = np.argmin(np.abs(np.array(epoch_counts) - load_epoch))
    else:
        idx = np.argmax(epoch_counts)
    chkpt_path = os.path.join(load_path, checkpoints[idx])
    print(f"Loading from {chkpt_path}")

    if cls_kwargs is None:
        cls_kwargs = __make_cls_kwargs()
    model = _cls.load_from_checkpoint(chkpt_path, **cls_kwargs)
    return model, chkpt_path


if __name__ == "__main__":
    # cli_main()
    chkpt_path = None
    load_path = os.path.join(LOGDIR, args.model, args.load) if args.load != "" else None
    _cls = get_model_cls(args.model)
    if load_path is not None and os.path.exists(load_path):
        model, chkpt_path = _load(
            _cls,
            load_path,
            load_from_pt=args.load_from_pt,
            load_epoch=args.load_epoch,
        )
    else:
        cls_kwargs = __make_cls_kwargs()
        model = _cls(**cls_kwargs)
        if args.kaiming:
            kaiming_init(model, a=0.1)

    log_path = os.path.join(LOGDIR, args.model, args.version)
    # summary(model, input_size=SAMPLE_INPUT.shape)
    logger = TensorBoardLogger(
        LOGDIR,
        name=args.model,
        log_graph=True,
        version=args.version,
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        accelerator="gpu",
        fast_dev_run=args.fast_dev,
        logger=logger,
        gradient_clip_val=args.grad_clip_val,
        gradient_clip_algorithm=args.grad_clip_alg,
        log_every_n_steps=10,
        sync_batchnorm=True,
        # strategy="ddp_find_unused_parameters_true",
        # overfit_batches=1,
        # detect_anomaly=True,
        callbacks=[
            # BatchSizeFinder(mode="binsearch", init_val=1000),
            # LearningRateFinder(min_lr=1e-6, max_lr=1e-1, num_training_steps=40),
            ModelSummary(max_depth=7),
            # BatchSizeScheduler(
            #     lambda epoch, min, max: min + (max - min) * (epoch + 1) / 100,
            #     max_batch_size=args.max_batch_size,
            #     min_batch_size=args.batch_size,
            # ),
            # BetaScheduler(T=50, R=0.5),
            # StochasticWeightAveraging(swa_epoch_start=100, swa_lrs=1e-3, device="cuda"),
        ],
        # deterministic=True,
    )

    transforms = getattr(model, "transforms", None)
    datamodule = get_data_module(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transforms=transforms,
        labels_path=args.labels_path,
        embedding_path=args.embedding_path,
    )
    if args.init_pseudos:
        datamodule.setup("fit")
        initial_vals = torch.empty((args.num_pseudos, *SAMPLE_INPUT.shape[1:]))
        dataset = iter(datamodule.train_dataloader().dataset)
        for i in range(args.num_pseudos):
            initial_vals[i] = torch.tensor(next(dataset)[0])
        model.init_pseudos(initial_vals)
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=chkpt_path if args.resume else None,
    )
