from torch import nn
import lightning as L
from .util import SBD, plot_to_image, generate_embedding, ScaleBlock, DeconvDecoder
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
from pytorch_metric_learning import losses, miners, reducers
import numpy as np
from torchvision.models import resnet50
from torchvision import transforms
from torch import Tensor
from typing import Literal


def get_model_cls(model_name) -> L.LightningModule:
    return {
        "vae": ModularVAE,
        "nvp": ModularNVP,
        "nvpw": ModularNVPW,
        "nvpwc": ModularNVPWC,
    }[model_name]


def get_encoder_cls(encoder_name) -> nn.Module:
    return {
        "basic": BasicEncoder,
        "resnet": ResnetEncoder,
    }[encoder_name]


def get_decoder_cls(decoder_name) -> nn.Module:
    return {
        "sbd": SBD,
    }[decoder_name]


# def get_model_cls(model_name) -> L.LightningModule:
#     return {
#         "vae": VAE,
#         "nvp": NVP,
#         "nvpw": NVPW,
#         "resnet": Resnet,
#         "resnet-nvp": ResnetNVP,
#         "resnet-50": Resnet50,
#     }[model_name]


def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    log_p = (
        -0.5 * torch.log(2.0 * torch.tensor(torch.pi))
        - 0.5 * log_var
        - 0.5 * torch.exp(-log_var) * (x - mu) ** 2.0
    )
    if reduction == "avg":
        return torch.mean(log_p, dim)
    elif reduction == "sum":
        return torch.sum(log_p, dim)
    else:
        return log_p


class ResnetEncoder(L.LightningModule):
    def __init__(
        self,
        sample_input_shape,
        blocks_per_scale=2,
        num_encoder_scales=4,
        num_decoder_scales=4,
        base_channels=8,
        encoder_depth_per_block=3,
        intermediate_channels=32,
        encoder_block_channels=3,
    ):
        super().__init__()
        self.sample_input_shape = sample_input_shape
        self.blocks_per_scale = blocks_per_scale
        self.num_encoder_scales = num_encoder_scales
        self.num_decoder_scales = num_decoder_scales
        self.base_channels = base_channels
        self.encoder_depth_per_block = encoder_depth_per_block
        self.intermediate_channels = intermediate_channels
        self.encoder_block_channels = encoder_block_channels

        self.scale_blocks = nn.ModuleList()
        in_channels = sample_input_shape[1]
        out_channels = base_channels
        for _ in range(num_encoder_scales):
            self.scale_blocks.append(
                ScaleBlock(
                    in_dim=in_channels,
                    mid_dim=intermediate_channels,
                    out_dim=out_channels,
                    block_dim=encoder_block_channels,
                    kernel_size=3,
                    block_per_scale=blocks_per_scale,
                    depth_per_block=encoder_depth_per_block,
                )
            )
            in_channels = out_channels
            out_channels *= 2
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)
        self.FC_shape = (
            (sample_input_shape[-1] // 2**self.num_encoder_scales) ** 2
            * 2 ** (self.num_encoder_scales - 1)
            * self.base_channels
        )

    def forward(self, x):
        for scale in self.scale_blocks:
            x = F.avg_pool2d(scale(x), (2, 2))

        return x.view(-1, self.FC_shape)


class BasicEncoder(L.LightningModule):
    def __init__(self, input_shape):
        super().__init__()
        self.input_channels = input_shape[1]
        self.input_length = input_shape[-1]
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 8, 3),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(8, 16, 2),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3),
            nn.LeakyReLU(),
        )
        self.finalConvLength = ((self.input_length - 2) // 2 - 1) // 2 - 4
        self.FC_shape = 64 * self.finalConvLength * self.finalConvLength

    def forward(self, x):
        return self.encoder(x).view(-1, self.FC_shape)


class ModularVAE(L.LightningModule):
    def __init__(
        self,
        sample_input,
        lsdim=16,
        beta=1,
        delta=1,
        use_labels=False,
        lr=1e-3,
        anneal_epochs=10,
        constant_lr=False,
        one_cycle=False,
        scheduler: Literal["one-cycle", "constant", "cosine"] = "one-cycle",
        momentum=0.9,
        weight_decay=4e-4,
        one_cycle_warmup=0.3,
        # data_mean: Tensor | None = torch.tensor((0.4914, 0.4822, 0.4465)),
        # data_std: Tensor | None = torch.tensor((0.247, 0.243, 0.261)),
        encoder_cls=None,
        decoder_cls=SBD,
        encoder_kwargs=None,
        decoder_kwargs=None,
    ):
        super().__init__()

        # self.register_buffer("data_mean", data_mean.view(1, 3, 1, 1))
        # self.register_buffer("data_std", data_std.view(1, 3, 1, 1))
        # self.transforms = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomVerticalFlip(),
        #         transforms.Normalize(mean=data_mean, std=data_std),
        #     ],
        # )

        self.example_input_array = sample_input
        self.input_length = sample_input.shape[-1]
        self.input_channels = sample_input.shape[1]
        self.lsdim = lsdim
        self.lr = lr
        self.use_labels = use_labels
        self.beta = beta
        self.anneal_epochs = anneal_epochs
        self.constant_lr = constant_lr
        self.delta = delta
        self.one_cycle = one_cycle
        self.scheduler = scheduler
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.one_cycle_warmup = one_cycle_warmup
        self.encoder_cls = encoder_cls
        self.decoder_cls = decoder_cls
        self.encoder_kwargs = encoder_kwargs if encoder_kwargs else {}
        self.decoder_kwargs = decoder_kwargs if decoder_kwargs else {}

        self.build_encoder()
        self.FC_shape = self.encoder.FC_shape
        self.build_decoder()

        self.log_gamma = nn.Parameter(torch.zeros(1))

        self.reducer = reducers.SumReducer()
        self.metric_loss_func = losses.NTXentLoss(temperature=0.1, reducer=self.reducer)
        self.miner = miners.BatchEasyHardMiner()

        self.save_hyperparameters()

    def build_decoder(self):
        self.decoder = self.decoder_cls(**self.decoder_kwargs)

    def build_encoder(self):
        self.encoder: nn.Module = self.encoder_cls(**self.encoder_kwargs)
        self.mean = nn.Linear(self.encoder.FC_shape, self.lsdim)
        self.logvar = nn.Linear(self.encoder.FC_shape, self.lsdim)

    def reconstruct_x(self, x):
        x_mean, _, _, _ = self.forward(x)
        return x_mean

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x):
        x = self.encoder(x)
        z_q_mean = self.mean(x)
        z_q_logvar = self.logvar(x)
        return (
            z_q_mean,
            z_q_logvar,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def output_activation(self, x):
        return F.sigmoid(x)

    # p(x|z)
    def p_x(self, z):
        return self.output_activation(self.decoder(z))

    def forward(self, x):

        # z~q(z|x)
        mu, logvar = self.q_z(x)
        z = self.reparameterize(mu, logvar)

        x_hat = self.p_x(z)
        return x_hat, mu, logvar, z

    def reconstruct(self, x):
        x_hat, *_ = self(x)
        return x_hat

    def metric_loss(self, z, labels):
        mined = self.miner(z, labels)
        return self.metric_loss_func(z, labels, mined)

    def mse_loss(self, x, x_hat):
        recon_loss = torch.square((x - x_hat) / torch.exp(self.log_gamma))
        recon_loss += self.log_gamma
        recon_loss /= 2
        return torch.sum(recon_loss)

    def recon_loss(self, x, x_hat):
        return self.mse_loss(x, x_hat)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(
        self,
        x,
        x_hat,
        mu,
        logvar,
        z,
        labels=None,
    ):
        recon_loss = self.recon_loss(x, x_hat)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        metric_loss = torch.tensor(0.0).to(self.device)
        if labels is not None:
            metric_loss = self.metric_loss(z, labels)

        # KL
        kl_loss = self.unit_kl(mu, logvar)

        return (
            recon_loss / x.size(0),
            kl_loss / x.size(0),
            metric_loss / x.size(0),
        )

    def general_kl(self, mu_1, logvar_1, mu_2, logvar_2):
        return -0.5 * torch.sum(
            1
            + logvar_1
            - logvar_2
            - (torch.square(mu_1 - mu_2) + torch.exp(logvar_1)) / torch.exp(logvar_2)
        )

    # KL Divergence between a parameterized gaussian, and a unit gaussian
    def unit_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - torch.square(mu) - torch.exp(logvar))

    def training_step(self, batch, batch_idx):
        x, labels = batch
        x_hat, mu_z, logvar, z = self(x)
        recon_loss, kl_loss, metric_loss = self.loss_function(
            x=x,
            x_hat=x_hat,
            mu=mu_z,
            logvar=logvar,
            z=z,
            labels=labels if self.use_labels else None,
        )

        loss = recon_loss + self.beta * kl_loss + self.delta * metric_loss
        if batch_idx % 10 == 0:
            logs = {
                "kl_loss": kl_loss,
                "recon_loss": recon_loss,
                "loss": loss,
                "metric_loss": metric_loss,
                "learning rate": self.trainer.optimizers[0].param_groups[0]["lr"],
                "raw_MSE": F.mse_loss(x_hat, x, reduction="mean"),
                "gamma": torch.exp(self.log_gamma),
            }
            self.log_dict(logs, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        if self.global_rank != 0:
            return

        tb = self.trainer.logger.experiment
        self.eval()
        fig, axes = plt.subplots(2, 4)
        sample_images = next(iter(self.trainer.train_dataloader))[0]
        sample_images = sample_images[: axes.shape[1]]
        reconstructions = self.reconstruct(sample_images.to(self.device))

        data_mean = self.data_mean.type_as(sample_images)
        data_std = self.data_std.type_as(sample_images)

        sample_images *= data_std
        sample_images += data_mean

        reconstructions *= data_std
        reconstructions += data_mean

        sample_images = sample_images.cpu().detach().numpy()
        reconstructions = reconstructions.cpu().detach().numpy()

        if reconstructions.shape[1] == 1:
            sample_images = sample_images.squeeze(1)
            reconstructions = reconstructions.squeeze(1)
        elif reconstructions.shape[1] == 3:
            sample_images = sample_images.transpose(0, 2, 3, 1)
            reconstructions = reconstructions.transpose(0, 2, 3, 1)
        for i in range(axes.shape[1]):
            axes[0, i].imshow(sample_images[i])
            axes[1, i].imshow(reconstructions[i])

        tb.add_image(
            f"Reconstructions",
            plot_to_image(fig),
            self.current_epoch,
        )
        if (
            self.local_rank == 0
            and self.current_epoch % 10 == 0
            and self.current_epoch > 0
        ):
            generate_embedding(
                self.trainer.train_dataloader,
                lambda x: self(x)[3],
                self.device,
                tb,
                self.current_epoch,
                mu_p=None,
                logvar_p=None,
            )
        self.train()

    def _make_optim_params(self):
        optim_params = [
            {"params": self.parameters(), "lr": self.lr},
        ]
        return optim_params

    def configure_optimizers(self):
        optim_params = self._make_optim_params()
        optim = torch.optim.SGD(
            optim_params,
            momentum=self.momentum,
            nesterov=self.momentum > 0,
            weight_decay=self.weight_decay,
        )
        if self.scheduler == "constant":
            return optim
        STEPS_PER_EPOCH = int(
            np.ceil((50000 // 4) / self.trainer.datamodule.batch_size)
        )
        WARMUP_PERIOD = 10
        lr_sched = {
            "one-cycle": {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optim,
                    [spec["lr"] for spec in optim_params],
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    max_momentum=self.momentum,
                    div_factor=25,
                    pct_start=self.one_cycle_warmup,
                ),
                "interval": "step",
            },
            "cosine": torch.optim.lr_scheduler.SequentialLR(
                optimizer=optim,
                schedulers=[
                    torch.optim.lr_scheduler.LambdaLR(
                        optimizer=optim,
                        lr_lambda=lambda epoch: (epoch + 1) / (WARMUP_PERIOD + 1),
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer=optim, T_max=self.anneal_epochs
                    ),
                ],
                milestones=[WARMUP_PERIOD],
            ),
            "plateau": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optim,
                    mode="min",
                    factor=0.85,
                    patience=5,
                ),
                "monitor": "loss",
                "interval": "epoch",
            },
        }[self.scheduler]
        return {"optimizer": optim, "lr_scheduler": lr_sched}

    def trainable_params(self):
        return [param for param in self.parameters() if param.requires_grad]


class ModularNVP(ModularVAE):
    def __init__(
        self,
        sample_input,
        num_pseudos=4,
        alpha=1,
        eta=0,
        sample_size=1,
        pseudo_lr=None,
        freeze_pseudos=0,
        train_pseudos=False,
        **kwargs,
    ):
        self.num_pseudos = num_pseudos
        self.eta = eta
        self.alpha = alpha
        self.sample_size = sample_size
        self.pseudo_lr = pseudo_lr
        self.freeze_pseudos = freeze_pseudos
        self.train_pseudos = train_pseudos

        super().__init__(
            sample_input=sample_input,
            **kwargs,
        )

        self.build_pseudos()
        self.empirical_freq = torch.zeros(self.num_pseudos)
        self.register_buffer(
            "log_n_pseudos",
            torch.log(torch.tensor(self.num_pseudos)),
        )
        if self.train_pseudos:
            self._setup_train_pseudos()

        self.save_hyperparameters()

    def _setup_train_pseudos(self):
        # We disable the encoder, but retain decoder for backprop to pseudos.
        for p in self.parameters():
            p.requires_grad_(True)
        self.encoder.requires_grad_(False)

    def _pseudos_params(self):
        return [self.pseudos]

    def build_pseudos(self):
        initial_val = torch.randn(
            (
                self.num_pseudos,
                self.input_channels,
                self.input_length,
                self.input_length,
            )
        )
        self.pseudos = nn.Parameter(initial_val, requires_grad=self.freeze_pseudos == 0)

    def gmm_likelihood(self, z, mean, logvar):
        z = z.unsqueeze(2)  # Pseudos dim
        mean = mean.unsqueeze(1)  # batch-size
        logvar = logvar.unsqueeze(1)  # batch-size

        reduced_log_likelihood = log_normal_diag(
            z, mean, logvar, reduction="sum", dim=-1
        )

        # (batch-size,)
        sample_log_likelihood = torch.logsumexp(
            reduced_log_likelihood - self.log_n_pseudos,
            dim=-1,
        )
        return sample_log_likelihood

    def posterior_likelihood(self, z, mean, logvar):
        return torch.mean(log_normal_diag(z, mean, logvar, reduction="sum", dim=-1))

    def log_p_z(self, z):
        # Generate posterios for pseudos to serve as components for the prior
        mu_p, logvar_p = self.q_z(self.get_pseudos())  # C x M
        return torch.mean(
            self.gmm_likelihood(z, mu_p.unsqueeze(0), logvar_p.unsqueeze(0))
        )

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(
        self,
        x,
        x_hat,
        mu,
        logvar,
        z,
        labels=None,
        reduction="mean",
    ):
        recon_loss = self.mse_loss(x, x_hat)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        metric_loss = torch.tensor(0.0).to(self.device)
        if labels is not None:
            metric_loss = self.metric_loss(z, labels)

        # KL
        if self.sample_size > 1:
            idle = torch.randn((self.sample_size - 1, *z.shape)).type_as(
                z
            )  # (sample_size, batch-size, lsdim)
            stochastic_samples = (
                mu.unsqueeze(0) + torch.exp(logvar.unsqueeze(0) / 2) * idle
            )
            stochastic_samples = torch.concat(
                [z.unsqueeze(0), stochastic_samples], dim=0
            )
        else:
            stochastic_samples = z.unsqueeze(0)
        log_p_z = self.log_p_z(stochastic_samples)
        log_q_z = self.posterior_likelihood(
            stochastic_samples, mu.unsqueeze(0), logvar.unsqueeze(0)
        )
        kl_loss = log_q_z - log_p_z

        if reduction == "mean":
            recon_loss /= x.size(0)

        return (
            recon_loss,
            kl_loss,
            metric_loss,
        )

    def pseudos_activation(self, x):
        return F.sigmoid(x)

    def get_pseudos(self):
        return self.pseudos_activation(self.pseudos)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        x_hat, mu_z, logvar, z = self(x)
        pseudos = self.get_pseudos()
        recon_pseudos, mu_p, logvar_p, z_p = self(pseudos)
        recon_loss, kl_loss, metric_loss = self.loss_function(
            x=x,
            x_hat=x_hat,
            mu=mu_z,
            logvar=logvar,
            z=z,
            labels=labels if self.use_labels else None,
        )

        pseudo_recon_loss, pseudo_kl_loss, _ = self.loss_function(
            x=pseudos,
            x_hat=recon_pseudos,
            mu=mu_p,
            logvar=logvar_p,
            z=z_p,
            labels=None,
        )
        loss = (
            (1 - self.eta) * (recon_loss + self.beta * kl_loss)
            + self.eta * pseudo_recon_loss
            + self.delta * metric_loss
        )
        if batch_idx % 10 == 0:
            logs = {
                "kl_loss": kl_loss,
                "recon_loss": recon_loss,
                "loss": loss,
                "metric_loss": metric_loss,
                "learning rate": self.trainer.optimizers[0].param_groups[0]["lr"],
                "raw_MSE": F.mse_loss(x_hat, x, reduction="mean"),
                "pseudo_MSE": F.mse_loss(recon_pseudos, pseudos, reduction="mean").to(
                    self.device
                )
                / self.num_pseudos,
                "pseudo_recon_loss": pseudo_recon_loss,
                "pseudo_kl_loss": pseudo_kl_loss,
                "beta": self.beta,
            }
            if hasattr(self, "log_gamma"):
                logs["gamma"] = torch.exp(self.log_gamma)
            self.log_dict(logs, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        if self.global_rank != 0:
            return
        tb = self.trainer.logger.experiment
        self.eval()
        fig, axes = plt.subplots(2, 4)
        sample_images = next(iter(self.trainer.train_dataloader))[0].to(self.device)
        sample_images = sample_images[: axes.shape[1]]
        reconstructions = self.reconstruct(sample_images)
        assert (
            reconstructions.device == sample_images.device
        ), f"{reconstructions.device} != {sample_images.device}"
        # data_mean = self.data_mean.type_as(sample_images)
        # data_std = self.data_std.type_as(sample_images)

        # sample_images *= data_std
        # sample_images += data_mean

        # reconstructions *= data_std
        # reconstructions += data_mean

        sample_images = sample_images.cpu().detach().numpy()
        reconstructions = reconstructions.cpu().detach().numpy()

        if reconstructions.shape[1] == 1:
            sample_images = sample_images.squeeze(1)
            reconstructions = reconstructions.squeeze(1)
        elif reconstructions.shape[1] == 3:
            sample_images = sample_images.transpose(0, 2, 3, 1)
            reconstructions = reconstructions.transpose(0, 2, 3, 1)
        for i in range(axes.shape[1]):
            axes[0, i].imshow(sample_images[i])
            axes[1, i].imshow(reconstructions[i])
        tb.add_image(
            f"Reconstructions",
            plot_to_image(fig),
            self.current_epoch,
        )

        if not self.train_pseudos:
            return

        grid_len = int(np.ceil(np.sqrt(self.num_pseudos)))
        fig, axes = plt.subplots(grid_len, grid_len)
        fig.set_size_inches(8, 8)
        pseudos = self.get_pseudos()
        for i in range(self.num_pseudos):
            sample_image = pseudos[i]
            row = i // grid_len
            col = i % grid_len
            if self.input_channels == 1:
                sample_image = sample_image.squeeze(0)
            else:
                sample_image = torch.permute(sample_image, (1, 2, 0))
            axes[row, col].imshow(sample_image.cpu().detach().numpy())

        tb.add_image(
            f"Pseudos",
            plot_to_image(fig),
            self.current_epoch,
        )

        self.train()

    def _make_optim_params(self):
        pseudo_lr = self.lr if self.pseudo_lr is None else self.pseudo_lr
        optim_params = [
            {"params": self.pseudos, "lr": pseudo_lr},
            {
                "params": [
                    *self.mean.parameters(),
                    *self.logvar.parameters(),
                    self.log_gamma,
                ],
                "lr": self.lr,
            },
        ]
        if not self.train_pseudos:
            optim_params.append(
                {
                    "params": [*self.encoder.parameters(), *self.decoder.parameters()],
                    "lr": self.lr,
                }
            )

        return optim_params


class ModularNVPW(ModularNVP):
    def __init__(
        self,
        sample_input,
        **kwargs,
    ):
        super().__init__(
            sample_input=sample_input,
            **kwargs,
        )

    def _make_optim_params(self):
        pseudo_lr = self.lr if self.pseudo_lr is None else self.pseudo_lr
        optim_params = [
            {
                "params": [self.pseudos],
                "lr": pseudo_lr,
            },
            {
                "params": [*self.encoder.parameters(), *self.w_evidence.parameters()],
                "lr": self.lr,
            },
        ]
        if not self.train_pseudos:
            optim_params.append(
                {
                    "params": [
                        *self.mean.parameters(),
                        *self.logvar.parameters(),
                        *self.decoder.parameters(),
                        self.log_gamma,
                    ],
                    "lr": self.lr,
                }
            )

        return optim_params

    def build_pseudos(self):
        super().build_pseudos()

        self.w_evidence = nn.Linear(self.FC_shape, self.num_pseudos)
        self.register_buffer("log_w_min", torch.tensor(-5))

    def get_log_w(self, x):
        return F.log_softmax(self.w_evidence(x), dim=1)

    def log_p_z(self, z, log_w):
        # Generate posterios for pseudos to serve as components for the prior
        mu_p, logvar_p, _ = self.q_z(self.get_pseudos())  # C x M
        return torch.mean(
            self.gmm_likelihood(z, mu_p.unsqueeze(0), logvar_p.unsqueeze(0), log_w)
        )

    def q_z(self, x):
        x = self.encoder(x)
        z_q_mean = self.mean(x)
        z_q_logvar = self.logvar(x)
        return z_q_mean, z_q_logvar, self.get_log_w(x)

    def forward(self, x):

        # z~q(z|x)
        mu, logvar, log_w = self.q_z(x)
        z = self.reparameterize(mu, logvar)

        x_hat = self.p_x(z)
        # decode code
        return x_hat, mu, logvar, z, log_w

    def gmm_likelihood(self, z, mean, logvar, log_w):
        z = z.unsqueeze(2)  # Pseudos dim
        mean = mean.unsqueeze(1)  # batch-size
        logvar = logvar.unsqueeze(1)  # batch-size

        reduced_log_likelihood = log_normal_diag(
            z, mean, logvar, reduction="sum", dim=-1
        )

        # (batch-size,)
        sample_log_likelihood = torch.logsumexp(
            reduced_log_likelihood + log_w,
            dim=-1,
        )
        return sample_log_likelihood

    def training_step(self, batch, batch_idx):
        x, labels = batch
        x_hat, mu_z, logvar, z, log_w = self(x)
        pseudos = self.get_pseudos()
        recon_pseudos, mu_p, logvar_p, z_p, pseudo_log_w = self(pseudos)
        recon_loss, kl_loss, metric_loss = self.loss_function(
            x=x,
            x_hat=x_hat,
            log_w=log_w,
            mu=mu_z,
            logvar=logvar,
            z=z,
            labels=labels if self.use_labels else None,
        )

        pseudo_recon_loss, pseudo_kl_loss, _ = self.loss_function(
            x=pseudos,
            x_hat=recon_pseudos,
            log_w=pseudo_log_w,
            mu=mu_p,
            logvar=logvar_p,
            z=z_p,
            labels=None,
        )
        # We want to minimize sample entropy (i.e. every sample should have a
        # sharp preference for its prior) while maximizing batch_entropy (i.e.
        # across the batch, the priors should be diverse)
        batch_entropy = -torch.mean(torch.sum(torch.exp(log_w) * log_w, dim=1))
        aggregate_log_w = torch.mean(log_w, dim=0)
        sample_entropy = -torch.sum(torch.exp(aggregate_log_w) * aggregate_log_w)
        entropy_loss = -batch_entropy
        loss = (
            (1 - self.eta) * (recon_loss + self.beta * kl_loss)
            + self.eta * (pseudo_recon_loss + self.beta * pseudo_kl_loss)
            + self.delta * metric_loss
            + self.alpha * entropy_loss
        )
        if batch_idx % 10 == 0:
            logs = {
                "kl_loss": kl_loss,
                "recon_loss": recon_loss,
                "loss": loss,
                "metric_loss": metric_loss,
                "learning rate": self.trainer.optimizers[0].param_groups[-1]["lr"],
                "raw_MSE": F.mse_loss(x_hat, x, reduction="mean"),
                "pseudo_MSE": F.mse_loss(recon_pseudos, pseudos, reduction="mean"),
                "pseudo_recon_loss": pseudo_recon_loss,
                "pseudo_kl_loss": pseudo_kl_loss,
                "beta": self.beta,
                "batch_entropy": batch_entropy,
                "sample_entropy": sample_entropy,
                "entropy_loss": entropy_loss,
            }
            if hasattr(self, "log_gamma"):
                logs["gamma"] = torch.exp(self.log_gamma)
            self.log_dict(logs, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def loss_function(
        self,
        x,
        x_hat,
        mu,
        logvar,
        z,
        labels=None,
        reduction="mean",
        log_w=None,
    ):
        recon_loss = self.mse_loss(x, x_hat)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        metric_loss = torch.tensor(0.0).to(self.device)
        if labels is not None:
            metric_loss = self.metric_loss(z, labels)

        # KL
        if self.sample_size > 1:
            idle = torch.randn((self.sample_size - 1, *z.shape)).type_as(
                z
            )  # (sample_size, batch-size, lsdim)
            stochastic_samples = (
                mu.unsqueeze(0) + torch.exp(logvar.unsqueeze(0) / 2) * idle
            )
            stochastic_samples = torch.concat(
                [z.unsqueeze(0), stochastic_samples], dim=0
            )
        else:
            stochastic_samples = z.unsqueeze(0)
        log_p_z = self.log_p_z(stochastic_samples, log_w)
        log_q_z = self.posterior_likelihood(
            stochastic_samples, mu.unsqueeze(0), logvar.unsqueeze(0)
        )
        kl_loss = log_q_z - log_p_z

        if reduction == "mean":
            recon_loss /= x.size(0)

        return (
            recon_loss,
            kl_loss,
            metric_loss,
        )


class ModularNVPWC(ModularNVPW):
    def __init__(
        self,
        sample_input,
        **kwargs,
    ):
        super().__init__(
            sample_input=sample_input,
            **kwargs,
        )

    def build_pseudos(self):
        ModularNVP.build_pseudos(self)
        self.w_evidence = nn.Parameter(torch.zeros(self.num_pseudos))

    def get_log_w(self, x):
        return F.log_softmax(self.w_evidence, dim=0).unsqueeze(0)

    def _make_optim_params(self):
        pseudo_lr = self.lr if self.pseudo_lr is None else self.pseudo_lr
        optim_params = [
            {
                "params": [self.pseudos],
                "lr": pseudo_lr,
            },
            {
                "params": [*self.encoder.parameters(), self.w_evidence],
                "lr": self.lr,
            },
        ]
        if not self.train_pseudos:
            optim_params[1]["params"].extend(
                [
                    *self.mean.parameters(),
                    *self.logvar.parameters(),
                    *self.decoder.parameters(),
                    self.log_gamma,
                ]
            )

        return optim_params


class VAE(L.LightningModule):
    """
    :math:`\gamma` - simple
    """

    def __init__(
        self,
        sample_input,
        lsdim=16,
        beta=1,
        delta=1,
        use_labels=False,
        lr=1e-3,
        sbd_kernel_size=3,
        anneal_epochs=10,
        num_decoder_blocks=6,
        num_decoder_layers_per_block=3,
        sbd_channels=16,
        constant_lr=False,
        one_cycle=False,
        scheduler: Literal["one-cycle", "constant", "cosine"] = "one-cycle",
        momentum=0.9,
        one_cycle_warmup=0.3,
        **kwargs,
    ):
        super().__init__()

        self.example_input_array = sample_input
        self.input_length = sample_input.shape[-1]
        self.input_channels = sample_input.shape[1]
        self.lsdim = lsdim
        self.lr = lr
        self.use_labels = use_labels
        self.beta = beta
        self.sbd_kernel_size = sbd_kernel_size
        self.anneal_epochs = anneal_epochs
        self.constant_lr = constant_lr
        self.delta = delta
        self.num_decoder_blocks = num_decoder_blocks
        self.num_decoder_layers_per_block = num_decoder_layers_per_block
        self.sbd_channels = sbd_channels
        self.one_cycle = one_cycle
        self.scheduler = scheduler
        self.momentum = momentum
        self.one_cycle_warmup = one_cycle_warmup

        self.build_encoder()
        self.build_decoder()

        self.log_gamma = nn.Parameter(torch.zeros(1))

        self.reducer = reducers.SumReducer()
        self.metric_loss_func = losses.NTXentLoss(temperature=0.1, reducer=self.reducer)
        self.miner = miners.BatchEasyHardMiner()

        self.save_hyperparameters()

    def build_decoder(self):
        self.sbd = SBD(
            input_shape=self.example_input_array.shape,
            lsdim=self.lsdim,
            num_blocks=self.num_decoder_blocks,
            kernel_size=self.sbd_kernel_size,
            layers_per_block=self.num_decoder_layers_per_block,
            channels_per_layer=self.sbd_channels,
        )

    def build_encoder(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 8, 3),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(8, 16, 2),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3),
            nn.LeakyReLU(),
        )
        self.finalConvLength = ((self.input_length - 2) // 2 - 1) // 2 - 4
        self.FC_shape = 64 * self.finalConvLength * self.finalConvLength
        self.mean = nn.Linear(self.FC_shape, self.lsdim)
        self.logvar = nn.Linear(self.FC_shape, self.lsdim)

    def reconstruct_x(self, x):
        x_mean, _, _, _ = self.forward(x)
        return x_mean

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.FC_shape)

        z_q_mean = self.mean(x)
        z_q_logvar = self.logvar(x)
        return (
            z_q_mean,
            z_q_logvar,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def output_activation(self, x):
        return F.sigmoid(x)

    # p(x|z)
    def p_x(self, z):
        return self.output_activation(self.sbd(z))

    def forward(self, x):

        # z~q(z|x)
        mu, logvar = self.q_z(x)
        z = self.reparameterize(mu, logvar)

        x_hat = self.p_x(z)
        return x_hat, mu, logvar, z

    def reconstruct(self, x):
        x_hat, *_ = self(x)
        return x_hat

    def metric_loss(self, z, labels):
        mined = self.miner(z, labels)
        return self.metric_loss_func(z, labels, mined)

    def mse_loss(self, x, x_hat):
        recon_loss = torch.square((x - x_hat) / torch.exp(self.log_gamma))
        recon_loss += self.log_gamma
        recon_loss /= 2
        return torch.sum(recon_loss)

    def recon_loss(self, x, x_hat):
        return self.mse_loss(x, x_hat)
        # return F.binary_cross_entropy(x, x_hat, reduction="sum")

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(
        self,
        x,
        x_hat,
        mu,
        logvar,
        z,
        labels=None,
    ):
        recon_loss = self.recon_loss(x, x_hat)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        metric_loss = torch.tensor(0.0).to(self.device)
        if labels is not None:
            metric_loss = self.metric_loss(z, labels)

        # KL
        kl_loss = self.unit_kl(mu, logvar)

        return (
            recon_loss / x.size(0),
            kl_loss / x.size(0),
            metric_loss / x.size(0),
        )

    def general_kl(self, mu_1, logvar_1, mu_2, logvar_2):
        return -0.5 * torch.sum(
            1
            + logvar_1
            - logvar_2
            - (torch.square(mu_1 - mu_2) + torch.exp(logvar_1)) / torch.exp(logvar_2)
        )

    # KL Divergence between a parameterized gaussian, and a unit gaussian
    def unit_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - torch.square(mu) - torch.exp(logvar))

    def training_step(self, batch, batch_idx):
        x, labels = batch
        x_hat, mu_z, logvar, z = self(x)
        recon_loss, kl_loss, metric_loss = self.loss_function(
            x=x,
            x_hat=x_hat,
            mu=mu_z,
            logvar=logvar,
            z=z,
            labels=labels if self.use_labels else None,
        )

        loss = recon_loss + self.beta * kl_loss + self.delta * metric_loss
        if batch_idx % 10 == 0:
            batch_size = x.size(0)
            logs = {
                "kl_loss": kl_loss,
                "recon_loss": recon_loss,
                "loss": loss,
                "metric_loss": metric_loss,
                "learning rate": self.trainer.optimizers[0].param_groups[0]["lr"],
                "raw_MSE": F.mse_loss(x_hat, x, reduction="sum") / batch_size,
                "gamma": torch.exp(self.log_gamma),
            }
            self.log_dict(logs, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        tb = self.trainer.logger.experiment
        self.eval()
        fig, axes = plt.subplots(2, 4)
        sample_images = next(iter(self.trainer.train_dataloader))[0]
        sample_images = sample_images[: axes.shape[1]]
        reconstructions = self.reconstruct(sample_images.to(self.device))

        sample_images = sample_images.cpu().detach().numpy()
        reconstructions = reconstructions.cpu().detach().numpy()

        if reconstructions.shape[1] == 1:
            sample_images = sample_images.squeeze(1)
            reconstructions = reconstructions.squeeze(1)
        elif reconstructions.shape[1] == 3:
            sample_images = sample_images.transpose(0, 2, 3, 1)
            reconstructions = reconstructions.transpose(0, 2, 3, 1)
        for i in range(axes.shape[1]):
            axes[0, i].imshow(sample_images[i])
            axes[1, i].imshow(reconstructions[i])

        tb.add_image(
            f"Reconstructions",
            plot_to_image(fig),
            self.current_epoch,
        )
        if (
            self.local_rank == 0
            and self.current_epoch % 10 == 0
            and self.current_epoch > 0
        ):
            generate_embedding(
                self.trainer.train_dataloader,
                lambda x: self(x)[3],
                self.device,
                tb,
                self.current_epoch,
                mu_p=None,
                logvar_p=None,
            )
        self.train()

    def configure_optimizers(self):
        # optim = torch.optim.NAdam(
        #     self.trainable_params(),
        #     lr=self.lr,
        #     weight_decay=0.00002,
        # )

        optim = torch.optim.SGD(
            self.trainable_params(),
            lr=self.lr,
            momentum=self.momentum,
            nesterov=True,
            weight_decay=0.02,
        )
        if self.scheduler == "constant":
            return optim
        STEPS_PER_EPOCH = int(
            np.ceil((50000 // 4) / self.trainer.datamodule.batch_size)
        )
        WARMUP_PERIOD = 10
        lr_sched = {
            "one-cycle": {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optim,
                    self.lr,
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    div_factor=25,
                    pct_start=self.one_cycle_warmup,
                ),
                "interval": "step",
            },
            "cosine": torch.optim.lr_scheduler.SequentialLR(
                optimizer=optim,
                schedulers=[
                    torch.optim.lr_scheduler.LambdaLR(
                        optimizer=optim,
                        lr_lambda=lambda epoch: (epoch + 1) / (WARMUP_PERIOD + 1),
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer=optim, T_max=self.anneal_epochs
                    ),
                ],
                milestones=[WARMUP_PERIOD],
            ),
            "plateau": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optim,
                    mode="min",
                    factor=0.85,
                    patience=5,
                ),
                "monitor": "loss",
                "interval": "epoch",
            },
        }[self.scheduler]
        return {"optimizer": optim, "lr_scheduler": lr_sched}

    def trainable_params(self):
        return [param for param in self.parameters() if param.requires_grad]


class NVP(VAE):
    def __init__(
        self,
        sample_input,
        num_pseudos=4,
        alpha=1,
        eta=0,
        sample_size=1,
        pseudo_lr=None,
        freeze_pseudos=0,
        train_pseudos=False,
        **kwargs,
    ):
        self.num_pseudos = num_pseudos
        self.eta = eta
        self.alpha = alpha
        self.sample_size = sample_size
        self.pseudo_lr = pseudo_lr
        self.freeze_pseudos = freeze_pseudos
        self.train_pseudos = train_pseudos

        super().__init__(
            sample_input=sample_input,
            **kwargs,
        )

        self.build_pseudos()
        self.empirical_freq = torch.zeros(self.num_pseudos)
        self.register_buffer(
            "log_n_pseudos",
            torch.log(torch.tensor(self.num_pseudos)),
        )
        if self.train_pseudos:
            self._setup_train_pseudos()

        self.save_hyperparameters()

    def _setup_train_pseudos(self):
        # We disable the encoder, relying on it to implicitly regularize the
        # encoding and learned manifold.
        for p in self.parameters():
            p.requires_grad_(True)
        self.sbd.requires_grad_(False)

    def _pseudos_params(self):
        return [self.pseudos]

    def build_pseudos(self):
        initial_val = torch.randn(
            (
                self.num_pseudos,
                self.input_channels,
                self.input_length,
                self.input_length,
            )
        )
        self.pseudos = nn.Parameter(initial_val, requires_grad=self.freeze_pseudos == 0)

    def gmm_likelihood(self, z, mean, logvar):
        z = z.unsqueeze(2)  # Pseudos dim
        mean = mean.unsqueeze(1)  # batch-size
        logvar = logvar.unsqueeze(1)  # batch-size

        reduced_log_likelihood = log_normal_diag(
            z, mean, logvar, reduction="sum", dim=-1
        )

        # # (batch-size, num-pseudos, lsdim)
        # log_likelihood = -(logvar + torch.square(z - mean) / torch.exp(logvar)) / 2

        # # (batch-size, num-pseudos)
        # reduced_log_likelihood = torch.sum(log_likelihood, 2)

        # (batch-size,)
        sample_log_likelihood = torch.logsumexp(
            reduced_log_likelihood - self.log_n_pseudos,
            dim=-1,
        )
        return sample_log_likelihood

    def posterior_likelihood(self, z, mean, logvar):
        return torch.mean(log_normal_diag(z, mean, logvar, reduction="sum", dim=-1))

    def log_p_z(self, z):
        # Generate posterios for pseudos to serve as components for the prior
        mu_p, logvar_p = self.q_z(self.get_pseudos())  # C x M
        return torch.mean(
            self.gmm_likelihood(z, mu_p.unsqueeze(0), logvar_p.unsqueeze(0))
        )

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(
        self,
        x,
        x_hat,
        mu,
        logvar,
        z,
        labels=None,
        reduction="mean",
    ):
        recon_loss = self.mse_loss(x, x_hat)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        metric_loss = torch.tensor(0.0).to(self.device)
        if labels is not None:
            metric_loss = self.metric_loss(z, labels)

        # KL
        if self.sample_size > 1:
            idle = torch.randn((self.sample_size - 1, *z.shape)).type_as(
                z
            )  # (sample_size, batch-size, lsdim)
            stochastic_samples = (
                mu.unsqueeze(0) + torch.exp(logvar.unsqueeze(0) / 2) * idle
            )
            stochastic_samples = torch.concat(
                [z.unsqueeze(0), stochastic_samples], dim=0
            )
        else:
            stochastic_samples = z.unsqueeze(0)
        log_p_z = self.log_p_z(stochastic_samples)
        log_q_z = self.posterior_likelihood(
            stochastic_samples, mu.unsqueeze(0), logvar.unsqueeze(0)
        )
        kl_loss = log_q_z - log_p_z

        if reduction == "mean":
            recon_loss /= x.size(0)

        return (
            recon_loss,
            kl_loss,
            metric_loss,
        )

    def get_pseudos(self):
        return self.output_activation(self.pseudos)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        x_hat, mu_z, logvar, z = self(x)
        pseudos = self.get_pseudos()
        recon_pseudos, mu_p, logvar_p, z_p = self(pseudos)
        recon_loss, kl_loss, metric_loss = self.loss_function(
            x=x,
            x_hat=x_hat,
            mu=mu_z,
            logvar=logvar,
            z=z,
            labels=labels if self.use_labels else None,
        )

        pseudo_recon_loss, pseudo_kl_loss, _ = self.loss_function(
            x=pseudos,
            x_hat=recon_pseudos,
            mu=mu_p,
            logvar=logvar_p,
            z=z_p,
            labels=None,
        )
        loss = (
            (1 - self.eta) * (recon_loss + self.beta * kl_loss)
            + self.eta * pseudo_recon_loss
            + self.delta * metric_loss
        )
        if batch_idx % 10 == 0:
            batch_size = x.size(0)
            logs = {
                "kl_loss": kl_loss.to(self.device),
                "recon_loss": recon_loss.to(self.device),
                "loss": loss.to(self.device),
                "metric_loss": metric_loss.to(self.device),
                "learning rate": self.trainer.optimizers[0].param_groups[0]["lr"],
                "raw_MSE": F.mse_loss(x_hat, x, reduction="sum").to(self.device)
                / batch_size,
                "pseudo_MSE": F.mse_loss(recon_pseudos, pseudos, reduction="sum").to(
                    self.device
                )
                / self.num_pseudos,
                "pseudo_recon_loss": pseudo_recon_loss.to(self.device),
                "pseudo_kl_loss": pseudo_kl_loss.to(self.device),
                "beta": self.beta,
            }
            if hasattr(self, "log_gamma"):
                logs["gamma"] = torch.exp(self.log_gamma).to(self.device)
            self.log_dict(logs, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        if self.global_rank != 0:
            return

        tb = self.trainer.logger.experiment
        self.eval()
        fig, axes = plt.subplots(2, 4)
        sample_images = next(iter(self.trainer.train_dataloader))[0]
        sample_images = sample_images[: axes.shape[1]]
        reconstructions = self.reconstruct(sample_images.to(self.device))

        sample_images = sample_images.cpu().detach().numpy()
        reconstructions = reconstructions.cpu().detach().numpy()

        if reconstructions.shape[1] == 1:
            sample_images = sample_images.squeeze(1)
            reconstructions = reconstructions.squeeze(1)
        elif reconstructions.shape[1] == 3:
            sample_images = sample_images.transpose(0, 2, 3, 1)
            reconstructions = reconstructions.transpose(0, 2, 3, 1)
        for i in range(axes.shape[1]):
            axes[0, i].imshow(sample_images[i])
            axes[1, i].imshow(reconstructions[i])
        tb.add_image(
            f"Reconstructions",
            plot_to_image(fig),
            self.current_epoch,
        )
        grid_len = int(np.ceil(np.sqrt(self.num_pseudos)))
        fig, axes = plt.subplots(grid_len, grid_len)
        pseudos = self.get_pseudos()
        for i in range(self.num_pseudos):
            sample_image = pseudos[i]
            row = i // grid_len
            col = i % grid_len
            if self.input_channels == 1:
                sample_image = sample_image.squeeze(0)
            else:
                sample_image = torch.permute(sample_image, (1, 2, 0))
            axes[row, col].imshow(sample_image.cpu().detach().numpy())

        tb.add_image(
            f"Pseudos",
            plot_to_image(fig),
            self.current_epoch,
        )

        # if (
        #     self.local_rank == 0
        #     and self.current_epoch % 10 == 0
        #     and self.current_epoch > 0
        # ):
        #     _, mu_p, logvar_p, *_ = self(pseudos)
        #     generate_embedding(
        #         self.trainer.train_dataloader,
        #         lambda x: self(x)[3],
        #         self.device,
        #         tb,
        #         self.current_epoch,
        #         mu_p=mu_p,
        #         logvar_p=logvar_p,
        #     )
        self.train()

    def _make_optim_params(self):
        pseudo_lr = self.lr if self.pseudo_lr is None else self.pseudo_lr
        base_params = [
            param
            for param in self.parameters()
            if id(param) not in map(id, self._pseudos_params())
        ]

        optim_params = [
            {"params": self._pseudos_params(), "lr": pseudo_lr},
        ]
        if not self.train_pseudos:
            optim_params.append({"params": base_params, "lr": self.lr})

        return optim_params

    def configure_optimizers(self):
        # optim = torch.optim.NAdam(
        #     self.trainable_params(),
        #     lr=self.lr,
        #     weight_decay=0.00002,
        # )
        optim_params = self._make_optim_params()

        optim = torch.optim.SGD(
            optim_params,
            momentum=self.momentum,
            nesterov=self.momentum > 0,
            weight_decay=0.0004,
        )
        if self.scheduler == "constant":
            return optim
        STEPS_PER_EPOCH = int(
            np.ceil((50000 // 4) / self.trainer.datamodule.batch_size)
        )
        WARMUP_PERIOD = 10
        lr_sched = {
            "one-cycle": {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optim,
                    [spec["lr"] for spec in optim_params],
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    max_momentum=self.momentum,
                    div_factor=100,
                    pct_start=self.one_cycle_warmup,
                ),
                "interval": "step",
            },
            "cosine": torch.optim.lr_scheduler.SequentialLR(
                optimizer=optim,
                schedulers=[
                    torch.optim.lr_scheduler.LambdaLR(
                        optimizer=optim,
                        lr_lambda=lambda epoch: (epoch + 1) / (WARMUP_PERIOD + 1),
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer=optim, T_max=self.anneal_epochs
                    ),
                ],
                milestones=[WARMUP_PERIOD],
            ),
            "plateau": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optim,
                    mode="min",
                    factor=0.85,
                    patience=5,
                ),
                "monitor": "loss",
                "interval": "epoch",
            },
        }[self.scheduler]
        return {"optimizer": optim, "lr_scheduler": lr_sched}


class NVPW(NVP):
    def __init__(
        self,
        sample_input,
        **kwargs,
    ):
        super().__init__(
            sample_input=sample_input,
            **kwargs,
        )

    def _make_optim_params(self):
        pseudo_lr = self.lr if self.pseudo_lr is None else self.pseudo_lr
        base_params = [
            *self.encoder.parameters(),
            *self.mean.parameters(),
            *self.logvar.parameters(),
            *self.sbd.parameters(),
        ]
        optim_params = [
            {
                "params": [self.pseudos],
                "lr": pseudo_lr,
            },
            {"params": self.w_evidence.parameters(), "lr": self.lr},
        ]
        if not self.train_pseudos:
            optim_params.append({"params": base_params, "lr": self.lr})

        return optim_params

    def _pseudos_params(self):
        return

    def build_pseudos(self):
        super().build_pseudos()
        # self.w_evidence = nn.Parameter(torch.zeros(self.num_pseudos))
        self.w_evidence = nn.Linear(self.FC_shape, self.num_pseudos)
        self.register_buffer("log_w_min", torch.tensor(-5))

    def get_log_w(self, x):
        return F.log_softmax(self.w_evidence(x), dim=1)

    def log_p_z(self, z, log_w):
        # Generate posterios for pseudos to serve as components for the prior
        mu_p, logvar_p, _ = self.q_z(self.get_pseudos())  # C x M
        return torch.mean(
            self.gmm_likelihood(z, mu_p.unsqueeze(0), logvar_p.unsqueeze(0), log_w)
        )

    def q_z(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.FC_shape)
        z_q_mean = self.mean(x)
        z_q_logvar = self.logvar(x)
        return z_q_mean, z_q_logvar, self.get_log_w(x)

    def forward(self, x):

        # z~q(z|x)
        mu, logvar, log_w = self.q_z(x)
        z = self.reparameterize(mu, logvar)

        x_hat = self.p_x(z)
        # decode code
        return x_hat, mu, logvar, z, log_w

    def gmm_likelihood(self, z, mean, logvar, log_w):
        z = z.unsqueeze(2)  # Pseudos dim
        mean = mean.unsqueeze(1)  # batch-size
        logvar = logvar.unsqueeze(1)  # batch-size
        if log_w is None:
            log_w = self.log_n_pseudos

        reduced_log_likelihood = log_normal_diag(
            z, mean, logvar, reduction="sum", dim=-1
        )

        # # (batch-size, num-pseudos, lsdim)
        # log_likelihood = -(logvar + torch.square(z - mean) / torch.exp(logvar)) / 2

        # # (batch-size, num-pseudos)
        # reduced_log_likelihood = torch.sum(log_likelihood, 2)

        # (batch-size,)
        sample_log_likelihood = torch.logsumexp(
            reduced_log_likelihood + log_w,
            dim=-1,
        )
        return sample_log_likelihood

    def training_step(self, batch, batch_idx):
        x, labels = batch
        x_hat, mu_z, logvar, z, log_w = self(x)
        pseudos = self.get_pseudos()
        recon_pseudos, mu_p, logvar_p, z_p, _ = self(pseudos)
        recon_loss, kl_loss, metric_loss = self.loss_function(
            x=x,
            x_hat=x_hat,
            log_w=log_w,
            mu=mu_z,
            logvar=logvar,
            z=z,
            labels=labels if self.use_labels else None,
        )

        pseudo_recon_loss, pseudo_kl_loss, _ = self.loss_function(
            x=pseudos,
            x_hat=recon_pseudos,
            mu=mu_p,
            logvar=logvar_p,
            z=z_p,
            labels=None,
        )
        # We want to minimize sample entropy (i.e. every sample should have a
        # sharp preference for its prior) while maximizing batch_entropy (i.e.
        # across the batch, the priors should be diverse)
        batch_entropy = -torch.mean(torch.sum(torch.exp(log_w) * log_w, dim=1))
        aggregate_log_w = torch.mean(log_w, dim=0)
        sample_entropy = -torch.sum(torch.exp(aggregate_log_w) * aggregate_log_w)
        entropy_loss = sample_entropy - batch_entropy
        loss = (
            (1 - self.eta) * (recon_loss + self.beta * kl_loss)
            + self.eta * (pseudo_recon_loss + self.beta * pseudo_kl_loss)
            + self.delta * metric_loss
            + self.alpha * entropy_loss
        )
        if batch_idx % 10 == 0:
            logs = {
                "kl_loss": kl_loss,
                "recon_loss": recon_loss,
                "loss": loss,
                "metric_loss": metric_loss,
                "learning rate": self.trainer.optimizers[0].param_groups[0]["lr"],
                "raw_MSE": F.mse_loss(x_hat, x, reduction="mean"),
                "pseudo_MSE": F.mse_loss(recon_pseudos, pseudos, reduction="mean"),
                "pseudo_recon_loss": pseudo_recon_loss,
                "pseudo_kl_loss": pseudo_kl_loss,
                "beta": self.beta,
                "batch_entropy": batch_entropy,
                "sample_entropy": sample_entropy,
                "entropy_loss": entropy_loss,
            }
            if hasattr(self, "log_gamma"):
                logs["gamma"] = torch.exp(self.log_gamma)
            self.log_dict(logs, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def loss_function(
        self,
        x,
        x_hat,
        mu,
        logvar,
        z,
        labels=None,
        reduction="mean",
        log_w=None,
    ):
        recon_loss = self.mse_loss(x, x_hat)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        metric_loss = torch.tensor(0.0).to(self.device)
        if labels is not None:
            metric_loss = self.metric_loss(z, labels)

        # KL
        if self.sample_size > 1:
            idle = torch.randn((self.sample_size - 1, *z.shape)).type_as(
                z
            )  # (sample_size, batch-size, lsdim)
            stochastic_samples = (
                mu.unsqueeze(0) + torch.exp(logvar.unsqueeze(0) / 2) * idle
            )
            stochastic_samples = torch.concat(
                [z.unsqueeze(0), stochastic_samples], dim=0
            )
        else:
            stochastic_samples = z.unsqueeze(0)
        log_p_z = self.log_p_z(stochastic_samples, log_w)
        log_q_z = self.posterior_likelihood(
            stochastic_samples, mu.unsqueeze(0), logvar.unsqueeze(0)
        )
        kl_loss = log_q_z - log_p_z

        if reduction == "mean":
            recon_loss /= x.size(0)

        return (
            recon_loss,
            kl_loss,
            metric_loss,
        )


class Resnet50(VAE):
    def __init__(
        self,
        sample_input: Tensor,
        pretrained: bool = True,
        freeze_gamma: int = 0,
        freeze_encoder: int = 0,
        data_mean: Tensor | None = torch.tensor((0.4914, 0.4822, 0.4465)),
        data_std: Tensor | None = torch.tensor((0.247, 0.243, 0.261)),
        **kwargs,
    ):
        self.pretrained = pretrained
        self.freeze_gamma = freeze_gamma
        self.freeze_encoder = freeze_encoder
        super().__init__(sample_input, **kwargs)

        self.register_buffer("data_mean", data_mean.view(1, 3, 1, 1))
        self.register_buffer("data_std", data_std.view(1, 3, 1, 1))
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(mean=data_mean, std=data_std),
            ],
        )
        if self.freeze_gamma != 0:
            self.log_gamma.requires_grad_(False)
        if self.freeze_encoder != 0:
            self.encoder.requires_grad_(False)

    def build_encoder(self):
        self.encoder = resnet50(
            weights=torch.load("resnet_c10_aa_ls.pth") if self.pretrained else None
        )
        self.FC_shape = self.encoder.fc.in_features
        self.mean = nn.Linear(self.FC_shape, self.lsdim)
        self.logvar = nn.Linear(self.FC_shape, self.lsdim)
        delattr(self.encoder, "fc")

    def q_z(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)

        z_q_mean = self.mean(x)
        z_q_logvar = self.logvar(x)
        return z_q_mean, z_q_logvar

    def output_activation(self, x):
        return x

    def p_x(self, z):
        return self.output_activation(self.sbd(z))

    def training_step(self, batch, batch_idx):
        x, labels = batch
        x_hat, mu_z, logvar, z = self(x)
        recon_loss, kl_loss, metric_loss = self.loss_function(
            x=x,
            x_hat=x_hat,
            mu=mu_z,
            logvar=logvar,
            z=z,
            labels=labels if self.use_labels else None,
        )

        loss = recon_loss + self.beta * kl_loss + self.delta * metric_loss
        if batch_idx % 10 == 0:
            logs = {
                "kl_loss": kl_loss,
                "recon_loss": recon_loss,
                "loss": loss,
                "metric_loss": metric_loss,
                "learning rate": self.trainer.optimizers[0].param_groups[0]["lr"],
                "raw_MSE": F.mse_loss(x_hat, x, reduction="mean"),
                "gamma": torch.exp(self.log_gamma),
            }
            self.log_dict(logs, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        if self.current_epoch == self.freeze_gamma:
            self.log_gamma.requires_grad_(True)
        if self.current_epoch == self.freeze_encoder:
            self.encoder.requires_grad_(True)
        if self.local_rank != 0:
            return

        tb = self.trainer.logger.experiment
        self.eval()
        fig, axes = plt.subplots(2, 4)
        sample_images = next(iter(self.trainer.train_dataloader))[0]
        num_samples = axes.shape[1]
        sample_images = sample_images[:num_samples].to(self.device)
        reconstructions = self.reconstruct(sample_images)
        data_mean = self.data_mean.type_as(sample_images)
        data_std = self.data_std.type_as(sample_images)

        sample_images *= data_std
        sample_images += data_mean

        reconstructions *= data_std
        reconstructions += data_mean

        sample_images = sample_images.cpu().detach().numpy()
        reconstructions = reconstructions.cpu().detach().numpy()

        if reconstructions.shape[1] == 1:
            sample_images = sample_images.squeeze(1)
            reconstructions = reconstructions.squeeze(1)
        elif reconstructions.shape[1] == 3:
            sample_images = sample_images.transpose(0, 2, 3, 1)
            reconstructions = reconstructions.transpose(0, 2, 3, 1)
        for i in range(axes.shape[1]):
            axes[0, i].imshow(sample_images[i])
            axes[1, i].imshow(reconstructions[i])

        tb.add_image(
            f"Reconstructions",
            plot_to_image(fig),
            self.current_epoch,
        )
        self.train()

    def configure_optimizers(self):
        encoder_params = self.encoder.parameters()
        ignored_params = list(map(id, encoder_params))
        base_params = filter(lambda p: id(p) not in ignored_params, self.parameters())
        optim = torch.optim.SGD(
            [
                {"params": base_params, "lr": self.lr},
                {"params": encoder_params, "lr": self.lr / 10},
            ],
            momentum=self.momentum,
            nesterov=True,
            weight_decay=0.0004,
        )
        if self.scheduler == "constant":
            return optim
        STEPS_PER_EPOCH = int(
            np.ceil((50000 // 4) / self.trainer.datamodule.batch_size)
        )
        WARMUP_PERIOD = 10
        lr_sched = {
            "one-cycle": {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optim,
                    self.lr,
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    div_factor=100,
                    pct_start=self.one_cycle_warmup,
                ),
                "interval": "step",
            },
            "cosine": torch.optim.lr_scheduler.SequentialLR(
                optimizer=optim,
                schedulers=[
                    torch.optim.lr_scheduler.LambdaLR(
                        optimizer=optim,
                        lr_lambda=lambda epoch: (epoch + 1) / (WARMUP_PERIOD + 1),
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer=optim, T_max=self.anneal_epochs
                    ),
                ],
                milestones=[WARMUP_PERIOD],
            ),
        }[self.scheduler]
        return {"optimizer": optim, "lr_scheduler": lr_sched}


class Resnet(VAE):
    def __init__(
        self,
        sample_input,
        blocks_per_scale=2,
        num_encoder_scales=4,
        num_decoder_scales=4,
        base_channels=8,
        encoder_depth_per_block=3,
        intermediate_channels=32,
        encoder_block_channels=3,
        **kwargs,
    ):
        self.blocks_per_scale = blocks_per_scale
        self.num_encoder_scales = num_encoder_scales
        self.num_decoder_scales = num_decoder_scales
        self.base_channels = base_channels
        self.encoder_depth_per_block = encoder_depth_per_block
        self.intermediate_channels = intermediate_channels
        self.encoder_block_channels = encoder_block_channels
        super().__init__(
            sample_input,
            **kwargs,
        )

    def build_encoder(self):
        self.scale_blocks = nn.ModuleList()
        in_channels = self.input_channels
        out_channels = self.base_channels
        for _ in range(self.num_encoder_scales):
            self.scale_blocks.append(
                ScaleBlock(
                    in_dim=in_channels,
                    mid_dim=self.intermediate_channels,
                    out_dim=out_channels,
                    block_dim=self.encoder_block_channels,
                    kernel_size=3,
                    block_per_scale=self.blocks_per_scale,
                    depth_per_block=self.encoder_depth_per_block,
                )
            )
            in_channels = out_channels
            out_channels *= 2

        self.FC_shape = (
            (self.input_length // 2**self.num_encoder_scales) ** 2
            * 2 ** (self.num_encoder_scales - 1)
            * self.base_channels
        )
        self.mean = nn.Linear(self.FC_shape, self.lsdim)
        self.logvar = nn.Linear(self.FC_shape, self.lsdim)

    def q_z(self, x):

        for scale in self.scale_blocks:
            x = F.avg_pool2d(scale(x), (2, 2))

        x = x.view(-1, self.FC_shape)

        z_q_mean = self.mean(x)
        z_q_logvar = self.logvar(x)
        return z_q_mean, z_q_logvar


class ResnetNVP(NVP):
    def __init__(
        self,
        sample_input,
        blocks_per_scale=2,
        num_encoder_scales=4,
        num_decoder_scales=4,
        base_channels=8,
        **kwargs,
    ):
        self.blocks_per_scale = blocks_per_scale
        self.num_encoder_scales = num_encoder_scales
        self.num_decoder_scales = num_decoder_scales
        self.base_channels = base_channels
        super().__init__(
            sample_input,
            **kwargs,
        )

    def build_encoder(self):
        self.scale_blocks = nn.ModuleList()
        in_channels = 1
        out_channels = self.base_channels
        for _ in range(self.num_encoder_scales):
            self.scale_blocks.append(
                ScaleBlock(
                    in_dim=in_channels,
                    out_dim=out_channels,
                    kernel_size=3,
                    block_per_scale=self.blocks_per_scale,
                    depth_per_block=3,
                )
            )
            in_channels = out_channels
            out_channels *= 2

        self.FC_shape = (
            (self.input_length // 2**self.num_encoder_scales) ** 2
            * 2 ** (self.num_encoder_scales - 1)
            * self.base_channels
        )

    def q_z(self, x):

        for scale in self.scale_blocks:
            x = F.avg_pool2d(scale(x), (2, 2))

        x = x.view(-1, self.FC_shape)

        z_q_mean = self.mean(x)
        z_q_logvar = self.logvar(x)
        # log_w = (
        #     torch.ones((x.shape[0], self.num_pseudos), device=self.device)
        #     / self.num_pseudos
        # )
        log_w = F.log_softmax(self.prior_coefficients(x), dim=1)
        # self.empirical_freq = self.empirical_freq.to(x.device)
        # self.empirical_freq += torch.sum(log_w.detach(), dim=0)
        return z_q_mean, z_q_logvar, log_w
