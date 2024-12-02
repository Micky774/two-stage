import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import lightning as L
from torch import nn
from torchvision.datasets import MNIST
import os
from .util import ScaleBlock, Downsample, Upsample, ScaleFCBlock

HALF_LOG_TWO_PI = 0.91893


# TODO: Implement optimizer
# TODO: Implement logging
# TODO: Implement first VAE
class TwoStageVaeModel(L.LightningModule):
    def __init__(
        self,
        sample_input: torch.Tensor,
        latent_dim: int = 64,
        second_depth: int = 3,
        second_dim: int = 1024,
        cross_entropy_loss: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sample_input = sample_input
        self.input_channels = sample_input.shape[1]
        self.batch_size = sample_input.shape[0]
        self.latent_dim = latent_dim
        self.second_dim = second_dim
        self.second_depth = second_depth
        self.cross_entropy_loss = cross_entropy_loss
        self.is_training = None
        self.loggamma_x = torch.zeros_like(self.sample_input, dtype=torch.float32)

        self.build_distribution_encoder()
        self.build_distribution_decoder()

    # Must implement in subclass
    def build_manifold_encoder(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "A subclass should implement this based on the dataset"
        )

    def build_manifold_decoder(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "A subclass should implement this based on the dataset"
        )

    def apply_manifold_encoder(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "A subclass should implement this based on the dataset"
        )

    def apply_manifold_decoder(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "A subclass should implement this based on the dataset"
        )

    def cross_entropy_loss(self, x, x_hat):
        return -torch.sum(
            x * torch.log(torch.maximum(x_hat, 1e-8))
            + (1 - x) * torch.log(torch.maximum(1 - x_hat, 1e-8))
        ) / float(self.batch_size)

    def MSE_loss(self, x, x_hat):
        print(f"DEBUG *** {x.shape} {x_hat.shape} {self.loggamma_x.shape}")
        return torch.sum(
            torch.square((x - x_hat) / torch.exp(self.loggamma_x)) / 2.0
            + self.loggamma_x
            + HALF_LOG_TWO_PI
        ) / float(self.batch_size)

    def KL_loss(self, mu, log_sd):
        return (
            torch.sum(
                torch.square(mu) + torch.square(torch.exp(log_sd)) - 2 * log_sd - 1
            )
            / 2.0
            / float(self.batch_size)
        )

    def MSE_gamma_loss(self, x, x_hat):
        return torch.sum(
            torch.square((x - x_hat) / torch.exp(self.loggamma_x)) / 2.0
            + self.loggamma_x
            + HALF_LOG_TWO_PI
        ) / float(self.batch_size)

    def apply_manifold_loss(self, mu_z, logsd_z, x_hat, x):
        manifold_kl_loss = self.KL_loss(mu_z, logsd_z)
        if not self.cross_entropy_loss:
            recon_loss = self.MSE_loss(x, x_hat)
        else:
            recon_loss = self.cross_entropy_loss(x, x_hat)
        return manifold_kl_loss + recon_loss

    def apply_distribution_loss(self, mu_u, logsd_u, z_hat, z):
        distribution_kl_loss = self.KL_loss(mu_u, logsd_u)
        recon_loss = self.MSE_gamma_loss(z, z_hat)
        return distribution_kl_loss + recon_loss

    def build_distribution_encoder(self):
        self.distribution_encoder_layers = nn.ModuleList()
        self.distribution_encoder_layers.append(
            nn.Linear(self.latent_dim, self.second_dim)
        )
        self.distribution_encoder_layers.extend(
            [
                nn.Linear(self.second_dim, self.second_dim)
                for _ in range(self.second_depth)
            ]
        )
        self.mu_u = nn.Linear(self.latent_dim + self.second_dim, self.latent_dim)
        self.logsd_u = nn.Linear(self.latent_dim + self.second_dim, self.latent_dim)

    def apply_distribution_encoder(self, z):
        t = z
        for layer in self.distribution_encoder_layers:
            t = F.leaky_relu(layer(t))
        mu_u = self.mu_u(t)
        logsd_u = self.logsd_u(t)
        u_dist = torch.distributions.Normal(mu_u, torch.exp(logsd_u))
        self.sample_u = u_dist.rsample()
        return

    def build_distribution_decoder(self):
        self.distribution_decoder_layers = nn.ModuleList()
        self.distribution_decoder_layers.append(
            nn.Linear(self.latent_dim, self.second_dim)
        )

        self.z_hat = nn.Linear(self.latent_dim, self.latent_dim)
        self.loggamma_z = torch.zeros(size=(1, self.latent_dim), dtype=torch.float32)
        self.gamma_z = torch.exp(self.loggamma_z)

    def apply_distribution_decoder(self, u):
        t = u
        for layer in self.distribution_decoder_layers:
            t = F.leaky_relu(layer(t))
        t = torch.concat([u, t], -1)
        z_hat = self.z_hat(t)
        z_dist = torch.distributions.Normal(z_hat, self.gamma_z)
        self.sample_z = z_dist.rsample()
        return

    def reconstruct(self, x):
        z, _, _ = self.apply_manifold_encoder(x)
        x_hat = self.apply_manifold_decoder(z)
        return x_hat

    def generate(self, num_samples):
        # u ~ N(0, I)
        u = np.random.normal(0, 1, [num_samples, self.latent_dim])

        # z ~ N(f_2(u), \gamma_z I)
        z = self.apply_distribution_decoder(u)
        z = z + torch.exp(self.loggamma_z) * np.random.normal(
            0, 1, [num_samples, self.latent_dim]
        )
        return self.apply_manifold_decoder(z)

    def generate_raw_manifold_sample(self, n_samples):
        # sample_input = f_1(z)
        z = np.random.normal(0, 1, [n_samples, self.latent_dim])
        return self.apply_manifold_decoder(z)


class Resnet(TwoStageVaeModel):
    def __init__(
        self,
        sample_input,
        num_scale,
        block_per_scale=1,
        depth_per_block=2,
        kernel_size=3,
        base_dim=16,
        fc_dim=512,
        latent_dim=64,
        second_depth=3,
        second_dim=1024,
        cross_entropy_loss=False,
    ):
        super().__init__(
            sample_input, latent_dim, second_depth, second_dim, cross_entropy_loss
        )
        self.num_scale = num_scale
        self.block_per_scale = block_per_scale
        self.depth_per_block = depth_per_block
        self.kernel_size = kernel_size
        self.base_dim = base_dim
        self.fc_dim = fc_dim

        self.build_manifold_encoder()
        self.build_manifold_decoder()

    def build_manifold_encoder(self):
        dim = self.base_dim
        self.encoder_scale_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        input_channels = self.input_channels

        for i in range(self.num_scale):
            next_scale_block = ScaleBlock(
                input_channels,
                dim,
                self.block_per_scale,
                self.depth_per_block,
                self.kernel_size,
            )

            self.encoder_scale_blocks.append(next_scale_block)

            if i != self.num_scale - 1:
                next_downsample_block = Downsample(dim, dim, self.kernel_size)
                input_channels = dim
                dim *= 2
                self.downsample_blocks.append(next_downsample_block)

        self.scale_fc = ScaleFCBlock(dim, self.fc_dim, 1, self.depth_per_block)
        self.mu_z = nn.Linear(self.fc_dim, self.latent_dim)
        self.logsd_z = nn.Linear(self.fc_dim, self.latent_dim)

    def apply_manifold_encoder(self, x):
        t = x
        for i in range(self.num_scale):
            t = self.encoder_scale_blocks[i](t)
            if i != self.num_scale - 1:
                t = self.downsample_blocks[i](t)

        t = t.mean([2, 3])
        t = self.scale_fc(t)
        mu_z = self.mu_z(t)
        logsd_z = self.logsd_z(t)
        sd_z = torch.exp(logsd_z)
        z = (
            mu_z
            + torch.distributions.Normal(0, 1)
            .sample(sample_shape=[self.batch_size, self.latent_dim])
            .to(mu_z.device)
            * sd_z
        )
        return z, mu_z, logsd_z

    def build_manifold_decoder(self):
        desired_scale = self.sample_input.shape[-1]
        scale_steps = int(np.log2(desired_scale + 1))
        scales = [2**s for s in range(1, 1 + scale_steps)]
        dims = [min(self.base_dim * (2**s), 1024) for s in range(scale_steps)]
        self.dims = dims[::-1]

        z = self.apply_manifold_encoder(self.sample_input)[0]
        data_depth = self.sample_input.shape[1]
        fc_dim = 2 * 2 * self.dims[0]
        self.flatten_map = nn.Linear(z.shape[-1], fc_dim)
        self.upsample_blocks = nn.ModuleList()
        self.decoder_scale_blocks = nn.ModuleList()
        for i in range(len(scales) - 1):
            self.upsample_blocks.append(
                Upsample(self.dims[i], self.dims[i], self.kernel_size)
            )
            self.decoder_scale_blocks.append(
                ScaleBlock(
                    self.dims[i],
                    self.dims[i + 1],
                    self.is_training,
                    self.block_per_scale,
                    self.depth_per_block,
                    self.kernel_size,
                )
            )
        self.out_conv = nn.Conv2d(
            self.dims[-1], data_depth, self.kernel_size, 1, "same"
        )

    def apply_manifold_decoder(self, z):
        y = self.flatten_map(z)
        y = y.reshape(-1, self.dims[0], 2, 2)
        print(f"DEBUG *** Reshaped {y.shape=}")
        for i in range(len(self.upsample_blocks)):
            y = self.upsample_blocks[i](y)
            y = self.decoder_scale_blocks[i](y)
        print(f"DEBUG *** {y.shape=}")
        y = self.out_conv(y)
        print(f"DEBUG *** {y.shape=}")
        return y

    def forward(self, x):
        z, mu_z, logsd_z = self.apply_manifold_encoder(x)
        x_hat = self.apply_manifold_decoder(z)
        return x_hat, mu_z, logsd_z

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat, mu_z, logsd_z = self(x)
        return self.apply_manifold_loss(mu_z, logsd_z, x_hat, x)

    def train_dataloader(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(mnist_train, batch_size=64, num_workers=-1)

    def test_dataloader(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        return DataLoader(mnist_test, batch_size=64, num_workers=-1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
