import numpy as np
import torch
from torch.utils.module_tracker import ModuleTracker
from torch import nn
import torch.nn.functional as F
import lightning as L
from .util import ScaleBlock, Downsample, ScaleFCBlock
from lightning.pytorch.utilities import grad_norm

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
        use_cross_entropy_loss: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sample_input = sample_input
        self.input_channels = sample_input.shape[1]
        self.latent_dim = latent_dim
        self.second_dim = second_dim
        self.second_depth = second_depth
        self.use_cross_entropy_loss = use_cross_entropy_loss
        self.is_training = None
        self.BCELogitsLoss = nn.BCELoss()

        self.build_distribution_encoder()
        self.build_distribution_decoder()

        self.save_hyperparameters()

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

    def MSE_loss(self, x, x_hat, log_gamma):
        error = torch.square((x - x_hat) / torch.exp(log_gamma))
        error += 2 * log_gamma
        # We instead take the mean across all elements to implicitly normalize
        # by the data dimensionality to ensure a well-conditioned loss scale
        # error = error.sum([1, 2, 3])
        return error.mean()

    def KL_loss(self, mu, log_sd):
        reduced = (
            torch.sum(torch.square(mu) + torch.exp(2 * log_sd) - 2 * log_sd - 1, dim=1)
            / 2.0
        )
        return reduced.mean()

    def apply_manifold_loss(self, mu_z, logsd_z, x_hat, x):
        manifold_kl_loss = self.KL_loss(mu_z, logsd_z)
        if self.use_cross_entropy_loss:
            recon_loss = self.BCELoss(F.sigmoid(x_hat), x)
        else:
            recon_loss = self.MSE_loss(x, x_hat, self.log_gamma_x)
        return manifold_kl_loss, recon_loss

    def apply_distribution_loss(self, mu_u, logsd_u, z_hat, z):
        distribution_kl_loss = self.KL_loss(mu_u, logsd_u)
        recon_loss = self.MSE_loss(z, z_hat, self.log_gamma_z)
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
        # self.z_std_id = torch.ones(size=(1, self.latent_dim), dtype=torch.float32)
        self.log_gamma_z = nn.Parameter(torch.tensor(0.0))

    def apply_distribution_decoder(self, u):
        t = u
        for layer in self.distribution_decoder_layers:
            t = F.leaky_relu(layer(t))
        t = torch.concat([u, t], -1)
        z_hat = self.z_hat(t)
        z_dist = torch.distributions.Normal(z_hat, torch.exp(self.log_gamma_z))
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
        z = z + torch.exp(self.log_gamma_z) * np.random.normal(
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
        num_encoder_scales,
        num_decoder_scales=3,
        block_per_scale=1,
        depth_per_block=2,
        kernel_size=3,
        base_dim=16,
        fc_dim=512,
        latent_dim=64,
        second_depth=3,
        second_dim=1024,
        use_cross_entropy_loss=False,
    ):
        super().__init__(
            sample_input, latent_dim, second_depth, second_dim, use_cross_entropy_loss
        )
        self.num_encoder_scales = num_encoder_scales
        self.block_per_scale = block_per_scale
        self.depth_per_block = depth_per_block
        self.kernel_size = kernel_size
        self.base_dim = base_dim
        self.fc_dim = fc_dim
        self.num_decoder_scales = num_decoder_scales
        with ModuleTracker() as tracker:
            self.build_manifold_encoder()
            self.build_manifold_decoder()
        _nan_hook = self.nan_hook(tracker)
        for submodule in self.modules():
            submodule.register_forward_hook(_nan_hook)

    def build_manifold_encoder(self):
        dim = self.base_dim
        self.encoder_scale_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        input_channels = self.input_channels

        for i in range(self.num_encoder_scales):
            next_scale_block = ScaleBlock(
                input_channels,
                dim,
                self.block_per_scale,
                self.depth_per_block,
                self.kernel_size,
            )

            self.encoder_scale_blocks.append(next_scale_block)

            if i != self.num_encoder_scales - 1:
                next_downsample_block = Downsample(dim, dim, self.kernel_size)
                input_channels = dim
                dim *= 2
                self.downsample_blocks.append(next_downsample_block)

        self.scale_fc = ScaleFCBlock(dim, self.fc_dim, 1, self.depth_per_block)
        self.mu_z = nn.Linear(self.fc_dim, self.latent_dim)
        self.logsd_z = nn.Linear(self.fc_dim, self.latent_dim)

    def apply_manifold_encoder(self, x):
        t = x
        for i in range(self.num_encoder_scales):
            t = self.encoder_scale_blocks[i](t)
            if i != self.num_encoder_scales - 1:
                t = self.downsample_blocks[i](t)

        t = t.mean([2, 3])
        t = self.scale_fc(t)
        mu_z = self.mu_z(t)
        logsd_z = self.logsd_z(t)
        sd_z = torch.exp(logsd_z)
        z = (
            mu_z
            + torch.distributions.Normal(
                torch.tensor(0, dtype=torch.float32).to(device=self.device),
                torch.tensor(1, dtype=torch.float32).to(device=self.device),
            ).sample(sample_shape=[x.shape[0], self.latent_dim])
            * sd_z
        )
        return z, mu_z, logsd_z

    def build_manifold_decoder(self):
        data_depth = self.sample_input.shape[1]
        self.decoder_scale_blocks = nn.ModuleList()
        data_depth_bits = np.ceil(np.log2(data_depth))
        dims = [
            2**i for i in range(int(np.log2(self.latent_dim))) if i >= data_depth_bits
        ]
        self.dims = dims[::-1]
        # To account for spatial mesh grid coordinate channels
        in_dim = self.latent_dim + 2
        for out_dim in dims:
            self.decoder_scale_blocks.append(
                ScaleBlock(
                    in_dim,
                    out_dim,
                    self.is_training,
                    self.block_per_scale,
                    self.depth_per_block,
                    self.kernel_size,
                )
            )
            in_dim = out_dim
        self.log_gamma_x = nn.Parameter(torch.tensor(0.0))
        self.out_conv = nn.Conv2d(in_dim, data_depth, self.kernel_size, 1, "same")

    def apply_manifold_decoder(self, z):
        t = z.unsqueeze(2).unsqueeze(3)
        t = t.tile(1, 1, *self.sample_input.shape[-2:])
        length = self.sample_input.shape[-1]
        # breakpoint()
        xs = torch.linspace(-1, 1, steps=length).to(self.device)
        ys = torch.linspace(-1, 1, steps=length).to(self.device)
        xs, ys = torch.meshgrid(xs, ys, indexing="xy")
        xs = xs.unsqueeze(0).unsqueeze(0).tile(z.shape[0], 1, 1, 1)
        ys = ys.unsqueeze(0).unsqueeze(0).tile(z.shape[0], 1, 1, 1)
        # breakpoint()
        t = torch.concat((t, xs, ys), dim=1).to(self.device)
        for block in self.decoder_scale_blocks:
            t = block(t)
        t = self.out_conv(t)
        return F.sigmoid(t)

    def forward(self, x):
        z, mu_z, logsd_z = self.apply_manifold_encoder(x)
        x_hat = self.apply_manifold_decoder(z)
        return x_hat, mu_z, logsd_z

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu_z, logsd_z = self(x)
        kl_loss, recon_loss = self.apply_manifold_loss(mu_z, logsd_z, x_hat, x)
        loss = kl_loss + recon_loss
        logs = {
            "kl_loss": kl_loss,
            "recon_loss": recon_loss,
            "loss": loss,
            "gamma_x": torch.exp(self.log_gamma_x),
            "learning rate": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        self.log_dict(logs, on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_end(self):
        tb = self.trainer.logger.experiment
        sample_image = next(iter(self.trainer.train_dataloader))[0][0].unsqueeze(0)
        tb.add_image(
            f"Sample_{self.current_epoch}/ground_truth",
            sample_image.squeeze(0),
            self.current_epoch,
        )
        tb.add_image(
            f"Sample_{self.current_epoch}/reconstruction",
            self.reconstruct(sample_image.to(self.device)).squeeze(0),
            self.current_epoch,
        )

    def configure_optimizers(self):
        optim = torch.optim.Adagrad(self.parameters(), lr=1e-3)
        decay_sched = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optim, gamma=0.985
        )
        cos_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optim, T_0=10, T_mult=2
        )
        lr_sched = torch.optim.lr_scheduler.ChainedScheduler([cos_sched, decay_sched])

        return {"optimizer": optim, "lr_scheduler": lr_sched}

    def nan_hook(self, tracker: ModuleTracker):
        def _nan_hook(module, args, output):
            if not isinstance(output, tuple):
                outputs = [output]
            else:
                outputs = output

            for i, out in enumerate(outputs):
                nan_mask = torch.isnan(out)
                if nan_mask.any():
                    print("In", self.__class__.__name__)
                    raise RuntimeError(f"Found NAN in output of {tracker.parents}")
            return None

        return _nan_hook

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer):
        self.log_dict(
            grad_norm(self, norm_type=2),
            on_step=True,
            on_epoch=True,
        )
