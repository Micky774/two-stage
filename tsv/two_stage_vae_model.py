import numpy as np
import torch
from torch.utils.module_tracker import ModuleTracker
from torch import nn
import torch.nn.functional as F
import lightning as L
from .util import (
    ScaleBlock,
    Downsample,
    ScaleFCBlock,
    plot_to_image,
    generate_embedding,
    SBD,
)
from matplotlib import pyplot as plt

HALF_LOG_TWO_PI = 0.91893


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
        self.example_input_array = sample_input
        self.input_channels = sample_input.shape[1]
        self.latent_dim = latent_dim
        self.second_dim = second_dim
        self.second_depth = second_depth
        self.use_cross_entropy_loss = use_cross_entropy_loss
        self.is_training = None
        self.BCELoss = nn.BCELoss(reduction="sum")
        self.log_gamma_x = nn.Parameter(torch.zeros(1))

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

    def MSE_loss(
        self, x: torch.Tensor, x_hat: torch.Tensor, log_gamma: torch.Tensor
    ) -> torch.Tensor:
        nn.MSELoss
        error = torch.square((x - x_hat) / torch.exp(log_gamma))
        error += 2 * log_gamma
        # We can instead take the mean across all elements to implicitly normalize
        # by the data dimensionality to ensure a well-conditioned loss scale
        # error = error.sum([1, 2, 3])
        return error.sum() / 2

    def KL_loss(self, mu, log_sd):
        reduced = (
            torch.sum(torch.square(mu) + torch.exp(2 * log_sd) - 2 * log_sd - 1, dim=1)
            / 2.0
        )
        return reduced.sum()

    def apply_manifold_loss(self, mu_z, logsd_z, x_hat, x):
        manifold_kl_loss = self.KL_loss(mu_z, logsd_z)
        assert 0 <= x.min() <= x.max() <= 1
        if self.use_cross_entropy_loss:
            recon_loss = self.BCELoss(x_hat, x)
        else:
            recon_loss = self.MSE_loss(x, x_hat, self.log_gamma_x)
        return manifold_kl_loss, recon_loss

    def apply_distribution_loss(self, mu_u, logsd_u, z_hat, z):
        distribution_kl_loss = self.KL_loss(mu_u, logsd_u)
        recon_loss = self.MSE_loss(z, z_hat, self.log_gamma_z)
        return distribution_kl_loss + recon_loss

    def build_distribution_encoder(self):
        self.distribution_encoder_fcs = nn.ModuleList()
        self.distribution_encoder_prelus = nn.ModuleList()
        self.distribution_encoder_fcs.append(
            nn.Linear(self.latent_dim, self.second_dim)
        )
        self.distribution_encoder_fcs.extend(
            [
                nn.Linear(self.second_dim, self.second_dim)
                for _ in range(self.second_depth)
            ]
        )
        self.distribution_encoder_prelus.extend(
            [nn.PReLU() for _ in range(self.second_depth)]
        )
        self.mu_u_fc = nn.Linear(self.latent_dim + self.second_dim, self.latent_dim)
        self.logsd_u_fc = nn.Linear(self.latent_dim + self.second_dim, self.latent_dim)

    def apply_distribution_encoder(self, z):
        t = z
        for layer, prelu in zip(
            self.distribution_encoder_fcs, self.distribution_encoder_prelus
        ):
            t = prelu(layer(t))
        mu_u = self.mu_u_fc(t)
        logsd_u = self.logsd_u_fc(t)
        u_dist = torch.distributions.Normal(mu_u, torch.exp(logsd_u))
        self.sample_u = u_dist.rsample()
        return

    def build_distribution_decoder(self):
        self.distribution_decoder_fcs = nn.ModuleList()
        self.distribution_decoder_prelus = nn.ModuleList()
        self.distribution_decoder_fcs.append(
            nn.Linear(self.latent_dim, self.second_dim)
        )
        self.distribution_decoder_prelus.extend(
            [nn.PReLU() for _ in range(len(self.distribution_decoder_fcs))]
        )

        self.z_hat = nn.Linear(self.latent_dim, self.latent_dim)
        # self.z_std_id = torch.ones(size=(1, self.latent_dim), dtype=torch.float32)
        self.log_gamma_z = nn.Parameter(torch.tensor(0.0))

    def apply_distribution_decoder(self, u):
        t = u
        for layer, prelu in zip(
            self.distribution_decoder_fcs, self.distribution_decoder_prelus
        ):
            t = prelu(layer(t))
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
        lr=1e-3,
        num_decoder_scales=3,
        num_fc_scales=1,
        decoder_channels=64,
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
        self.num_fc_scales = num_fc_scales
        self.lr = lr
        self.decoder_channels = decoder_channels
        self.block_per_scale = block_per_scale
        self.depth_per_block = depth_per_block
        self.kernel_size = kernel_size
        self.base_dim = base_dim
        self.fc_dim = fc_dim
        self.num_decoder_scales = num_decoder_scales
        self.sbd = SBD(
            input_length=sample_input.shape[-1],
            lsdim=self.latent_dim,
            channels_per_layer=self.decoder_channels,
            num_layers=self.num_decoder_scales,
        )
        with ModuleTracker() as tracker:
            self.build_manifold_encoder()
            self.build_manifold_decoder()
            # self.build_distribution_encoder()
            # self.build_distribution_decoder()

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
                block_per_scale=self.block_per_scale,
                depth_per_block=self.depth_per_block,
                kernel_size=self.kernel_size,
            )

            self.encoder_scale_blocks.append(next_scale_block)

            if i != self.num_encoder_scales - 1:
                next_downsample_block = Downsample(dim, dim, self.kernel_size)
                input_channels = dim
                dim *= 2
                self.downsample_blocks.append(next_downsample_block)

        self.scale_fc = ScaleFCBlock(
            dim, self.fc_dim, self.num_fc_scales, self.depth_per_block
        )
        self.mu_z_fc = nn.Linear(self.fc_dim, self.latent_dim)
        self.mu_z_prelu = nn.PReLU()
        self.logsd_z_fc = nn.Linear(self.fc_dim, self.latent_dim)
        self.logsd_z_prelu = nn.PReLU()

    def apply_manifold_encoder(self, x):
        t = x
        for i in range(self.num_encoder_scales):
            t = self.encoder_scale_blocks[i](t)
            if i != self.num_encoder_scales - 1:
                t = self.downsample_blocks[i](t)

        t = t.mean([2, 3])
        t = self.scale_fc(t)
        mu_z = self.mu_z_prelu(self.mu_z_fc(t))
        logsd_z = self.logsd_z_prelu(self.logsd_z_fc(t))
        logsd_z = torch.clamp(logsd_z, max=2)
        sd_z = torch.exp(logsd_z)
        z = mu_z + torch.randn_like(sd_z) * sd_z
        return z, mu_z, logsd_z

    def build_manifold_decoder(self):
        return

    def apply_manifold_decoder(self, z):
        return self.sbd(z)

    def forward(self, x):
        z, mu_z, logsd_z = self.apply_manifold_encoder(x)
        x_hat = self.apply_manifold_decoder(z)
        return x_hat, mu_z, logsd_z

    def training_step(self, batch, batch_idx):
        x, _ = batch
        mu_z: torch.Tensor
        x_hat, mu_z, logsd_z = self(x)
        kl_loss, recon_loss = self.apply_manifold_loss(mu_z, logsd_z, x_hat, x)
        loss = kl_loss + recon_loss
        batch_size = x.size(0)
        logs = {
            "kl_loss": kl_loss.item() / batch_size,
            "recon_loss": recon_loss.item() / batch_size,
            "loss": loss.item() / batch_size,
            "learning rate": self.trainer.optimizers[0].param_groups[0]["lr"],
            "raw_Mse": F.mse_loss(x_hat, x, reduction="sum").item() / batch_size,
        }
        if hasattr(self, "log_gamma_x"):
            logs["log_gamma"] = self.log_gamma_x.item()
        self.log_dict(logs, on_epoch=False, on_step=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        tb = self.trainer.logger.experiment

        fig, axes = plt.subplots(2, 4)
        sample_images = next(iter(self.trainer.train_dataloader))[0]
        for i in range(axes.shape[1]):
            sample_image = sample_images[i].unsqueeze(1)
            reconstruction = self.reconstruct(sample_image.to(self.device)).squeeze(0)
            axes[0, i].imshow(sample_image.squeeze(0).squeeze(0).cpu().detach().numpy())
            axes[1, i].imshow(
                reconstruction.squeeze(0).squeeze(0).cpu().detach().numpy()
            )

        tb.add_image(
            f"Reconstructions",
            plot_to_image(fig),
            self.current_epoch,
        )

        if self.global_rank == 0 and self.latent_dim == 2:
            generate_embedding(
                self.trainer.train_dataloader,
                self.apply_manifold_encoder,
                self.device,
                tb,
                self.current_epoch,
            )

    def configure_optimizers(self):
        optim = torch.optim.NAdam(self.parameters(), lr=self.lr, weight_decay=0.01)
        WARMUP_PERIOD = 10
        warmup_sched = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optim, lr_lambda=lambda epoch: (epoch + 1) / (WARMUP_PERIOD + 1)
        )
        cos_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optim, T_0=10, T_mult=2
        )
        lr_sched = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optim,
            schedulers=[warmup_sched, cos_sched],
            milestones=[WARMUP_PERIOD],
        )
        lr_sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optim, max_lr=self.lr, steps_per_epoch=469, epochs=1000
        )

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
