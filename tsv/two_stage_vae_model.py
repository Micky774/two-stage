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
)
from lightning.pytorch.utilities import grad_norm
from matplotlib import pyplot as plt

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
        self.example_input_array = sample_input
        self.input_channels = sample_input.shape[1]
        self.latent_dim = latent_dim
        self.second_dim = second_dim
        self.second_depth = second_depth
        self.use_cross_entropy_loss = use_cross_entropy_loss
        self.is_training = None
        self.BCELoss = nn.BCELoss()

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
        error = error.sum([1, 2, 3])
        return error.mean()

    def KL_loss(self, mu, log_sd):
        reduced = (
            torch.sum(torch.square(mu) + torch.exp(2 * log_sd) - 2 * log_sd - 1, dim=1)
            / 2.0
        ) * 0
        return reduced.mean()

    def apply_manifold_loss(self, mu_z, logsd_z, x_hat, x):
        manifold_kl_loss = self.KL_loss(mu_z, logsd_z)
        assert 0 <= x.min() <= x.max() <= 1
        if self.use_cross_entropy_loss:
            recon_loss = self.BCELoss(x_hat, x)
        else:
            recon_loss = self.MSE_loss(x, x_hat, torch.tensor(0))
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
        self.sampler = None

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
        data_depth = self.example_input_array.shape[1]
        self.decoder_scale_blocks = nn.ModuleList()
        # To account for spatial mesh grid coordinate channels
        in_channels = self.latent_dim + 2
        for i in range(self.num_decoder_scales):
            self.decoder_scale_blocks.append(
                ScaleBlock(
                    in_channels if i == 0 else self.decoder_channels,
                    self.decoder_channels,
                    self.is_training,
                    self.block_per_scale,
                    self.depth_per_block,
                    self.kernel_size,
                )
            )
        # self.log_gamma_x = nn.Parameter(torch.tensor(0.0))
        self.out_conv = nn.Conv2d(self.decoder_channels, data_depth, 1, 1, "same")

    def apply_manifold_decoder(self, z):
        t = z.unsqueeze(2).unsqueeze(3)
        t = t.tile(1, 1, *self.example_input_array.shape[-2:])
        length = self.example_input_array.shape[-1]
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
        mu_z: torch.Tensor
        x_hat, mu_z, logsd_z = self(x)
        self.logger.experiment.add_histogram("mu_z", mu_z.detach(), batch_idx)
        self.logger.experiment.add_histogram(
            "sd_z", torch.exp(logsd_z).detach(), batch_idx
        )
        kl_loss, recon_loss = self.apply_manifold_loss(mu_z, logsd_z, x_hat, x)
        loss = kl_loss + recon_loss
        logs = {
            "kl_loss": kl_loss.item(),
            "recon_loss": recon_loss.item(),
            "loss": loss.item(),
            # "gamma_x": np.exp(self.log_gamma_x.item()),
            "learning rate": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        self.log_dict(logs, on_epoch=True, on_step=False, sync_dist=True)
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
        # if self.global_rank == 0:
        #     self.log_dict(
        #         grad_norm(self, norm_type="2"),
        #         self.current_epoch,
        #         rank_zero_only=True,
        #     )

        if self.global_rank == 0 and self.latent_dim == 2:
            generate_embedding(
                self.trainer.train_dataloader,
                self.apply_manifold_encoder,
                self.device,
                tb,
                self.current_epoch,
            )

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.1)
        decay_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.9)
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

    def on_after_backward(self):
        self.logger.experiment.add_histogram(
            "mu_z_fc", self.mu_z_fc.weight.data, self.current_epoch
        )
        # if self.mu_z_fc.weight.grad.data is not None:
        #     self.logger.experiment.add_histogram(
        #         "mu_z_fc.grad", self.mu_z_fc.weight.grad.data, self.current_epoch
        #     )
        self.logger.experiment.add_histogram(
            "logsd_z_fc", self.logsd_z_fc.weight.data, self.current_epoch
        )
        # if self.logsd_z_fc.weight.grad.data is not None:
        #     self.logger.experiment.add_histogram(
        #         "logsd_z_fc.grad", self.logsd_z_fc.weight.grad.data, self.current_epoch
        #     )
        # for name, param in self.named_parameters():
        #     if "weight" not in name: continue
        #     self.logger.experiment.add_histogram(name,param.data,self.current_epoch)
        #     self.logger.experiment.add_histogram(f"{name}.grad",param.grad.data,self.current_epoch)

    # def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer):
    #     self.log_dict(
    #         grad_norm(self, norm_type=2),
    #         on_step=True,
    #         on_epoch=True,
    #         sync_dist=True,
    #     )


class Simple(TwoStageVaeModel):
    def __init__(
        self,
        sample_input,
        num_encoder_scales,
        lr=1e-3,
        num_decoder_scales=3,
        num_fc_scales=1,
        decoder_channels=64,
        encoder_channels=64,
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
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.block_per_scale = block_per_scale
        self.depth_per_block = depth_per_block
        self.kernel_size = kernel_size
        self.base_dim = base_dim
        self.fc_dim = fc_dim
        self.num_decoder_scales = num_decoder_scales
        self.sampler = None

        with ModuleTracker() as tracker:
            self.build_manifold_encoder()
            self.build_manifold_decoder()
            # self.build_distribution_encoder()
            # self.build_distribution_decoder()

        _nan_hook = self.nan_hook(tracker)
        for submodule in self.modules():
            submodule.register_forward_hook(_nan_hook)

    def build_manifold_encoder(self):
        self.encoder_convs = nn.ModuleList()
        self.encoder_prelus = nn.ModuleList()
        in_channels = self.input_channels
        for _ in range(self.num_encoder_scales):
            self.encoder_convs.append(
                nn.Conv2d(
                    in_channels, self.encoder_channels, self.kernel_size, padding="same"
                )
            )
            in_channels = self.encoder_channels
            self.encoder_prelus.append(nn.PReLU(self.encoder_channels))
        self.encoder_pixelwise_prelu = nn.PReLU(self.encoder_channels)
        self.encoder_pixelwise_conv = nn.Conv2d(self.encoder_channels + self.input_channels, self.encoder_channels, 1)
        self.scale_fc = ScaleFCBlock(
            self.encoder_channels, self.fc_dim, self.num_fc_scales, self.depth_per_block
        )
        self.mu_z_fc = nn.Linear(self.fc_dim, self.latent_dim)
        self.mu_z_prelu = nn.PReLU()
        self.logsd_z_fc = nn.Linear(self.fc_dim, self.latent_dim)
        self.logsd_z_prelu = nn.PReLU()

    def apply_manifold_encoder(self, x):
        t = x
        for block, prelu in zip(self.encoder_convs, self.encoder_prelus):
            t = block(t)
            t = prelu(t)
        t = torch.concatenate(
            [t, x], dim=1
        )
        t = self.encoder_pixelwise_conv(t)
        t = self.encoder_pixelwise_prelu(t)

        t = t.mean([2, 3])
        t = self.scale_fc(t)
        mu_z = self.mu_z_prelu(self.mu_z_fc(t))
        logsd_z = self.logsd_z_prelu(self.logsd_z_fc(t))
        logsd_z = torch.clamp(logsd_z, max=2)
        sd_z = torch.exp(logsd_z)
        z = mu_z + torch.randn_like(sd_z) * sd_z
        return z, mu_z, logsd_z

    def build_manifold_decoder(self):
        data_depth = self.example_input_array.shape[1]
        self.decoder_scale_blocks = nn.ModuleList()
        # To account for spatial mesh grid coordinate channels
        in_channels = self.latent_dim + 2
        for i in range(self.num_decoder_scales):
            self.decoder_scale_blocks.append(
                ScaleBlock(
                    in_channels if i == 0 else self.decoder_channels,
                    self.decoder_channels,
                    self.is_training,
                    self.block_per_scale,
                    self.depth_per_block,
                    self.kernel_size,
                )
            )
        # self.log_gamma_x = nn.Parameter(torch.tensor(0.0))
        self.out_conv = nn.Conv2d(self.decoder_channels, data_depth, 1, 1, "same")

    def apply_manifold_decoder(self, z):
        t = z.unsqueeze(2).unsqueeze(3)
        t = t.tile(1, 1, *self.example_input_array.shape[-2:])
        length = self.example_input_array.shape[-1]
        xs = torch.linspace(-1, 1, steps=length).to(self.device)
        ys = torch.linspace(-1, 1, steps=length).to(self.device)
        xs, ys = torch.meshgrid(xs, ys, indexing="xy")
        xs = xs.unsqueeze(0).unsqueeze(0).tile(z.shape[0], 1, 1, 1)
        ys = ys.unsqueeze(0).unsqueeze(0).tile(z.shape[0], 1, 1, 1)
        t = torch.concat((t, xs, ys), dim=1)
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
        mu_z: torch.Tensor
        x_hat, mu_z, logsd_z = self(x)
        self.logger.experiment.add_histogram("mu_z", mu_z.detach(), batch_idx)
        self.logger.experiment.add_histogram(
            "sd_z", torch.exp(logsd_z).detach(), batch_idx
        )
        kl_loss, recon_loss = self.apply_manifold_loss(mu_z, logsd_z, x_hat, x)
        loss = kl_loss + recon_loss
        logs = {
            "kl_loss": kl_loss.item(),
            "recon_loss": recon_loss.item(),
            "loss": loss.item(),
            "learning rate": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        self.log_dict(logs, on_epoch=True, on_step=False, sync_dist=True)
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
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)
        decay_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.9)
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

    def on_after_backward(self):
        self.logger.experiment.add_histogram(
            "mu_z_fc", self.mu_z_fc.weight.data, self.current_epoch
        )
        self.logger.experiment.add_histogram(
            "logsd_z_fc", self.logsd_z_fc.weight.data, self.current_epoch
        )
