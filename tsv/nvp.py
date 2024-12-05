import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from .util import plot_to_image, SBD, generate_embedding
from lightning.pytorch.utilities import grad_norm
from matplotlib import pyplot as plt


class VAE(L.LightningModule):
    def __init__(self, input_length, lsdim, logvar_bound):
        super().__init__()

        self.input_length = input_length
        self.lsdim = lsdim
        self.logvar_bound = logvar_bound

        # Encoding Layers
        self.conv1A = nn.Conv2d(1, 64, 5, padding=2)
        self.conv1B = nn.Conv2d(64, 64, 5, padding=2)
        self.conv1C = nn.Conv2d(64, 64, 5, padding=2)
        self.conv1D = nn.Conv2d(64, 1, 5, padding=2)

        self.conv2A = nn.Conv2d(64, 64, 5, padding=2)
        self.conv2B = nn.Conv2d(64, 64, 5, padding=2)
        self.conv2C = nn.Conv2d(64, 64, 5, padding=2)
        self.conv2D = nn.Conv2d(64, 1, 5, padding=2)

        self.conv3A = nn.Conv2d(64, 64, 5, padding=2)
        self.conv3B = nn.Conv2d(64, 64, 5, padding=2)
        self.conv3C = nn.Conv2d(64, 64, 5, padding=2)
        self.conv3D = nn.Conv2d(64, 1, 5, padding=2)

        self.conv4A = nn.Conv2d(64, 64, 5, padding=2)
        self.conv4B = nn.Conv2d(64, 64, 5, padding=2)
        self.conv4C = nn.Conv2d(64, 64, 5, padding=2)
        self.conv4D = nn.Conv2d(64, 1, 5, padding=2)

        self.linear1 = nn.Linear(int(self.input_length / 2) ** 2, self.input_length**2)
        self.linear2 = nn.Linear(int(self.input_length / 4) ** 2, self.input_length**2)
        self.linear3 = nn.Linear(int(self.input_length / 8) ** 2, self.input_length**2)
        self.linear4 = nn.Linear(int(self.input_length / 16) ** 2, self.input_length**2)

        self.conv5 = nn.Conv2d(5, 64, 5, padding=2)

        self.conv6 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv7 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv8 = nn.Conv2d(64, 64, 5, padding=2)

        self.conv9 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv10 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv11 = nn.Conv2d(64, 64, 5, padding=2)

        self.conv12 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv13 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv14 = nn.Conv2d(64, 64, 5, padding=2)

        self.conv15 = nn.Conv2d(64, 4, 1)

        self.fcc = nn.Linear(int(self.input_length / 8) ** 2 * 4, 1000)

        self.mean = nn.Linear(1000, lsdim)
        self.logvar = nn.Linear(1000, lsdim)

        self.sbd = SBD(input_length=self.input_length, lsdim=self.lsdim)
        # Create an idle input for calling pseudo-inputs

    def reconstruct_x(self, x):
        x_mean, _, _, _ = self.forward(x)
        return x_mean

    def encoder(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        y1 = x.clone()

        x = F.leaky_relu(self.conv1A(x))
        x = F.leaky_relu(self.conv1B(x))
        x = F.leaky_relu(F.max_pool2d(self.conv1C(x), (2, 2)))
        y2 = F.leaky_relu(self.conv1D(x))

        x = F.leaky_relu(self.conv2A(x))
        x = F.leaky_relu(self.conv2B(x))
        x = F.leaky_relu(F.max_pool2d(self.conv2C(x), (2, 2)))
        y3 = F.leaky_relu(self.conv2D(x))

        x = F.leaky_relu(self.conv3A(x))
        x = F.leaky_relu(self.conv3B(x))
        x = F.leaky_relu(F.max_pool2d(self.conv3C(x), (2, 2)))
        y4 = F.leaky_relu(self.conv3D(x))

        x = F.leaky_relu(self.conv4A(x))
        x = F.leaky_relu(self.conv4B(x))
        x = F.leaky_relu(F.max_pool2d(self.conv4C(x), (2, 2)))
        y5 = F.leaky_relu(self.conv4D(x))

        y1 = y1
        y2 = F.leaky_relu(
            self.linear1(y2.view(-1, int(self.input_length / 2) ** 2))
        ).reshape(-1, 1, self.input_length, self.input_length)
        y3 = F.leaky_relu(
            self.linear2(y3.view(-1, int(self.input_length / 4) ** 2))
        ).reshape(-1, 1, self.input_length, self.input_length)
        y4 = F.leaky_relu(
            self.linear3(y4.view(-1, int(self.input_length / 8) ** 2))
        ).reshape(-1, 1, self.input_length, self.input_length)
        y5 = F.leaky_relu(
            self.linear4(y5.view(-1, int(self.input_length / 16) ** 2))
        ).reshape(-1, 1, self.input_length, self.input_length)

        x = torch.cat((y1, y2, y3, y4, y5), dim=1)

        x_ = F.leaky_relu(self.conv5(x))

        x = F.leaky_relu(self.conv6(x_))
        x = F.leaky_relu(self.conv7(x))
        x = F.leaky_relu(self.conv8(x))
        x_ = F.max_pool2d(x + x_, (2, 2))

        x = F.leaky_relu(self.conv9(x_))
        x = F.leaky_relu(self.conv10(x))
        x = F.leaky_relu(self.conv11(x))
        x_ = F.max_pool2d(x + x_, (2, 2))

        x = F.leaky_relu(self.conv12(x_))
        x = F.leaky_relu(self.conv13(x))
        x = F.leaky_relu(self.conv14(x))
        x = F.max_pool2d(x + x_, (2, 2))

        x = F.leaky_relu(self.conv15(x))

        x = x.view(-1, int(self.input_length / 8) ** 2 * 4)
        x = F.leaky_relu(self.fcc(x))

        z_mean = self.mean(x)
        z_logvar = F.elu(self.logvar(x), -1.0 * self.logvar_bound)
        return z_mean, z_logvar

    # Samples at least one z~Q(mu,logvar)
    # Can sample multiple by increasing n
    # Returns in shape [batch_size,num_samples,lsdim]
    def reparameterize(self, mu, logvar, n=1):

        # Converts log-variance to standard-deviation
        std = torch.exp(0.5 * logvar)

        # Tiles standard deviation and mean along new axis for sampling multiple z vectors
        std = std.unsqueeze(1).repeat(1, n, 1)
        mu = mu.unsqueeze(1).repeat(1, n, 1)

        # Samples from a unit gaussian in the shape of the provided standard-deviation vector
        eps = torch.randn_like(std)

        # Constructs multiple z vectors from random noise
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return torch.sigmoid(self.sbd(z))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z


class PseudoGen(L.LightningModule):
    def __init__(self, input_length, num_pseudos):
        super(PseudoGen, self).__init__()
        self.input_length = input_length
        self.num_pseudos = num_pseudos
        self.means = nn.Linear(num_pseudos, input_length * input_length, bias=False)
        self.idle_input = nn.Parameter(torch.eye(num_pseudos, num_pseudos))

    def forward(self, x):
        return torch.sigmoid(self.means(x)).reshape(
            self.num_pseudos, 1, self.input_length, self.input_length
        )


class NVP(L.LightningModule):
    def __init__(self, input_length, lsdim, num_pseudos, beta, gamma, logvar_bound):
        super(NVP, self).__init__()

        self.input_length = input_length
        self.num_pseudos = num_pseudos
        self.beta = beta
        self.gamma = gamma
        self.lsdim = lsdim

        self.vae = VAE(input_length, lsdim, logvar_bound)
        self.pseudoGen = PseudoGen(input_length, num_pseudos)

        self.idle_input = nn.Parameter(torch.eye(num_pseudos))

    def forward(self, x):
        return self.vae.forward(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon_x, mu, logvar, z_q = self.forward(x)
        pseudo = self.pseudoGen.forward(self.idle_input)
        recon_pseudo, p_mu, p_logvar, p_z = self.forward(pseudo)
        loss, recon_loss, kl_loss, pseduo_recon_loss, pkl_loss = self.loss(
            recon_x, x, mu, logvar, z_q, pseudo, recon_pseudo, p_mu, p_logvar, p_z
        )

        logs = {
            "kl_loss": kl_loss.item(),
            "recon_loss": recon_loss.item(),
            "pseduo_recon_loss": pseduo_recon_loss.item(),
            "pkl_loss": pkl_loss.item(),
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
            reconstruction = self.vae(sample_image.to(self.device))[0].squeeze(0)
            axes[0, i].imshow(sample_image.squeeze(0).squeeze(0).cpu().detach().numpy())
            axes[1, i].imshow(
                reconstruction.squeeze(0).squeeze(0).cpu().detach().numpy()
            )

        tb.add_image(
            f"Reconstructions",
            plot_to_image(fig),
            self.current_epoch,
        )
        if self.lsdim == 2:

            def _encode(x):
                mu, logvar = self.vae.encoder(x)
                z = self.vae.reparameterize(mu, logvar).squeeze(1)
                return z, mu, logvar

            generate_embedding(
                self.trainer.train_dataloader,
                _encode,
                self.device,
                tb,
                self.current_epoch,
            )

    # Note that this loss *sums* instead of averages, so scale learning rate with
    # the expected batch size to account for missing averaging. This is done to ensure consistent
    # learning across variable batch sizes without needing to calculate them live.
    def loss(
        self,
        recon_x,
        x,
        mu,
        logvar,
        z_q,
        pseudo,
        recon_pseudo,
        p_mu,
        p_logvar,
        p_z,
        gamma=None,
    ):
        if gamma is None:
            gamma = self.gamma
        # Reshape inputs for easier comparison
        x = x.view(-1, self.input_length * self.input_length)
        recon_x = recon_x.view(-1, self.input_length * self.input_length)

        recon_loss = F.mse_loss(recon_x, x, reduction="sum")

        # Log-likelihood of a given z-sample originating from the prior
        # [batch_size, sample_size]
        log_p_z = self.log_p_z(z_q)

        # Log-likelihood of a given z-sample originating from the posterior
        # [batch_size, sample_size]
        log_q_z = self.prior_likelihood(z_q, mu, logvar, dim=2)

        # Monte Carlo estimation of KL divergence from posterior to prior
        # computed over several samples per posterior
        kl_loss = (log_q_z - log_p_z).sum(axis=1).sum()

        # Reshape inputs for easier comparison
        pseudo = pseudo.view(-1, self.input_length * self.input_length)
        recon_pseudo = recon_pseudo.view(-1, self.input_length * self.input_length)

        pseduo_recon_loss = F.mse_loss(recon_pseudo, pseudo, reduction="sum")

        plog_p_z = self.log_p_z(p_z)
        plog_q_z = self.prior_likelihood(p_z, p_mu, p_logvar, dim=2)
        # KL divergence from posterior to prior for pseudoinputs
        pkl_loss = (plog_q_z - plog_p_z).sum(axis=1).sum()

        # L(Original Data)+gamma*L(Pseudo Inputs)
        lossSum = (recon_loss + self.beta * kl_loss) + (
            gamma * x.shape[0] / self.num_pseudos
        ) * (pseduo_recon_loss + self.beta * pkl_loss)
        return lossSum, recon_loss, kl_loss, pseduo_recon_loss, pkl_loss

    # Computes the log-likelihood of an `x` originating from a mixture of Gaussian distributions
    # parameterized by `mean` and `log_var` whose shapes are [batch_size, num_psuedos, lsdim]
    def prior_likelihood(self, x, mean, log_var, dim=None):

        # T:(batch-size, num-pseudos, sample_size, lsdim)
        # T[i,j,k,l]=element i, marginal probability along axis l in sample k for posterior j
        log_normal = -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))

        # T: (batch-size, num-pseudos, sample_size)
        # T[i,j,k]=log probability that sample k of element i originates from posterior j
        return torch.sum(log_normal, dim)

    def log_p_z(self, z):

        # calculate params
        X = self.pseudoGen.forward(self.idle_input)

        # calculate params for given data
        z_p_mean, z_p_logvar = self.vae.encoder(
            X.view(-1, 1, self.input_length, self.input_length)
        )

        # [batch_size, 1, sample_size, lsdim]
        z_expand = z.unsqueeze(1)
        # z_expand = z_expand.repeat(1, self.num_pseudos, 1)

        # [1, num_pseudos, 1, lsdim]
        z_p_mean = z_p_mean.unsqueeze(0)
        z_p_mean = z_p_mean.unsqueeze(2)
        # means = means.repeat(z_expand.shape[0], 1, 1)

        # [1, num_pseudos, 1, lsdim]
        z_p_logvar = z_p_logvar.unsqueeze(0)
        z_p_logvar = z_p_logvar.unsqueeze(2)

        # Computes log-likelihood across all components (weighted by number of pseudos)
        a = self.prior_likelihood(z_expand, z_p_mean, z_p_logvar, dim=3) - np.log(
            self.num_pseudos
        )

        # We will scale/shift by the maximum posterior probabilities in order to ensure numerical
        # stability when calculating the exp in the final step
        a_max, _ = torch.max(a, 1)  # MB x 1

        # Calculate the log-likelihood across all pseudo-inputs by taking the logarithm
        # of the sum of individual likelihoods across each pseudo-input posterior (component of the prior).
        # Use of a_max cancels out but is kept for numerical stability of exp operation
        log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))
        return log_prior

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-6)
