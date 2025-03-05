import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import io
import PIL.Image
from torchvision.transforms import ToTensor
import lightning as L
import umap
import os
from matplotlib.patches import Ellipse

ENV = os.environ


def spectral_norm(input_: torch.Tensor):
    """Performs Spectral Normalization on a weight tensor."""
    if input_.ndim < 2:
        raise ValueError(
            "Spectral norm can only be applied to multi-dimensional tensors"
        )

    # The paper says to flatten convnet kernel weights from (C_out, C_in, KH, KW)
    # to (C_out, C_in * KH * KW). But Sonnet's and Compare_gan's Conv2D kernel
    # weight shape is (KH, KW, C_in, C_out), so it should be reshaped to
    # (KH * KW * C_in, C_out), and similarly for other layers that put output
    # channels as last dimension.
    # n.b. this means that w here is equivalent to w.T in the paper.
    w = input_.reshape([-1, input_.shapee[-1]])

    # Persisted approximation of first left singular vector of matrix `w`.

    u_var = torch.normal(0, 1, size=(w.shape[0], 1), requires_grad=False)
    u = u_var

    # Use power iteration method to approximate spectral norm.
    # The authors suggest that "one round of power iteration was sufficient in the
    # actual experiment to achieve satisfactory performance". According to
    # observation, the spectral norm become very accurate after ~20 steps.

    power_iteration_rounds = 1
    for _ in range(power_iteration_rounds):
        # `v` approximates the first right singular vector of matrix `w`.
        v = F.normalize(w.T @ u, dim=None, epsilon=1e-12)
        u = F.normalize(w @ v, dim=None, epsilon=1e-12)

    # Update persisted approximation.
    u = nn.Identity(u)

    # The authors of SN-GAN chose to stop gradient propagating through u and v.
    # In johnme@'s experiments it wasn't clear that this helps, but it doesn't
    # seem to hinder either so it's kept in order to be a faithful implementation.
    u = torch.detach(u)
    v = torch.detach(v)

    # Largest singular value of `w`.
    norm_value = (u.T @ w) @ v
    w_normalized = w / norm_value

    # Unflatten normalized weights to match the unnormalized tensor.
    w_tensor_normalized = w_normalized.reshape(input_.shape)
    return w_tensor_normalized


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(
            in_dim, out_dim, kernel_size, stride=2, padding_mode="reflect"
        )
        self.prelu = nn.PReLU(out_dim)

    def forward(self, x):
        return self.prelu(self.conv(x))


class Upsample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size) -> None:
        super().__init__()
        self.convT = nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=2)
        self.prelu = nn.PReLU(out_dim)

    def forward(self, x):
        return self.prelu(self.convT(x))


class ResidualConv(nn.Module):
    def __init__(
        self, input_shape, in_dim, mid_dim, out_dim, depth=2, kernel_size=3
    ) -> None:
        super().__init__()

        num_channels = [in_dim] + [mid_dim] * depth + [out_dim]
        self.conv_modules = nn.ModuleList(
            [
                nn.Conv2d(
                    num_channels[i], num_channels[i + 1], kernel_size, padding="same"
                )
                for i in range(depth + 1)
            ]
        )
        # No activation on final output -- let the consumer decide
        self.activation_funcs = nn.ModuleList(
            [nn.LeakyReLU() for _ in range(depth)] + [nn.Identity()]
        )
        self.batch_norms = nn.ModuleList(
            [nn.LayerNorm((dim, *input_shape)) for dim in num_channels[:-1]]
        )
        self.bypass_activation_func = nn.LeakyReLU()
        self.bypass_conv = nn.Conv2d(in_dim, out_dim, 1, padding="same")
        self.bypass_batch_norm = nn.LayerNorm((in_dim, *input_shape))

    def forward(self, x):
        y = x
        for conv, activation, batch_norm in zip(
            self.conv_modules, self.activation_funcs, self.batch_norms
        ):
            y = batch_norm(y)
            y = activation(conv(y))
        x = self.bypass_batch_norm(x)
        return y + self.bypass_conv(x)


class ResidualFC(nn.Module):
    def __init__(self, in_dim, out_dim, depth=2) -> None:
        super().__init__()
        self.up_feature_fc = nn.Linear(in_dim, out_dim)
        self.fc_modules = nn.ModuleList(
            [nn.Linear(out_dim, out_dim) for _ in range(depth - 1)]
        )
        self.prelu_modules = nn.ModuleList(
            [nn.PReLU(out_dim) for _ in range(depth - 1)]
        )
        self.bypass_fc = nn.Linear(in_dim, out_dim)
        self.bypass_prelu = nn.PReLU(out_dim)

    def forward(self, x):
        y = self.up_feature_fc(x)
        for fc, prelu in zip(self.fc_modules, self.prelu_modules):
            y = prelu(y)
            y = fc(y)
        return self.bypass_prelu(y + self.bypass_fc(x))


class ScaleBlock(nn.Module):
    def __init__(
        self,
        input_shape,
        in_dim,
        mid_dim,
        out_dim,
        block_dim,
        block_per_scale=1,
        depth_per_block=2,
        kernel_size=3,
    ) -> None:
        super().__init__()
        n_dims = [in_dim] + [mid_dim] * block_per_scale + [out_dim]
        self.blocks = nn.ModuleList(
            [
                ResidualConv(
                    input_shape,
                    n_dims[i],
                    block_dim,
                    n_dims[i + 1],
                    depth_per_block,
                    kernel_size,
                )
                for i in range(block_per_scale + 1)
            ]
        )

    def forward(self, x):
        # No activation on final output -- let the consumer decide
        for block in self.blocks[:-1]:
            x = F.leaky_relu(block(x))
        return self.blocks[-1](x)


class ScaleFCBlock(nn.Module):
    def __init__(self, in_dim, out_dim, block_per_scale=1, depth_per_block=2) -> None:
        super().__init__()
        self.up_channel_block = ResidualFC(in_dim, out_dim, depth_per_block)
        self.blocks = nn.ModuleList(
            [
                ResidualFC(out_dim, out_dim, depth_per_block)
                for _ in range(block_per_scale - 1)
            ]
        )

    def forward(self, x):
        y = self.up_channel_block(x)
        for block in self.blocks:
            y = block(y)
        return y


def kaiming_init(model: nn.Module, a=0.01, debug=False):
    for name, param in model.named_parameters():
        init_as = None
        if name.endswith(".bias"):
            init_as = "bias"
            param.data.fill_(0)
        elif "prelu" in name:
            init_as = "prelu"
            param.data.fill_(0.25)
        elif "log_gamma" in name or "batch_norm" in name:
            pass
        elif "fc" in name:
            init_as = "fc"
            nn.init.kaiming_normal_(param.data, a=a)
        elif "conv" in name:
            init_as = "conv"
            nn.init.kaiming_normal_(param.data, a=a)
        else:
            init_as = "skip"
        if debug:
            print(f"Initialized {name} as {init_as}")


def plot_to_image(fig):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(fig)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image


class DeconvDecoder(L.LightningModule):
    def __init__(self, input_shape, lsdim, channels_per_layer=64):
        super().__init__()
        self.input_channels = input_shape[1]
        self.lsdim = lsdim

        # (3, 3)
        self.in_conv = nn.ConvTranspose2d(lsdim, channels_per_layer, kernel_size=3)

        # (8, 8)
        # (16, 16)
        # (32, 32)
        self.convs = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    channels_per_layer, channels_per_layer, kernel_size=3, stride=2
                )
                for _ in range(2)
            ]
            + [
                nn.ConvTranspose2d(
                    channels_per_layer, channels_per_layer, kernel_size=4, stride=2
                )
            ]
        )
        self.out_conv = nn.Conv2d(channels_per_layer, self.input_channels, 1)

    def forward(self, z):
        t = z.view(-1, self.lsdim, 1, 1)
        t = F.leaky_relu(self.in_conv(t))
        for conv in self.convs:
            t = F.leaky_relu(conv(t))
        t = self.out_conv(t)
        return t


class SBD(L.LightningModule):
    """
    Constructs spatial broadcast decoder

    @param input_length width of image
    @param lsdim dimensionality of latent space
    @param kernel_size size of size-preserving kernel. Must be odd.
    @param channels list of output-channels for each of the four size-preserving convolutional layers
    """

    def __init__(
        self,
        input_shape,
        lsdim,
        kernel_size=3,
        channels_per_layer=64,
        num_blocks=3,
        layers_per_block=3,
    ):
        super().__init__()
        self.input_length = input_shape[-1]
        self.input_channels = input_shape[1]
        self.lsdim = lsdim
        # Size-Preserving Convolutions
        num_channels = (
            [lsdim + 2] + [channels_per_layer] * num_blocks + [self.input_channels]
        )
        self.blocks = nn.ModuleList(
            [
                ResidualConv(
                    input_shape[-2:],
                    num_channels[i],
                    channels_per_layer,
                    num_channels[i + 1],
                    depth=layers_per_block,
                    kernel_size=kernel_size,
                )
                for i in range(num_blocks + 1)
            ]
        )
        stepTensor = torch.linspace(-1, 1, self.input_length)
        self.register_buffer("xAxisVector", stepTensor.view(1, 1, self.input_length, 1))
        self.register_buffer("yAxisVector", stepTensor.view(1, 1, 1, self.input_length))

        kaiming_init(self)

    """
    Applies the spatial broadcast decoder to a code z

    @param z the code to be decoded
    @return the decoding of z
    """

    def forward(self, z):
        baseVector = z.view(-1, self.lsdim, 1, 1)
        base = baseVector.repeat(1, 1, self.input_length, self.input_length)

        xPlane = self.xAxisVector.repeat(z.shape[0], 1, 1, self.input_length).to(
            self.device
        )
        yPlane = self.yAxisVector.repeat(z.shape[0], 1, self.input_length, 1).to(
            self.device
        )

        t = torch.cat((xPlane, yPlane, base), 1)
        for block in self.blocks:
            t = F.leaky_relu(block(t))
        return t


def generate_embedding(
    train_dataloader,
    encoder,
    device,
    tb,
    current_epoch,
    mu_p=None,
    logvar_p=None,
    transform=lambda x: x,
):
    embeddings = []
    targets = []
    for batch in train_dataloader:
        x, y = batch
        z = encoder(x.to(device))
        z = transform(z)
        embeddings.append(z.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
    embeddings = np.concatenate(embeddings)
    targets = np.concatenate(targets)
    fig, axes = plt.subplots(1, 1, figsize=(20, 20))
    if embeddings.shape[-1] > 2:
        if current_epoch % 50 != 0:
            rng = np.random.RandomState(current_epoch)
            sample_idxs = rng.choice(len(embeddings), 4000)
            embeddings = embeddings[sample_idxs]
            targets = targets[sample_idxs]
        reducer = umap.UMAP(min_dist=0)

        visual_embedding = reducer.fit_transform(embeddings)
        if mu_p is not None:
            mu_p = mu_p.detach().cpu().numpy()
            embedded_pseudos = reducer.transform(mu_p)
    else:
        if logvar_p is not None:
            logvar_p = logvar_p.detach().cpu().numpy()
            std_p = np.exp(0.5 * logvar_p)
        visual_embedding = embeddings
        if mu_p is not None:
            embedded_pseudos = mu_p.detach().cpu().numpy()
            for embedded_pseudo, std in zip(embedded_pseudos, std_p):
                axes.add_patch(
                    Ellipse(
                        xy=embedded_pseudo,
                        width=3 * std[0],
                        height=3 * std[1],
                        edgecolor="r",
                        fc="grey",
                        lw=2,
                    )
                )
    assert targets is not None
    axes.scatter(
        visual_embedding[:, 0],
        visual_embedding[:, 1],
        c=targets,
        s=45 / np.sqrt(len(targets)),
        cmap="tab10",
    )
    if mu_p is not None:
        axes.scatter(
            embedded_pseudos[:, 0],
            embedded_pseudos[:, 1],
            c="black",
            s=50,
            marker="x",
        )

    tb.add_image(
        f"Latent Space",
        plot_to_image(fig),
        current_epoch,
    )
