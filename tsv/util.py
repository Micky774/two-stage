import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import io
import PIL.Image
from torchvision.transforms import ToTensor
import lightning as L


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
    def __init__(self, in_dim, out_dim, depth=2, kernel_size=3) -> None:
        super().__init__()
        self.up_channel_conv = nn.Conv2d(in_dim, out_dim, kernel_size, padding="same")
        self.up_channel_prelu = nn.PReLU(out_dim)
        self.conv_modules = nn.ModuleList(
            [
                nn.Conv2d(out_dim, out_dim, kernel_size, padding="same")
                for _ in range(depth - 1)
            ]
        )
        self.prelu_modules = nn.ModuleList(
            [nn.PReLU(out_dim) for _ in range(depth - 1)]
        )
        self.batch_norm = nn.BatchNorm2d(out_dim)
        self.bypass_prelu = nn.PReLU(out_dim)
        self.bypass_conv = nn.Conv2d(in_dim, out_dim, kernel_size, padding="same")

    def forward(self, x):
        _x = x.clone()
        y = self.up_channel_conv(x)
        y = self.up_channel_prelu(y)
        for conv, prelu in zip(self.conv_modules, self.prelu_modules):
            y = self.batch_norm(y)
            y = prelu(conv(y))
        return y + self.bypass_prelu(self.bypass_conv(_x))


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
        _x = x.clone()
        y = self.up_feature_fc(x)
        for fc, prelu in zip(self.fc_modules, self.prelu_modules):
            y = prelu(y)
            y = fc(y)
        return self.bypass_prelu(y + self.bypass_fc(_x))


class ScaleBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        is_training,
        block_per_scale=1,
        depth_per_block=2,
        kernel_size=3,
    ) -> None:
        super().__init__()
        self.is_training = is_training
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.up_channel_block = ResidualConv(
            in_dim, out_dim, depth_per_block, kernel_size
        )
        self.blocks = nn.ModuleList(
            [
                ResidualConv(out_dim, out_dim, depth_per_block, kernel_size)
                for _ in range(block_per_scale - 1)
            ]
        )

    def forward(self, x):
        y = self.up_channel_block(x)
        for block in self.blocks:
            y = block(y)
        return y


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


def kaiming_init(model: nn.Module):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        elif "prelu" in name:
            param.data.fill_(0.25)
        elif "log_gamma" in name or "batch_norm" in name:
            pass
        elif "fc" in name:
            nn.init.kaiming_uniform_(param.data, a=0.25, mode="fan_out")
        elif "conv" in name:
            nn.init.kaiming_uniform_(param.data, a=0.25, mode="fan_out")
        else:
            print(f"Skipping {name} initialization")


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


class SBD(L.LightningModule):
    """
    Constructs spatial broadcast decoder

    @param input_length width of image
    @param lsdim dimensionality of latent space
    @param kernel_size size of size-preserving kernel. Must be odd.
    @param channels list of output-channels for each of the four size-preserving convolutional layers
    """

    def __init__(self, input_length, lsdim, kernel_size=3, channels_per_layer=64):
        super().__init__()
        self.input_length = input_length
        self.lsdim = lsdim
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = int((kernel_size - 1) / 2)
        # Size-Preserving Convolutions
        self.conv1 = nn.Conv2d(
            lsdim + 2, channels_per_layer, kernel_size=kernel_size, padding=padding
        )
        self.conv2 = nn.Conv2d(
            channels_per_layer,
            channels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv3 = nn.Conv2d(
            channels_per_layer,
            channels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv4 = nn.Conv2d(
            channels_per_layer,
            channels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv5 = nn.Conv2d(channels_per_layer, 1, 1)

    """
    Applies the spatial broadcast decoder to a code z

    @param z the code to be decoded
    @return the decoding of z
    """

    def forward(self, z):
        baseVector = z.view(-1, self.lsdim, 1, 1)
        base = baseVector.repeat(1, 1, self.input_length, self.input_length)

        stepTensor = torch.linspace(-1, 1, self.input_length)

        xAxisVector = stepTensor.view(1, 1, self.input_length, 1)
        yAxisVector = stepTensor.view(1, 1, 1, self.input_length)

        xPlane = xAxisVector.repeat(z.shape[0], 1, 1, self.input_length).to(self.device)
        yPlane = yAxisVector.repeat(z.shape[0], 1, self.input_length, 1).to(self.device)

        base = torch.cat((xPlane, yPlane, base), 1)

        x = F.leaky_relu(self.conv1(base))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))

        return x


def generate_embedding(train_dataloader, encoder, device, tb, current_epoch):
    fig, axes = plt.subplots(1, 1)
    embeddings = []
    labels = []
    for batch in train_dataloader:
        x, y = batch
        z, *_ = encoder(x.to(device))
        embeddings.append(z.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    axes.scatter(embeddings[:, 0], embeddings[:, 1], c=labels)
    tb.add_image(
        f"Latent Space",
        plot_to_image(fig),
        current_epoch,
    )
