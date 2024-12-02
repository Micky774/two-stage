import torch
import torch.nn.functional as F
from torch import nn


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

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size) -> None:
        super().__init__()
        self.convT = nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=2)

    def forward(self, x):
        return self.convT(x)


class ResidualConv(nn.Module):
    def __init__(self, in_dim, out_dim, depth=2, kernel_size=3) -> None:
        super().__init__()
        self.up_channel_conv = nn.Conv2d(in_dim, out_dim, kernel_size, padding="same")
        self.conv_modules = nn.ModuleList(
            [
                nn.Conv2d(out_dim, out_dim, kernel_size, padding="same")
                for _ in range(depth - 1)
            ]
        )
        self.batch_norm = nn.BatchNorm2d(out_dim)
        self.bypass = nn.Conv2d(in_dim, out_dim, kernel_size, padding="same")

    def forward(self, x):
        y = self.up_channel_conv(x)
        for conv in self.conv_modules:
            y = F.relu(self.batch_norm(y))
            y = conv(y)
        return y + self.bypass(x)


class ResidualFC(nn.Module):
    def __init__(self, in_dim, out_dim, depth=2) -> None:
        super().__init__()
        self.up_feature_fc = nn.Linear(in_dim, out_dim)
        self.fc_modules = nn.ModuleList(
            [nn.Linear(out_dim, out_dim) for _ in range(depth - 1)]
        )
        self.bypass = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        y = self.up_feature_fc(x)
        for fc in self.fc_modules:
            y = F.leaky_relu(y)
            y = fc(y)
        return y + self.bypass(x)


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
                ResidualFC(in_dim, out_dim, depth_per_block)
                for _ in range(block_per_scale - 1)
            ]
        )

    def forward(self, x):
        y = self.up_channel_block(x)
        for block in self.blocks:
            y = block(y)
        return y
