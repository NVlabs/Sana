"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com

Clean-room FVD implementation. I3D network adapted from VideoGPT (MIT,
https://github.com/wilson1yan/VideoGPT), itself derived from
piergiaj/pytorch-i3d (Apache-2.0). Protocol constants follow the
videogpt_i3d_logits_400 convention: 400-dim classifier logits features,
224x224 bilinear stretch (no aspect preservation, no crop), [-1,1] input
range, non-overlapping 16-frame clips, covariance denominator N-1,
eigh-based matrix sqrt with negative eigenvalues clamped to 0, no eps.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Original code from https://github.com/piergiaj/pytorch-i3d
class MaxPool3dSamePadding(nn.MaxPool3d):
    """Three-dimensional max pooling with TensorFlow-style same padding."""

    def compute_pad(self, dim: int, size: int) -> int:
        if size % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        return max(self.kernel_size[dim] - (size % self.stride[dim]), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, time, height, width = x.size()
        pad_t = self.compute_pad(0, time)
        pad_h = self.compute_pad(1, height)
        pad_w = self.compute_pad(2, width)

        pad_t_front = pad_t // 2
        pad_t_back = pad_t - pad_t_front
        pad_h_front = pad_h // 2
        pad_h_back = pad_h - pad_h_front
        pad_w_front = pad_w // 2
        pad_w_back = pad_w - pad_w_front

        x = F.pad(x, (pad_w_front, pad_w_back, pad_h_front, pad_h_back, pad_t_front, pad_t_back))
        return super().forward(x)


class Unit3D(nn.Module):
    """Three-dimensional convolution with optional batch normalization."""

    def __init__(
        self,
        in_channels: int,
        output_channels: int,
        kernel_shape: tuple[int, int, int] = (1, 1, 1),
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: int = 0,
        activation_fn=F.relu,
        use_batch_norm: bool = True,
        use_bias: bool = False,
        name: str = "unit_3d",
    ):
        super().__init__()
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=output_channels,
            kernel_size=kernel_shape,
            stride=stride,
            padding=0,
            bias=use_bias,
        )
        if use_batch_norm:
            self.bn = nn.BatchNorm3d(output_channels, eps=1e-5, momentum=0.001)

    def compute_pad(self, dim: int, size: int) -> int:
        if size % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        return max(self._kernel_shape[dim] - (size % self._stride[dim]), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, time, height, width = x.size()
        pad_t = self.compute_pad(0, time)
        pad_h = self.compute_pad(1, height)
        pad_w = self.compute_pad(2, width)

        pad_t_front = pad_t // 2
        pad_t_back = pad_t - pad_t_front
        pad_h_front = pad_h // 2
        pad_h_back = pad_h - pad_h_front
        pad_w_front = pad_w // 2
        pad_w_back = pad_w - pad_w_front

        x = F.pad(x, (pad_w_front, pad_w_back, pad_h_front, pad_h_back, pad_t_front, pad_t_back))
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    """I3D inception block."""

    def __init__(self, in_channels: int, out_channels: list[int], name: str):
        super().__init__()
        self.b0 = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_shape=(1, 1, 1),
            name=name + "/Branch_0/Conv3d_0a_1x1",
        )
        self.b1a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_shape=(1, 1, 1),
            name=name + "/Branch_1/Conv3d_0a_1x1",
        )
        self.b1b = Unit3D(
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_shape=(3, 3, 3),
            name=name + "/Branch_1/Conv3d_0b_3x3",
        )
        self.b2a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_shape=(1, 1, 1),
            name=name + "/Branch_2/Conv3d_0a_1x1",
        )
        self.b2b = Unit3D(
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_shape=(3, 3, 3),
            name=name + "/Branch_2/Conv3d_0b_3x3",
        )
        self.b3a = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_shape=(1, 1, 1),
            name=name + "/Branch_3/Conv3d_0b_1x1",
        )
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat((b0, b1, b2, b3), dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture with Kinetics classifier logits."""

    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c",
    )

    def __init__(
        self,
        num_classes: int = 400,
        spatial_squeeze: bool = True,
        name: str = "inception_i3d",
        in_channels: int = 3,
        dropout_keep_prob: float = 0.5,
    ):
        super().__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze

        self.end_points = {
            "Conv3d_1a_7x7": Unit3D(
                in_channels=in_channels,
                output_channels=64,
                kernel_shape=(7, 7, 7),
                stride=(2, 2, 2),
                padding=3,
                name=name + "Conv3d_1a_7x7",
            ),
            "MaxPool3d_2a_3x3": MaxPool3dSamePadding(
                kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0
            ),
            "Conv3d_2b_1x1": Unit3D(
                in_channels=64,
                output_channels=64,
                kernel_shape=(1, 1, 1),
                name=name + "Conv3d_2b_1x1",
            ),
            "Conv3d_2c_3x3": Unit3D(
                in_channels=64,
                output_channels=192,
                kernel_shape=(3, 3, 3),
                padding=1,
                name=name + "Conv3d_2c_3x3",
            ),
            "MaxPool3d_3a_3x3": MaxPool3dSamePadding(
                kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0
            ),
            "Mixed_3b": InceptionModule(192, [64, 96, 128, 16, 32, 32], name + "Mixed_3b"),
            "Mixed_3c": InceptionModule(256, [128, 128, 192, 32, 96, 64], name + "Mixed_3c"),
            "MaxPool3d_4a_3x3": MaxPool3dSamePadding(
                kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=0
            ),
            "Mixed_4b": InceptionModule(480, [192, 96, 208, 16, 48, 64], name + "Mixed_4b"),
            "Mixed_4c": InceptionModule(512, [160, 112, 224, 24, 64, 64], name + "Mixed_4c"),
            "Mixed_4d": InceptionModule(512, [128, 128, 256, 24, 64, 64], name + "Mixed_4d"),
            "Mixed_4e": InceptionModule(512, [112, 144, 288, 32, 64, 64], name + "Mixed_4e"),
            "Mixed_4f": InceptionModule(528, [256, 160, 320, 32, 128, 128], name + "Mixed_4f"),
            "MaxPool3d_5a_2x2": MaxPool3dSamePadding(
                kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0
            ),
            "Mixed_5b": InceptionModule(832, [256, 160, 320, 32, 128, 128], name + "Mixed_5b"),
            "Mixed_5c": InceptionModule(832, [384, 192, 384, 48, 128, 128], name + "Mixed_5c"),
        }
        for endpoint, module in self.end_points.items():
            self.add_module(endpoint, module)

        self.avg_pool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(
            in_channels=1024,
            output_channels=num_classes,
            kernel_shape=(1, 1, 1),
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for endpoint in self.VALID_ENDPOINTS:
            x = self._modules[endpoint](x)
        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            x = x.squeeze(3).squeeze(3)
        return x.mean(dim=2)


class FVDI3D(InceptionI3d):
    """I3D model for videogpt_i3d_logits_400 feature extraction."""

    feature_dim = 400

    def __init__(self):
        super().__init__(num_classes=400, in_channels=3)

    @torch.no_grad()
    def extract(self, videos: torch.Tensor) -> torch.Tensor:
        """Extract one logits feature vector per uint8 BTHWC video."""
        # Train mode would activate Dropout and batch-stat BN: silently corrupted
        # features AND mutated running stats. The model config must carry
        # ``runtime: {training: false}``.
        assert not self.training, "FVDI3D.extract requires eval mode (runtime: {training: false})"
        assert videos.dtype == torch.uint8
        assert videos.ndim == 5 and videos.shape[-1] == 3

        batch, time, height, width, channels = videos.shape
        x = videos.to(dtype=torch.float32).div(255.0)
        x = x.reshape(batch * time, height, width, channels).permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x.reshape(batch, time, channels, 224, 224)
        x = x.mul(2.0).sub(1.0)
        x = x.permute(0, 2, 1, 3, 4)

        parameter_dtype = next(self.parameters()).dtype
        if x.dtype != parameter_dtype:
            x = x.to(dtype=parameter_dtype)
        return super().forward(x).to(dtype=torch.float32)


def chunk_video(video: torch.Tensor, chunk_len: int = 16) -> torch.Tensor:
    """Split a THWC uint8 video into non-overlapping clips, dropping its remainder."""
    assert video.dtype == torch.uint8
    assert video.ndim == 4

    time, height, width, channels = video.shape
    num_chunks = time // chunk_len
    return video[: num_chunks * chunk_len].reshape(num_chunks, chunk_len, height, width, channels)


def gaussian_stats(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a feature mean and unbiased sample covariance in float64."""
    assert features.dtype == torch.float32
    assert features.ndim == 2

    features = features.to(dtype=torch.float64)
    mu = features.mean(dim=0)
    centered = features - mu
    sigma = centered.T @ centered / (features.shape[0] - 1)
    return mu, sigma


def _symmetric_matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(0.0)
    return (eigenvectors * eigenvalues.sqrt().unsqueeze(0)) @ eigenvectors.T


def frechet_distance(
    mu1: torch.Tensor,
    sigma1: torch.Tensor,
    mu2: torch.Tensor,
    sigma2: torch.Tensor,
) -> float:
    """Compute Fréchet distance between two Gaussian feature distributions."""
    mu1 = mu1.to(dtype=torch.float64)
    sigma1 = sigma1.to(dtype=torch.float64)
    mu2 = mu2.to(dtype=torch.float64)
    sigma2 = sigma2.to(dtype=torch.float64)

    mean_distance = (mu1 - mu2).square().sum()
    if not bool(torch.isfinite(sigma1).all()) or not bool(torch.isfinite(sigma2).all()):
        return mean_distance.item()

    sigma1 = (sigma1 + sigma1.T) / 2.0
    sigma2 = (sigma2 + sigma2.T) / 2.0
    sigma1_sqrt = _symmetric_matrix_sqrt(sigma1)
    covariance_product = sigma1_sqrt @ sigma2 @ sigma1_sqrt
    covariance_product_sqrt = _symmetric_matrix_sqrt(covariance_product)
    distance = mean_distance + torch.trace(sigma1 + sigma2 - 2.0 * covariance_product_sqrt)
    return distance.item()


def compute_fvd(feats_real: torch.Tensor, feats_gen: torch.Tensor) -> float:
    """Compute FVD from real and generated feature matrices."""
    mu_real, sigma_real = gaussian_stats(feats_real)
    mu_gen, sigma_gen = gaussian_stats(feats_gen)
    return frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
