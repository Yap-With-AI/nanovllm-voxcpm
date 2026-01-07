"""AudioVAE encoder/decoder for VoxCPM.

Provides causal VAE for encoding audio waveforms to latent representations
and decoding latents back to audio.
"""

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class CausalConv1d(nn.Conv1d):
    """Causal 1D convolution with left padding."""

    def __init__(self, *args, padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x, (self.__padding * 2, 0))
        return super().forward(x_pad)


class CausalTransposeConv1d(nn.ConvTranspose1d):
    """Causal transpose 1D convolution."""

    def __init__(self, *args, padding: int = 0, output_padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding = padding
        self.__output_padding = output_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)[..., : -(self.__padding * 2 - self.__output_padding)]


def WNCausalConv1d(*args, **kwargs):
    return weight_norm(CausalConv1d(*args, **kwargs))


def WNCausalTransposeConv1d(*args, **kwargs):
    return weight_norm(CausalTransposeConv1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    """Snake activation function for 1D signals."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha)


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class CausalResidualUnit(nn.Module):
    """Causal residual block with dilated convolutions."""

    def __init__(self, dim: int = 16, dilation: int = 1, kernel: int = 7, groups: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNCausalConv1d(
                dim,
                dim,
                kernel_size=kernel,
                dilation=dilation,
                padding=pad,
                groups=groups,
            ),
            Snake1d(dim),
            WNCausalConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        assert pad == 0
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class CausalEncoderBlock(nn.Module):
    """Encoder block with downsampling."""

    def __init__(self, output_dim: int = 16, input_dim: int = None, stride: int = 1, groups: int = 1):
        super().__init__()
        input_dim = input_dim or output_dim // 2
        self.block = nn.Sequential(
            CausalResidualUnit(input_dim, dilation=1, groups=groups),
            CausalResidualUnit(input_dim, dilation=3, groups=groups),
            CausalResidualUnit(input_dim, dilation=9, groups=groups),
            Snake1d(input_dim),
            WNCausalConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CausalEncoder(nn.Module):
    """Causal VAE encoder."""

    def __init__(
        self,
        d_model: int = 64,
        latent_dim: int = 32,
        strides: list = [2, 4, 8, 8],
        depthwise: bool = False,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNCausalConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            groups = d_model // 2 if depthwise else 1
            self.block += [CausalEncoderBlock(output_dim=d_model, stride=stride, groups=groups)]

        groups = d_model if depthwise else 1

        # Create two convolution, for mu and logvar
        self.fc_mu = WNCausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)
        self.fc_logvar = WNCausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)

        # Wrap block into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x: torch.Tensor) -> dict:
        hidden_state = self.block(x)
        return {
            "hidden_state": hidden_state,
            "mu": self.fc_mu(hidden_state),
            "logvar": self.fc_logvar(hidden_state),
        }


class NoiseBlock(nn.Module):
    """Noise injection block for decoder."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = WNCausalConv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        noise = torch.randn((B, 1, T), device=x.device, dtype=x.dtype)
        h = self.linear(x)
        n = noise * h
        x = x + n
        return x


class CausalDecoderBlock(nn.Module):
    """Decoder block with upsampling."""

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        groups: int = 1,
        use_noise_block: bool = False,
    ):
        super().__init__()
        layers = [
            Snake1d(input_dim),
            WNCausalTransposeConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
        ]
        if use_noise_block:
            layers.append(NoiseBlock(output_dim))
        layers.extend(
            [
                CausalResidualUnit(output_dim, dilation=1, groups=groups),
                CausalResidualUnit(output_dim, dilation=3, groups=groups),
                CausalResidualUnit(output_dim, dilation=9, groups=groups),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransposeLastTwoDim(torch.nn.Module):
    """Transpose last two dimensions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.transpose(x, -1, -2)


class CausalDecoder(nn.Module):
    """Causal VAE decoder."""

    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: List[int],
        depthwise: bool = False,
        d_out: int = 1,
        use_noise_block: bool = False,
    ):
        super().__init__()

        # Add first conv layer
        if depthwise:
            layers = [
                WNCausalConv1d(
                    input_channel,
                    input_channel,
                    kernel_size=7,
                    padding=3,
                    groups=input_channel,
                ),
                WNCausalConv1d(input_channel, channels, kernel_size=1),
            ]
        else:
            layers = [WNCausalConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            groups = output_dim if depthwise else 1
            layers += [
                CausalDecoderBlock(
                    input_dim,
                    output_dim,
                    stride,
                    groups=groups,
                    use_noise_block=use_noise_block,
                )
            ]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNCausalConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AudioVAEConfig(BaseModel):
    """Configuration for AudioVAE."""

    encoder_dim: int = 128
    encoder_rates: List[int] = [2, 5, 8, 8]
    latent_dim: int = 64
    decoder_dim: int = 1536
    decoder_rates: List[int] = [8, 8, 5, 2]
    depthwise: bool = True
    sample_rate: int = 16000
    use_noise_block: bool = False


class AudioVAE(nn.Module):
    """Audio Variational Autoencoder for VoxCPM."""

    def __init__(self, config: Optional[AudioVAEConfig] = None):
        # Use default config if none provided
        if config is None:
            config = AudioVAEConfig()

        super().__init__()

        encoder_dim = config.encoder_dim
        encoder_rates = config.encoder_rates
        latent_dim = config.latent_dim
        decoder_dim = config.decoder_dim
        decoder_rates = config.decoder_rates
        depthwise = config.depthwise
        sample_rate = config.sample_rate
        use_noise_block = config.use_noise_block

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.depthwise = depthwise

        self.use_noise_block = use_noise_block

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim
        self.hop_length = np.prod(encoder_rates)
        self.encoder = CausalEncoder(
            encoder_dim,
            latent_dim,
            encoder_rates,
            depthwise=depthwise,
        )

        self.decoder = CausalDecoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            depthwise=depthwise,
            use_noise_block=use_noise_block,
        )
        self.sample_rate = sample_rate
        self.chunk_size = math.prod(encoder_rates)

    def preprocess(self, audio_data: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Preprocess audio data with padding."""
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate
        pad_to = self.hop_length
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / pad_to) * pad_to - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent codes to audio.

        Args:
            z: Tensor[B x D x T] - Latent representation

        Returns:
            Tensor[B x 1 x length] - Decoded audio data
        """
        return self.decoder(z)

    def encode(self, audio_data: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Encode audio to latent representation.

        Args:
            audio_data: Tensor[B x 1 x T]
            sample_rate: int

        Returns:
            z: Tensor[B x D x T]
        """
        if audio_data.ndim == 2:
            audio_data = audio_data.unsqueeze(1)

        audio_data = self.preprocess(audio_data, sample_rate)
        return self.encoder(audio_data)["mu"]

