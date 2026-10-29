"""Building blocks of the hybrid ResNet-Transformer model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralFeatures(nn.Module):
    """Magnitude / phase features extracted from the one-sided spectrum."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv_freq = nn.Sequential(
            nn.Conv1d(in_channels * 2, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x.clone(), dim=2)
        features = torch.cat([torch.abs(spectrum), torch.angle(spectrum)], dim=1)
        return self.conv_freq(features)


class StatisticalFeatures(nn.Module):
    """Moments of the I/Q channels, broadcast to 32 positions."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        maximum = x.max(dim=2, keepdim=True)[0]
        minimum = x.min(dim=2, keepdim=True)[0]
        kurtosis = torch.mean((x - mean) ** 4, dim=2, keepdim=True) / (std**4)
        skewness = torch.mean((x - mean) ** 3, dim=2, keepdim=True) / (std**3)

        features = torch.cat([mean, std, maximum, minimum, kurtosis, skewness], dim=1)
        return self.conv(features.repeat(1, 1, 32))


class SEBlock(nn.Module):
    """Squeeze-and-excitation recalibration of a 1-D feature map."""

    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = x.size()
        scale = self.fc(self.avg_pool(x).view(batch, channels)).view(batch, channels, 1)
        return x * scale.expand_as(x)


class ResBlock(nn.Module):
    """Bottleneck residual block with squeezed excitation."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        hidden = out_channels // 4
        self.conv1 = nn.Conv1d(in_channels, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.conv3 = nn.Conv1d(hidden, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.Dropout(0.1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = F.gelu(self.bn2(self.conv2(out)))
        out = self.se(self.bn3(self.conv3(out)))
        return F.gelu(out + residual)


class AdaptiveResNet(nn.Module):
    """ResNet encoder whose down-sampling rate follows the input scale."""

    def __init__(self, in_channels: int, scale: int = 1) -> None:
        super().__init__()
        self.scale = scale
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, stride=2 * scale, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2 * scale, padding=1),
            ResBlock(64, 64, stride=1),
            ResBlock(64, 128, stride=2 * scale),
            ResBlock(128, 256, stride=2 * scale),
            ResBlock(256, 512, stride=2 * scale),
            nn.AdaptiveAvgPool1d(32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)


class PositionAwareTransformer(nn.Module):
    """Three stacked transformer encoders with learned relative positions."""

    def __init__(self, feat_dim: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.rel_pos_encoder = nn.Parameter(torch.randn(32, feat_dim))
        self.input_norm = nn.LayerNorm(feat_dim)
        self.transformers = nn.ModuleList(
            [
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=feat_dim,
                        nhead=num_heads,
                        dim_feedforward=2048,
                        dropout=0.2,
                        batch_first=True,
                    ),
                    num_layers // 3,
                )
                for _ in range(3)
            ]
        )
        self.final_norm = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor, data_len: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)

        pad_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        for i, length in enumerate(data_len):
            valid_len = min(int(length / (2e7 // 32)), 32)
            pad_mask[i, valid_len:] = True

        x = x + self.rel_pos_encoder.unsqueeze(0)
        for transformer in self.transformers:
            x = transformer(x, src_key_padding_mask=~pad_mask)
        return x.mean(dim=1)


class FeatureFusion(nn.Module):
    """1x1 fusion of two feature maps followed by channel attention."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels * 2, in_channels, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // 16, 1),
            nn.ReLU(),
            nn.Conv1d(in_channels // 16, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if x1.size(2) != x2.size(2):
            x2 = F.interpolate(x2, size=x1.size(2))
        fused = self.conv1x1(torch.cat([x1, x2], dim=1))
        return fused * self.attention(fused)


class ImprovedRegressor(nn.Module):
    """Three-layer MLP head with batch norm and dropout."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        return self.fc_out(x)
