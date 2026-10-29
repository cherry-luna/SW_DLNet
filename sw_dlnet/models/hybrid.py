"""The hybrid multi-scale ResNet + Transformer regression model."""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import (
    AdaptiveResNet,
    FeatureFusion,
    ImprovedRegressor,
    PositionAwareTransformer,
    SEBlock,
    SpectralFeatures,
    StatisticalFeatures,
)


class HybridModel(nn.Module):
    """Regress the symbol width of an I/Q window.

    Three ResNet encoders operate on the same signal at different
    down-sampling rates; their features are fused with spectral and
    statistical descriptors, encoded by a transformer and finally mapped
    to a single scalar.
    """

    def __init__(self, use_amp: bool = True) -> None:
        super().__init__()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.input_bn = nn.BatchNorm1d(2)

        self.resnet_small = AdaptiveResNet(2, scale=1)
        self.resnet_medium = AdaptiveResNet(2, scale=2)
        self.resnet_large = AdaptiveResNet(2, scale=4)
        self.fusion = FeatureFusion(512)

        self.spectral_features = SpectralFeatures(2)
        self.statistical_features = StatisticalFeatures()

        self.fusion_all = nn.Sequential(
            nn.Conv1d(512 + 256 + 128, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            SEBlock(512),
        )

        self.transformer = PositionAwareTransformer(
            feat_dim=512, num_heads=8, num_layers=12
        )
        self.regressor = ImprovedRegressor(512)

        self._init_weights()
        self.use_amp = use_amp

    def _init_weights(self) -> None:
        """Kaiming initialisation, as used by the original release."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

    def _forward_impl(self, x: torch.Tensor, data_len: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)

        small = self.resnet_small(x)
        medium = self.resnet_medium(x)
        large = self.resnet_large(x)

        spatial = self.fusion(small, medium)
        spatial = self.fusion(spatial, large)

        spectral = self.spectral_features(x)
        statistics = self.statistical_features(x)

        fused = torch.cat([spatial, spectral, statistics], dim=1)
        features = self.fusion_all(fused)

        pooled = self.transformer(features, data_len)
        return self.regressor(pooled).squeeze()

    def forward(self, x: torch.Tensor, data_len: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        if self.use_amp and x.is_cuda:
            with torch.amp.autocast(device_type="cuda"):
                return self._forward_impl(x, data_len)
        return self._forward_impl(x, data_len)