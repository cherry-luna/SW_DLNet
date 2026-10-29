"""Model composition, shapes and weight initialisation."""

from __future__ import annotations

import pytest
import torch

from sw_dlnet.models import HybridModel
from sw_dlnet.models.blocks import FeatureFusion, PositionAwareTransformer

TOTAL_PARAMETERS = 40_735_353
TOTAL_STATE_KEYS = 565


def test_parameter_budget():
    torch.manual_seed(0)
    model = HybridModel()
    assert sum(p.numel() for p in model.parameters()) == TOTAL_PARAMETERS
    assert len(model.state_dict()) == TOTAL_STATE_KEYS


def test_state_dict_keys_are_checkpoint_compatible():
    torch.manual_seed(0)
    keys = set(HybridModel().state_dict())
    for key in (
        "resnet_small.feature_extractor.0.weight",
        "fusion.conv1x1.weight",
        "fusion_all.0.weight",
        "transformer.rel_pos_encoder",
        "regressor.fc_out.weight",
    ):
        assert key in keys
    assert "transformer.input_norm.weight" in keys
    assert "transformer.final_norm.weight" in keys


def test_three_resnet_scales():
    model = HybridModel()
    strides = [
        model.resnet_small.feature_extractor[0].stride[0],
        model.resnet_medium.feature_extractor[0].stride[0],
        model.resnet_large.feature_extractor[0].stride[0],
    ]
    assert strides == [2, 4, 8]


@pytest.mark.parametrize("batch_size", [1, 2])
def test_forward_returns_one_scalar_per_sample(batch_size):
    torch.manual_seed(0)
    model = HybridModel(use_amp=False).eval()
    iq = torch.randn(batch_size, 2, 2048)
    data_len = torch.tensor([2048] * batch_size)
    with torch.no_grad():
        output = model(iq, data_len)
    # The head ends with squeeze(), so a single sample yields a 0-dim tensor.
    assert output.reshape(-1).shape == (batch_size,)
    assert torch.isfinite(output).all()


def test_initialisation_is_seeded():
    torch.manual_seed(123)
    first = HybridModel()
    torch.manual_seed(123)
    second = HybridModel()
    for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters()):
        assert torch.equal(a, b), name


def test_fusion_upsamples_with_nearest_neighbour():
    torch.manual_seed(0)
    fusion = FeatureFusion(32).eval()
    x1 = torch.randn(1, 32, 12)
    x2 = torch.randn(1, 32, 5)
    with torch.no_grad():
        output = fusion(x1, x2)
        upsampled = torch.nn.functional.interpolate(x2, size=12)
        manual = fusion.conv1x1(torch.cat([x1, upsampled], dim=1))
        manual = manual * fusion.attention(manual)
    assert output.shape == (1, 32, 12)
    torch.testing.assert_close(output, manual)


def test_transformer_pools_thirty_two_tokens():
    torch.manual_seed(0)
    block = PositionAwareTransformer(feat_dim=32, num_heads=8, num_layers=3).eval()
    features = torch.randn(2, 32, 32)
    data_len = torch.tensor([625_000, 2_000])
    with torch.no_grad():
        output = block(features, data_len)
    assert output.shape == (2, 32)
    assert block.rel_pos_encoder.shape == (32, 32)
