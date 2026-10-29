"""Optimizer, scheduler and criterion factories."""

from __future__ import annotations

import pytest
import torch

from sw_dlnet.utils import create_criterion, create_optimizer, create_scheduler, mape_loss


@pytest.fixture()
def params():
    return [torch.nn.Parameter(torch.zeros(2))]


def test_sgd_defaults(params):
    optimizer = create_optimizer("SGD", params, lr=5e-5, weight_decay=5e-5)
    defaults = optimizer.defaults
    assert defaults["momentum"] == 0.9
    assert defaults["nesterov"] is True
    assert defaults["weight_decay"] == 5e-5


@pytest.mark.parametrize("name", ["Adam", "AdamW", "RAdam"])
def test_adaptive_optimizers(name, params):
    optimizer = create_optimizer(name, params, lr=1e-4, weight_decay=1e-4)
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["weight_decay"] == 1e-4


def test_unknown_optimizer_raises(params):
    with pytest.raises(ValueError):
        create_optimizer("Lion", params, lr=1e-3)


def test_criterion_lookup():
    assert create_criterion("mape") is mape_loss
    assert isinstance(create_criterion("mse"), torch.nn.MSELoss)
    assert isinstance(create_criterion("huber"), torch.nn.HuberLoss)
    with pytest.raises(ValueError):
        create_criterion("l1")


def test_onecycle_scheduler_total_steps(params):
    optimizer = torch.optim.SGD(params, lr=0.1)
    scheduler = create_scheduler("onecycle", optimizer, max_lr=1e-3, total_steps=10)
    assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)


def test_plateau_scheduler_reacts_to_loss(params):
    optimizer = torch.optim.SGD(params, lr=0.1)
    scheduler = create_scheduler("plateau", optimizer, factor=0.5, patience=0)
    base_lr = optimizer.param_groups[0]["lr"]
    for _ in range(3):
        scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] < base_lr


def test_cosine_and_step_schedulers(params):
    optimizer = torch.optim.SGD(params, lr=0.1)
    assert isinstance(
        create_scheduler("cosine", optimizer, T_max=10), torch.optim.lr_scheduler.CosineAnnealingLR
    )
    assert isinstance(create_scheduler("step", optimizer), torch.optim.lr_scheduler.StepLR)


def test_unknown_scheduler_raises(params):
    with pytest.raises(ValueError):
        create_scheduler("warmup", torch.optim.SGD(params, lr=0.1))
