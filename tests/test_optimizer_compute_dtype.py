"""CPU checks of optimizer math; fused CUDA/MPS compilation needs GPU validation."""

from __future__ import annotations

import pytest
import torch

from nanochat import optim


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adamw_matches_fp32_reference_from_the_same_stored_state(monkeypatch, dtype):
    # Exercise the same kernel math without paying compiler startup in CPU tests.
    monkeypatch.setattr(optim, "adamw_step_fused", optim.adamw_step_fused.__wrapped__)
    generator = torch.Generator().manual_seed(42)
    parameter = torch.nn.Parameter(torch.randn(8, 5, generator=generator).to(dtype))
    reference = torch.nn.Parameter(parameter.detach().float().clone())
    hypers = dict(lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
    optimizer = optim.MuonAdamW([dict(kind="adamw", params=[parameter], **hypers)])
    reference_optimizer = torch.optim.AdamW([reference], foreach=False, **hypers)
    # Float32 tensor hyperparameters and the reference's Python scalars can
    # round to adjacent bf16 values; allow one stored-dtype relative ULP.
    rtol = max(1e-5, torch.finfo(dtype).eps)

    for _ in range(5):
        gradient = torch.randn(8, 5, generator=generator).to(dtype)
        parameter.grad = gradient
        reference.grad = gradient.float()
        optimizer.step()
        reference_optimizer.step()

        torch.testing.assert_close(parameter, reference.to(dtype), rtol=rtol, atol=1e-7)
        for name in ("exp_avg", "exp_avg_sq"):
            stored = optimizer.state[parameter][name]
            expected = reference_optimizer.state[reference][name]
            assert stored.dtype == dtype
            torch.testing.assert_close(stored, expected.to(dtype), rtol=rtol, atol=1e-7)
            # Start the next comparison from identical stored state, so a
            # one-ULP rounding difference cannot accumulate across steps.
            expected.copy_(stored)
        with torch.no_grad():
            reference.copy_(parameter)


@pytest.mark.parametrize("shape, red_dim", [((2, 8, 4), -1), ((2, 4, 8), -2)])
def test_muon_bf16_orthogonalization_updates_fp32_parameters(monkeypatch, shape, red_dim):
    monkeypatch.setattr(optim, "COMPUTE_DTYPE", torch.bfloat16)
    generator = torch.Generator().manual_seed(42)
    gradients = torch.randn(shape, generator=generator)
    parameters = torch.zeros(shape)
    momentum = torch.zeros_like(parameters)
    state_shape = (*shape[:-1], 1) if red_dim == -1 else (shape[0], 1, shape[-1])
    second_momentum = torch.zeros(state_shape)

    optim.muon_step_fused.__wrapped__(
        gradients, parameters, momentum, second_momentum,
        torch.tensor(0.95), torch.tensor(0.02), torch.tensor(0.0),
        torch.tensor(0.999), 5, red_dim,
    )

    assert parameters.dtype == torch.float32
    assert torch.isfinite(parameters).all()
    assert parameters.norm() > 0
    # bf16 cannot represent beta2=0.999 (it rounds to 1). Keeping the update in
    # fp32 is required for the variance accumulator to receive this first step.
    assert second_momentum.sum() > 0
