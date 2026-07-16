from __future__ import annotations

import math

import pytest

from nanochat.optimizer_schedule import (
    learning_rate_multiplier,
    muon_momentum,
    weight_decay_multiplier,
)


SOURCE_AND_TARGET_HORIZONS = (144, 399, 400, 401, 729, 1141, 1142, 1735, 4614)
REFERENCE_ITERATIONS = 2615
LR_RATIO = 40 / REFERENCE_ITERATIONS
MOMENTUM_RATIO = 400 / REFERENCE_ITERATIONS


def test_legacy_momentum_trajectory_exposes_the_short_horizon_regime_change() -> None:
    final_144 = muon_momentum(
        143, num_iterations=144, warmup_steps=400, warmdown_ratio=0.65
    )
    before_transition = muon_momentum(
        399, num_iterations=729, warmup_steps=400, warmdown_ratio=0.65
    )
    after_transition = muon_momentum(
        400, num_iterations=729, warmup_steps=400, warmdown_ratio=0.65
    )

    assert final_144 == pytest.approx(0.8929)
    assert before_transition == pytest.approx(0.9697)
    assert after_transition == pytest.approx(0.9485864979)
    assert before_transition - after_transition > 0.02


@pytest.mark.parametrize("num_iterations", SOURCE_AND_TARGET_HORIZONS)
def test_self_similar_momentum_warmup_finishes_before_warmdown(
    num_iterations: int,
) -> None:
    warmup_steps = max(1, round(MOMENTUM_RATIO * num_iterations))
    warmdown_start = num_iterations - round(0.65 * num_iterations)
    assert warmup_steps < warmdown_start

    before = muon_momentum(
        warmup_steps - 1,
        num_iterations=num_iterations,
        warmup_steps=warmup_steps,
        warmdown_ratio=0.65,
    )
    after = muon_momentum(
        warmup_steps,
        num_iterations=num_iterations,
        warmup_steps=warmup_steps,
        warmdown_ratio=0.65,
    )
    assert 0.0 <= after - before <= 0.12 / warmup_steps + 1e-12
    assert after == pytest.approx(0.97)


@pytest.mark.parametrize("num_iterations", SOURCE_AND_TARGET_HORIZONS)
def test_realized_self_similar_lr_and_weight_decay_are_finite(
    num_iterations: int,
) -> None:
    warmup_steps = max(1, round(LR_RATIO * num_iterations))
    lr_values = [
        learning_rate_multiplier(
            step,
            num_iterations=num_iterations,
            warmup_steps=warmup_steps,
            warmdown_ratio=0.65,
            final_lr_fraction=0.05,
        )
        for step in (0, warmup_steps - 1, num_iterations - 1, num_iterations)
    ]
    wd_values = [
        weight_decay_multiplier(step, num_iterations=num_iterations)
        for step in (0, num_iterations // 2, num_iterations)
    ]
    assert all(math.isfinite(value) for value in lr_values + wd_values)
    assert lr_values[0] > 0.0
    assert lr_values[-1] == pytest.approx(0.05)
    assert wd_values == pytest.approx([1.0, 0.5, 0.0], abs=0.006)
