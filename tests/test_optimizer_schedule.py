from __future__ import annotations

import math

import pytest

from nanochat.optimizer_schedule import (
    learning_rate_multiplier,
    muon_momentum,
    schedule_warmdown_steps,
    validate_schedule_geometry,
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
            schedule_version="exact_last_update_v1",
        )
        for step in (0, warmup_steps - 1, num_iterations - 1)
    ]
    wd_values = [
        weight_decay_multiplier(
            step,
            num_iterations=num_iterations,
            schedule_version="exact_last_update_v1",
        )
        for step in (0, (num_iterations - 1) // 2, num_iterations - 1)
    ]
    assert all(math.isfinite(value) for value in lr_values + wd_values)
    assert lr_values[0] > 0.0
    assert lr_values[-1] == pytest.approx(0.05)
    assert wd_values == pytest.approx([1.0, 0.5, 0.0], abs=0.006)


@pytest.mark.parametrize("num_iterations", SOURCE_AND_TARGET_HORIZONS)
def test_corrected_schedule_reaches_all_endpoints_on_last_real_update(
    num_iterations: int,
) -> None:
    step = num_iterations - 1
    assert learning_rate_multiplier(
        step,
        num_iterations=num_iterations,
        warmup_steps=max(1, round(0.05 * num_iterations)),
        warmdown_ratio=0.65,
        final_lr_fraction=0.05,
        schedule_version="exact_last_update_v1",
    ) == pytest.approx(0.05)
    assert muon_momentum(
        step,
        num_iterations=num_iterations,
        warmup_steps=max(1, round(0.15 * num_iterations)),
        warmdown_ratio=0.65,
        schedule_version="exact_last_update_v1",
    ) == pytest.approx(0.90)
    assert weight_decay_multiplier(
        step,
        num_iterations=num_iterations,
        schedule_version="exact_last_update_v1",
    ) == pytest.approx(0.0)


def test_legacy_and_corrected_endpoint_semantics_are_explicitly_distinct() -> None:
    legacy_last = learning_rate_multiplier(
        143,
        num_iterations=144,
        warmup_steps=5,
        warmdown_ratio=0.65,
        final_lr_fraction=0.05,
        schedule_version="legacy_v0",
    )
    corrected_last = learning_rate_multiplier(
        143,
        num_iterations=144,
        warmup_steps=5,
        warmdown_ratio=0.65,
        final_lr_fraction=0.05,
        schedule_version="exact_last_update_v1",
    )
    assert legacy_last > corrected_last
    assert corrected_last == pytest.approx(0.05)


def test_warmdown_step_rounding_is_explicit_half_up() -> None:
    assert schedule_warmdown_steps(0.65, 10) == 7
    assert schedule_warmdown_steps(0.65, 9) == 6


def test_exact_schedule_rejects_lr_overlap_but_allows_historical_muon_overlap() -> None:
    with pytest.raises(ValueError, match="at least two"):
        validate_schedule_geometry(
            num_iterations=1,
            warmup_steps=0,
            muon_momentum_warmup_steps=0,
            warmdown_ratio=0.65,
            schedule_version="exact_last_update_v1",
        )
    with pytest.raises(ValueError, match="LR warmup overlaps"):
        validate_schedule_geometry(
            num_iterations=10,
            warmup_steps=4,
            muon_momentum_warmup_steps=3,
            warmdown_ratio=0.65,
            schedule_version="exact_last_update_v1",
        )
    assert validate_schedule_geometry(
        num_iterations=216,
        warmup_steps=11,
        muon_momentum_warmup_steps=400,
        warmdown_ratio=0.65,
        schedule_version="exact_last_update_v1",
    ) == 140
    assert muon_momentum(
        215,
        num_iterations=216,
        warmup_steps=400,
        warmdown_ratio=0.65,
        schedule_version="exact_last_update_v1",
    ) == pytest.approx(0.85 + (0.97 - 0.85) * 215 / 400)
    assert learning_rate_multiplier(
        215,
        num_iterations=216,
        warmup_steps=11,
        warmdown_ratio=0.65,
        final_lr_fraction=0.05,
        schedule_version="exact_last_update_v1",
    ) == pytest.approx(0.05)
    assert weight_decay_multiplier(
        215,
        num_iterations=216,
        schedule_version="exact_last_update_v1",
    ) == pytest.approx(0.0)
