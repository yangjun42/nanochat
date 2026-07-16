"""Pure optimizer-schedule functions shared by training and experiment audits."""

from __future__ import annotations

import math


def learning_rate_multiplier(
    step: int,
    *,
    num_iterations: int,
    warmup_steps: int,
    warmdown_ratio: float,
    final_lr_fraction: float,
) -> float:
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    if not 0.0 <= warmdown_ratio < 1.0:
        raise ValueError("warmdown_ratio must be in [0, 1)")
    if not 0.0 <= final_lr_fraction <= 1.0:
        raise ValueError("final_lr_fraction must be in [0, 1]")
    warmdown_steps = round(warmdown_ratio * num_iterations)
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if warmdown_steps == 0 or step <= num_iterations - warmdown_steps:
        return 1.0
    progress = (num_iterations - step) / warmdown_steps
    return progress + (1.0 - progress) * final_lr_fraction


def muon_momentum(
    step: int,
    *,
    num_iterations: int,
    warmup_steps: int,
    warmdown_ratio: float,
) -> float:
    """Return the historical nanochat Muon trajectory.

    Warmup precedence is intentionally preserved for backward compatibility.
    If warmup overlaps warmdown, this legacy rule can jump when warmup ends.
    Self-similar experiment plans prevent overlap by construction.
    """

    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    if not 0.0 <= warmdown_ratio < 1.0:
        raise ValueError("warmdown_ratio must be in [0, 1)")
    warmdown_steps = round(warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_steps
    if warmup_steps > 0 and step < warmup_steps:
        fraction = step / warmup_steps
        return (1.0 - fraction) * 0.85 + fraction * 0.97
    if warmdown_steps > 0 and step >= warmdown_start:
        progress = (step - warmdown_start) / warmdown_steps
        return 0.97 * (1.0 - progress) + 0.90 * progress
    return 0.97


def weight_decay_multiplier(step: int, *, num_iterations: int) -> float:
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")
    return 0.5 * (1.0 + math.cos(math.pi * step / num_iterations))
