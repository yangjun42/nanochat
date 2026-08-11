"""Pure optimizer-schedule functions shared by training and experiment audits."""

from __future__ import annotations

import math


SCHEDULE_VERSIONS = ("legacy_v0", "exact_last_update_v1")


def schedule_warmdown_steps(warmdown_ratio: float, num_iterations: int) -> int:
    """Convert a nonnegative ratio to steps with explicit round-half-up semantics."""

    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")
    if not 0.0 <= warmdown_ratio < 1.0:
        raise ValueError("warmdown_ratio must be in [0, 1)")
    return int(math.floor(warmdown_ratio * num_iterations + 0.5))


def validate_schedule_geometry(
    *,
    num_iterations: int,
    warmup_steps: int,
    muon_momentum_warmup_steps: int,
    warmdown_ratio: float,
    schedule_version: str,
) -> int:
    """Reject exact schedules whose warmup masks their declared endpoint."""

    if schedule_version not in SCHEDULE_VERSIONS:
        raise ValueError(f"unknown optimizer schedule version: {schedule_version}")
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")
    if warmup_steps < 0 or muon_momentum_warmup_steps < 0:
        raise ValueError("optimizer warmup steps must be nonnegative")
    warmdown_steps = schedule_warmdown_steps(warmdown_ratio, num_iterations)
    if schedule_version == "exact_last_update_v1":
        if num_iterations < 2:
            raise ValueError("exact_last_update_v1 requires at least two updates")
        if warmdown_steps < 1:
            raise ValueError(
                "exact_last_update_v1 requires at least one warmdown step"
            )
        warmdown_start = num_iterations - warmdown_steps
        if warmup_steps > warmdown_start:
            raise ValueError("LR warmup overlaps the exact warmdown interval")
        if muon_momentum_warmup_steps > warmdown_start:
            raise ValueError("Muon warmup overlaps the exact warmdown interval")
    return warmdown_steps


def _validate_schedule_step(step: int, num_iterations: int, schedule_version: str) -> None:
    if schedule_version not in SCHEDULE_VERSIONS:
        raise ValueError(f"unknown optimizer schedule version: {schedule_version}")
    maximum = num_iterations if schedule_version == "legacy_v0" else num_iterations - 1
    if step < 0 or step > maximum:
        raise ValueError(
            f"step {step} is outside the {schedule_version} schedule range 0..{maximum}"
        )


def learning_rate_multiplier(
    step: int,
    *,
    num_iterations: int,
    warmup_steps: int,
    warmdown_ratio: float,
    final_lr_fraction: float,
    schedule_version: str = "legacy_v0",
) -> float:
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    if not 0.0 <= warmdown_ratio < 1.0:
        raise ValueError("warmdown_ratio must be in [0, 1)")
    if not 0.0 <= final_lr_fraction <= 1.0:
        raise ValueError("final_lr_fraction must be in [0, 1]")
    _validate_schedule_step(step, num_iterations, schedule_version)
    warmdown_steps = schedule_warmdown_steps(warmdown_ratio, num_iterations)
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule_version == "legacy_v0":
        if warmdown_steps == 0 or step <= num_iterations - warmdown_steps:
            return 1.0
        progress = (num_iterations - step) / warmdown_steps
        return progress + (1.0 - progress) * final_lr_fraction
    if warmdown_steps == 0:
        return 1.0
    warmdown_start = num_iterations - warmdown_steps
    if step < warmdown_start:
        return 1.0
    if warmdown_steps == 1:
        return final_lr_fraction
    progress = (step - warmdown_start) / (warmdown_steps - 1)
    return (1.0 - progress) + progress * final_lr_fraction


def muon_momentum(
    step: int,
    *,
    num_iterations: int,
    warmup_steps: int,
    warmdown_ratio: float,
    schedule_version: str = "legacy_v0",
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
    _validate_schedule_step(step, num_iterations, schedule_version)
    warmdown_steps = schedule_warmdown_steps(warmdown_ratio, num_iterations)
    warmdown_start = num_iterations - warmdown_steps
    if warmup_steps > 0 and step < warmup_steps:
        fraction = step / warmup_steps
        return (1.0 - fraction) * 0.85 + fraction * 0.97
    if warmdown_steps > 0 and step >= warmdown_start:
        if schedule_version == "exact_last_update_v1" and warmdown_steps == 1:
            return 0.90
        denominator = (
            warmdown_steps - 1
            if schedule_version == "exact_last_update_v1"
            else warmdown_steps
        )
        progress = (step - warmdown_start) / denominator
        return 0.97 * (1.0 - progress) + 0.90 * progress
    return 0.97


def weight_decay_multiplier(
    step: int,
    *,
    num_iterations: int,
    schedule_version: str = "legacy_v0",
) -> float:
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")
    _validate_schedule_step(step, num_iterations, schedule_version)
    denominator = (
        max(num_iterations - 1, 1)
        if schedule_version == "exact_last_update_v1"
        else num_iterations
    )
    return 0.5 * (1.0 + math.cos(math.pi * step / denominator))
