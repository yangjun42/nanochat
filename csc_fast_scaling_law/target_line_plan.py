#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

from csc_fast_scaling_law.model_size_probe import make_formula_model_size_row


TARGET_LINE_FIELDNAMES = [
    "stage",
    "split",
    "recipe_band",
    "depth",
    "aspect_ratio",
    "head_dim",
    "model_dim",
    "seed",
    "num_iterations",
    "target_tokens",
    "total_batch_size",
    "device_batch_size",
    "max_seq_len",
    "warmup_steps",
    "fp8",
    "window_pattern",
    "model_tag",
    "n_index",
    "d_index",
    "planned_family",
    "planned_model_depth",
    "planned_data_tokens",
    "planned_total_batch_size",
    "planned_note",
    "sizing_family",
    "target_n_scaling",
    "preflight_n_scaling",
    "target_n_scaling_rel_error",
    "n_geometric_rel_error",
    "target_budget_b",
    "target_budget_c",
    "target_budget_line_residual",
    "target_d_actual",
    "target_line_index",
    "target_num_iterations_float",
    "verification_role",
]


DEFAULT_B = 1.305337
X5_C_TARGET = 42.8322801227841


def _format_float(value: float) -> str:
    return repr(float(value))


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def _budget_c_for_multiplier(multiplier: float) -> float:
    if multiplier <= 0.0:
        raise ValueError("budget multiplier must be positive")
    return X5_C_TARGET + math.log(multiplier / 5.0)


def make_target_line_rows(
    *,
    budget_multiplier: float,
    seeds: Iterable[int],
    model_dims: Iterable[int],
    run_slug: str,
    depth: int = 4,
    aspect_ratio: int = 72,
    head_dim: int = 64,
    total_batch_size: int = 524288,
    device_batch_size: int = 16,
    max_seq_len: int = 2048,
    warmup_steps: int = 40,
    fp8: int = 1,
    window_pattern: str = "L",
    b: float = DEFAULT_B,
) -> list[dict[str, str]]:
    """Create measured target-budget-line plan rows in the existing Roihu CSV schema."""

    c_target = _budget_c_for_multiplier(budget_multiplier)
    dims = [int(value) for value in model_dims]
    seed_values = [int(value) for value in seeds]
    rows: list[dict[str, str]] = []
    for seed in seed_values:
        for target_line_index, model_dim in enumerate(dims):
            size = make_formula_model_size_row(
                depth=depth,
                aspect_ratio=aspect_ratio,
                head_dim=head_dim,
                model_dim=model_dim,
                max_seq_len=max_seq_len,
                window_pattern=window_pattern,
            )
            n_scaling = int(size["N_scaling"])
            target_d_actual = math.exp((c_target - math.log(n_scaling)) / b)
            target_num_iterations_float = target_d_actual / total_batch_size
            num_iterations = max(1, int(round(target_num_iterations_float)))
            target_tokens = num_iterations * total_batch_size
            residual = math.log(n_scaling) + b * math.log(target_tokens) - c_target
            stage = (
                f"{run_slug}_u{target_line_index}_dep{depth}_md{model_dim}"
                f"_it{num_iterations}_s{seed}"
            ).replace("-", "_")
            model_tag = (
                f"{run_slug}-u{target_line_index}-dep{depth}-md{model_dim}"
                f"-it{num_iterations}-s{seed}"
            )
            rows.append(
                {
                    "stage": stage,
                    "split": "target_line_referee",
                    "recipe_band": "canonical",
                    "depth": str(depth),
                    "aspect_ratio": str(aspect_ratio),
                    "head_dim": str(head_dim),
                    "model_dim": str(model_dim),
                    "seed": str(seed),
                    "num_iterations": str(num_iterations),
                    "target_tokens": str(target_tokens),
                    "total_batch_size": str(total_batch_size),
                    "device_batch_size": str(device_batch_size),
                    "max_seq_len": str(max_seq_len),
                    "warmup_steps": str(warmup_steps),
                    "fp8": str(fp8),
                    "window_pattern": window_pattern,
                    "model_tag": model_tag,
                    "n_index": str(target_line_index),
                    "d_index": str(target_line_index),
                    "planned_family": "roihu_target_line_measured_oracle",
                    "planned_model_depth": str(depth),
                    "planned_data_tokens": str(target_tokens),
                    "planned_total_batch_size": str(total_batch_size),
                    "planned_note": (
                        "Held-out measured target-budget-line referee row; "
                        "not used for estimator fitting unless explicitly charged as overhead."
                    ),
                    "sizing_family": "controlled_model_dim_target_line",
                    "target_n_scaling": str(n_scaling),
                    "preflight_n_scaling": str(n_scaling),
                    "target_n_scaling_rel_error": "0.0",
                    "n_geometric_rel_error": "",
                    "target_budget_b": _format_float(b),
                    "target_budget_c": _format_float(c_target),
                    "target_budget_line_residual": _format_float(residual),
                    "target_d_actual": _format_float(target_d_actual),
                    "target_line_index": str(target_line_index),
                    "target_num_iterations_float": _format_float(target_num_iterations_float),
                    "verification_role": "target_line_oracle",
                }
            )
    return rows


def write_target_line_plan(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TARGET_LINE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-multiplier", type=float, required=True)
    parser.add_argument("--seeds", type=str, required=True, help="Comma-separated seeds.")
    parser.add_argument("--model-dims", type=str, required=True, help="Comma-separated explicit model_dim values.")
    parser.add_argument("--run-slug", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = make_target_line_rows(
        budget_multiplier=args.budget_multiplier,
        seeds=_parse_int_list(args.seeds),
        model_dims=_parse_int_list(args.model_dims),
        run_slug=args.run_slug,
    )
    write_target_line_plan(args.out, rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
