#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


METRIC_FIELDS = [
    "stage",
    "split",
    "recipe_band",
    "depth",
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
    "N_total",
    "N_scaling",
    "D_actual",
    "flops_per_token_est",
    "C_6ND",
    "C_arch",
    "gpu_seconds_train",
    "gpu_hours_train",
    "train_wall_seconds",
    "eval_wall_seconds",
    "train_bpb_final",
    "val_bpb_final",
    "train_exit_code",
    "eval_exit_code",
    "train_partition",
    "eval_partition",
    "train_log_path",
    "eval_log_path",
]


def fnum(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_stage_metrics(path: Path) -> dict[str, dict[str, Any]]:
    jsonl = path / "stage_metrics.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    if not jsonl.exists():
        return rows
    with jsonl.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["stage"])] = row
    return rows


def parse_log(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "N_total": r"\bN_total\b\s*[:=]\s*([0-9.eE+-]+)",
        "N_scaling": r"\bN_scaling\b\s*[:=]\s*([0-9.eE+-]+)",
        "flops_per_token_est": r"\bflops_per_token_est\b\s*[:=]\s*([0-9.eE+-]+)",
        "flops_per_token_est_report": r"Estimated FLOPs per token:\s*([0-9.eE+-]+)",
        "train_bpb_final": r"train bpb:\s*([0-9.eE+-]+)",
        "train_bpb_minimum": r"Minimum validation bpb:\s*([0-9.eE+-]+)",
        "val_bpb_final": r"val bpb:\s*([0-9.eE+-]+)",
    }
    out: dict[str, float] = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            canonical_key = {
                "flops_per_token_est_report": "flops_per_token_est",
                "train_bpb_minimum": "train_bpb_final",
            }.get(key, key)
            out[canonical_key] = float(matches[-1].replace(",", ""))

    parameter_counts: dict[str, float] = {}
    for key, raw in re.findall(r"^([A-Za-z0-9_]+)\s*:\s*([0-9][0-9,]*)\s*$", text, flags=re.MULTILINE):
        parameter_counts[key] = float(raw.replace(",", ""))
    if "N_total" not in out and "total" in parameter_counts:
        out["N_total"] = parameter_counts["total"]
    if "N_scaling" not in out:
        if "transformer_matrices" in parameter_counts and "lm_head" in parameter_counts:
            out["N_scaling"] = parameter_counts["transformer_matrices"] + parameter_counts["lm_head"]
        elif "total" in parameter_counts:
            out["N_scaling"] = parameter_counts["total"]
    return out


def _path_from_metric(row: dict[str, Any] | None, fallback_dir: Path, stage: str) -> Path | None:
    if row and row.get("log_path"):
        return Path(str(row["log_path"]))
    candidate = fallback_dir / f"{stage}.log"
    return candidate if candidate.exists() else None


def merge_rows(
    *,
    plan_csv: Path,
    train_metrics_dir: Path,
    eval_metrics_dir: Path,
) -> list[dict[str, Any]]:
    train_metrics = read_stage_metrics(train_metrics_dir)
    eval_metrics = read_stage_metrics(eval_metrics_dir)
    rows: list[dict[str, Any]] = []
    for plan_row in read_csv(plan_csv):
        stage = plan_row["stage"]
        eval_stage = f"bpb_{stage}"
        train_metric = train_metrics.get(stage, {})
        eval_metric = eval_metrics.get(eval_stage, {})
        train_log = _path_from_metric(train_metric, train_metrics_dir.parent / "train_logs", stage)
        eval_log = _path_from_metric(eval_metric, eval_metrics_dir.parent / "eval_logs", eval_stage)

        row: dict[str, Any] = dict(plan_row)
        row.update(parse_log(train_log))
        row.update(parse_log(eval_log))
        row.update(
            {
                "D_actual": fnum(plan_row.get("target_tokens")),
                "train_wall_seconds": fnum(train_metric.get("elapsed_seconds")),
                "eval_wall_seconds": fnum(eval_metric.get("elapsed_seconds")),
                "train_exit_code": train_metric.get("exit_code"),
                "eval_exit_code": eval_metric.get("exit_code"),
                "train_partition": train_metric.get("partition"),
                "eval_partition": eval_metric.get("partition"),
                "train_log_path": str(train_log) if train_log else "",
                "eval_log_path": str(eval_log) if eval_log else "",
            }
        )
        n_scaling = fnum(row.get("N_scaling"))
        d_actual = fnum(row.get("D_actual"))
        flops_per_token = fnum(row.get("flops_per_token_est"))
        train_wall = fnum(row.get("train_wall_seconds"))
        if n_scaling is not None and d_actual is not None:
            row["C_6ND"] = 6.0 * n_scaling * d_actual
        if flops_per_token is not None and d_actual is not None:
            row["C_arch"] = flops_per_token * d_actual
        if train_wall is not None:
            row["gpu_seconds_train"] = train_wall * 4.0
            row["gpu_hours_train"] = row["gpu_seconds_train"] / 3600.0
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def collect_campaign(
    *,
    plan_csv: Path,
    train_metrics_dir: Path,
    eval_metrics_dir: Path,
    out_dir: Path,
) -> dict[str, Path]:
    rows = merge_rows(plan_csv=plan_csv, train_metrics_dir=train_metrics_dir, eval_metrics_dir=eval_metrics_dir)
    fit_ready = [
        row
        for row in rows
        if fnum(row.get("N_scaling")) is not None
        and fnum(row.get("D_actual")) is not None
        and fnum(row.get("val_bpb_final")) is not None
    ]

    metrics_csv = out_dir / "metrics.csv"
    metrics_jsonl = out_dir / "metrics.jsonl"
    fit_ready_cells_csv = out_dir / "fit_ready_cells.csv"
    write_csv(metrics_csv, rows, METRIC_FIELDS)
    write_jsonl(metrics_jsonl, rows)
    write_csv(fit_ready_cells_csv, fit_ready, METRIC_FIELDS)
    return {
        "metrics_csv": metrics_csv,
        "metrics_jsonl": metrics_jsonl,
        "fit_ready_cells_csv": fit_ready_cells_csv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Roihu nanochat scaling-law campaign metrics.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--train-metrics-dir", type=Path, required=True)
    parser.add_argument("--eval-metrics-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = collect_campaign(
        plan_csv=args.plan,
        train_metrics_dir=args.train_metrics_dir,
        eval_metrics_dir=args.eval_metrics_dir,
        out_dir=args.out_dir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
