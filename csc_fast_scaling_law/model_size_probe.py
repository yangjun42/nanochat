#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from nanochat.gpt import GPT, GPTConfig


SIZE_FIELDNAMES = [
    "depth",
    "aspect_ratio",
    "head_dim",
    "max_seq_len",
    "vocab_size",
    "window_pattern",
    "n_embd",
    "n_head",
    "N_total",
    "N_scaling",
    "flops_per_token_est",
    "wte",
    "value_embeds",
    "lm_head",
    "transformer_matrices",
    "scalars",
]

TRIPLET_FIELDNAMES = [
    "rank",
    "control_changes",
    "spacing_error",
    "geometric_rel_error",
    "n_ratio_01",
    "n_ratio_12",
    "n_total_ratio",
    "depths",
    "aspect_ratios",
    "head_dims",
    "n_embds",
    "n_heads",
    "n_values",
    "flops_per_token_est_values",
]


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(item))
    return sorted(set(values))


def _rounded_model_dim(*, depth: int, aspect_ratio: int, head_dim: int) -> int:
    base_dim = depth * aspect_ratio
    return ((base_dim + head_dim - 1) // head_dim) * head_dim


def _compact_values(values: Iterable[Any]) -> str:
    return ",".join(f"{float(value):.12g}" if isinstance(value, float) else str(value) for value in values)


def make_model_size_row(
    *,
    depth: int,
    aspect_ratio: int = 64,
    head_dim: int = 128,
    max_seq_len: int = 2048,
    vocab_size: int = 32768,
    window_pattern: str = "L",
) -> dict[str, int | float | str]:
    """Return nanochat model size metadata without allocating real weights."""

    if min(depth, aspect_ratio, head_dim, max_seq_len, vocab_size) <= 0:
        raise ValueError("depth, aspect_ratio, head_dim, max_seq_len, and vocab_size must be positive")

    model_dim = _rounded_model_dim(depth=depth, aspect_ratio=aspect_ratio, head_dim=head_dim)
    n_head = model_dim // head_dim
    config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )
    with torch.device("meta"):
        model = GPT(config)

    counts = model.num_scaling_params()
    n_scaling = counts["transformer_matrices"] + counts["lm_head"]
    return {
        "depth": depth,
        "aspect_ratio": aspect_ratio,
        "head_dim": head_dim,
        "max_seq_len": max_seq_len,
        "vocab_size": vocab_size,
        "window_pattern": window_pattern,
        "n_embd": model_dim,
        "n_head": n_head,
        "N_total": counts["total"],
        "N_scaling": n_scaling,
        "flops_per_token_est": model.estimate_flops(),
        "wte": counts["wte"],
        "value_embeds": counts["value_embeds"],
        "lm_head": counts["lm_head"],
        "transformer_matrices": counts["transformer_matrices"],
        "scalars": counts["scalars"],
    }


def make_model_size_table(
    *,
    depths: Iterable[int],
    aspect_ratios: Iterable[int],
    head_dim: int = 128,
    max_seq_len: int = 2048,
    vocab_size: int = 32768,
    window_pattern: str = "L",
) -> list[dict[str, int | float | str]]:
    rows = []
    for aspect_ratio, depth in itertools.product(sorted(aspect_ratios), sorted(depths)):
        rows.append(
            make_model_size_row(
                depth=depth,
                aspect_ratio=aspect_ratio,
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                vocab_size=vocab_size,
                window_pattern=window_pattern,
            )
        )
    return rows


def select_geometric_triplet(
    rows: Iterable[dict[str, Any]],
    *,
    value_key: str = "N_scaling",
    control_keys: tuple[str, ...] = ("aspect_ratio", "head_dim", "max_seq_len", "window_pattern"),
    min_step_ratio: float = 1.25,
    min_total_ratio: float = 2.0,
    max_value: float | None = None,
) -> list[dict[str, Any]]:
    """Select the three rows whose measured size values are closest to geometric."""

    ranked = rank_geometric_triplets(
        rows,
        value_key=value_key,
        control_keys=control_keys,
        min_step_ratio=min_step_ratio,
        min_total_ratio=min_total_ratio,
        max_value=max_value,
        top_k=1,
    )
    if not ranked:
        raise ValueError("no non-degenerate size triplet satisfies the minimum ratio constraints")
    return ranked[0]["rows"]


def rank_geometric_triplets(
    rows: Iterable[dict[str, Any]],
    *,
    value_key: str = "N_scaling",
    control_keys: tuple[str, ...] = ("aspect_ratio", "head_dim", "max_seq_len", "window_pattern"),
    min_step_ratio: float = 1.25,
    min_total_ratio: float = 2.0,
    max_value: float | None = None,
    max_control_changes: int | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Rank candidate model-size triplets by fixed controls, spacing, and span."""

    candidates = [
        row
        for row in rows
        if row.get(value_key) is not None and math.isfinite(float(row[value_key])) and float(row[value_key]) > 0
    ]
    if max_value is not None:
        candidates = [row for row in candidates if float(row[value_key]) <= max_value]
    if len(candidates) < 3:
        raise ValueError("at least three positive size rows are required")

    if max_control_changes == 0:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in candidates:
            grouped.setdefault(tuple(row.get(key) for key in control_keys), []).append(row)
        combo_iterables = [itertools.combinations(group, 3) for group in grouped.values() if len(group) >= 3]
        combinations_iter = itertools.chain.from_iterable(combo_iterables)
    else:
        combinations_iter = itertools.combinations(candidates, 3)

    scored: list[tuple[tuple[int, float, float], list[dict[str, Any]], list[float], list[float]]] = []
    for combo in combinations_iter:
        ordered = sorted(combo, key=lambda row: float(row[value_key]))
        values = [float(row[value_key]) for row in ordered]
        if values[1] / values[0] < min_step_ratio or values[2] / values[1] < min_step_ratio:
            continue
        if values[2] / values[0] < min_total_ratio:
            continue
        log_ratios = [math.log(values[1] / values[0]), math.log(values[2] / values[1])]
        spacing_error = abs(log_ratios[1] - log_ratios[0])
        control_changes = len({tuple(row.get(key) for key in control_keys) for row in ordered}) - 1
        if max_control_changes is not None and control_changes > max_control_changes:
            continue
        span = math.log(values[2] / values[0])
        score = (control_changes, spacing_error, -span)
        scored.append((score, ordered, values, [values[1] / values[0], values[2] / values[1]]))
    if not scored:
        raise ValueError("no non-degenerate size triplet satisfies the minimum ratio constraints")

    ranked: list[dict[str, Any]] = []
    for rank, (score, ordered, values, ratios) in enumerate(sorted(scored, key=lambda item: item[0])[:top_k], start=1):
        ratio_denom = max(abs(ratios[0]), abs(ratios[1]), 1e-300)
        ranked.append(
            {
                "rank": rank,
                "control_changes": score[0],
                "spacing_error": score[1],
                "geometric_rel_error": abs(ratios[1] - ratios[0]) / ratio_denom,
                "n_ratio_01": ratios[0],
                "n_ratio_12": ratios[1],
                "n_total_ratio": values[2] / values[0],
                "depths": _compact_values(row.get("depth") for row in ordered),
                "aspect_ratios": _compact_values(row.get("aspect_ratio") for row in ordered),
                "head_dims": _compact_values(row.get("head_dim") for row in ordered),
                "n_embds": _compact_values(row.get("n_embd") for row in ordered),
                "n_heads": _compact_values(row.get("n_head") for row in ordered),
                "n_values": _compact_values(values),
                "flops_per_token_est_values": _compact_values(
                    row.get("flops_per_token_est", "") for row in ordered
                ),
                "rows": ordered,
            }
        )
    return ranked


def write_size_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=SIZE_FIELDNAMES,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_triplet_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=TRIPLET_FIELDNAMES,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe nanochat model sizes without training.")
    parser.add_argument("--depths", default="1-12", help="Comma list and/or inclusive ranges, e.g. 1,2,4 or 1-12.")
    parser.add_argument("--aspect-ratios", default="64", help="Comma list and/or inclusive ranges.")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--window-pattern", default="L")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--print-triplet", action="store_true")
    parser.add_argument("--triplet-out", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-step-ratio", type=float, default=1.25)
    parser.add_argument("--min-total-ratio", type=float, default=2.0)
    parser.add_argument("--max-n-scaling", type=float, default=None)
    parser.add_argument("--max-control-changes", type=int, default=None)
    args = parser.parse_args()

    rows = make_model_size_table(
        depths=_parse_int_list(args.depths),
        aspect_ratios=_parse_int_list(args.aspect_ratios),
        head_dim=args.head_dim,
        max_seq_len=args.max_seq_len,
        vocab_size=args.vocab_size,
        window_pattern=args.window_pattern,
    )
    write_size_table(args.out, rows)
    ranked_triplets = rank_geometric_triplets(
        rows,
        min_step_ratio=args.min_step_ratio,
        min_total_ratio=args.min_total_ratio,
        max_value=args.max_n_scaling,
        max_control_changes=args.max_control_changes,
        top_k=args.top_k,
    )
    if args.triplet_out:
        write_triplet_table(args.triplet_out, ranked_triplets)
    if args.print_triplet:
        for row in ranked_triplets[0]["rows"]:
            print(
                f"depth={row['depth']} aspect_ratio={row['aspect_ratio']} "
                f"N_scaling={row['N_scaling']}"
            )


if __name__ == "__main__":
    main()
