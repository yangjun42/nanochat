#!/usr/bin/env python3
"""Run one manifest-backed nanochat train+BPB task inside a partial directory.

This runner contains no campaign values.  It accepts only the ``prepared.json``
created by :mod:`campaign_contract`, builds an explicit allow-listed CLI, and
writes a compact machine-readable metric.  The outer contract decides whether
the resulting task directory is complete enough to publish atomically.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from csc_fast_scaling_law.campaign_contract import (
    ContractError,
    TASK_ID_RE,
    atomic_write_json,
    sha256_file,
)
from csc_fast_scaling_law.collect_metrics import parse_log
from csc_fast_scaling_law.model_shape import resolve_model_shape


RUNNER_SCHEMA_VERSION = "nanochat-task-runner-v1"
ENVIRONMENT_SCHEMA_VERSION = "nanochat-environment-v1"
SCHEDULE_SCHEMA_VERSION = "nanochat-schedule-manifest-v1"
SAFE_SHARD_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
INTEGER_SCHEDULE_FIELDS = (
    "target_tokens",
    "total_batch_size",
    "warmup_steps",
    "muon_momentum_warmup_steps",
    "weight_decay_horizon_tokens",
)
FLOAT_SCHEDULE_FIELDS = (
    "warmdown_ratio",
    "final_lr_frac",
    "embedding_lr",
    "unembedding_lr",
    "weight_decay",
    "matrix_lr",
    "scalar_lr",
    "target_param_data_ratio",
    "reference_param_data_ratio",
)
STRING_SCHEDULE_FIELDS = (
    "optimizer_schedule_version",
    "attention_backend",
    "compute_dtype",
)
BOOLEAN_SCHEDULE_FIELDS = ("known_schedule_overlap",)
REQUIRED_RUN_FIELDS = (
    "model_tag",
    "depth",
    "aspect_ratio",
    "head_dim",
    "model_dim",
    "max_seq_len",
    "window_pattern",
    "target_tokens",
    "target_param_data_ratio",
    "reference_param_data_ratio",
    "total_batch_size",
    "device_batch_size",
    "eval_device_batch_size",
    "eval_tokens_per_shard",
    "warmup_steps",
    "muon_momentum_warmup_steps",
    "weight_decay_horizon_tokens",
    "warmdown_ratio",
    "final_lr_frac",
    "embedding_lr",
    "unembedding_lr",
    "weight_decay",
    "matrix_lr",
    "scalar_lr",
    "fp8",
    "fp8_recipe",
    "optimizer_schedule_version",
    "allow_data_reuse",
    "attention_backend",
    "compute_dtype",
    "known_schedule_overlap",
    "n_scaling",
    "dataset_key",
    "dataset_manifest_sha256",
    "validation_manifest_sha256",
)


def _integer(row: Mapping[str, str], name: str, *, minimum: int | None = None) -> int:
    try:
        value = int(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"plan field {name} must be an integer") from exc
    if minimum is not None and value < minimum:
        raise ContractError(f"plan field {name} must be >= {minimum}")
    return value


def _number(row: Mapping[str, str], name: str, *, minimum: float | None = None) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"plan field {name} must be numeric") from exc
    if not math.isfinite(value):
        raise ContractError(f"plan field {name} must be finite")
    if minimum is not None and value < minimum:
        raise ContractError(f"plan field {name} must be >= {minimum}")
    return value


def _boolean(row: Mapping[str, str], name: str) -> bool:
    raw = row.get(name, "").strip().lower()
    if raw in {"1", "true", "yes"}:
        return True
    if raw in {"0", "false", "no"}:
        return False
    raise ContractError(f"plan field {name} must be one of 0/1/true/false/yes/no")


def _option(command: list[str], name: str, value: Any) -> None:
    command.append(f"--{name}={value}")


def _validate_row(row: Mapping[str, str]) -> None:
    absent = [name for name in REQUIRED_RUN_FIELDS if not row.get(name)]
    if absent:
        raise ContractError(f"nanochat task row lacks required fields: {', '.join(absent)}")
    if not TASK_ID_RE.fullmatch(row["model_tag"]):
        raise ContractError("model_tag must be a path-safe identifier")
    if not TASK_ID_RE.fullmatch(row["dataset_key"]):
        raise ContractError("dataset_key must be a path-safe identifier")
    for field in ("dataset_manifest_sha256", "validation_manifest_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", row[field]):
            raise ContractError(f"{field} must be a lowercase SHA-256 digest")
    if not re.fullmatch(r"[LS]+", row["window_pattern"]):
        raise ContractError("window_pattern may contain only L and S")
    if row["fp8_recipe"] not in {"tensorwise", "rowwise"}:
        raise ContractError("fp8_recipe must be tensorwise or rowwise")
    if row["optimizer_schedule_version"] not in {"legacy_v0", "exact_last_update_v1"}:
        raise ContractError(
            "optimizer_schedule_version must be legacy_v0 or exact_last_update_v1"
        )
    if row["attention_backend"] != "fa3":
        raise ContractError("formal contracted training requires attention_backend=fa3")
    if row["compute_dtype"] != "bfloat16":
        raise ContractError("formal contracted training requires compute_dtype=bfloat16")
    target_tokens = _integer(row, "target_tokens", minimum=1)
    total_batch_size = _integer(row, "total_batch_size", minimum=1)
    if target_tokens % total_batch_size:
        raise ContractError("target_tokens must be exactly divisible by total_batch_size")
    num_iterations = target_tokens // total_batch_size
    if num_iterations < 2:
        raise ContractError("contracted schedules require at least two optimizer updates")
    _number(row, "target_param_data_ratio", minimum=sys.float_info.min)
    _number(row, "reference_param_data_ratio", minimum=sys.float_info.min)
    for name in (
        "depth",
        "aspect_ratio",
        "head_dim",
        "max_seq_len",
        "device_batch_size",
        "eval_device_batch_size",
        "eval_tokens_per_shard",
    ):
        _integer(row, name, minimum=1)
    _integer(row, "model_dim", minimum=-1)
    gpu_count = _integer(row, "gpu_count", minimum=1)
    sequence_length = _integer(row, "max_seq_len", minimum=1)
    device_batch_size = _integer(row, "device_batch_size", minimum=1)
    eval_device_batch_size = _integer(row, "eval_device_batch_size", minimum=1)
    eval_tokens_per_shard = _integer(row, "eval_tokens_per_shard", minimum=1)
    train_micro_positions = device_batch_size * sequence_length * gpu_count
    if total_batch_size % train_micro_positions:
        raise ContractError(
            "total_batch_size must be divisible by device_batch_size*max_seq_len*gpu_count"
        )
    eval_micro_positions = eval_device_batch_size * sequence_length * gpu_count
    if eval_tokens_per_shard % eval_micro_positions:
        raise ContractError(
            "eval_tokens_per_shard must be divisible by "
            "eval_device_batch_size*max_seq_len*gpu_count"
        )
    warmup_steps = _integer(row, "warmup_steps", minimum=0)
    muon_warmup_steps = _integer(row, "muon_momentum_warmup_steps", minimum=0)
    horizon = _integer(row, "weight_decay_horizon_tokens", minimum=-1)
    if horizon == 0:
        raise ContractError("weight_decay_horizon_tokens may be -1 or positive")
    n_scaling = _integer(row, "n_scaling", minimum=1)
    if row["optimizer_schedule_version"] == "legacy_v0":
        if horizon != 12 * n_scaling:
            raise ContractError(
                "legacy_v0 requires weight_decay_horizon_tokens=12*n_scaling"
            )
    elif horizon != target_tokens:
        raise ContractError(
            "exact_last_update_v1 requires weight_decay_horizon_tokens=target_tokens"
        )
    warmdown_ratio = _number(row, "warmdown_ratio", minimum=0.0)
    if warmdown_ratio >= 1.0:
        raise ContractError("warmdown_ratio must be in [0, 1)")
    final_lr_frac = _number(row, "final_lr_frac", minimum=0.0)
    if final_lr_frac > 1.0:
        raise ContractError("final_lr_frac must be in [0, 1]")
    warmdown_steps = math.floor(num_iterations * warmdown_ratio + 0.5)
    warmdown_start = num_iterations - warmdown_steps
    observed_overlap = warmup_steps > warmdown_start or muon_warmup_steps > warmdown_start
    if _boolean(row, "known_schedule_overlap") is not observed_overlap:
        raise ContractError("known_schedule_overlap does not match the schedule geometry")
    if row["optimizer_schedule_version"] == "exact_last_update_v1" and observed_overlap:
        raise ContractError("exact schedule warmup overlaps the warmdown interval")
    for name in (
        "embedding_lr",
        "unembedding_lr",
        "weight_decay",
        "matrix_lr",
        "scalar_lr",
    ):
        _number(row, name, minimum=0.0)
    _boolean(row, "fp8")
    if _boolean(row, "allow_data_reuse"):
        raise ContractError("contracted training requires allow_data_reuse=false")
    if row.get("bpb_splits", "val") != "val":
        raise ContractError("contracted evaluation requires bpb_splits=val")
    if _boolean(row, "fp8") and row["fp8_recipe"] != "tensorwise":
        raise ContractError("current nanochat FP8 execution supports only fp8_recipe=tensorwise")


def _load_bound_plan_rows(prepared: Mapping[str, Any]) -> list[dict[str, str]]:
    try:
        plan_ref = prepared["file_provenance"]["plan"]
        plan_path = Path(plan_ref["path"]).resolve()
        plan_sha256 = plan_ref["sha256"]
    except (KeyError, TypeError) as exc:
        raise ContractError("prepared record lacks plan provenance") from exc
    if not plan_path.is_file() or sha256_file(plan_path) != plan_sha256:
        raise ContractError("bound plan is missing or hash-mismatched")
    if plan_sha256 != prepared.get("plan_sha256"):
        raise ContractError("prepared plan hashes disagree")
    try:
        with plan_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise ContractError(f"cannot parse bound plan {plan_path}: {exc}") from exc
    if not rows:
        raise ContractError("bound plan contains no tasks")
    task_ids = [row.get("task_id", "") for row in rows]
    if any(not TASK_ID_RE.fullmatch(task_id) for task_id in task_ids):
        raise ContractError("bound plan contains an invalid task_id")
    if len(task_ids) != len(set(task_ids)):
        raise ContractError("bound plan contains duplicate task_id values")
    dataset_keys = {row.get("dataset_key", "") for row in rows}
    if len(dataset_keys) != 1 or not TASK_ID_RE.fullmatch(next(iter(dataset_keys))):
        raise ContractError("bound plan must contain exactly one valid dataset_key")
    expected_data_sha = prepared["file_provenance"]["data_manifest"]["sha256"]
    expected_validation_sha = prepared["file_provenance"]["validation_manifest"]["sha256"]
    for row in rows:
        if row.get("dataset_manifest_sha256") != expected_data_sha:
            raise ContractError(
                f"plan task {row['task_id']} data manifest hash does not match campaign"
            )
        if row.get("validation_manifest_sha256") != expected_validation_sha:
            raise ContractError(
                f"plan task {row['task_id']} validation manifest hash does not match campaign"
            )
    return rows


def _json_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{label} must be a JSON integer")
    return value


def _json_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be a JSON number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ContractError(f"{label} must be finite")
    return parsed


def validate_schedule_manifest(prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Bind every plan task to one exact, typed optimizer schedule record."""

    try:
        manifest_ref = prepared["file_provenance"]["schedule_manifest"]
        manifest_path = Path(manifest_ref["path"]).resolve()
    except (KeyError, TypeError) as exc:
        raise ContractError("prepared record lacks schedule_manifest provenance") from exc
    if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_ref.get(
        "sha256"
    ):
        raise ContractError("schedule manifest is missing or hash-mismatched")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid schedule manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEDULE_SCHEMA_VERSION:
        raise ContractError(f"schedule manifest schema must be {SCHEDULE_SCHEMA_VERSION}")
    schedules = manifest.get("schedules")
    if not isinstance(schedules, list) or not schedules:
        raise ContractError("schedule manifest schedules must be a non-empty list")
    plan_rows = _load_bound_plan_rows(prepared)
    expected_ids = [row["task_id"] for row in plan_rows]
    observed_ids = [
        entry.get("task_id") if isinstance(entry, dict) else None for entry in schedules
    ]
    if observed_ids != expected_ids:
        raise ContractError(
            "schedule manifest task IDs/order do not exactly match the bound plan"
        )
    expected_keys = {
        "task_id",
        "schedule_id",
        *INTEGER_SCHEDULE_FIELDS,
        *FLOAT_SCHEDULE_FIELDS,
        *STRING_SCHEDULE_FIELDS,
        *BOOLEAN_SCHEDULE_FIELDS,
    }
    for index, (row, entry) in enumerate(zip(plan_rows, schedules, strict=True)):
        if not isinstance(entry, dict) or set(entry) != expected_keys:
            raise ContractError(
                f"schedule manifest schedules[{index}] must contain exactly the frozen fields"
            )
        if entry["task_id"] != row["task_id"] or entry["schedule_id"] != row.get(
            "schedule_id"
        ):
            raise ContractError(f"schedule identity mismatch for task {row['task_id']}")
        for field in INTEGER_SCHEDULE_FIELDS:
            observed = _json_integer(
                entry[field], label=f"schedule {row['task_id']} {field}"
            )
            if observed != _integer(row, field):
                raise ContractError(
                    f"schedule {field} mismatch for task {row['task_id']}"
                )
        for field in FLOAT_SCHEDULE_FIELDS:
            observed = _json_number(
                entry[field], label=f"schedule {row['task_id']} {field}"
            )
            if observed != _number(row, field):
                raise ContractError(
                    f"schedule {field} mismatch for task {row['task_id']}"
                )
        for field in STRING_SCHEDULE_FIELDS:
            if not isinstance(entry[field], str) or entry[field] != row.get(field):
                raise ContractError(
                    f"schedule {field} mismatch for task {row['task_id']}"
                )
        for field in BOOLEAN_SCHEDULE_FIELDS:
            if not isinstance(entry[field], bool) or entry[field] is not _boolean(row, field):
                raise ContractError(
                    f"schedule {field} mismatch for task {row['task_id']}"
                )
        _validate_row(row)
    return {
        "path": str(manifest_path),
        "sha256": manifest_ref["sha256"],
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "task_count": len(schedules),
    }


def build_commands(
    prepared: Mapping[str, Any], *, python_executable: str
) -> tuple[list[str], list[str]]:
    row = prepared.get("plan_row")
    if not isinstance(row, dict):
        raise ContractError("prepared record has no plan_row object")
    _validate_row(row)
    file_provenance = prepared.get("file_provenance")
    if not isinstance(file_provenance, dict):
        raise ContractError("prepared record has no file_provenance object")
    try:
        dataset_manifest = file_provenance["data_manifest"]["path"]
        validation_manifest = file_provenance["validation_manifest"]["path"]
    except (KeyError, TypeError) as exc:
        raise ContractError("prepared record lacks dataset/validation manifest provenance") from exc
    gpu_count = int(prepared["schedule"]["gpu_count"])
    if gpu_count < 1:
        raise ContractError("prepared schedule gpu_count must be positive")
    distributed = [
        python_executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpu_count}",
    ]
    train = [*distributed, "-m", "scripts.base_train"]
    _option(train, "run", "dummy")
    _option(train, "model-tag", row["model_tag"])
    _option(train, "dataset-manifest", dataset_manifest)
    _option(train, "validation-manifest", validation_manifest)
    _option(train, "init-seed", int(row["init_seed"]))
    _option(train, "data-seed", int(row["data_seed"]))
    _option(train, "require-attention-backend", row["attention_backend"])
    _option(train, "require-compute-dtype", row["compute_dtype"])
    train.extend(["--no-save-optimizer", "--no-allow-data-reuse"])
    for cli_name, row_name in (
        ("depth", "depth"),
        ("aspect-ratio", "aspect_ratio"),
        ("head-dim", "head_dim"),
        ("model-dim", "model_dim"),
        ("max-seq-len", "max_seq_len"),
        ("window-pattern", "window_pattern"),
        ("target-tokens", "target_tokens"),
        ("target-param-data-ratio", "target_param_data_ratio"),
        ("reference-param-data-ratio", "reference_param_data_ratio"),
        ("total-batch-size", "total_batch_size"),
        ("device-batch-size", "device_batch_size"),
        ("eval-tokens-per-shard", "eval_tokens_per_shard"),
        ("warmup-steps", "warmup_steps"),
        ("muon-momentum-warmup-steps", "muon_momentum_warmup_steps"),
        ("weight-decay-horizon-tokens", "weight_decay_horizon_tokens"),
        ("warmdown-ratio", "warmdown_ratio"),
        ("final-lr-frac", "final_lr_frac"),
        ("embedding-lr", "embedding_lr"),
        ("unembedding-lr", "unembedding_lr"),
        ("weight-decay", "weight_decay"),
        ("matrix-lr", "matrix_lr"),
        ("scalar-lr", "scalar_lr"),
        ("fp8-recipe", "fp8_recipe"),
        ("optimizer-schedule-version", "optimizer_schedule_version"),
    ):
        _option(train, cli_name, row[row_name])
    train.extend(
        [
            "--eval-every=-1",
            "--sample-every=-1",
            "--core-metric-every=-1",
            "--save-every=-1",
        ]
    )
    if _boolean(row, "fp8"):
        train.append("--fp8")

    evaluate = [*distributed, "-m", "scripts.base_eval"]
    _option(evaluate, "eval", "bpb")
    _option(evaluate, "model-tag", row["model_tag"])
    _option(evaluate, "device-batch-size", row["eval_device_batch_size"])
    _option(evaluate, "eval-tokens-per-shard", row["eval_tokens_per_shard"])
    _option(evaluate, "bpb-splits", row.get("bpb_splits", "val"))
    _option(evaluate, "dataset-manifest", dataset_manifest)
    _option(evaluate, "validation-manifest", validation_manifest)
    _option(evaluate, "data-seed", int(row["data_seed"]))
    _option(evaluate, "require-attention-backend", row["attention_backend"])
    _option(evaluate, "require-compute-dtype", row["compute_dtype"])
    return train, evaluate


def validate_data_preflight_receipt(prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a once-per-campaign full-hash receipt plus cheap current stats."""

    try:
        receipt_ref = prepared["file_provenance"]["data_preflight_receipt"]
        receipt_path = Path(receipt_ref["path"]).resolve()
        receipt_sha256 = receipt_ref["sha256"]
    except (KeyError, TypeError) as exc:
        raise ContractError("prepared record lacks data_preflight_receipt provenance") from exc
    if not receipt_path.is_file() or sha256_file(receipt_path) != receipt_sha256:
        raise ContractError("data preflight receipt is missing or hash-mismatched")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid data preflight receipt {receipt_path}: {exc}") from exc
    if not isinstance(receipt, dict):
        raise ContractError("data preflight receipt must be a JSON object")
    if receipt.get("schema_version") != "nanochat-data-preflight-v1":
        raise ContractError("unsupported data preflight receipt schema")
    if receipt.get("status") != "verified":
        raise ContractError("data preflight receipt status is not verified")
    if receipt.get("code_commit") != prepared.get("code", {}).get("commit"):
        raise ContractError("data preflight receipt was produced by a different code commit")
    split_refs = {
        "train": "data_manifest",
        "validation": "validation_manifest",
    }
    summary: dict[str, Any] = {
        "path": str(receipt_path),
        "sha256": receipt_sha256,
        "verified_unix_time": receipt.get("verified_unix_time"),
        "splits": {},
    }

    def declared_manifest(path: Path, expected_split: str) -> tuple[str, list[tuple[Any, ...]]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"cannot parse bound {expected_split} manifest {path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ContractError(f"bound {expected_split} manifest schema_version must be 1")
        if payload.get("split") != expected_split:
            raise ContractError(f"bound manifest split mismatch for {expected_split}")
        dataset_id = payload.get("dataset_id")
        raw_shards = payload.get("shards")
        if not isinstance(dataset_id, str) or not dataset_id or not isinstance(raw_shards, list):
            raise ContractError(f"bound {expected_split} manifest is incomplete")
        declared: list[tuple[Any, ...]] = []
        seen_names: set[str] = set()
        for index, shard in enumerate(raw_shards):
            if not isinstance(shard, dict) or not isinstance(shard.get("path"), str):
                raise ContractError(f"bound {expected_split} manifest shard {index} is invalid")
            shard_path = Path(shard["path"]).expanduser()
            if not shard_path.is_absolute():
                shard_path = path.parent / shard_path
            shard_path = shard_path.resolve()
            name = shard.get("name", shard_path.stem)
            digest = shard.get("sha256")
            if (
                not isinstance(name, str)
                or not SAFE_SHARD_NAME_RE.fullmatch(name)
                or name in seen_names
                or not re.fullmatch(r"[0-9a-fA-F]{64}", str(digest))
            ):
                raise ContractError(f"bound {expected_split} manifest shard {index} lacks identity/hash")
            seen_names.add(name)
            declared.append(
                (name, str(shard_path), str(digest).lower(), shard.get("size_bytes"))
            )
        if not declared:
            raise ContractError(f"bound {expected_split} manifest has no shards")
        return dataset_id, declared

    for split, ref_name in split_refs.items():
        split_receipt = receipt.get(split)
        if not isinstance(split_receipt, dict):
            raise ContractError(f"data preflight receipt lacks {split} provenance")
        manifest_ref = prepared["file_provenance"][ref_name]
        if split_receipt.get("manifest_sha256") != manifest_ref["sha256"]:
            raise ContractError(f"data preflight {split} manifest hash mismatch")
        if Path(split_receipt.get("manifest_path", "")).resolve() != Path(
            manifest_ref["path"]
        ).resolve():
            raise ContractError(f"data preflight {split} manifest path mismatch")
        shards = split_receipt.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ContractError(f"data preflight {split} has no verified shards")
        dataset_id, declared_shards = declared_manifest(Path(manifest_ref["path"]), split)
        if split_receipt.get("dataset_id") != dataset_id or split_receipt.get("split") != split:
            raise ContractError(f"data preflight {split} dataset identity mismatch")
        receipt_shards = [
            (
                shard.get("name") if isinstance(shard, dict) else None,
                str(Path(str(shard.get("path", ""))).resolve())
                if isinstance(shard, dict)
                else None,
                str(shard.get("sha256", "")).lower() if isinstance(shard, dict) else None,
                shard.get("size_bytes") if isinstance(shard, dict) else None,
            )
            for shard in shards
        ]
        if receipt_shards != declared_shards:
            raise ContractError(
                f"data preflight {split} shard list/order does not exactly match its bound manifest"
            )
        for shard in shards:
            if not isinstance(shard, dict) or not re.fullmatch(
                r"[0-9a-f]{64}", str(shard.get("sha256", ""))
            ):
                raise ContractError(f"data preflight {split} has an invalid shard hash")
            shard_path = Path(str(shard.get("path", ""))).resolve()
            if not shard_path.is_file():
                raise ContractError(f"preflight-verified shard is missing: {shard_path}")
            stat = shard_path.stat()
            if stat.st_size != shard.get("observed_size_bytes"):
                raise ContractError(f"preflight-verified shard size changed: {shard_path}")
            if stat.st_mtime_ns != shard.get("observed_mtime_ns"):
                raise ContractError(f"preflight-verified shard mtime changed: {shard_path}")
        summary["splits"][split] = {
            "manifest_sha256": split_receipt["manifest_sha256"],
            "shard_count": len(shards),
        }
    return summary


def validate_text_overlap_audit(prepared: Mapping[str, Any]) -> dict[str, Any]:
    try:
        audit_ref = prepared["file_provenance"]["text_overlap_audit"]
        audit_path = Path(audit_ref["path"]).resolve()
    except (KeyError, TypeError) as exc:
        raise ContractError("prepared record lacks text_overlap_audit provenance") from exc
    if not audit_path.is_file() or sha256_file(audit_path) != audit_ref.get("sha256"):
        raise ContractError("text-overlap audit is missing or hash-mismatched")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid text-overlap audit {audit_path}: {exc}") from exc
    expected = {
        "schema_version": 1,
        "algorithm": "sha256_utf8_sqlite_exact_v1",
        "verdict": "PASS",
        "code_commit": prepared.get("code", {}).get("commit"),
        "train_manifest_sha256": prepared["file_provenance"]["data_manifest"]["sha256"],
        "validation_manifest_sha256": prepared["file_provenance"]["validation_manifest"][
            "sha256"
        ],
    }
    for key, value in expected.items():
        if not isinstance(audit, dict) or audit.get(key) != value:
            raise ContractError(
                f"text-overlap audit {key} mismatch: expected {value!r}, observed "
                f"{audit.get(key)!r}"
            )
    cross_split = audit.get("cross_split")
    if not isinstance(cross_split, dict) or any(
        cross_split.get(key) != 0
        for key in (
            "overlapping_unique_texts",
            "overlapping_train_rows",
            "overlapping_validation_rows",
        )
    ):
        raise ContractError("text-overlap audit PASS is inconsistent with overlap counts")
    return {
        "path": str(audit_path),
        "sha256": audit_ref["sha256"],
        "algorithm": audit["algorithm"],
        "verdict": audit["verdict"],
    }


def _load_validation_shard_names(prepared: Mapping[str, Any]) -> list[str]:
    try:
        manifest_ref = prepared["file_provenance"]["validation_manifest"]
        manifest_path = Path(manifest_ref["path"]).resolve()
    except (KeyError, TypeError) as exc:
        raise ContractError("prepared record lacks validation_manifest provenance") from exc
    if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_ref.get(
        "sha256"
    ):
        raise ContractError("validation manifest is missing or hash-mismatched")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid validation manifest {manifest_path}: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("split") != "validation"
        or not isinstance(manifest.get("shards"), list)
        or not manifest["shards"]
    ):
        raise ContractError("validation manifest is incomplete or has the wrong schema/split")
    names: list[str] = []
    for index, shard in enumerate(manifest["shards"]):
        name = shard.get("name") if isinstance(shard, dict) else None
        if not isinstance(name, str) or not SAFE_SHARD_NAME_RE.fullmatch(name):
            raise ContractError(
                f"validation shard {index} name is not unique and log-safe"
            )
        names.append(name)
    if len(names) != len(set(names)):
        raise ContractError("validation manifest contains duplicate shard names")
    return names


def validate_data_capacity_receipt(
    prepared: Mapping[str, Any], *, environment_observed: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    try:
        receipt_ref = prepared["file_provenance"]["data_capacity_receipt"]
        receipt_path = Path(receipt_ref["path"]).resolve()
        tokenizer_manifest_path = Path(
            prepared["file_provenance"]["tokenizer_manifest"]["path"]
        ).resolve()
    except (KeyError, TypeError) as exc:
        raise ContractError("prepared record lacks data-capacity/tokenizer provenance") from exc
    if not receipt_path.is_file() or sha256_file(receipt_path) != receipt_ref.get("sha256"):
        raise ContractError("data-capacity receipt is missing or hash-mismatched")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        tokenizer_manifest = json.loads(tokenizer_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid data-capacity/tokenizer JSON: {exc}") from exc
    tokenizer_files = tokenizer_manifest.get("files") if isinstance(tokenizer_manifest, dict) else None
    tokenizer_hashes = [
        spec.get("sha256")
        for spec in tokenizer_files or []
        if isinstance(spec, dict) and spec.get("name") == "tokenizer.pkl"
    ]
    if len(tokenizer_hashes) != 1:
        raise ContractError("tokenizer manifest must bind exactly one tokenizer.pkl")
    expected = {
        "schema_version": "nanochat-data-capacity-v1",
        "status": "verified",
        "algorithm": "exact_bestfit_row_replay_v1",
        "code_commit": prepared.get("code", {}).get("commit"),
        "plan_frozen_sha256": prepared["file_provenance"]["plan_frozen"]["sha256"],
        "dataset_key": prepared["plan_row"]["dataset_key"],
        "plan_sha256": prepared["plan_sha256"],
        "train_manifest_sha256": prepared["file_provenance"]["data_manifest"]["sha256"],
        "validation_manifest_sha256": prepared["file_provenance"]["validation_manifest"][
            "sha256"
        ],
        "tokenizer_sha256": tokenizer_hashes[0],
    }
    if not isinstance(receipt, dict):
        raise ContractError("data-capacity receipt must be a JSON object")
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ContractError(
                f"data-capacity receipt {key} mismatch: expected {value!r}, "
                f"observed {receipt.get(key)!r}"
            )
    if environment_observed is not None:
        expected_engine = {
            "rustbpe_version": environment_observed.get("rustbpe_version"),
            "tiktoken_version": environment_observed.get("tiktoken_version"),
            "tokenizers_version": environment_observed.get("tokenizers_version"),
        }
        if receipt.get("tokenizer_engine") != expected_engine:
            raise ContractError(
                "data-capacity tokenizer engine differs from the child training runtime"
            )
    packing = receipt.get("packing")
    if packing != {
        "buffer_size": 1000,
        "tokenizer_batch_size": 128,
        "allow_data_reuse": False,
    }:
        raise ContractError("data-capacity receipt does not prove the production packer contract")
    plan_rows = _load_bound_plan_rows(prepared)
    train_groups: dict[tuple[int, int, int], dict[str, Any]] = {}
    validation_groups: dict[tuple[int, int], dict[str, Any]] = {}
    for row in plan_rows:
        _validate_row(row)
        if _boolean(row, "allow_data_reuse"):
            raise ContractError("capacity proof cannot cover allow_data_reuse=true")
        data_seed = _integer(row, "data_seed", minimum=0)
        sequence_length = _integer(row, "max_seq_len", minimum=1)
        world_size = _integer(row, "gpu_count", minimum=1)
        device_batch_size = _integer(row, "device_batch_size", minimum=1)
        target_tokens = _integer(row, "target_tokens", minimum=1)
        required_train_rows = device_batch_size + target_tokens // (
            sequence_length * world_size
        )
        train_key = (data_seed, sequence_length, world_size)
        train_group = train_groups.setdefault(
            train_key,
            {
                "data_seed": data_seed,
                "sequence_length": sequence_length,
                "world_size": world_size,
                "required_rows_per_rank": 0,
                "covering_tasks": [],
            },
        )
        train_group["required_rows_per_rank"] = max(
            train_group["required_rows_per_rank"], required_train_rows
        )
        train_group["covering_tasks"].append(
            {
                "task_id": row["task_id"],
                "required_rows_per_rank": required_train_rows,
            }
        )

        eval_tokens = _integer(row, "eval_tokens_per_shard", minimum=1)
        required_validation_rows = eval_tokens // (sequence_length * world_size)
        validation_key = (sequence_length, world_size)
        validation_group = validation_groups.setdefault(
            validation_key,
            {
                "sequence_length": sequence_length,
                "world_size": world_size,
                "required_rows_per_rank": 0,
                "covering_tasks": [],
            },
        )
        validation_group["required_rows_per_rank"] = max(
            validation_group["required_rows_per_rank"], required_validation_rows
        )
        validation_group["covering_tasks"].append(
            {
                "task_id": row["task_id"],
                "required_rows_per_rank": required_validation_rows,
                "eval_tokens_per_shard": eval_tokens,
            }
        )

    def validate_rank_proofs(
        ranks: Any, *, world_size: int, required_rows: int, label: str
    ) -> None:
        if not isinstance(ranks, list) or len(ranks) != world_size:
            raise ContractError(f"{label} has incomplete rank coverage")
        rank_ids = [rank.get("rank") if isinstance(rank, dict) else None for rank in ranks]
        if rank_ids != list(range(world_size)):
            raise ContractError(f"{label} rank IDs/order are not exact")
        for rank in ranks:
            if (
                rank.get("rows_replayed") != required_rows
                or rank.get("final_source_epoch") != 1
            ):
                raise ContractError(
                    f"{label} did not replay the exact required rows in source epoch one"
                )

    proofs = receipt.get("proofs")
    expected_train = [train_groups[key] for key in sorted(train_groups)]
    if not isinstance(proofs, list) or len(proofs) != len(expected_train):
        raise ContractError("data-capacity train proof groups do not exactly cover the plan")
    for index, (proof, group) in enumerate(zip(proofs, expected_train, strict=True)):
        if not isinstance(proof, dict):
            raise ContractError(f"data-capacity train proof {index} must be an object")
        for key, expected_value in group.items():
            if proof.get(key) != expected_value:
                raise ContractError(
                    f"data-capacity train proof {index} {key} does not match the bound plan"
                )
        validate_rank_proofs(
            proof.get("ranks"),
            world_size=group["world_size"],
            required_rows=group["required_rows_per_rank"],
            label=f"data-capacity train proof {index}",
        )

    validation_proofs = receipt.get("validation_proofs")
    expected_validation = [validation_groups[key] for key in sorted(validation_groups)]
    validation_names = _load_validation_shard_names(prepared)
    if not isinstance(validation_proofs, list) or len(validation_proofs) != len(
        expected_validation
    ):
        raise ContractError(
            "data-capacity validation proof groups do not exactly cover the plan"
        )
    for index, (proof, group) in enumerate(
        zip(validation_proofs, expected_validation, strict=True)
    ):
        if not isinstance(proof, dict):
            raise ContractError(f"data-capacity validation proof {index} must be an object")
        for key, expected_value in group.items():
            if proof.get(key) != expected_value:
                raise ContractError(
                    f"data-capacity validation proof {index} {key} does not match the plan"
                )
        shards = proof.get("shards")
        if not isinstance(shards, list) or [
            shard.get("shard_name") if isinstance(shard, dict) else None
            for shard in shards
        ] != validation_names:
            raise ContractError(
                f"data-capacity validation proof {index} shard set/order is incomplete"
            )
        for shard in shards:
            validate_rank_proofs(
                shard.get("ranks"),
                world_size=group["world_size"],
                required_rows=group["required_rows_per_rank"],
                label=(
                    f"data-capacity validation proof {index} shard "
                    f"{shard.get('shard_name')}"
                ),
            )
    return {
        "path": str(receipt_path),
        "sha256": receipt_ref["sha256"],
        "algorithm": receipt["algorithm"],
        "proof_group_count": len(proofs),
        "validation_proof_group_count": len(validation_proofs),
    }


ENVIRONMENT_PROBE = r"""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
from pathlib import Path
import re
import sys
import pyarrow
import torch

def distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

from nanochat.common import COMPUTE_DTYPE
import nanochat.flash_attention as flash_attention_runtime
from nanochat.flash_attention import HAS_FA3, USE_FA3
nanochat_spec = importlib.util.find_spec("nanochat")
base_train_spec = importlib.util.find_spec("scripts.base_train")

fa3_interface = getattr(flash_attention_runtime, "_fa3", None)
fa3_origin = getattr(fa3_interface, "__file__", None)
fa3_root = Path(fa3_origin).resolve().parent if fa3_origin else None
fa3_loaded_files = []
for module_name, module in sorted(sys.modules.items()):
    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        continue
    path = Path(raw_path).resolve()
    in_fa3_root = fa3_root is not None and (path == fa3_root or fa3_root in path.parents)
    if not in_fa3_root:
        continue
    if not path.is_file() or path.suffix not in {".py", ".so"}:
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    fa3_loaded_files.append({"module": module_name, "path": str(path), "sha256": digest})
covered_fa3_paths = {item["path"] for item in fa3_loaded_files}
for raw_path in getattr(fa3_interface, "__fa3_artifact_files__", ()):
    path = Path(raw_path).resolve()
    if str(path) in covered_fa3_paths:
        continue
    if not path.is_file() or path.suffix not in {".py", ".so"}:
        raise RuntimeError(f"invalid declared FA3 artifact: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    fa3_loaded_files.append({
        "module": f"artifact:{path.name}",
        "path": str(path),
        "sha256": digest,
    })
    covered_fa3_paths.add(str(path))
fa3_loaded_files.sort(key=lambda item: (item["path"], item["module"]))
revision_candidates = set()
for candidate_path in [
    fa3_origin,
    *getattr(fa3_interface, "__fa3_artifact_files__", ()),
]:
    if candidate_path:
        revision_candidates.update(re.findall(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", candidate_path))
declared_revision = getattr(fa3_interface, "__revision__", None)
if isinstance(declared_revision, str) and re.fullmatch(r"[0-9a-f]{40}", declared_revision):
    revision_candidates.add(declared_revision)
resolved_revision = next(iter(revision_candidates)) if len(revision_candidates) == 1 else None

try:
    nccl_version = torch.cuda.nccl.version() if torch.cuda.is_available() else None
except Exception:
    nccl_version = None
device_count = torch.cuda.device_count()
devices = []
for index in range(device_count):
    properties = torch.cuda.get_device_properties(index)
    devices.append({
        "name": properties.name,
        "capability": list(torch.cuda.get_device_capability(index)),
        "total_memory": int(properties.total_memory),
    })
print(json.dumps({
    "python_version": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "python_executable": sys.executable,
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "pyarrow_version": pyarrow.__version__,
    "rustbpe_version": distribution_version("rustbpe"),
    "tokenizers_version": distribution_version("tokenizers"),
    "tiktoken_version": distribution_version("tiktoken"),
    "torchao_version": distribution_version("torchao"),
    "kernels_version": distribution_version("kernels"),
    "fa3_interface_origin": str(Path(fa3_origin).resolve()) if fa3_origin else None,
    "fa3_loaded_files": fa3_loaded_files,
    "fa3_requested_revision": os.environ.get("NANOCHAT_FA3_REVISION"),
    "fa3_resolved_revision": resolved_revision,
    "nanochat_origin": nanochat_spec.origin if nanochat_spec else None,
    "base_train_origin": base_train_spec.origin if base_train_spec else None,
    "attention_backend": "fa3" if USE_FA3 else "sdpa",
    "flash_attention_3_available": bool(HAS_FA3),
    "compute_dtype": str(COMPUTE_DTYPE).removeprefix("torch."),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": device_count,
    "cuda_devices": devices,
    "cuda_device_names_unique": sorted({device["name"] for device in devices}),
    "cuda_capabilities_unique": sorted({tuple(device["capability"]) for device in devices}),
    "nccl_version": nccl_version,
    "cudnn_version": torch.backends.cudnn.version(),
}, sort_keys=True))
"""


def probe_environment(
    python_executable: str, *, checkout: Path | None = None
) -> dict[str, Any]:
    resolved = shutil.which(python_executable) or python_executable
    executable_path = Path(resolved).expanduser().resolve()
    if not executable_path.is_file():
        raise ContractError(f"NANOCHAT_PYTHON is not a file: {executable_path}")
    probe_environment = dict(os.environ)
    probe_environment.pop("PYTHONPATH", None)
    probe_environment["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [str(executable_path), "-c", ENVIRONMENT_PROBE],
        capture_output=True,
        text=True,
        cwd=str(checkout.resolve()) if checkout is not None else None,
        env=probe_environment,
    )
    if result.returncode:
        raise ContractError(
            "environment probe failed before training: "
            + (result.stderr.strip() or f"exit code {result.returncode}")
        )
    try:
        last_line = result.stdout.strip().splitlines()[-1]
        observed = json.loads(last_line)
    except (IndexError, json.JSONDecodeError) as exc:
        raise ContractError(f"environment probe returned invalid JSON: {exc}") from exc
    if not isinstance(observed, dict):
        raise ContractError("environment probe must return a JSON object")
    observed["python_executable_sha256"] = sha256_file(executable_path)
    return observed


def validate_environment_observation(
    prepared: Mapping[str, Any], observed: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        lock_ref = prepared["file_provenance"]["environment_lock"]
        lock_path = Path(lock_ref["path"]).resolve()
    except (KeyError, TypeError) as exc:
        raise ContractError("prepared record lacks environment_lock provenance") from exc
    if not lock_path.is_file() or sha256_file(lock_path) != lock_ref.get("sha256"):
        raise ContractError("environment lock is missing or hash-mismatched")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid environment lock {lock_path}: {exc}") from exc
    if not isinstance(lock, dict) or lock.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        raise ContractError(f"environment lock schema must be {ENVIRONMENT_SCHEMA_VERSION}")
    expected = lock.get("expected")
    required_expected = {
        "python_version",
        "python_implementation",
        "python_executable_sha256",
        "torch_version",
        "torch_cuda_version",
        "pyarrow_version",
        "rustbpe_version",
        "tokenizers_version",
        "tiktoken_version",
        "torchao_version",
        "kernels_version",
        "fa3_interface_origin",
        "fa3_loaded_files",
        "fa3_requested_revision",
        "fa3_resolved_revision",
        "attention_backend",
        "flash_attention_3_available",
        "compute_dtype",
        "cuda_available",
        "cuda_device_names_unique",
        "cuda_capabilities_unique",
        "nccl_version",
        "cudnn_version",
    }
    if not isinstance(expected, dict) or set(expected) != required_expected:
        raise ContractError(
            "environment lock expected must contain exactly the required runtime fingerprint keys"
        )
    for key, value in expected.items():
        if observed.get(key) != value:
            raise ContractError(
                f"environment {key} mismatch: expected {value!r}, observed {observed.get(key)!r}"
            )
    gpu_count = int(prepared["schedule"]["gpu_count"])
    if observed.get("cuda_available") is not True or observed.get("cuda_device_count") != gpu_count:
        raise ContractError(
            f"visible CUDA device count mismatch: expected {gpu_count}, "
            f"observed {observed.get('cuda_device_count')}"
        )
    row = prepared.get("plan_row")
    if not isinstance(row, dict):
        raise ContractError("prepared record lacks plan_row for runtime validation")
    if observed.get("attention_backend") != row.get("attention_backend"):
        raise ContractError("observed attention backend does not match the plan")
    if observed.get("compute_dtype") != row.get("compute_dtype"):
        raise ContractError("observed compute dtype does not match the plan")
    if row.get("attention_backend") == "fa3" and observed.get(
        "flash_attention_3_available"
    ) is not True:
        raise ContractError("plan requires FA3 but the child runtime did not load it")
    if row.get("attention_backend") == "fa3":
        if not observed.get("kernels_version") or not observed.get("fa3_interface_origin"):
            raise ContractError("FA3 runtime lacks a pinned kernels/interface fingerprint")
        loaded_files = observed.get("fa3_loaded_files")
        if not isinstance(loaded_files, list) or not loaded_files:
            raise ContractError("FA3 runtime did not expose hashable loaded implementation files")
        if not any(
            isinstance(item, dict)
            and item.get("path") == observed.get("fa3_interface_origin")
            and re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))
            for item in loaded_files
        ):
            raise ContractError("FA3 interface origin is not covered by a loaded-file hash")
        requested_revision = observed.get("fa3_requested_revision")
        resolved_revision = observed.get("fa3_resolved_revision")
        if (
            not isinstance(requested_revision, str)
            or not re.fullmatch(r"[0-9a-f]{40}", requested_revision)
            or resolved_revision != requested_revision
        ):
            raise ContractError("FA3 requested/resolved revisions are not the same pinned commit")
    if _boolean(row, "fp8") and not observed.get("torchao_version"):
        raise ContractError("FP8 plan requires an installed torchao distribution")
    if (
        not observed.get("rustbpe_version")
        or not observed.get("tiktoken_version")
        or not observed.get("tokenizers_version")
    ):
        raise ContractError("tokenizer engine distributions are not installed in child runtime")
    try:
        checkout = Path(prepared["observed_code"]["checkout"]).resolve()
    except (KeyError, TypeError) as exc:
        raise ContractError("prepared record lacks observed checkout path") from exc
    for field in ("nanochat_origin", "base_train_origin"):
        origin = observed.get(field)
        if not isinstance(origin, str) or not Path(origin).resolve().is_file():
            raise ContractError(f"child runtime did not resolve {field}")
        try:
            Path(origin).resolve().relative_to(checkout)
        except ValueError as exc:
            raise ContractError(
                f"child runtime {field} is outside the verified checkout: {origin}"
            ) from exc
    lockfiles = lock.get("lockfiles")
    if not isinstance(lockfiles, list) or not lockfiles:
        raise ContractError("environment lock must bind at least one dependency lockfile")
    verified_lockfiles: list[dict[str, str]] = []
    for index, spec in enumerate(lockfiles):
        if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
            raise ContractError(f"environment lockfiles[{index}] is invalid")
        path = Path(spec["path"]).expanduser()
        if not path.is_absolute():
            path = lock_path.parent / path
        path = path.resolve()
        digest = spec.get("sha256")
        if not path.is_file() or not isinstance(digest, str) or sha256_file(path) != digest:
            raise ContractError(f"environment dependency lockfile is missing or changed: {path}")
        verified_lockfiles.append({"path": str(path), "sha256": digest})
    return {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "environment_lock": {"path": str(lock_path), "sha256": lock_ref["sha256"]},
        "observed": dict(observed),
        "verified_lockfiles": verified_lockfiles,
    }


def _resolve_manifest_file(manifest_path: Path, raw_path: Any, *, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ContractError(f"{label} path must be a non-empty string")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def validate_tokenizer_consistency_receipt(
    prepared: Mapping[str, Any], *, environment_observed: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        receipt_ref = prepared["file_provenance"]["tokenizer_consistency_receipt"]
        receipt_path = Path(receipt_ref["path"]).resolve()
        manifest_ref = prepared["file_provenance"]["tokenizer_manifest"]
        manifest_path = Path(manifest_ref["path"]).resolve()
    except (KeyError, TypeError) as exc:
        raise ContractError("prepared record lacks tokenizer consistency provenance") from exc
    if not receipt_path.is_file() or sha256_file(receipt_path) != receipt_ref.get("sha256"):
        raise ContractError("tokenizer consistency receipt is missing or hash-mismatched")
    if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_ref.get("sha256"):
        raise ContractError("tokenizer manifest is missing or hash-mismatched")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid tokenizer receipt/manifest JSON: {exc}") from exc
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or not isinstance(files, list)
    ):
        raise ContractError("tokenizer manifest schema_version must be 1 with files")
    by_name = {
        spec.get("name"): spec for spec in files if isinstance(spec, dict)
    }
    if set(by_name) != {"tokenizer.pkl", "token_bytes.pt"} or len(files) != 2:
        raise ContractError(
            "formal tokenizer manifest must contain exactly tokenizer.pkl and token_bytes.pt"
        )
    expected = {
        "schema_version": "nanochat-tokenizer-consistency-v1",
        "status": "verified",
        "algorithm": "tiktoken_raw_token_bytes_v1",
        "code_commit": prepared.get("code", {}).get("commit"),
        "tokenizer_manifest_sha256": manifest_ref["sha256"],
        "tokenizer_pickle_sha256": by_name["tokenizer.pkl"].get("sha256"),
        "token_bytes_sha256": by_name["token_bytes.pt"].get("sha256"),
        "engine": {
            "rustbpe_version": environment_observed.get("rustbpe_version"),
            "tiktoken_version": environment_observed.get("tiktoken_version"),
            "tokenizers_version": environment_observed.get("tokenizers_version"),
            "torch_version": environment_observed.get("torch_version"),
        },
    }
    if not isinstance(receipt, dict):
        raise ContractError("tokenizer consistency receipt must be an object")
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ContractError(
                f"tokenizer consistency receipt {key} mismatch: expected {value!r}, "
                f"observed {receipt.get(key)!r}"
            )
    vocab_size = receipt.get("vocab_size")
    special_ids = receipt.get("special_token_ids")
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size < 1
        or not isinstance(special_ids, list)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= vocab_size
            for value in special_ids
        )
        or special_ids != sorted(set(special_ids))
    ):
        raise ContractError("tokenizer consistency receipt vocab/special IDs are invalid")
    return {
        "path": str(receipt_path),
        "sha256": receipt_ref["sha256"],
        "algorithm": receipt["algorithm"],
        "vocab_size": vocab_size,
        "tokenizer_pickle_sha256": receipt["tokenizer_pickle_sha256"],
        "token_bytes_sha256": receipt["token_bytes_sha256"],
    }


def stage_tokenizer(prepared: Mapping[str, Any], work_dir: Path) -> list[dict[str, Any]]:
    try:
        raw_path = prepared["file_provenance"]["tokenizer_manifest"]["path"]
    except (KeyError, TypeError) as exc:
        raise ContractError("prepared record lacks tokenizer manifest provenance") from exc
    manifest_path = Path(raw_path).resolve()
    expected_manifest_sha256 = prepared["file_provenance"]["tokenizer_manifest"].get(
        "sha256"
    )
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_manifest_sha256:
        raise ContractError("tokenizer manifest is missing or hash-mismatched")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load tokenizer manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ContractError("tokenizer manifest schema_version must be 1")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ContractError("tokenizer manifest files must be a non-empty list")
    tokenizer_dir = work_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=False)
    staged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, spec in enumerate(files):
        if not isinstance(spec, dict):
            raise ContractError(f"tokenizer files[{index}] must be an object")
        name = spec.get("name")
        digest = spec.get("sha256")
        if not isinstance(name, str) or Path(name).name != name or not name:
            raise ContractError(f"invalid tokenizer filename at index {index}")
        if name in seen:
            raise ContractError(f"duplicate tokenizer filename: {name}")
        seen.add(name)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ContractError(f"invalid tokenizer SHA-256 for {name}")
        source = _resolve_manifest_file(manifest_path, spec.get("path"), label=name)
        if not source.is_file() or sha256_file(source) != digest:
            raise ContractError(f"tokenizer file is missing or hash-mismatched: {source}")
        destination = tokenizer_dir / name
        destination.symlink_to(source)
        staged.append({"name": name, "source": str(source), "sha256": digest})
    if seen != {"tokenizer.pkl", "token_bytes.pt"}:
        raise ContractError(
            "formal tokenizer manifest must contain exactly tokenizer.pkl and token_bytes.pt"
        )
    return staged


def _run_logged(
    command: list[str],
    log_path: Path,
    environment: Mapping[str, str],
    *,
    checkout: Path,
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log_path.open("wb") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(environment),
            cwd=str(checkout),
        )
        log.flush()
        os.fsync(log.fileno())
    elapsed = time.monotonic() - start
    if result.returncode:
        raise ContractError(
            f"command failed with exit code {result.returncode}; inspect {log_path}"
        )
    return elapsed


def validate_checkpoint_contract(
    checkpoint_meta: Mapping[str, Any], prepared: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove that the checkpoint was produced under the frozen plan contract."""

    row = prepared.get("plan_row")
    if not isinstance(row, dict):
        raise ContractError("prepared record lacks a plan row")
    _validate_row(row)
    user_config = checkpoint_meta.get("user_config")
    if not isinstance(user_config, dict):
        raise ContractError("checkpoint metadata lacks user_config")

    integer_fields = (
        "init_seed",
        "data_seed",
        "depth",
        "aspect_ratio",
        "head_dim",
        "model_dim",
        "max_seq_len",
        "target_tokens",
        "total_batch_size",
        "device_batch_size",
        "eval_tokens_per_shard",
        "warmup_steps",
        "muon_momentum_warmup_steps",
        "weight_decay_horizon_tokens",
    )
    float_fields = FLOAT_SCHEDULE_FIELDS
    string_fields = (
        "window_pattern",
        "fp8_recipe",
        "optimizer_schedule_version",
    )
    for field in integer_fields:
        observed = _json_integer(user_config.get(field), label=f"user_config.{field}")
        expected = _integer(row, field)
        if observed != expected:
            raise ContractError(
                f"checkpoint user_config.{field} mismatch: expected {expected}, observed {observed}"
            )
    for field in float_fields:
        observed = _json_number(user_config.get(field), label=f"user_config.{field}")
        expected = _number(row, field)
        if observed != expected:
            raise ContractError(
                f"checkpoint user_config.{field} mismatch: expected {expected}, observed {observed}"
            )
    for field in string_fields:
        if user_config.get(field) != row.get(field):
            raise ContractError(f"checkpoint user_config.{field} mismatch")
    if user_config.get("fp8") is not _boolean(row, "fp8"):
        raise ContractError("checkpoint user_config.fp8 mismatch")
    expected_fixed_config = {
        "model_tag": row["model_tag"],
        "run": "dummy",
        "seed": _integer(row, "init_seed"),
        "num_iterations": -1,
        "target_flops": -1.0,
        "save_optimizer": False,
        "allow_data_reuse": False,
        "train_row_group_start": 0,
        "verify_shard_hashes": False,
        "eval_every": -1,
        "sample_every": -1,
        "core_metric_every": -1,
        "save_every": -1,
        "require_attention_backend": row["attention_backend"],
        "require_compute_dtype": row["compute_dtype"],
    }
    for field, expected in expected_fixed_config.items():
        if user_config.get(field) != expected:
            raise ContractError(
                f"checkpoint user_config.{field} mismatch: expected {expected!r}, "
                f"observed {user_config.get(field)!r}"
            )
    for field, ref_name in (
        ("dataset_manifest", "data_manifest"),
        ("validation_manifest", "validation_manifest"),
    ):
        observed_path = user_config.get(field)
        if not isinstance(observed_path, str) or Path(observed_path).resolve() != Path(
            prepared["file_provenance"][ref_name]["path"]
        ).resolve():
            raise ContractError(f"checkpoint user_config.{field} path mismatch")

    num_iterations = _integer(row, "target_tokens") // _integer(row, "total_batch_size")
    expected_top_level = {
        "step": num_iterations,
        "device_batch_size": _integer(row, "device_batch_size"),
        "max_seq_len": _integer(row, "max_seq_len"),
        "total_batch_size": _integer(row, "total_batch_size"),
        "prediction_positions_completed": _integer(row, "target_tokens"),
    }
    for field, expected in expected_top_level.items():
        if checkpoint_meta.get(field) != expected:
            raise ContractError(
                f"checkpoint {field} mismatch: expected {expected}, "
                f"observed {checkpoint_meta.get(field)!r}"
            )

    shape = resolve_model_shape(
        depth=_integer(row, "depth"),
        aspect_ratio=_integer(row, "aspect_ratio"),
        head_dim=_integer(row, "head_dim"),
        model_dim=_integer(row, "model_dim"),
    )
    model_config = checkpoint_meta.get("model_config")
    expected_model_config = {
        "sequence_len": _integer(row, "max_seq_len"),
        "n_layer": _integer(row, "depth"),
        "n_head": shape.n_head,
        "n_kv_head": shape.n_head,
        "n_embd": shape.model_dim,
        "window_pattern": row["window_pattern"],
    }
    if not isinstance(model_config, dict):
        raise ContractError("checkpoint lacks model_config")
    for field, expected in expected_model_config.items():
        if model_config.get(field) != expected:
            raise ContractError(
                f"checkpoint model_config.{field} mismatch: expected {expected!r}, "
                f"observed {model_config.get(field)!r}"
            )

    policy = checkpoint_meta.get("checkpoint_policy")
    if not isinstance(policy, dict) or policy.get("save_optimizer") is not False:
        raise ContractError("checkpoint policy does not declare save_optimizer=false")
    if policy.get("resume_mode") != "restart_only":
        raise ContractError("checkpoint policy does not declare restart_only")
    runtime_contract = checkpoint_meta.get("runtime_contract")
    if runtime_contract != {
        "attention_backend": row["attention_backend"],
        "compute_dtype": row["compute_dtype"],
    }:
        raise ContractError("checkpoint runtime_contract does not exactly match the plan")

    for metadata_field, ref_name in (
        ("dataset_manifest", "data_manifest"),
        ("validation_manifest", "validation_manifest"),
    ):
        provenance = checkpoint_meta.get(metadata_field)
        reference = prepared["file_provenance"][ref_name]
        if not isinstance(provenance, dict) or provenance.get(
            "manifest_sha256"
        ) != reference.get("sha256"):
            raise ContractError(f"checkpoint {metadata_field} hash mismatch")
        if Path(str(provenance.get("manifest_path", ""))).resolve() != Path(
            reference["path"]
        ).resolve():
            raise ContractError(f"checkpoint {metadata_field} path mismatch")

    loader_state = checkpoint_meta.get("dataloader_state_dict")
    if not isinstance(loader_state, dict):
        raise ContractError("checkpoint lacks dataloader_state_dict")
    data_contract = loader_state.get("data_contract")
    expected_data_contract = {
        "loader_state_version": 2,
        "train_manifest_sha256": prepared["file_provenance"]["data_manifest"]["sha256"],
        "validation_manifest_sha256": prepared["file_provenance"]["validation_manifest"][
            "sha256"
        ],
        "data_seed": _integer(row, "data_seed", minimum=0),
        "split": "train",
        "validation_shard": None,
        "device_batch_size": _integer(row, "device_batch_size"),
        "sequence_length": _integer(row, "max_seq_len"),
        "tokenizer_batch_size": 128,
        "packing_buffer_size": 1000,
        "ddp_world_size": _integer(row, "gpu_count"),
        "allow_data_reuse": False,
    }
    if data_contract != expected_data_contract:
        raise ContractError("checkpoint dataloader data_contract does not exactly match the plan")
    if loader_state.get("loader_state_version") != 2:
        raise ContractError("checkpoint dataloader loader_state_version mismatch")
    return {
        "step": num_iterations,
        "prediction_positions_completed": _integer(row, "target_tokens"),
        "model_dim": shape.model_dim,
        "n_head": shape.n_head,
        "data_contract": data_contract,
    }


def _parse_eval_bpb(
    log_path: Path,
    *,
    expected_validation_names: list[str],
    expected_runtime_contract: Mapping[str, str],
) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    runtime_lines = re.findall(r"^EVAL_RUNTIME_CONTRACT:\s*(\{.*\})\s*$", text, re.MULTILINE)
    if len(runtime_lines) != 1:
        raise ContractError("evaluation log must contain exactly one runtime contract record")
    try:
        runtime_contract = json.loads(runtime_lines[0])
    except json.JSONDecodeError as exc:
        raise ContractError("evaluation runtime contract is invalid JSON") from exc
    if runtime_contract != dict(expected_runtime_contract):
        raise ContractError(
            "evaluation runtime contract does not exactly match the frozen plan"
        )
    matches = re.findall(
        r"^([^\n]+?) bpb:\s*([0-9.eE+-]+)\s*$", text, flags=re.MULTILINE
    )
    observed_keys = [key.strip() for key, _ in matches]
    expected_keys = [*(f"val/{name}" for name in expected_validation_names), "val"]
    if observed_keys != expected_keys:
        raise ContractError(
            "evaluation BPB keys/order do not exactly match every validation manifest shard: "
            f"expected {expected_keys!r}, observed {observed_keys!r}"
        )
    values: dict[str, float] = {}
    for key, raw in matches:
        key = key.strip()
        try:
            value = float(raw)
        except ValueError as exc:
            raise ContractError(f"invalid BPB for {key}") from exc
        if not math.isfinite(value):
            raise ContractError(f"non-finite BPB for {key}")
        values[key] = value
    shards = {name: values[f"val/{name}"] for name in expected_validation_names}
    recomputed = sum(shards.values()) / len(shards)
    if not math.isclose(values["val"], recomputed, rel_tol=0.0, abs_tol=2e-6):
        raise ContractError(
            "aggregate val BPB does not equal the arithmetic mean of all heldout shards"
        )
    return {
        "val_bpb": values["val"],
        "train_bpb": None,
        "val_bpb_by_shard": shards,
        "runtime_contract": runtime_contract,
    }


def _single_file(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1 or not matches[0].is_file():
        raise ContractError(f"expected exactly one {label} in {directory}, found {len(matches)}")
    return matches[0]


def run_task(prepared_path: Path, *, python_executable: str | None = None) -> Path:
    prepared_path = prepared_path.resolve()
    try:
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load prepared task {prepared_path}: {exc}") from exc
    if not isinstance(prepared, dict) or prepared.get("status") != "prepared":
        raise ContractError("runner requires a prepared task record")
    staging_dir = Path(prepared["staging_dir"]).resolve()
    if prepared_path.parent != staging_dir:
        raise ContractError("prepared task path does not match its staging directory")
    work_dir = staging_dir / "work"
    logs_dir = staging_dir / "logs"
    metrics_dir = staging_dir / "metrics"
    work_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    executable = python_executable or os.environ.get("NANOCHAT_PYTHON") or sys.executable
    try:
        checkout = Path(prepared["observed_code"]["checkout"]).resolve()
    except (KeyError, TypeError) as exc:
        raise ContractError("prepared record lacks verified checkout provenance") from exc
    schedule_contract = validate_schedule_manifest(prepared)
    environment_observed = validate_environment_observation(
        prepared, probe_environment(executable, checkout=checkout)
    )
    atomic_write_json(staging_dir / "environment_observed.json", environment_observed)
    data_preflight = validate_data_preflight_receipt(prepared)
    text_overlap_audit = validate_text_overlap_audit(prepared)
    data_capacity = validate_data_capacity_receipt(
        prepared, environment_observed=environment_observed["observed"]
    )
    tokenizer_consistency = validate_tokenizer_consistency_receipt(
        prepared, environment_observed=environment_observed["observed"]
    )
    tokenizer_files = stage_tokenizer(prepared, work_dir)

    train_command, eval_command = build_commands(prepared, python_executable=executable)
    atomic_write_json(
        staging_dir / "commands.json",
        {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "python_executable": executable,
            "environment_observed": environment_observed,
            "train": train_command,
            "eval": eval_command,
            "data_preflight": data_preflight,
            "text_overlap_audit": text_overlap_audit,
            "data_capacity": data_capacity,
            "tokenizer_consistency": tokenizer_consistency,
            "schedule_contract": schedule_contract,
            "tokenizer_files": tokenizer_files,
        },
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["NANOCHAT_BASE_DIR"] = str(work_dir)
    environment.setdefault("WANDB_MODE", "disabled")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    train_log = logs_dir / "train.log"
    eval_log = logs_dir / "eval.log"
    train_seconds = _run_logged(
        train_command, train_log, environment, checkout=checkout
    )
    eval_seconds = _run_logged(eval_command, eval_log, environment, checkout=checkout)

    row = prepared["plan_row"]
    checkpoint_dir = work_dir / "base_checkpoints" / row["model_tag"]
    model_path = _single_file(checkpoint_dir, "model_*.pt", "model checkpoint")
    meta_path = _single_file(checkpoint_dir, "meta_*.json", "checkpoint metadata")
    optimizer_paths = sorted(checkpoint_dir.glob("optim_*.pt"))
    if optimizer_paths:
        raise ContractError("--no-save-optimizer contract violated: optimizer shards exist")
    try:
        checkpoint_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid checkpoint metadata {meta_path}: {exc}") from exc
    target_tokens = int(row["target_tokens"])
    checkpoint_contract = validate_checkpoint_contract(checkpoint_meta, prepared)

    train_metrics = parse_log(train_log)
    eval_metrics = _parse_eval_bpb(
        eval_log,
        expected_validation_names=_load_validation_shard_names(prepared),
        expected_runtime_contract={
            "attention_backend": row["attention_backend"],
            "compute_dtype": row["compute_dtype"],
        },
    )
    n_scaling = train_metrics.get("N_scaling")
    n_total = train_metrics.get("N_total")
    flops_per_token = train_metrics.get("flops_per_token_est")
    for label, value in (
        ("N_scaling", n_scaling),
        ("N_total", n_total),
        ("flops_per_token_est", flops_per_token),
    ):
        if value is None or not math.isfinite(float(value)):
            raise ContractError(f"training log has no finite {label}")
    expected_n_scaling = _integer(row, "n_scaling", minimum=1)
    if float(n_scaling) != expected_n_scaling:
        raise ContractError(
            f"training log N_scaling mismatch: expected {expected_n_scaling}, "
            f"observed {n_scaling!r}"
        )
    horizon = _integer(row, "weight_decay_horizon_tokens")
    if row["optimizer_schedule_version"] == "legacy_v0":
        if horizon != 12 * int(float(n_scaling)):
            raise ContractError("legacy weight-decay horizon does not match actual N_scaling")
    elif horizon != target_tokens:
        raise ContractError("exact weight-decay horizon does not match target_tokens")
    gpu_count = int(prepared["schedule"]["gpu_count"])
    metric = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "status": "success",
        "task_id": prepared["task_id"],
        "N_total": n_total,
        "N_scaling": n_scaling,
        "D_actual": target_tokens,
        "flops_per_token_est": flops_per_token,
        "C_6ND": 6.0 * float(n_scaling) * target_tokens,
        "C_arch": float(flops_per_token) * target_tokens,
        "val_bpb": eval_metrics["val_bpb"],
        "train_bpb": eval_metrics["train_bpb"],
        "val_bpb_by_shard": eval_metrics["val_bpb_by_shard"],
        "train_wall_seconds": train_seconds,
        "eval_wall_seconds": eval_seconds,
        "gpu_seconds_train": train_seconds * gpu_count,
        "gpu_seconds_eval": eval_seconds * gpu_count,
        "model_checkpoint": model_path.relative_to(staging_dir).as_posix(),
        "checkpoint_meta": meta_path.relative_to(staging_dir).as_posix(),
        "checkpoint_policy": "restart_only",
        "attention_backend": row["attention_backend"],
        "compute_dtype": row["compute_dtype"],
        "optimizer_schedule_version": row["optimizer_schedule_version"],
        "checkpoint_contract": checkpoint_contract,
        "eval_runtime_contract": eval_metrics["runtime_contract"],
    }
    metric_path = metrics_dir / "final.json"
    atomic_write_json(metric_path, metric)
    return metric_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--python", dest="python_executable")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        print(run_task(args.prepared, python_executable=args.python_executable))
        return 0
    except ContractError as exc:
        print(f"TASK ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
