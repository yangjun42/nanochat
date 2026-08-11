#!/usr/bin/env python3
"""Replay the real packer on CPU to prove a no-reuse plan fits one data epoch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from nanochat.dataloader import (
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.dataset import load_parquet_manifest, validate_manifest_separation
from nanochat.tokenizer import RustBPETokenizer


CAPACITY_SCHEMA_VERSION = "nanochat-data-capacity-v1"


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(row: Mapping[str, str], name: str, minimum: int = 1) -> int:
    try:
        value = int(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"plan field {name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"plan field {name} must be >= {minimum}")
    return value


def _false(row: Mapping[str, str], name: str) -> bool:
    raw = row.get(name, "false").strip().lower()
    if raw in {"0", "false", "no"}:
        return False
    if raw in {"1", "true", "yes"}:
        return True
    raise ValueError(f"plan field {name} must be boolean")


def capacity_requirements(
    rows: Iterable[Mapping[str, str]],
) -> dict[tuple[int, int, int], dict[str, Any]]:
    """Collapse a plan to the longest exact row replay per seed/T/world tuple."""

    groups: dict[tuple[int, int, int], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if _false(row, "allow_data_reuse"):
            raise ValueError("capacity proof only covers allow_data_reuse=false plans")
        data_seed = _integer(row, "data_seed", minimum=0)
        sequence_length = _integer(row, "max_seq_len")
        world_size = _integer(row, "gpu_count")
        device_batch_size = _integer(row, "device_batch_size")
        total_batch_size = _integer(row, "total_batch_size")
        target_tokens = _integer(row, "target_tokens")
        micro_positions = device_batch_size * sequence_length * world_size
        if total_batch_size % micro_positions:
            raise ValueError(
                f"plan row {index} total_batch_size is not divisible by its global microbatch"
            )
        if target_tokens % total_batch_size:
            raise ValueError(
                f"plan row {index} target_tokens is not divisible by total_batch_size"
            )
        # Training prefetches one complete device batch, then one replacement
        # per real microstep. Packing is sequential by row, so proving with
        # B=1 is exactly equivalent and lets different device B values share a proof.
        required_rows_per_rank = (
            device_batch_size + target_tokens // (sequence_length * world_size)
        )
        key = (data_seed, sequence_length, world_size)
        task_id = row.get("task_id") or row.get("model_tag") or f"row-{index}"
        group = groups.setdefault(
            key,
            {
                "data_seed": data_seed,
                "sequence_length": sequence_length,
                "world_size": world_size,
                "required_rows_per_rank": 0,
                "covering_tasks": [],
            },
        )
        group["required_rows_per_rank"] = max(
            int(group["required_rows_per_rank"]), required_rows_per_rank
        )
        group["covering_tasks"].append(
            {
                "task_id": str(task_id),
                "required_rows_per_rank": required_rows_per_rank,
            }
        )
    if not groups:
        raise ValueError("plan contains no rows")
    return groups


def validation_capacity_requirements(
    rows: Iterable[Mapping[str, str]],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Collapse heldout evaluation to the longest row replay per T/world tuple."""

    groups: dict[tuple[int, int], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        sequence_length = _integer(row, "max_seq_len")
        world_size = _integer(row, "gpu_count")
        eval_device_batch_size = _integer(row, "eval_device_batch_size")
        eval_tokens_per_shard = _integer(row, "eval_tokens_per_shard")
        eval_micro_positions = eval_device_batch_size * sequence_length * world_size
        if eval_tokens_per_shard % eval_micro_positions:
            raise ValueError(
                f"plan row {index} eval_tokens_per_shard is not divisible by "
                "its distributed evaluation batch"
            )
        required_rows_per_rank = eval_tokens_per_shard // (
            sequence_length * world_size
        )
        key = (sequence_length, world_size)
        task_id = row.get("task_id") or row.get("model_tag") or f"row-{index}"
        group = groups.setdefault(
            key,
            {
                "sequence_length": sequence_length,
                "world_size": world_size,
                "required_rows_per_rank": 0,
                "covering_tasks": [],
            },
        )
        group["required_rows_per_rank"] = max(
            int(group["required_rows_per_rank"]), required_rows_per_rank
        )
        group["covering_tasks"].append(
            {
                "task_id": str(task_id),
                "required_rows_per_rank": required_rows_per_rank,
                "eval_tokens_per_shard": eval_tokens_per_shard,
            }
        )
    if not groups:
        raise ValueError("plan contains no rows")
    return groups


def _with_rank_environment(rank: int, world_size: int) -> dict[str, str | None]:
    prior = {key: os.environ.get(key) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")}
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    return prior


def _restore_rank_environment(prior: Mapping[str, str | None]) -> None:
    for key, value in prior.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def prove_capacity(
    *,
    plan_path: Path,
    train_manifest_path: Path,
    validation_manifest_path: Path,
    tokenizer,
    tokenizer_sha256: str,
    receipt_path: Path,
    code_commit: str,
    plan_frozen_sha256: str,
    dataset_key: str | None = None,
    buffer_size: int = 1000,
    tokenizer_batch_size: int = 128,
) -> dict[str, Any]:
    if buffer_size <= 0 or tokenizer_batch_size <= 0:
        raise ValueError("buffer sizes must be positive")
    if not re.fullmatch(r"[0-9a-f]{40,64}", code_commit):
        raise ValueError("code_commit must be a lowercase Git commit digest")
    if not re.fullmatch(r"[0-9a-f]{64}", plan_frozen_sha256):
        raise ValueError("plan_frozen_sha256 must be a lowercase SHA-256 digest")
    if receipt_path.exists():
        raise FileExistsError(
            f"capacity receipt is immutable; choose a new path: {receipt_path}"
        )
    with plan_path.open("r", encoding="utf-8", newline="") as handle:
        plan_rows = list(csv.DictReader(handle))
    dataset_keys = {row.get("dataset_key", "") for row in plan_rows}
    if len(dataset_keys) != 1 or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", next(iter(dataset_keys), "")
    ):
        raise ValueError("capacity plan must contain exactly one valid dataset_key")
    dataset_key = next(iter(dataset_keys))
    train = load_parquet_manifest(train_manifest_path, expected_split="train")
    validation = load_parquet_manifest(
        validation_manifest_path, expected_split="validation"
    )
    validate_manifest_separation(train, validation)
    for index, row in enumerate(plan_rows):
        if row.get("dataset_manifest_sha256") != train.manifest_sha256:
            raise ValueError(f"plan row {index} train manifest hash mismatch")
        if row.get("validation_manifest_sha256") != validation.manifest_sha256:
            raise ValueError(f"plan row {index} validation manifest hash mismatch")
    if dataset_key is not None:
        plan_rows = [row for row in plan_rows if row.get("dataset_key") == dataset_key]
        if not plan_rows:
            raise ValueError(f"plan contains no rows for dataset_key={dataset_key!r}")
    for index, row in enumerate(plan_rows):
        declared_train = row.get("dataset_manifest_sha256")
        declared_validation = row.get("validation_manifest_sha256")
        if declared_train and declared_train != train.manifest_sha256:
            raise ValueError(f"plan row {index} declares a different train manifest")
        if declared_validation and declared_validation != validation.manifest_sha256:
            raise ValueError(
                f"plan row {index} declares a different validation manifest"
            )
    requirements = capacity_requirements(plan_rows)
    validation_requirements = validation_capacity_requirements(plan_rows)

    started = time.monotonic()
    proofs: list[dict[str, Any]] = []
    for key in sorted(requirements):
        group = requirements[key]
        rank_proofs: list[dict[str, Any]] = []
        for rank in range(group["world_size"]):
            prior = _with_rank_environment(rank, group["world_size"])
            rank_started = time.monotonic()
            try:
                loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
                    tokenizer,
                    B=1,
                    T=group["sequence_length"],
                    split="train",
                    device="cpu",
                    tokenizer_threads=4,
                    tokenizer_batch_size=tokenizer_batch_size,
                    buffer_size=buffer_size,
                    dataset_manifest=str(train_manifest_path),
                    validation_manifest=str(validation_manifest_path),
                    data_seed=group["data_seed"],
                    allow_data_reuse=False,
                )
                final_state = None
                for row_index in range(group["required_rows_per_rank"]):
                    _, _, final_state = next(loader)
                    if row_index and row_index % 10_000 == 0:
                        print(
                            f"capacity seed={group['data_seed']} rank={rank}/"
                            f"{group['world_size']} rows={row_index:,}/"
                            f"{group['required_rows_per_rank']:,}",
                            flush=True,
                        )
            except (RuntimeError, StopIteration) as exc:
                raise RuntimeError(
                    "one-epoch data capacity is insufficient for the frozen plan: "
                    f"seed={group['data_seed']} rank={rank} "
                    f"required_rows={group['required_rows_per_rank']}"
                ) from exc
            finally:
                _restore_rank_environment(prior)
            assert final_state is not None
            rank_proofs.append(
                {
                    "rank": rank,
                    "rows_replayed": group["required_rows_per_rank"],
                    "final_source_epoch": final_state["epoch"],
                    "final_packing_stats": final_state["packing_after_batch"],
                    "elapsed_seconds": time.monotonic() - rank_started,
                }
            )
        proofs.append({**group, "ranks": rank_proofs})

    validation_proofs: list[dict[str, Any]] = []
    for key in sorted(validation_requirements):
        group = validation_requirements[key]
        shard_proofs: list[dict[str, Any]] = []
        for shard in validation.shards:
            rank_proofs = []
            for rank in range(group["world_size"]):
                prior = _with_rank_environment(rank, group["world_size"])
                rank_started = time.monotonic()
                try:
                    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
                        tokenizer,
                        B=1,
                        T=group["sequence_length"],
                        split="val",
                        device="cpu",
                        tokenizer_threads=4,
                        tokenizer_batch_size=tokenizer_batch_size,
                        buffer_size=buffer_size,
                        dataset_manifest=str(train_manifest_path),
                        validation_manifest=str(validation_manifest_path),
                        validation_shard=shard.name,
                        allow_data_reuse=False,
                    )
                    final_state = None
                    for _ in range(group["required_rows_per_rank"]):
                        _, _, final_state = next(loader)
                except (RuntimeError, StopIteration) as exc:
                    raise RuntimeError(
                        "one-epoch heldout capacity is insufficient for the frozen plan: "
                        f"shard={shard.name} rank={rank} "
                        f"required_rows={group['required_rows_per_rank']}"
                    ) from exc
                finally:
                    _restore_rank_environment(prior)
                assert final_state is not None
                rank_proofs.append(
                    {
                        "rank": rank,
                        "rows_replayed": group["required_rows_per_rank"],
                        "final_source_epoch": final_state["epoch"],
                        "final_packing_stats": final_state["packing_after_batch"],
                        "elapsed_seconds": time.monotonic() - rank_started,
                    }
                )
            shard_proofs.append({"shard_name": shard.name, "ranks": rank_proofs})
        validation_proofs.append({**group, "shards": shard_proofs})

    receipt = {
        "schema_version": CAPACITY_SCHEMA_VERSION,
        "status": "verified",
        "algorithm": "exact_bestfit_row_replay_v1",
        "code_commit": code_commit,
        "plan_frozen_sha256": plan_frozen_sha256,
        "dataset_key": dataset_key,
        "dataset_id": train.dataset_id,
        "plan_sha256": _sha256_file(plan_path),
        "train_manifest_sha256": train.manifest_sha256,
        "validation_manifest_sha256": validation.manifest_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "tokenizer_engine": {
            "rustbpe_version": _distribution_version("rustbpe"),
            "tiktoken_version": _distribution_version("tiktoken"),
            "tokenizers_version": _distribution_version("tokenizers"),
        },
        "packing": {
            "buffer_size": buffer_size,
            "tokenizer_batch_size": tokenizer_batch_size,
            "allow_data_reuse": False,
        },
        "proofs": proofs,
        "validation_proofs": validation_proofs,
        "elapsed_seconds": time.monotonic() - started,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    partial = receipt_path.with_name(receipt_path.name + ".partial")
    partial.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    partial.replace(receipt_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--plan-frozen-sha256", required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--tokenizer-batch-size", type=int, default=128)
    args = parser.parse_args()
    tokenizer_path = args.tokenizer_dir / "tokenizer.pkl"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer.pkl is missing: {tokenizer_path}")
    tokenizer_sha256 = _sha256_file(tokenizer_path)
    tokenizer = RustBPETokenizer.from_directory(args.tokenizer_dir)
    receipt = prove_capacity(
        plan_path=args.plan,
        train_manifest_path=args.train_manifest,
        validation_manifest_path=args.validation_manifest,
        tokenizer=tokenizer,
        tokenizer_sha256=tokenizer_sha256,
        receipt_path=args.receipt,
        code_commit=args.code_commit,
        plan_frozen_sha256=args.plan_frozen_sha256,
        dataset_key=args.dataset_key,
        buffer_size=args.buffer_size,
        tokenizer_batch_size=args.tokenizer_batch_size,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
