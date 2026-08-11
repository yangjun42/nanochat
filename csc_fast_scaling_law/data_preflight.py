#!/usr/bin/env python3
"""Verify campaign parquet inputs once and emit a hash-bound receipt.

Run this on a CPU node before authoring the immutable campaign manifest.  GPU
tasks validate the receipt plus cheap file size/mtime observations instead of
hashing the full corpus independently on every distributed rank.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from csc_fast_scaling_law.campaign_contract import ContractError, atomic_write_json


DATA_PREFLIGHT_SCHEMA_VERSION = "nanochat-data-preflight-v1"


def _observed_provenance(manifest: Any) -> dict[str, Any]:
    provenance = manifest.provenance()
    for shard in provenance["shards"]:
        stat = Path(shard["path"]).stat()
        shard["observed_size_bytes"] = stat.st_size
        shard["observed_mtime_ns"] = stat.st_mtime_ns
    return provenance


def build_receipt(
    train: Any,
    validation: Any,
    *,
    code_commit: str,
    verified_unix_time: int | None = None,
) -> dict[str, Any]:
    """Build the exact receipt consumed by the GPU-side validator."""

    return {
        "schema_version": DATA_PREFLIGHT_SCHEMA_VERSION,
        "status": "verified",
        "code_commit": code_commit,
        "verified_unix_time": (
            int(time.time()) if verified_unix_time is None else int(verified_unix_time)
        ),
        "train": _observed_provenance(train),
        "validation": _observed_provenance(validation),
    }


def verify_data(
    *,
    dataset_manifest: Path,
    validation_manifest: Path,
    code_commit: str,
    out_path: Path,
) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40,64}", code_commit):
        raise ContractError("code_commit must be a lowercase Git commit digest")
    out_path = out_path.expanduser().resolve()
    if out_path.exists():
        raise ContractError(f"preflight receipt already exists; choose a new path: {out_path}")
    try:
        from nanochat.dataset import load_parquet_manifest, validate_manifest_separation

        train = load_parquet_manifest(
            dataset_manifest,
            expected_split="train",
            verify_shard_hashes=True,
            require_shard_hashes=True,
        )
        validation = load_parquet_manifest(
            validation_manifest,
            expected_split="validation",
            verify_shard_hashes=True,
            require_shard_hashes=True,
        )
        validate_manifest_separation(train, validation)
    except (ImportError, OSError, ValueError) as exc:
        raise ContractError(f"dataset preflight failed: {exc}") from exc
    receipt = build_receipt(train, validation, code_commit=code_commit)
    atomic_write_json(out_path, receipt)
    return out_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        print(
            verify_data(
                dataset_manifest=args.dataset_manifest,
                validation_manifest=args.validation_manifest,
                code_commit=args.code_commit,
                out_path=args.out,
            )
        )
        return 0
    except ContractError as exc:
        print(f"PREFLIGHT ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
