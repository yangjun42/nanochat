#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import requests

from nanochat.dataset import (
    load_parquet_manifest,
    validate_manifest_separation,
    validate_parquet_text_column,
)
from csc_fast_scaling_law.audit_dataset_text import audit_text_overlap


def parse_indices(specification: str) -> tuple[int, ...]:
    """Parse a comma-separated collection of integers and inclusive ranges."""

    values: set[int] = set()
    for raw_part in specification.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            pieces = part.split("-")
            if len(pieces) != 2:
                raise ValueError(f"invalid shard range: {part!r}")
            start, end = (int(piece) for piece in pieces)
            if start < 0 or end < start:
                raise ValueError(f"invalid shard range: {part!r}")
            values.update(range(start, end + 1))
        else:
            value = int(part)
            if value < 0:
                raise ValueError("shard indices must be nonnegative")
            values.add(value)
    if not values:
        raise ValueError("at least one shard index is required")
    return tuple(sorted(values))


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_parquet(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"downloaded shard is empty or missing: {path}")
    parquet = pq.ParquetFile(path)
    if parquet.num_row_groups <= 0 or parquet.metadata.num_rows <= 0:
        raise ValueError(f"downloaded parquet has no rows: {path}")
    validate_parquet_text_column(parquet, path)


def _download_one(*, url: str, destination: Path, timeout_s: float) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _validate_parquet(destination)
        return destination
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        partial.unlink()
    try:
        with requests.get(url, stream=True, timeout=timeout_s) as response:
            response.raise_for_status()
            with partial.open("xb") as handle:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        _validate_parquet(partial)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


def materialize_shards(
    *,
    base_url: str,
    shard_dir: Path,
    indices: Iterable[int],
    filename_template: str = "shard_{index:05d}.parquet",
    workers: int = 4,
    timeout_s: float = 120.0,
) -> dict[int, Path]:
    indices = tuple(sorted(set(int(index) for index in indices)))
    if not indices or min(indices) < 0:
        raise ValueError("indices must contain nonnegative shard ids")
    if workers <= 0:
        raise ValueError("workers must be positive")
    root = base_url.rstrip("/")
    destinations = {
        index: shard_dir / filename_template.format(index=index) for index in indices
    }
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        pending = {
            executor.submit(
                _download_one,
                url=f"{root}/{path.name}",
                destination=path,
                timeout_s=float(timeout_s),
            ): index
            for index, path in destinations.items()
        }
        for future in as_completed(pending):
            future.result()
    return destinations


def build_manifest(
    *,
    dataset_id: str,
    split: str,
    shard_paths: Iterable[Path],
    output_path: Path,
) -> Path:
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    paths = tuple(sorted((Path(path).resolve() for path in shard_paths), key=lambda p: p.name))
    if not paths:
        raise ValueError("a manifest requires at least one shard")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shards = []
    seen_hashes: set[str] = set()
    for path in paths:
        _validate_parquet(path)
        digest = _sha256_file(path)
        if digest in seen_hashes:
            raise ValueError(f"exact duplicate shard content detected: {path}")
        seen_hashes.add(digest)
        shards.append(
            {
                "name": path.stem,
                "path": os.path.relpath(path, output_path.parent),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    payload = {
        "schema_version": 1,
        "dataset_id": str(dataset_id),
        "split": split,
        "shards": shards,
    }
    temporary = output_path.with_name(output_path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(output_path)
    return output_path


def materialize_dataset(
    *,
    dataset_id: str,
    code_commit: str,
    base_url: str,
    output_root: Path,
    train_indices: Iterable[int],
    validation_indices: Iterable[int],
    train_manifest_name: str = "train.manifest.json",
    validation_manifest_name: str = "validation.manifest.json",
    workers: int = 4,
    timeout_s: float = 120.0,
    audit_receipt_name: str = "text_overlap.audit.json",
) -> tuple[Path, Path]:
    train_ids = tuple(sorted(set(int(index) for index in train_indices)))
    validation_ids = tuple(sorted(set(int(index) for index in validation_indices)))
    overlap = sorted(set(train_ids) & set(validation_ids))
    if overlap:
        raise ValueError(f"train and validation shard ids overlap: {overlap}")
    all_paths = materialize_shards(
        base_url=base_url,
        shard_dir=output_root / "shards",
        indices=(*train_ids, *validation_ids),
        workers=workers,
        timeout_s=timeout_s,
    )
    manifest_dir = output_root / "manifests"
    train_path = build_manifest(
        dataset_id=dataset_id,
        split="train",
        shard_paths=(all_paths[index] for index in train_ids),
        output_path=manifest_dir / train_manifest_name,
    )
    validation_path = build_manifest(
        dataset_id=dataset_id,
        split="validation",
        shard_paths=(all_paths[index] for index in validation_ids),
        output_path=manifest_dir / validation_manifest_name,
    )
    train = load_parquet_manifest(
        train_path,
        expected_split="train",
        verify_shard_hashes=True,
        require_shard_hashes=True,
    )
    validation = load_parquet_manifest(
        validation_path,
        expected_split="validation",
        verify_shard_hashes=True,
        require_shard_hashes=True,
    )
    validate_manifest_separation(train, validation)
    audit_text_overlap(
        train_manifest_path=train_path,
        validation_manifest_path=validation_path,
        receipt_path=manifest_dir / audit_receipt_name,
        code_commit=code_commit,
        # build_manifest just read and hashed every byte; avoid an immediate
        # redundant full-file hash pass before the row-level scan.
        verify_shard_hashes=False,
    )
    return train_path, validation_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atomically download parquet shards and write hashed split manifests."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-indices", required=True)
    parser.add_argument("--validation-indices", required=True)
    parser.add_argument("--train-manifest-name", default="train.manifest.json")
    parser.add_argument("--validation-manifest-name", default="validation.manifest.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--audit-receipt-name", default="text_overlap.audit.json")
    args = parser.parse_args()
    train_path, validation_path = materialize_dataset(
        dataset_id=args.dataset_id,
        code_commit=args.code_commit,
        base_url=args.base_url,
        output_root=args.output_root,
        train_indices=parse_indices(args.train_indices),
        validation_indices=parse_indices(args.validation_indices),
        train_manifest_name=args.train_manifest_name,
        validation_manifest_name=args.validation_manifest_name,
        workers=args.workers,
        timeout_s=args.timeout_seconds,
        audit_receipt_name=args.audit_receipt_name,
    )
    print(train_path)
    print(validation_path)


if __name__ == "__main__":
    main()
