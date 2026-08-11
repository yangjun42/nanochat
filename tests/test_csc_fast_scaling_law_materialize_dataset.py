from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from csc_fast_scaling_law.materialize_dataset import build_manifest, parse_indices
from nanochat.dataset import load_parquet_manifest, validate_manifest_separation


def _write_parquet(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"text": [text]}), path)


def test_parse_indices_is_sorted_deduplicated_and_inclusive() -> None:
    assert parse_indices("4,1-3,2") == (1, 2, 3, 4)
    with pytest.raises(ValueError, match="invalid shard range"):
        parse_indices("4-2")


def test_build_manifests_hashes_and_validates_separation(tmp_path: Path) -> None:
    train_shard = tmp_path / "shards" / "shard_00000.parquet"
    validation_shard = tmp_path / "shards" / "shard_00001.parquet"
    _write_parquet(train_shard, "train")
    _write_parquet(validation_shard, "validation")
    train_path = build_manifest(
        dataset_id="synthetic",
        split="train",
        shard_paths=[train_shard],
        output_path=tmp_path / "manifests" / "train.json",
    )
    validation_path = build_manifest(
        dataset_id="synthetic",
        split="validation",
        shard_paths=[validation_shard],
        output_path=tmp_path / "manifests" / "validation.json",
    )
    train = load_parquet_manifest(
        train_path, expected_split="train", verify_shard_hashes=True
    )
    validation = load_parquet_manifest(
        validation_path, expected_split="validation", verify_shard_hashes=True
    )
    validate_manifest_separation(train, validation)
    assert train.shards[0].sha256
    assert validation.shards[0].size_bytes == validation_shard.stat().st_size


def test_exact_content_overlap_is_rejected_even_at_different_paths(tmp_path: Path) -> None:
    train_shard = tmp_path / "a" / "train.parquet"
    validation_shard = tmp_path / "b" / "validation.parquet"
    _write_parquet(train_shard, "same")
    validation_shard.parent.mkdir(parents=True)
    shutil.copy2(train_shard, validation_shard)
    train_path = build_manifest(
        dataset_id="synthetic",
        split="train",
        shard_paths=[train_shard],
        output_path=tmp_path / "manifests" / "train.json",
    )
    validation_path = build_manifest(
        dataset_id="synthetic",
        split="validation",
        shard_paths=[validation_shard],
        output_path=tmp_path / "manifests" / "validation.json",
    )
    train = load_parquet_manifest(train_path, expected_split="train")
    validation = load_parquet_manifest(validation_path, expected_split="validation")
    with pytest.raises(ValueError, match="exact duplicate"):
        validate_manifest_separation(train, validation)
