from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nanochat.dataset import (
    load_parquet_manifest,
    validate_manifest_separation,
    validation_shard_names,
)


def _write_parquet(path: Path, values: list[str]) -> None:
    table = pa.table({"text": values})
    pq.write_table(table, path, row_group_size=1)


def _write_manifest(
    path: Path,
    *,
    split: str,
    shards: list[dict[str, object]],
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "unit-test-corpus",
                "split": split,
                "shards": shards,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def test_manifest_resolves_paths_and_records_raw_fingerprint(tmp_path: Path) -> None:
    shard = tmp_path / "train.parquet"
    _write_parquet(shard, ["a", "b"])
    shard_sha = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest_path = _write_manifest(
        tmp_path / "train.json",
        split="train",
        shards=[
            {
                "name": "train-000",
                "path": shard.name,
                "sha256": shard_sha,
                "size_bytes": shard.stat().st_size,
            }
        ],
    )

    manifest = load_parquet_manifest(
        manifest_path,
        expected_split="train",
        verify_shard_hashes=True,
    )

    assert manifest.manifest_sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert manifest.shards[0].path == str(shard.resolve())
    assert manifest.provenance()["shards"][0]["sha256"] == shard_sha


def test_validation_manifest_names_independent_files_in_declared_order(tmp_path: Path) -> None:
    first = tmp_path / "heldout-a.parquet"
    second = tmp_path / "heldout-b.parquet"
    _write_parquet(first, ["a"])
    _write_parquet(second, ["b"])
    manifest_path = _write_manifest(
        tmp_path / "validation.json",
        split="validation",
        shards=[
            {"name": "heldout-a", "path": first.name},
            {"name": "heldout-b", "path": second.name},
        ],
    )

    assert validation_shard_names(manifest_path) == ["heldout-a", "heldout-b"]


def test_manifest_rejects_duplicate_paths_and_wrong_split(tmp_path: Path) -> None:
    shard = tmp_path / "data.parquet"
    _write_parquet(shard, ["a"])
    manifest_path = _write_manifest(
        tmp_path / "bad.json",
        split="validation",
        shards=[
            {"name": "a", "path": shard.name},
            {"name": "b", "path": shard.name},
        ],
    )

    with pytest.raises(ValueError, match="split mismatch"):
        load_parquet_manifest(manifest_path, expected_split="train")
    with pytest.raises(ValueError, match="duplicate shard path"):
        load_parquet_manifest(manifest_path, expected_split="validation")


def test_manifest_separation_rejects_exact_train_validation_overlap(tmp_path: Path) -> None:
    train_shard = tmp_path / "train.parquet"
    validation_shard = tmp_path / "validation.parquet"
    _write_parquet(train_shard, ["same"])
    validation_shard.write_bytes(train_shard.read_bytes())
    duplicate_sha = hashlib.sha256(train_shard.read_bytes()).hexdigest()
    train_path = _write_manifest(
        tmp_path / "train.json",
        split="train",
        shards=[{"path": train_shard.name, "sha256": duplicate_sha}],
    )
    validation_path = _write_manifest(
        tmp_path / "validation.json",
        split="validation",
        shards=[{"path": validation_shard.name, "sha256": duplicate_sha}],
    )

    train_manifest = load_parquet_manifest(
        train_path, expected_split="train", require_shard_hashes=True
    )
    validation_manifest = load_parquet_manifest(
        validation_path,
        expected_split="validation",
        require_shard_hashes=True,
    )

    with pytest.raises(ValueError, match="duplicate shard content"):
        validate_manifest_separation(train_manifest, validation_manifest)


@pytest.mark.parametrize(
    ("values", "message"),
    [([1, 2], "must be string"), (["ok", None], "null text values")],
)
def test_manifest_rejects_invalid_text_contract(
    tmp_path: Path, values: list[object], message: str
) -> None:
    shard = tmp_path / "invalid.parquet"
    pq.write_table(pa.table({"text": values}), shard, row_group_size=1)
    manifest_path = _write_manifest(
        tmp_path / "invalid.json",
        split="train",
        shards=[{"path": shard.name}],
    )
    with pytest.raises(ValueError, match=message):
        load_parquet_manifest(manifest_path, expected_split="train")


def test_manifest_separation_rejects_mismatched_dataset_ids(tmp_path: Path) -> None:
    train_shard = tmp_path / "train.parquet"
    validation_shard = tmp_path / "validation.parquet"
    _write_parquet(train_shard, ["train"])
    _write_parquet(validation_shard, ["validation"])
    train_path = _write_manifest(
        tmp_path / "train.json", split="train", shards=[{"path": train_shard.name}]
    )
    validation_path = _write_manifest(
        tmp_path / "validation.json",
        split="validation",
        shards=[{"path": validation_shard.name}],
    )
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    payload["dataset_id"] = "a-different-corpus"
    validation_path.write_text(json.dumps(payload), encoding="utf-8")
    train = load_parquet_manifest(train_path, expected_split="train")
    validation = load_parquet_manifest(
        validation_path, expected_split="validation"
    )
    with pytest.raises(ValueError, match="different dataset_id"):
        validate_manifest_separation(train, validation)
