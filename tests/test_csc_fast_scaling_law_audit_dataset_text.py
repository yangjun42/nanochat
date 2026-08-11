from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from csc_fast_scaling_law.audit_dataset_text import audit_text_overlap


def _manifest(tmp_path: Path, name: str, split: str, texts: list[str]) -> Path:
    shard = tmp_path / f"{name}.parquet"
    pq.write_table(pa.table({"text": texts}), shard, row_group_size=2)
    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "synthetic",
                "split": split,
                "shards": [
                    {
                        "name": name,
                        "path": shard.name,
                        "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                        "size_bytes": shard.stat().st_size,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_text_audit_reports_within_split_duplicates(tmp_path: Path) -> None:
    train = _manifest(tmp_path, "train", "train", ["a", "a", "b", ""])
    validation = _manifest(
        tmp_path, "validation", "validation", ["heldout-a", "heldout-b"]
    )
    receipt_path = tmp_path / "audit.json"
    receipt = audit_text_overlap(
        train_manifest_path=train,
        validation_manifest_path=validation,
        receipt_path=receipt_path,
        code_commit="a" * 40,
        batch_size=2,
    )
    assert receipt["verdict"] == "PASS"
    assert receipt["train"]["total_rows"] == 4
    assert receipt["train"]["unique_texts"] == 3
    assert receipt["train"]["duplicate_rows"] == 1
    assert receipt["train"]["empty_rows"] == 1
    assert receipt["cross_split"]["overlapping_unique_texts"] == 0
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


def test_text_audit_writes_failure_receipt_for_cross_split_leakage(
    tmp_path: Path,
) -> None:
    train = _manifest(tmp_path, "train", "train", ["shared", "train-only"])
    validation = _manifest(
        tmp_path, "validation", "validation", ["shared", "shared", "val-only"]
    )
    receipt_path = tmp_path / "audit.json"
    with pytest.raises(ValueError, match="exact overlapping text"):
        audit_text_overlap(
            train_manifest_path=train,
            validation_manifest_path=validation,
            receipt_path=receipt_path,
            code_commit="a" * 40,
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["verdict"] == "FAIL"
    assert receipt["cross_split"] == {
        "overlapping_unique_texts": 1,
        "overlapping_train_rows": 1,
        "overlapping_validation_rows": 2,
    }
    assert not list(tmp_path.glob("*.sqlite.partial"))
