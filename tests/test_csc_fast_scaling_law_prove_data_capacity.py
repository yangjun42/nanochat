from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from csc_fast_scaling_law.prove_data_capacity import (
    capacity_requirements,
    prove_capacity,
    validation_capacity_requirements,
)


class IntegerTokenizer:
    def get_bos_token_id(self) -> int:
        return 99

    def encode(self, batch, prepend, num_threads):
        del num_threads
        return [[prepend, int(text)] for text in batch]


def _data(tmp_path: Path) -> tuple[Path, Path]:
    train_shard = tmp_path / "train.parquet"
    validation_shard = tmp_path / "validation.parquet"
    pq.write_table(
        pa.table({"text": [str(index) for index in range(8)]}),
        train_shard,
        row_group_size=1,
    )
    pq.write_table(
        pa.table({"text": ["9", "10"]}), validation_shard, row_group_size=1
    )
    manifests = []
    for name, split, shard in (
        ("train", "train", train_shard),
        ("validation", "validation", validation_shard),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dataset_id": "synthetic",
                    "split": split,
                    "shards": [{"name": name, "path": shard.name}],
                }
            ),
            encoding="utf-8",
        )
        manifests.append(path)
    return manifests[0], manifests[1]


def _plan(
    tmp_path: Path, target_tokens: int, train_manifest: Path, validation_manifest: Path
) -> Path:
    path = tmp_path / f"plan-{target_tokens}.csv"
    fields = [
        "task_id",
        "data_seed",
        "max_seq_len",
        "gpu_count",
        "device_batch_size",
        "total_batch_size",
        "target_tokens",
        "eval_device_batch_size",
        "eval_tokens_per_shard",
        "allow_data_reuse",
        "dataset_key",
        "dataset_manifest_sha256",
        "validation_manifest_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "task_id": "task-a",
                "data_seed": 7,
                "max_seq_len": 1,
                "gpu_count": 2,
                "device_batch_size": 1,
                "total_batch_size": 2,
                "target_tokens": target_tokens,
                "eval_device_batch_size": 1,
                "eval_tokens_per_shard": 2,
                "allow_data_reuse": "false",
                "dataset_key": "synthetic-data",
                "dataset_manifest_sha256": hashlib.sha256(
                    train_manifest.read_bytes()
                ).hexdigest(),
                "validation_manifest_sha256": hashlib.sha256(
                    validation_manifest.read_bytes()
                ).hexdigest(),
            }
        )
    return path


def test_capacity_requirements_share_a_row_replay_across_device_batches() -> None:
    groups = capacity_requirements(
        [
            {
                "task_id": "short",
                "data_seed": "3",
                "max_seq_len": "8",
                "gpu_count": "4",
                "device_batch_size": "2",
                "total_batch_size": "64",
                "target_tokens": "128",
            },
            {
                "task_id": "long",
                "data_seed": "3",
                "max_seq_len": "8",
                "gpu_count": "4",
                "device_batch_size": "4",
                "total_batch_size": "128",
                "target_tokens": "256",
            },
        ]
    )
    assert list(groups) == [(3, 8, 4)]
    assert groups[(3, 8, 4)]["required_rows_per_rank"] == 12


def test_validation_capacity_groups_by_sequence_and_world_size() -> None:
    groups = validation_capacity_requirements(
        [
            {
                "task_id": "a",
                "max_seq_len": "8",
                "gpu_count": "4",
                "eval_device_batch_size": "2",
                "eval_tokens_per_shard": "64",
            },
            {
                "task_id": "b",
                "max_seq_len": "8",
                "gpu_count": "4",
                "eval_device_batch_size": "4",
                "eval_tokens_per_shard": "128",
            },
        ]
    )
    assert list(groups) == [(8, 4)]
    assert groups[(8, 4)]["required_rows_per_rank"] == 4


def test_exact_capacity_proof_passes_and_binds_inputs(tmp_path: Path) -> None:
    train, validation = _data(tmp_path)
    plan = _plan(tmp_path, target_tokens=4, train_manifest=train, validation_manifest=validation)
    receipt_path = tmp_path / "capacity.json"
    receipt = prove_capacity(
        plan_path=plan,
        train_manifest_path=train,
        validation_manifest_path=validation,
        tokenizer=IntegerTokenizer(),
        tokenizer_sha256="a" * 64,
        receipt_path=receipt_path,
        code_commit="b" * 40,
        plan_frozen_sha256="c" * 64,
        buffer_size=1,
        tokenizer_batch_size=1,
    )
    assert receipt["status"] == "verified"
    assert receipt["proofs"][0]["required_rows_per_rank"] == 3
    assert [rank["rows_replayed"] for rank in receipt["proofs"][0]["ranks"]] == [3, 3]
    assert [
        rank["rows_replayed"]
        for rank in receipt["validation_proofs"][0]["shards"][0]["ranks"]
    ] == [1, 1]
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


def test_exact_capacity_proof_fails_before_writing_receipt(tmp_path: Path) -> None:
    train, validation = _data(tmp_path)
    plan = _plan(tmp_path, target_tokens=8, train_manifest=train, validation_manifest=validation)
    receipt_path = tmp_path / "capacity.json"
    with pytest.raises(RuntimeError, match="capacity is insufficient"):
        prove_capacity(
            plan_path=plan,
            train_manifest_path=train,
            validation_manifest_path=validation,
            tokenizer=IntegerTokenizer(),
            tokenizer_sha256="a" * 64,
            receipt_path=receipt_path,
            code_commit="b" * 40,
            plan_frozen_sha256="c" * 64,
            buffer_size=1,
            tokenizer_batch_size=1,
        )
    assert not receipt_path.exists()
