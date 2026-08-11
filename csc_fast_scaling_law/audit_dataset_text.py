#!/usr/bin/env python3
"""Exact, streaming text-overlap audit for contracted parquet datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from nanochat.dataset import load_parquet_manifest, validate_manifest_separation


AUDIT_SCHEMA_VERSION = 1
AUDIT_ALGORITHM = "sha256_utf8_sqlite_exact_v1"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    partial.replace(path)


def _scan_split(
    connection: sqlite3.Connection,
    manifest,
    *,
    count_column: str,
    batch_size: int,
) -> dict[str, Any]:
    if count_column not in {"train_count", "validation_count"}:
        raise ValueError(f"invalid count column: {count_column}")
    sequence_digest = hashlib.sha256()
    total_rows = 0
    empty_rows = 0
    shard_summaries: list[dict[str, Any]] = []
    statement = (
        f"INSERT INTO documents(digest, {count_column}) VALUES (?, 1) "
        f"ON CONFLICT(digest) DO UPDATE SET {count_column}={count_column}+1"
    )
    for shard in manifest.shards:
        parquet = pq.ParquetFile(shard.path)
        shard_sequence_digest = hashlib.sha256()
        shard_rows = 0
        shard_empty_rows = 0
        for record_batch in parquet.iter_batches(
            batch_size=batch_size, columns=["text"], use_threads=False
        ):
            texts = record_batch.column(0).to_pylist()
            rows: list[tuple[bytes]] = []
            for text in texts:
                if not isinstance(text, str):
                    raise ValueError(
                        f"non-string text escaped parquet validation in {shard.path}"
                    )
                encoded = text.encode("utf-8")
                digest = hashlib.sha256(encoded).digest()
                sequence_digest.update(digest)
                shard_sequence_digest.update(digest)
                rows.append((digest,))
                total_rows += 1
                shard_rows += 1
                if not text:
                    empty_rows += 1
                    shard_empty_rows += 1
            connection.executemany(statement, rows)
        connection.commit()
        shard_summaries.append(
            {
                "name": shard.name,
                "rows": shard_rows,
                "empty_rows": shard_empty_rows,
                "ordered_text_digest_sha256": shard_sequence_digest.hexdigest(),
            }
        )
    return {
        "total_rows": total_rows,
        "empty_rows": empty_rows,
        "ordered_text_digest_sha256": sequence_digest.hexdigest(),
        "shards": shard_summaries,
    }


def audit_text_overlap(
    *,
    train_manifest_path: Path,
    validation_manifest_path: Path,
    receipt_path: Path,
    code_commit: str,
    work_database_path: Path | None = None,
    batch_size: int = 65_536,
    verify_shard_hashes: bool = True,
) -> dict[str, Any]:
    """Audit exact text duplication and fail closed on train/validation leakage."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not re.fullmatch(r"[0-9a-f]{40,64}", code_commit):
        raise ValueError("code_commit must be a lowercase Git commit digest")
    if receipt_path.exists():
        raise FileExistsError(
            f"text audit receipt is immutable; choose a new path: {receipt_path}"
        )
    train = load_parquet_manifest(
        train_manifest_path,
        expected_split="train",
        require_shard_hashes=True,
        verify_shard_hashes=verify_shard_hashes,
    )
    validation = load_parquet_manifest(
        validation_manifest_path,
        expected_split="validation",
        require_shard_hashes=True,
        verify_shard_hashes=verify_shard_hashes,
    )
    validate_manifest_separation(train, validation)

    database_path = work_database_path or receipt_path.with_suffix(
        receipt_path.suffix + ".sqlite"
    )
    database_path.parent.mkdir(parents=True, exist_ok=True)
    partial_database = database_path.with_name(database_path.name + ".partial")
    partial_database.unlink(missing_ok=True)
    connection = sqlite3.connect(partial_database)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            """
            CREATE TABLE documents (
                digest BLOB PRIMARY KEY,
                train_count INTEGER NOT NULL DEFAULT 0,
                validation_count INTEGER NOT NULL DEFAULT 0
            ) WITHOUT ROWID
            """
        )
        validation_summary = _scan_split(
            connection,
            validation,
            count_column="validation_count",
            batch_size=batch_size,
        )
        train_summary = _scan_split(
            connection,
            train,
            count_column="train_count",
            batch_size=batch_size,
        )
        unique_train, unique_validation, cross_unique = connection.execute(
            """
            SELECT
                SUM(CASE WHEN train_count > 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN validation_count > 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN train_count > 0 AND validation_count > 0 THEN 1 ELSE 0 END)
            FROM documents
            """
        ).fetchone()
        unique_train = int(unique_train or 0)
        unique_validation = int(unique_validation or 0)
        cross_unique = int(cross_unique or 0)
        cross_train_rows, cross_validation_rows = connection.execute(
            """
            SELECT
                COALESCE(SUM(train_count), 0),
                COALESCE(SUM(validation_count), 0)
            FROM documents
            WHERE train_count > 0 AND validation_count > 0
            """
        ).fetchone()
        train_summary.update(
            {
                "unique_texts": unique_train,
                "duplicate_rows": train_summary["total_rows"] - unique_train,
                "duplicate_fraction": (
                    (train_summary["total_rows"] - unique_train)
                    / train_summary["total_rows"]
                ),
            }
        )
        validation_summary.update(
            {
                "unique_texts": unique_validation,
                "duplicate_rows": validation_summary["total_rows"]
                - unique_validation,
                "duplicate_fraction": (
                    (validation_summary["total_rows"] - unique_validation)
                    / validation_summary["total_rows"]
                ),
            }
        )
        receipt = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "algorithm": AUDIT_ALGORITHM,
            "code_commit": code_commit,
            "dataset_id": train.dataset_id,
            "train_manifest_sha256": train.manifest_sha256,
            "validation_manifest_sha256": validation.manifest_sha256,
            "train": train_summary,
            "validation": validation_summary,
            "cross_split": {
                "overlapping_unique_texts": cross_unique,
                "overlapping_train_rows": int(cross_train_rows),
                "overlapping_validation_rows": int(cross_validation_rows),
            },
            "verdict": "PASS" if cross_unique == 0 else "FAIL",
        }
        _atomic_write_json(receipt_path, receipt)
    finally:
        connection.close()
        partial_database.unlink(missing_ok=True)
    if receipt["verdict"] != "PASS":
        raise ValueError(
            "training and validation contain exact overlapping text rows; "
            f"see audit receipt: {receipt_path}"
        )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream parquet text through an exact on-disk overlap audit."
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--work-database", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--skip-shard-hash-verification", action="store_true")
    args = parser.parse_args()
    receipt = audit_text_overlap(
        train_manifest_path=args.train_manifest,
        validation_manifest_path=args.validation_manifest,
        receipt_path=args.receipt,
        code_commit=args.code_commit,
        work_database_path=args.work_database,
        batch_size=args.batch_size,
        verify_shard_hashes=not args.skip_shard_hash_verification,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
