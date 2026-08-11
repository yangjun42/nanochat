#!/usr/bin/env python3
"""Prove that a hash-bound token_bytes.pt exactly matches tokenizer.pkl."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

import torch

from csc_fast_scaling_law.campaign_contract import sha256_file


SCHEMA_VERSION = "nanochat-tokenizer-consistency-v1"
ALGORITHM = "tiktoken_raw_token_bytes_v1"


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _resolve_file(manifest_path: Path, spec: dict[str, Any]) -> Path:
    raw = spec.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError("tokenizer manifest file path must be non-empty")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def verify_tokenizer_consistency(
    *,
    manifest_path: Path,
    code_commit: str,
    receipt_path: Path,
    corrected_token_bytes_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    receipt_path = receipt_path.expanduser().resolve()
    if receipt_path.exists():
        raise FileExistsError(f"tokenizer receipt is immutable: {receipt_path}")
    if not re.fullmatch(r"[0-9a-f]{40,64}", code_commit):
        raise ValueError("code_commit must be a lowercase Git commit digest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid tokenizer manifest {manifest_path}: {exc}") from exc
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if manifest.get("schema_version") != 1 or not isinstance(files, list):
        raise ValueError("tokenizer manifest schema_version must be 1 with files")
    by_name = {
        spec.get("name"): spec for spec in files if isinstance(spec, dict)
    }
    if set(by_name) != {"tokenizer.pkl", "token_bytes.pt"} or len(files) != 2:
        raise ValueError(
            "formal tokenizer manifest must contain exactly tokenizer.pkl and token_bytes.pt"
        )
    resolved: dict[str, tuple[Path, str]] = {}
    for name, spec in by_name.items():
        digest = spec.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"tokenizer manifest lacks a valid SHA-256 for {name}")
        path = _resolve_file(manifest_path, spec)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"tokenizer artifact is missing or hash-mismatched: {path}")
        resolved[name] = (path, digest)

    with resolved["tokenizer.pkl"][0].open("rb") as handle:
        encoding = pickle.load(handle)
    special_ids = sorted(
        encoding.encode_single_token(token) for token in encoding.special_tokens_set
    )
    expected = []
    special_id_set = set(special_ids)
    for token_id in range(encoding.n_vocab):
        if token_id in special_id_set:
            expected.append(0)
        else:
            try:
                expected.append(len(encoding.decode_single_token_bytes(token_id)))
            except KeyError as exc:
                raise ValueError(f"tokenizer vocabulary has a missing token id {token_id}") from exc
    try:
        observed = torch.load(
            resolved["token_bytes.pt"][0], map_location="cpu", weights_only=True
        )
    except TypeError:
        observed = torch.load(resolved["token_bytes.pt"][0], map_location="cpu")
    if not isinstance(observed, torch.Tensor):
        raise ValueError("token_bytes.pt must contain one torch.Tensor")
    if observed.dtype != torch.int32 or observed.ndim != 1:
        raise ValueError("token_bytes tensor must be one-dimensional torch.int32")
    if observed.numel() != len(expected):
        raise ValueError(
            f"token_bytes length mismatch: expected {len(expected)}, observed {observed.numel()}"
        )
    expected_tensor = torch.tensor(expected, dtype=torch.int32)
    mismatch = torch.nonzero(observed != expected_tensor, as_tuple=False).flatten()
    if mismatch.numel():
        first = int(mismatch[0].item())
        correction_note = ""
        if corrected_token_bytes_path is not None:
            corrected = corrected_token_bytes_path.expanduser().resolve()
            if corrected.exists():
                raise FileExistsError(
                    f"refusing to overwrite corrected token-bytes output: {corrected}"
                )
            corrected.parent.mkdir(parents=True, exist_ok=True)
            partial = corrected.with_name(corrected.name + ".partial")
            if partial.exists():
                raise FileExistsError(f"stale correction partial already exists: {partial}")
            with partial.open("wb") as handle:
                torch.save(expected_tensor, handle)
                handle.flush()
                os.fsync(handle.fileno())
            partial.replace(corrected)
            correction_note = (
                f" Corrected immutable candidate written to {corrected} "
                f"(sha256={sha256_file(corrected)}); create a new tokenizer manifest "
                "and rerun the audit."
            )
        raise ValueError(
            "token_bytes does not match raw tokenizer bytes: "
            f"token_id={first}, expected={expected[first]}, observed={int(observed[first])}."
            + correction_note
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
        "algorithm": ALGORITHM,
        "code_commit": code_commit,
        "tokenizer_manifest_sha256": sha256_file(manifest_path),
        "tokenizer_pickle_sha256": resolved["tokenizer.pkl"][1],
        "token_bytes_sha256": resolved["token_bytes.pt"][1],
        "vocab_size": len(expected),
        "special_token_ids": special_ids,
        "engine": {
            "rustbpe_version": _distribution_version("rustbpe"),
            "tiktoken_version": _distribution_version("tiktoken"),
            "tokenizers_version": _distribution_version("tokenizers"),
            "torch_version": torch.__version__,
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    partial = receipt_path.with_name(receipt_path.name + ".partial")
    partial.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    partial.replace(receipt_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--corrected-token-bytes-out", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_tokenizer_consistency(
                manifest_path=args.tokenizer_manifest,
                code_commit=args.code_commit,
                receipt_path=args.receipt,
                corrected_token_bytes_path=args.corrected_token_bytes_out,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
