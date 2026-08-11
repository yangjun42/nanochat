from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pytest
import tiktoken
import torch

from csc_fast_scaling_law.tokenizer_consistency import verify_tokenizer_consistency


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mismatched_token_bytes_generates_new_candidate_then_verifies(
    tmp_path: Path,
) -> None:
    encoding = tiktoken.Encoding(
        name="synthetic-raw-bytes",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([value]): value for value in range(256)},
        special_tokens={"<|bos|>": 256},
    )
    tokenizer_path = tmp_path / "tokenizer.pkl"
    with tokenizer_path.open("wb") as handle:
        pickle.dump(encoding, handle)
    stale = torch.ones(257, dtype=torch.int32)
    stale[128] = 3
    stale[256] = 0
    stale_path = tmp_path / "token_bytes_stale.pt"
    torch.save(stale, stale_path)
    manifest = tmp_path / "tokenizer.json"

    def write_manifest(token_bytes_path: Path) -> None:
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "tokenizer_id": "synthetic",
                    "files": [
                        {
                            "name": "tokenizer.pkl",
                            "path": str(tokenizer_path),
                            "sha256": _sha(tokenizer_path),
                        },
                        {
                            "name": "token_bytes.pt",
                            "path": str(token_bytes_path),
                            "sha256": _sha(token_bytes_path),
                        },
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    write_manifest(stale_path)
    corrected = tmp_path / "sealed-v2" / "token_bytes.pt"
    receipt = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="Corrected immutable candidate written"):
        verify_tokenizer_consistency(
            manifest_path=manifest,
            code_commit="a" * 40,
            receipt_path=receipt,
            corrected_token_bytes_path=corrected,
        )
    assert corrected.is_file()
    assert not receipt.exists()
    corrected_tensor = torch.load(corrected, map_location="cpu", weights_only=True)
    assert corrected_tensor[128].item() == 1
    assert corrected_tensor[256].item() == 0

    write_manifest(corrected)
    verified = verify_tokenizer_consistency(
        manifest_path=manifest,
        code_commit="a" * 40,
        receipt_path=receipt,
    )
    assert verified["status"] == "verified"
    assert verified["token_bytes_sha256"] == _sha(corrected)
