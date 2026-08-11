#!/usr/bin/env python3
"""Capture the exact child-Python environment expected by contracted GPU tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from csc_fast_scaling_law.campaign_contract import atomic_write_json, sha256_file
from csc_fast_scaling_law.nanochat_task_runner import (
    ENVIRONMENT_SCHEMA_VERSION,
    probe_environment,
)


EXPECTED_KEYS = (
    "python_version",
    "python_implementation",
    "python_executable_sha256",
    "torch_version",
    "torch_cuda_version",
    "pyarrow_version",
    "rustbpe_version",
    "tokenizers_version",
    "tiktoken_version",
    "torchao_version",
    "kernels_version",
    "fa3_interface_origin",
    "fa3_loaded_files",
    "fa3_requested_revision",
    "fa3_resolved_revision",
    "attention_backend",
    "flash_attention_3_available",
    "compute_dtype",
    "cuda_available",
    "cuda_device_names_unique",
    "cuda_capabilities_unique",
    "nccl_version",
    "cudnn_version",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--lockfile", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    observed = probe_environment(args.python, checkout=args.checkout)
    if observed.get("cuda_available") is not True or observed.get("cuda_device_count", 0) < 1:
        parser.error("capture the environment lock inside a GPU allocation")
    lockfiles = []
    for raw_path in args.lockfile:
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            parser.error(f"dependency lockfile is missing: {path}")
        lockfiles.append({"path": str(path), "sha256": sha256_file(path)})
    payload = {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "expected": {key: observed[key] for key in EXPECTED_KEYS},
        "lockfiles": lockfiles,
    }
    atomic_write_json(args.out.expanduser().resolve(), payload)
    print(args.out.expanduser().resolve())


if __name__ == "__main__":
    main()
