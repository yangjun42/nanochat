from __future__ import annotations

import csv
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from csc_fast_scaling_law.campaign_contract import ContractError
from csc_fast_scaling_law.data_preflight import verify_data
from csc_fast_scaling_law.nanochat_task_runner import (
    _parse_eval_bpb,
    build_commands,
    probe_environment,
    stage_tokenizer,
    validate_checkpoint_contract,
    validate_data_capacity_receipt,
    validate_data_preflight_receipt,
    validate_environment_observation,
    validate_schedule_manifest,
    validate_text_overlap_audit,
)


def test_environment_probe_preserves_symlink_venv_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_python = tmp_path / "base-python"
    base_python.write_text(
        "#!/bin/sh\n"
        'test "$0" = "$EXPECTED_PYTHON_LAUNCHER" || exit 97\n'
        "printf '%s\\n' '{\"launcher_preserved\": true}'\n",
        encoding="utf-8",
    )
    base_python.chmod(0o755)
    launcher = tmp_path / "campaign-venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(base_python)
    monkeypatch.setenv("EXPECTED_PYTHON_LAUNCHER", str(launcher))

    observed = probe_environment(str(launcher))

    assert observed["launcher_preserved"] is True
    assert observed["python_executable_sha256"] == hashlib.sha256(
        base_python.read_bytes()
    ).hexdigest()


def _prepared(tmp_path: Path) -> dict[str, object]:
    data_manifest = tmp_path / "data.json"
    validation_manifest = tmp_path / "validation.json"
    tokenizer_manifest = tmp_path / "tokenizer.json"
    tokenizer_manifest.write_text("{}\n", encoding="utf-8")
    data_manifest.write_text("{}\n", encoding="utf-8")
    validation_manifest.write_text("{}\n", encoding="utf-8")
    row = {
        "task_id": "synthetic-task",
        "init_seed": "101",
        "data_seed": "202",
        "schedule_id": "synthetic-schedule",
        "gpu_count": "2",
        "target_state": "source",
        "model_tag": "synthetic-model",
        "depth": "4",
        "aspect_ratio": "64",
        "head_dim": "64",
        "model_dim": "256",
        "max_seq_len": "1024",
        "window_pattern": "L",
        "target_tokens": "4194304",
        "target_param_data_ratio": "12",
        "reference_param_data_ratio": "12",
        "total_batch_size": "524288",
        "device_batch_size": "8",
        "eval_device_batch_size": "8",
        "eval_tokens_per_shard": "1048576",
        "warmup_steps": "2",
        "muon_momentum_warmup_steps": "2",
        "weight_decay_horizon_tokens": "4194304",
        "warmdown_ratio": "0.65",
        "final_lr_frac": "0.05",
        "embedding_lr": "0.3",
        "unembedding_lr": "0.008",
        "weight_decay": "0.28",
        "matrix_lr": "0.02",
        "scalar_lr": "0.5",
        "fp8": "0",
        "fp8_recipe": "tensorwise",
        "optimizer_schedule_version": "exact_last_update_v1",
        "allow_data_reuse": "false",
        "attention_backend": "fa3",
        "compute_dtype": "bfloat16",
        "known_schedule_overlap": "false",
        "n_scaling": "1000",
        "dataset_key": "synthetic-data",
        "dataset_manifest_sha256": hashlib.sha256(data_manifest.read_bytes()).hexdigest(),
        "validation_manifest_sha256": hashlib.sha256(
            validation_manifest.read_bytes()
        ).hexdigest(),
        "bpb_splits": "val",
    }
    plan = tmp_path / "plan.csv"
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    schedule_fields = (
        "target_tokens",
        "total_batch_size",
        "warmup_steps",
        "muon_momentum_warmup_steps",
        "weight_decay_horizon_tokens",
    )
    numeric_schedule_fields = (
        "warmdown_ratio",
        "final_lr_frac",
        "embedding_lr",
        "unembedding_lr",
        "weight_decay",
        "matrix_lr",
        "scalar_lr",
        "target_param_data_ratio",
        "reference_param_data_ratio",
    )
    schedule_manifest = tmp_path / "schedule.json"
    schedule_manifest.write_text(
        json.dumps(
            {
                "schema_version": "nanochat-schedule-manifest-v1",
                "schedules": [
                    {
                        "task_id": row["task_id"],
                        "schedule_id": row["schedule_id"],
                        **{field: int(row[field]) for field in schedule_fields},
                        **{field: float(row[field]) for field in numeric_schedule_fields},
                        "optimizer_schedule_version": row["optimizer_schedule_version"],
                        "attention_backend": row["attention_backend"],
                        "compute_dtype": row["compute_dtype"],
                        "known_schedule_overlap": False,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    plan_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    plan_frozen = tmp_path / "PLAN_FROZEN"
    plan_frozen.write_text("synthetic frozen-plan anchor\n", encoding="utf-8")
    return {
        "status": "prepared",
        "task_id": "synthetic-task",
        "staging_dir": str(tmp_path / "staging"),
        "schedule": {"id": "synthetic-schedule", "gpu_count": 2},
        "plan_sha256": plan_sha,
        "code": {"commit": "0" * 40, "dirty": False},
        "observed_code": {"checkout": str(tmp_path)},
        "file_provenance": {
            "plan": {"path": str(plan), "sha256": plan_sha},
            "plan_frozen": {
                "path": str(plan_frozen),
                "sha256": hashlib.sha256(plan_frozen.read_bytes()).hexdigest(),
            },
            "data_manifest": {
                "path": str(data_manifest),
                "sha256": hashlib.sha256(data_manifest.read_bytes()).hexdigest(),
            },
            "validation_manifest": {
                "path": str(validation_manifest),
                "sha256": hashlib.sha256(validation_manifest.read_bytes()).hexdigest(),
            },
            "tokenizer_manifest": {
                "path": str(tokenizer_manifest),
                "sha256": hashlib.sha256(tokenizer_manifest.read_bytes()).hexdigest(),
            },
            "schedule_manifest": {
                "path": str(schedule_manifest),
                "sha256": hashlib.sha256(schedule_manifest.read_bytes()).hexdigest(),
            },
        },
        "plan_row": row,
    }


def test_commands_enforce_exact_tokens_separate_seeds_and_no_optimizer(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    train, evaluate = build_commands(prepared, python_executable="/synthetic/python")
    assert "--nproc_per_node=2" in train
    assert "--target-tokens=4194304" in train
    assert "--target-param-data-ratio=12" in train
    assert "--reference-param-data-ratio=12" in train
    assert "--target-param-data-ratio=-1" not in train
    assert not any(part.startswith("--num-iterations") for part in train)
    assert "--init-seed=101" in train
    assert "--data-seed=202" in train
    assert "--no-save-optimizer" in train
    assert "--no-allow-data-reuse" in train
    assert "--optimizer-schedule-version=exact_last_update_v1" in train
    assert "--require-attention-backend=fa3" in train
    assert "--require-compute-dtype=bfloat16" in train
    assert "--verify-shard-hashes" not in train
    assert "--data-seed=202" in evaluate
    assert "--eval-tokens-per-shard=1048576" in evaluate
    assert "--bpb-splits=val" in evaluate
    assert "--require-attention-backend=fa3" in evaluate
    assert "--require-compute-dtype=bfloat16" in evaluate
    assert "--verify-shard-hashes" not in evaluate


def test_commands_reject_target_token_rounding(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    prepared["plan_row"]["target_tokens"] = "4194305"  # type: ignore[index]
    with pytest.raises(ContractError, match="exactly divisible"):
        build_commands(prepared, python_executable="/synthetic/python")


def test_commands_reject_negative_reference_data_ratio(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    prepared["plan_row"]["target_param_data_ratio"] = "-1"  # type: ignore[index]
    with pytest.raises(ContractError, match="target_param_data_ratio"):
        build_commands(prepared, python_executable="/synthetic/python")


def test_schedule_geometry_allows_declared_legacy_overlap_but_rejects_exact(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    row = prepared["plan_row"]  # type: ignore[assignment]
    row["warmup_steps"] = "4"
    row["known_schedule_overlap"] = "true"
    with pytest.raises(ContractError, match="exact schedule warmup overlaps"):
        build_commands(prepared, python_executable="/synthetic/python")
    row["optimizer_schedule_version"] = "legacy_v0"
    row["weight_decay_horizon_tokens"] = "12000"
    train, _ = build_commands(prepared, python_executable="/synthetic/python")
    assert "--optimizer-schedule-version=legacy_v0" in train


def test_schedule_manifest_exactly_binds_every_plan_task(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    assert validate_schedule_manifest(prepared)["task_count"] == 1
    path = Path(prepared["file_provenance"]["schedule_manifest"]["path"])  # type: ignore[index]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schedules"][0]["final_lr_frac"] = 0.2
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    prepared["file_provenance"]["schedule_manifest"]["sha256"] = hashlib.sha256(  # type: ignore[index]
        path.read_bytes()
    ).hexdigest()
    with pytest.raises(ContractError, match="final_lr_frac mismatch"):
        validate_schedule_manifest(prepared)


def test_bound_plan_rejects_mixed_or_mismatched_dataset_identity(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    plan_path = Path(prepared["file_provenance"]["plan"]["path"])  # type: ignore[index]
    first = dict(prepared["plan_row"])  # type: ignore[arg-type]

    def rewrite(rows: list[dict[str, str]]) -> None:
        with plan_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        prepared["file_provenance"]["plan"]["sha256"] = digest  # type: ignore[index]
        prepared["plan_sha256"] = digest

    mismatched = dict(first)
    mismatched["dataset_manifest_sha256"] = "f" * 64
    rewrite([mismatched])
    with pytest.raises(ContractError, match="data manifest hash"):
        validate_schedule_manifest(prepared)

    second = dict(first)
    second["task_id"] = "synthetic-task-2"
    second["dataset_key"] = "different-data"
    rewrite([first, second])
    with pytest.raises(ContractError, match="exactly one valid dataset_key"):
        validate_schedule_manifest(prepared)


def test_tokenizer_manifest_is_hash_verified_before_staging(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    tokenizer = tmp_path / "tokenizer.pkl"
    tokenizer.write_bytes(b"synthetic tokenizer; not a real vocabulary")
    digest = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    token_bytes = tmp_path / "token_bytes.pt"
    token_bytes.write_bytes(b"synthetic byte lengths; not a torch tensor")
    token_bytes_digest = hashlib.sha256(token_bytes.read_bytes()).hexdigest()
    manifest_path = Path(prepared["file_provenance"]["tokenizer_manifest"]["path"])  # type: ignore[index]
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tokenizer_id": "synthetic",
                    "files": [
                        {"name": "tokenizer.pkl", "path": str(tokenizer), "sha256": digest},
                        {
                            "name": "token_bytes.pt",
                            "path": str(token_bytes),
                            "sha256": token_bytes_digest,
                        },
                    ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prepared["file_provenance"]["tokenizer_manifest"]["sha256"] = hashlib.sha256(  # type: ignore[index]
        manifest_path.read_bytes()
    ).hexdigest()
    staged = stage_tokenizer(prepared, tmp_path / "work")
    assert staged == [
        {"name": "tokenizer.pkl", "source": str(tokenizer), "sha256": digest},
        {
            "name": "token_bytes.pt",
            "source": str(token_bytes),
            "sha256": token_bytes_digest,
        },
    ]
    assert (tmp_path / "work" / "tokenizer" / "tokenizer.pkl").is_symlink()

    tokenizer.write_bytes(b"changed")
    with pytest.raises(ContractError, match="hash-mismatched"):
        stage_tokenizer(prepared, tmp_path / "second-work")


def test_data_preflight_receipt_avoids_per_rank_full_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    train_shard = tmp_path / "train.parquet"
    validation_shard = tmp_path / "validation.parquet"
    train_shard.write_bytes(b"synthetic train bytes")
    validation_shard.write_bytes(b"synthetic validation bytes")
    data_manifest = Path(prepared["file_provenance"]["data_manifest"]["path"])  # type: ignore[index]
    validation_manifest = Path(  # type: ignore[index]
        prepared["file_provenance"]["validation_manifest"]["path"]  # type: ignore[index]
    )
    data_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "synthetic-train",
                "split": "train",
                "shards": [
                    {
                        "name": "train",
                        "path": str(train_shard),
                        "sha256": hashlib.sha256(train_shard.read_bytes()).hexdigest(),
                        "size_bytes": None,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    validation_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "synthetic-validation",
                "split": "validation",
                "shards": [
                    {
                        "name": "validation",
                        "path": str(validation_shard),
                        "sha256": hashlib.sha256(validation_shard.read_bytes()).hexdigest(),
                        "size_bytes": None,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    data_sha = hashlib.sha256(data_manifest.read_bytes()).hexdigest()
    validation_sha = hashlib.sha256(validation_manifest.read_bytes()).hexdigest()
    prepared["file_provenance"]["data_manifest"]["sha256"] = data_sha  # type: ignore[index]
    prepared["file_provenance"]["validation_manifest"]["sha256"] = validation_sha  # type: ignore[index]
    prepared["code"] = {"commit": "0" * 40, "dirty": False}

    class FakeManifest:
        def __init__(self, manifest: Path, split: str, shard: Path, manifest_sha: str) -> None:
            self.manifest = manifest
            self.split = split
            self.shard = shard
            self.manifest_sha = manifest_sha

        def provenance(self) -> dict[str, object]:
            return {
                "schema_version": 1,
                "dataset_id": f"synthetic-{self.split}",
                "split": self.split,
                "manifest_path": str(self.manifest),
                "manifest_sha256": self.manifest_sha,
                "shards": [
                    {
                        "name": self.shard.stem,
                        "path": str(self.shard),
                        "sha256": hashlib.sha256(self.shard.read_bytes()).hexdigest(),
                        "size_bytes": None,
                    }
                ],
            }

    fake_dataset = types.ModuleType("nanochat.dataset")

    def load_parquet_manifest(path: Path, *, expected_split: str, **_: object) -> FakeManifest:
        if expected_split == "train":
            return FakeManifest(data_manifest, "train", train_shard, data_sha)
        return FakeManifest(
            validation_manifest,
            "validation",
            validation_shard,
            validation_sha,
        )

    fake_dataset.load_parquet_manifest = load_parquet_manifest  # type: ignore[attr-defined]
    fake_dataset.validate_manifest_separation = lambda train, validation: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nanochat.dataset", fake_dataset)
    receipt = tmp_path / "preflight.json"
    verify_data(
        dataset_manifest=data_manifest,
        validation_manifest=validation_manifest,
        code_commit="0" * 40,
        out_path=receipt,
    )
    prepared["file_provenance"]["data_preflight_receipt"] = {  # type: ignore[index]
        "path": str(receipt),
        "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
    }
    summary = validate_data_preflight_receipt(prepared)
    assert summary["splits"]["train"]["shard_count"] == 1
    assert summary["splits"]["validation"]["shard_count"] == 1

    audit = tmp_path / "text_overlap_audit.json"
    audit_payload = {
        "schema_version": 1,
        "algorithm": "sha256_utf8_sqlite_exact_v1",
        "verdict": "PASS",
        "code_commit": "0" * 40,
        "train_manifest_sha256": data_sha,
        "validation_manifest_sha256": validation_sha,
        "cross_split": {
            "overlapping_unique_texts": 0,
            "overlapping_train_rows": 0,
            "overlapping_validation_rows": 0,
        },
    }
    audit.write_text(json.dumps(audit_payload) + "\n", encoding="utf-8")
    prepared["file_provenance"]["text_overlap_audit"] = {  # type: ignore[index]
        "path": str(audit),
        "sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
    }
    assert validate_text_overlap_audit(prepared)["verdict"] == "PASS"

    train_shard.write_bytes(b"changed")
    with pytest.raises(ContractError, match="size changed|mtime changed"):
        validate_data_preflight_receipt(prepared)


def test_environment_observation_is_bound_to_expected_runtime(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    nanochat_origin = tmp_path / "nanochat" / "__init__.py"
    base_train_origin = tmp_path / "scripts" / "base_train.py"
    nanochat_origin.parent.mkdir()
    base_train_origin.parent.mkdir()
    nanochat_origin.write_text("# synthetic\n", encoding="utf-8")
    base_train_origin.write_text("# synthetic\n", encoding="utf-8")
    fa3_origin = tmp_path / "kernel_cache" / "flash_attn_interface.py"
    fa3_origin.parent.mkdir()
    fa3_origin.write_text("# synthetic FA3 interface\n", encoding="utf-8")
    fa3_sha = hashlib.sha256(fa3_origin.read_bytes()).hexdigest()
    fa3_revision = "d" * 40
    dependency_lock = tmp_path / "dependency.lock"
    dependency_lock.write_text("synthetic dependency lock\n", encoding="utf-8")
    observed = {
        "python_version": "3.12.0",
        "python_implementation": "CPython",
        "python_executable": "/synthetic/python",
        "python_executable_sha256": "a" * 64,
        "torch_version": "2.10.0",
        "torch_cuda_version": "12.8",
        "pyarrow_version": "20.0.0",
        "rustbpe_version": "0.1.0",
        "tokenizers_version": "0.21.0",
        "tiktoken_version": "0.9.0",
        "torchao_version": "0.12.0",
        "kernels_version": "0.11.0",
        "fa3_interface_origin": str(fa3_origin),
        "fa3_loaded_files": [
            {
                "module": "flash_attn_interface",
                "path": str(fa3_origin),
                "sha256": fa3_sha,
            }
        ],
        "fa3_requested_revision": fa3_revision,
        "fa3_resolved_revision": fa3_revision,
        "nanochat_origin": str(nanochat_origin),
        "base_train_origin": str(base_train_origin),
        "attention_backend": "fa3",
        "flash_attention_3_available": True,
        "compute_dtype": "bfloat16",
        "cuda_available": True,
        "cuda_device_count": 2,
        "cuda_devices": [
            {"name": "Synthetic GPU", "capability": [9, 0], "total_memory": 1},
            {"name": "Synthetic GPU", "capability": [9, 0], "total_memory": 1},
        ],
        "cuda_device_names_unique": ["Synthetic GPU"],
        "cuda_capabilities_unique": [[9, 0]],
        "nccl_version": [2, 27, 0],
        "cudnn_version": 91000,
    }
    expected_keys = {
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
    }
    environment_lock = tmp_path / "environment.json"
    environment_lock.write_text(
        json.dumps(
            {
                "schema_version": "nanochat-environment-v1",
                "expected": {key: observed[key] for key in expected_keys},
                "lockfiles": [
                    {
                        "path": str(dependency_lock),
                        "sha256": hashlib.sha256(dependency_lock.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prepared["file_provenance"]["environment_lock"] = {  # type: ignore[index]
        "path": str(environment_lock),
        "sha256": hashlib.sha256(environment_lock.read_bytes()).hexdigest(),
    }
    validated = validate_environment_observation(prepared, observed)
    assert validated["observed"]["torch_version"] == "2.10.0"

    changed = dict(observed)
    changed["torch_version"] = "different"
    with pytest.raises(ContractError, match="torch_version mismatch"):
        validate_environment_observation(prepared, changed)


def test_eval_parser_requires_every_heldout_shard_and_recomputed_mean(
    tmp_path: Path,
) -> None:
    log = tmp_path / "eval.log"
    log.write_text(
        'EVAL_RUNTIME_CONTRACT: {"attention_backend": "fa3", "compute_dtype": "bfloat16"}\n'
        "val/heldout-a bpb: 1.000000\n"
        "val/heldout-b bpb: 1.200000\n"
        "val bpb: 1.100000\n",
        encoding="utf-8",
    )
    parsed = _parse_eval_bpb(
        log,
        expected_validation_names=["heldout-a", "heldout-b"],
        expected_runtime_contract={"attention_backend": "fa3", "compute_dtype": "bfloat16"},
    )
    assert parsed["val_bpb_by_shard"] == {
        "heldout-a": 1.0,
        "heldout-b": 1.2,
    }
    log.write_text(
        'EVAL_RUNTIME_CONTRACT: {"attention_backend": "fa3", "compute_dtype": "bfloat16"}\n'
        "val/heldout-a bpb: 1.000000\nval bpb: 1.000000\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="keys/order"):
        _parse_eval_bpb(
            log,
            expected_validation_names=["heldout-a", "heldout-b"],
            expected_runtime_contract={
                "attention_backend": "fa3",
                "compute_dtype": "bfloat16",
            },
        )


def test_checkpoint_metadata_must_exactly_match_plan_and_data_contract(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    row = prepared["plan_row"]  # type: ignore[assignment]
    integer_fields = (
        "init_seed",
        "data_seed",
        "depth",
        "aspect_ratio",
        "head_dim",
        "model_dim",
        "max_seq_len",
        "target_tokens",
        "total_batch_size",
        "device_batch_size",
        "eval_tokens_per_shard",
        "warmup_steps",
        "muon_momentum_warmup_steps",
        "weight_decay_horizon_tokens",
    )
    float_fields = (
        "warmdown_ratio",
        "final_lr_frac",
        "embedding_lr",
        "unembedding_lr",
        "weight_decay",
        "matrix_lr",
        "scalar_lr",
        "target_param_data_ratio",
        "reference_param_data_ratio",
    )
    user_config = {
        **{field: int(row[field]) for field in integer_fields},
        **{field: float(row[field]) for field in float_fields},
        "window_pattern": row["window_pattern"],
        "fp8_recipe": row["fp8_recipe"],
        "optimizer_schedule_version": row["optimizer_schedule_version"],
        "fp8": False,
        "model_tag": row["model_tag"],
        "run": "dummy",
        "seed": int(row["init_seed"]),
        "num_iterations": -1,
        "target_flops": -1.0,
        "save_optimizer": False,
        "allow_data_reuse": False,
        "train_row_group_start": 0,
        "verify_shard_hashes": False,
        "eval_every": -1,
        "sample_every": -1,
        "core_metric_every": -1,
        "save_every": -1,
        "require_attention_backend": "fa3",
        "require_compute_dtype": "bfloat16",
        "dataset_manifest": prepared["file_provenance"]["data_manifest"]["path"],  # type: ignore[index]
        "validation_manifest": prepared["file_provenance"]["validation_manifest"]["path"],  # type: ignore[index]
    }
    data_contract = {
        "loader_state_version": 2,
        "train_manifest_sha256": prepared["file_provenance"]["data_manifest"]["sha256"],  # type: ignore[index]
        "validation_manifest_sha256": prepared["file_provenance"]["validation_manifest"]["sha256"],  # type: ignore[index]
        "data_seed": int(row["data_seed"]),
        "split": "train",
        "validation_shard": None,
        "device_batch_size": int(row["device_batch_size"]),
        "sequence_length": int(row["max_seq_len"]),
        "tokenizer_batch_size": 128,
        "packing_buffer_size": 1000,
        "ddp_world_size": int(row["gpu_count"]),
        "allow_data_reuse": False,
    }
    meta = {
        "step": 8,
        "device_batch_size": 8,
        "max_seq_len": 1024,
        "total_batch_size": 524288,
        "prediction_positions_completed": 4194304,
        "user_config": user_config,
        "model_config": {
            "sequence_len": 1024,
            "n_layer": 4,
            "n_head": 4,
            "n_kv_head": 4,
            "n_embd": 256,
            "window_pattern": "L",
        },
        "checkpoint_policy": {"save_optimizer": False, "resume_mode": "restart_only"},
        "runtime_contract": {"attention_backend": "fa3", "compute_dtype": "bfloat16"},
        "dataset_manifest": {
            "manifest_path": prepared["file_provenance"]["data_manifest"]["path"],  # type: ignore[index]
            "manifest_sha256": prepared["file_provenance"]["data_manifest"]["sha256"],  # type: ignore[index]
        },
        "validation_manifest": {
            "manifest_path": prepared["file_provenance"]["validation_manifest"]["path"],  # type: ignore[index]
            "manifest_sha256": prepared["file_provenance"]["validation_manifest"]["sha256"],  # type: ignore[index]
        },
        "dataloader_state_dict": {
            "loader_state_version": 2,
            "data_contract": data_contract,
        },
    }
    assert validate_checkpoint_contract(meta, prepared)["step"] == 8
    meta["dataloader_state_dict"]["data_contract"]["data_seed"] = 999
    with pytest.raises(ContractError, match="data_contract"):
        validate_checkpoint_contract(meta, prepared)


def test_capacity_receipt_must_exactly_cover_plan_ranks_and_validation_shards(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    validation_manifest = Path(
        prepared["file_provenance"]["validation_manifest"]["path"]  # type: ignore[index]
    )
    validation_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "synthetic-validation",
                "split": "validation",
                "shards": [
                    {"name": "heldout-a", "path": "a.parquet", "sha256": "a" * 64},
                    {"name": "heldout-b", "path": "b.parquet", "sha256": "b" * 64},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prepared["file_provenance"]["validation_manifest"]["sha256"] = hashlib.sha256(  # type: ignore[index]
        validation_manifest.read_bytes()
    ).hexdigest()
    prepared["plan_row"]["validation_manifest_sha256"] = prepared["file_provenance"][  # type: ignore[index]
        "validation_manifest"
    ]["sha256"]
    plan_path = Path(prepared["file_provenance"]["plan"]["path"])  # type: ignore[index]
    with plan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prepared["plan_row"]))  # type: ignore[arg-type]
        writer.writeheader()
        writer.writerow(prepared["plan_row"])  # type: ignore[arg-type]
    new_plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    prepared["file_provenance"]["plan"]["sha256"] = new_plan_sha  # type: ignore[index]
    prepared["plan_sha256"] = new_plan_sha
    tokenizer = tmp_path / "tokenizer.pkl"
    tokenizer.write_bytes(b"synthetic tokenizer")
    tokenizer_sha = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    tokenizer_manifest = Path(
        prepared["file_provenance"]["tokenizer_manifest"]["path"]  # type: ignore[index]
    )
    tokenizer_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {"name": "tokenizer.pkl", "path": str(tokenizer), "sha256": tokenizer_sha}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prepared["file_provenance"]["tokenizer_manifest"]["sha256"] = hashlib.sha256(  # type: ignore[index]
        tokenizer_manifest.read_bytes()
    ).hexdigest()

    def ranks(required: int) -> list[dict[str, int]]:
        return [
            {"rank": rank, "rows_replayed": required, "final_source_epoch": 1}
            for rank in range(2)
        ]

    receipt_payload = {
        "schema_version": "nanochat-data-capacity-v1",
        "status": "verified",
        "algorithm": "exact_bestfit_row_replay_v1",
        "code_commit": "0" * 40,
        "plan_frozen_sha256": prepared["file_provenance"]["plan_frozen"]["sha256"],  # type: ignore[index]
        "dataset_key": "synthetic-data",
        "plan_sha256": prepared["plan_sha256"],
        "train_manifest_sha256": prepared["file_provenance"]["data_manifest"]["sha256"],  # type: ignore[index]
        "validation_manifest_sha256": prepared["file_provenance"]["validation_manifest"]["sha256"],  # type: ignore[index]
        "tokenizer_sha256": tokenizer_sha,
        "tokenizer_engine": {
            "rustbpe_version": "0.1.0",
            "tiktoken_version": "0.9.0",
            "tokenizers_version": "0.21.0",
        },
        "packing": {
            "buffer_size": 1000,
            "tokenizer_batch_size": 128,
            "allow_data_reuse": False,
        },
        "proofs": [
            {
                "data_seed": 202,
                "sequence_length": 1024,
                "world_size": 2,
                "required_rows_per_rank": 2056,
                "covering_tasks": [
                    {"task_id": "synthetic-task", "required_rows_per_rank": 2056}
                ],
                "ranks": ranks(2056),
            }
        ],
        "validation_proofs": [
            {
                "sequence_length": 1024,
                "world_size": 2,
                "required_rows_per_rank": 512,
                "covering_tasks": [
                    {
                        "task_id": "synthetic-task",
                        "required_rows_per_rank": 512,
                        "eval_tokens_per_shard": 1048576,
                    }
                ],
                "shards": [
                    {"shard_name": "heldout-a", "ranks": ranks(512)},
                    {"shard_name": "heldout-b", "ranks": ranks(512)},
                ],
            }
        ],
    }
    receipt = tmp_path / "capacity.json"
    receipt.write_text(json.dumps(receipt_payload) + "\n", encoding="utf-8")
    prepared["file_provenance"]["data_capacity_receipt"] = {  # type: ignore[index]
        "path": str(receipt),
        "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
    }
    observed = {
        "rustbpe_version": "0.1.0",
        "tiktoken_version": "0.9.0",
        "tokenizers_version": "0.21.0",
    }
    assert validate_data_capacity_receipt(
        prepared, environment_observed=observed
    )["validation_proof_group_count"] == 1
    receipt_payload["proofs"][0]["ranks"] = [ranks(2056)[0], ranks(2056)[0]]
    receipt.write_text(json.dumps(receipt_payload) + "\n", encoding="utf-8")
    prepared["file_provenance"]["data_capacity_receipt"]["sha256"] = hashlib.sha256(  # type: ignore[index]
        receipt.read_bytes()
    ).hexdigest()
    with pytest.raises(ContractError, match="rank IDs/order"):
        validate_data_capacity_receipt(prepared, environment_observed=observed)
