import hashlib
import json
from pathlib import Path

import pytest
import torch

from csc_fast_scaling_law.gpu_preflight import (
    FORMAL_CAMPAIGN_GPU_COUNT,
    PREFLIGHT_WORLD_SIZES,
    PreflightContractError,
    aggregate_preflight,
    build_parser,
    load_manifest,
    load_token_batches,
    parameter_state_sha256,
    partition_global_batches,
    prepare_preflight,
    validate_bound_runtime_environment,
)


COMMIT = "a" * 40
FA3_COMMIT = "d" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _write_environment_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    dependency = tmp_path / "dependency.lock"
    dependency.write_text("locked\n", encoding="utf-8")
    fa3_origin = tmp_path / "flash_attn_interface.py"
    fa3_origin.write_text("# synthetic\n", encoding="utf-8")
    expected = {
        "python_version": "3.12.0",
        "python_implementation": "CPython",
        "python_executable_sha256": "1" * 64,
        "torch_version": "2.10.0",
        "torch_cuda_version": "12.8",
        "pyarrow_version": "20.0.0",
        "rustbpe_version": "0.1.0",
        "tokenizers_version": "0.21.0",
        "tiktoken_version": "0.9.0",
        "torchao_version": None,
        "kernels_version": "0.11.0",
        "fa3_interface_origin": str(fa3_origin),
        "fa3_loaded_files": [
            {
                "module": "flash_attn_interface",
                "path": str(fa3_origin),
                "sha256": _sha256(fa3_origin),
            }
        ],
        "fa3_requested_revision": FA3_COMMIT,
        "fa3_resolved_revision": FA3_COMMIT,
        "attention_backend": "fa3",
        "flash_attention_3_available": True,
        "compute_dtype": "bfloat16",
        "cuda_available": True,
        "cuda_device_names_unique": ["Synthetic GH200"],
        "cuda_capabilities_unique": [[9, 0]],
        "nccl_version": [2, 27, 0],
        "cudnn_version": 91000,
    }
    environment_lock = tmp_path / "source_environment_lock.json"
    environment_lock.write_text(
        json.dumps(
            {
                "schema_version": "nanochat-environment-v1",
                "expected": expected,
                "lockfiles": [{"path": str(dependency), "sha256": _sha256(dependency)}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bootstrap = tmp_path / "source_runtime_bootstrap.sh"
    bootstrap.write_text("export NANOCHAT_FA3_REVISION=" + FA3_COMMIT + "\n", encoding="utf-8")
    profile = tmp_path / "source_formal_runtime_profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": "nanochat_formal_runtime_profile/v1",
                "gpu_count": 4,
                "compute_dtype": "bfloat16",
                "attention_backend": "fa3",
                "compile_model": True,
                "fp8_recipe": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return environment_lock, bootstrap, profile


def _prepare(tmp_path: Path) -> Path:
    environment_lock, bootstrap, profile = _write_environment_inputs(tmp_path)
    args = build_parser().parse_args(
        [
            "prepare",
            "--output-dir",
            str(tmp_path / "inputs"),
            "--code-commit",
            COMMIT,
            "--environment-lock",
            str(environment_lock),
            "--runtime-bootstrap",
            str(bootstrap),
            "--formal-runtime-profile",
            str(profile),
            "--num-updates",
            "4",
            "--global-batch-sequences",
            "8",
            "--sequence-length",
            "8",
            "--vocab-size",
            "32",
            "--n-embd",
            "32",
            "--n-head",
            "2",
            "--n-kv-head",
            "2",
            "--device-batch-sequences",
            "1",
            "--warmup-steps",
            "1",
            "--muon-momentum-warmup-steps",
            "1",
            "--timing-warmup-updates",
            "1",
        ]
    )
    return prepare_preflight(args)


def test_prepare_freezes_four_gpu_formal_campaign_and_one_token_tensor(tmp_path):
    manifest_path = _prepare(tmp_path)
    manifest = load_manifest(manifest_path)

    assert manifest["preflight_only"] is True
    assert manifest["formal_campaign_gpu_count"] == FORMAL_CAMPAIGN_GPU_COUNT == 4
    assert manifest["world_sizes"] == list(PREFLIGHT_WORLD_SIZES) == [1, 2, 4]
    assert set(manifest["tolerances"].values()) == {0.0}
    assert manifest["runtime"] == {
        "compute_dtype": "bfloat16",
        "attention_backend": "fa3",
        "compile_model": True,
        "fp8_recipe": None,
    }
    assert manifest["environment_lock"]["file"] == "environment_lock.json"
    assert manifest["runtime_bootstrap"]["file"] == "runtime_bootstrap.sh"
    assert manifest["formal_runtime_profile"]["file"] == "formal_runtime_profile.json"
    tokens = load_token_batches(manifest_path, manifest)
    assert tokens.shape == (4, 8, 9)
    assert tokens.dtype == torch.int32


def test_partitions_reconstruct_the_exact_global_batches_for_every_world_size(tmp_path):
    manifest_path = _prepare(tmp_path)
    manifest = load_manifest(manifest_path)
    tokens = load_token_batches(manifest_path, manifest)

    for world_size in PREFLIGHT_WORLD_SIZES:
        pieces = [
            partition_global_batches(
                tokens,
                world_size=world_size,
                rank=rank,
                device_batch_sequences=1,
            )
            for rank in range(world_size)
        ]
        assert torch.equal(torch.cat(pieces, dim=1), tokens)


def test_token_artifact_hash_is_fail_closed(tmp_path):
    manifest_path = _prepare(tmp_path)
    token_path = manifest_path.parent / "global_token_batches.pt"
    with token_path.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(PreflightContractError, match="SHA-256 mismatch"):
        load_manifest(manifest_path)


def _environment_receipt(manifest_path: Path) -> dict[str, object]:
    manifest = load_manifest(manifest_path)
    lock_path = manifest_path.parent / manifest["environment_lock"]["file"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    observed = dict(lock["expected"])
    observed.update(
        {
            "python_executable": "/synthetic/python",
            "cuda_device_count": 4,
            "cuda_devices": [],
            "nanochat_origin": "/synthetic/checkout/nanochat/__init__.py",
            "base_train_origin": "/synthetic/checkout/scripts/base_train.py",
        }
    )
    return {
        "schema_version": "nanochat_gpu_preflight_environment/v1",
        "gpu_count": 4,
        "environment_lock_sha256": manifest["environment_lock"]["sha256"],
        "runtime_bootstrap_sha256": manifest["runtime_bootstrap"]["sha256"],
        "formal_validation": {
            "schema_version": "nanochat-environment-v1",
            "environment_lock": {
                "path": str(lock_path),
                "sha256": manifest["environment_lock"]["sha256"],
            },
            "observed": observed,
            "verified_lockfiles": lock["lockfiles"],
        },
    }


def _write_worker_result(
    *,
    manifest_path: Path,
    root: Path,
    world_size: int,
    initial: dict[str, torch.Tensor],
    final: dict[str, torch.Tensor],
    losses: list[float],
    throughput: float,
) -> None:
    manifest = load_manifest(manifest_path)
    directory = root / f"world{world_size}"
    directory.mkdir(parents=True)
    initial_path = directory / "initial_parameters.pt"
    final_path = directory / "final_parameters.pt"
    torch.save(initial, initial_path)
    torch.save(final, final_path)
    deltas = [
        final[name].to(torch.float64) - initial[name].to(torch.float64)
        for name in sorted(initial)
    ]
    delta_l2 = sum(float(torch.sum(delta * delta).item()) for delta in deltas) ** 0.5
    delta_max = max(float(delta.abs().max().item()) for delta in deltas)
    environment_receipt = _environment_receipt(manifest_path)
    orchestrator_receipt = root / "orchestrator_environment.json"
    if not orchestrator_receipt.exists():
        orchestrator_receipt.write_text(
            json.dumps(environment_receipt) + "\n", encoding="utf-8"
        )
    result = {
        "schema_version": "nanochat_gpu_preflight_worker/v1",
        "preflight_only": True,
        "formal_campaign_gpu_count": 4,
        "manifest_sha256": _sha256(manifest_path),
        "code_commit": manifest["code_commit"],
        "world_size": world_size,
        "environment_contract": {
            "environment_lock_sha256": manifest["environment_lock"]["sha256"],
            "runtime_bootstrap_sha256": manifest["runtime_bootstrap"]["sha256"],
            "formal_runtime_profile_sha256": manifest["formal_runtime_profile"][
                "sha256"
            ],
            "validation_receipt_sha256": _canonical_sha256(environment_receipt),
            "validation_receipt": environment_receipt,
        },
        "runtime": {
            "compute_dtype": manifest["runtime"]["compute_dtype"],
            "attention_backend": manifest["runtime"]["attention_backend"],
            "compile_model": manifest["runtime"]["compile_model"],
            "fp8_recipe": manifest["runtime"]["fp8_recipe"],
        },
        "input_contract": {
            "token_sha256": manifest["token_batches"]["sha256"],
            "global_batch_sequences": manifest["token_batches"]["global_batch_sequences"],
            "sequence_length": manifest["token_batches"]["sequence_length"],
            "num_updates": manifest["token_batches"]["num_updates"],
            "device_batch_sequences": manifest["execution"]["device_batch_sequences"],
            "gradient_accumulation_steps": manifest["token_batches"][
                "global_batch_sequences"
            ]
            // (world_size * manifest["execution"]["device_batch_sequences"]),
        },
        "numerical": {
            "losses": losses,
            "initial_parameter_sha256": parameter_state_sha256(initial),
            "final_parameter_sha256": parameter_state_sha256(final),
            "parameter_delta_l2": delta_l2,
            "parameter_delta_max_abs": delta_max,
            "rank_parameters_exactly_synchronized": True,
        },
        "throughput": {
            "gate_role": "informational_only",
            "median_prediction_tokens_per_second": throughput,
        },
        "artifacts": {
            "initial_parameters": {
                "file": initial_path.name,
                "sha256": _sha256(initial_path),
            },
            "final_parameters": {
                "file": final_path.name,
                "sha256": _sha256(final_path),
            },
        },
    }
    (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (directory / "SUCCESS").write_text("PASS\n", encoding="utf-8")


def test_throughput_is_informational_and_cannot_fail_numerical_pass(tmp_path):
    manifest_path = _prepare(tmp_path)
    root = tmp_path / "results"
    initial = {"weight": torch.tensor([0.0, 0.0])}
    final = {"weight": torch.tensor([1.0, 2.0])}
    for world_size, throughput in ((1, 100.0), (2, 1.0), (4, 0.5)):
        _write_worker_result(
            manifest_path=manifest_path,
            root=root,
            world_size=world_size,
            initial=initial,
            final=final,
            losses=[3.0, 2.0, 1.5, 1.0],
            throughput=throughput,
        )

    summary = aggregate_preflight(manifest_path, root, root / "summary.json")

    assert summary["overall_status"] == "PASS"
    assert summary["numerical_equivalence"]["status"] == "PASS"
    assert summary["throughput"]["status"] == "MEASURED_NOT_GATED"
    assert summary["throughput"]["rows"][-1]["parallel_efficiency_vs_1_gpu"] == pytest.approx(
        0.00125
    )


def test_fast_candidate_still_fails_when_parameter_delta_disagrees(tmp_path):
    manifest_path = _prepare(tmp_path)
    root = tmp_path / "results"
    initial = {"weight": torch.tensor([0.0, 0.0])}
    reference_final = {"weight": torch.tensor([1.0, 2.0])}
    for world_size in PREFLIGHT_WORLD_SIZES:
        final = (
            {"weight": torch.tensor([1.5, 2.0])}
            if world_size == 4
            else reference_final
        )
        _write_worker_result(
            manifest_path=manifest_path,
            root=root,
            world_size=world_size,
            initial=initial,
            final=final,
            losses=[3.0, 2.0, 1.5, 1.0],
            throughput=float(world_size * 1000),
        )

    summary = aggregate_preflight(manifest_path, root, root / "summary.json")

    assert summary["overall_status"] == "FAIL"
    comparison = summary["numerical_equivalence"]["comparisons"][1]
    assert comparison["candidate_world_size"] == 4
    assert comparison["parameter_delta"]["pass"] is False
    assert summary["throughput"]["rows"][-1]["parallel_efficiency_vs_1_gpu"] == 1.0


def test_manifest_rejects_any_attempt_to_change_formal_gpu_count(tmp_path):
    manifest_path = _prepare(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["formal_campaign_gpu_count"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PreflightContractError, match="frozen at 4"):
        load_manifest(manifest_path, verify_token_file=False)


def test_v1_rejects_nonzero_tolerance_instead_of_auto_waiving_drift(tmp_path):
    manifest_path = _prepare(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tolerances"]["parameter_relative_l2"] = 0.019
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PreflightContractError, match="frozen at zero"):
        load_manifest(manifest_path, verify_token_file=False)


def test_any_nonzero_loss_difference_fails_exact_only_gate(tmp_path):
    manifest_path = _prepare(tmp_path)
    root = tmp_path / "results"
    initial = {"weight": torch.tensor([0.0, 0.0])}
    final = {"weight": torch.tensor([1.0, 2.0])}
    for world_size in PREFLIGHT_WORLD_SIZES:
        losses = [3.0, 2.0, 1.5, 1.0]
        if world_size == 2:
            losses[1] += 1e-12
        _write_worker_result(
            manifest_path=manifest_path,
            root=root,
            world_size=world_size,
            initial=initial,
            final=final,
            losses=losses,
            throughput=100.0,
        )

    summary = aggregate_preflight(manifest_path, root, root / "summary.json")
    assert summary["overall_status"] == "FAIL"
    assert summary["numerical_equivalence"]["comparisons"][0]["loss"]["pass"] is False


def test_environment_lock_and_runtime_profile_are_hash_bound(tmp_path):
    manifest_path = _prepare(tmp_path)
    lock_path = manifest_path.parent / "environment_lock.json"
    lock_path.write_text(lock_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(PreflightContractError, match="environment_lock SHA-256 mismatch"):
        load_manifest(manifest_path)

    other = tmp_path / "other"
    other.mkdir()
    second_manifest = _prepare(other)
    profile_path = second_manifest.parent / "formal_runtime_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["compile_model"] = False
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(PreflightContractError, match="formal_runtime_profile SHA-256 mismatch"):
        load_manifest(second_manifest)


def test_runtime_validation_reuses_formal_probe_and_validator(
    tmp_path, monkeypatch
):
    manifest_path = _prepare(tmp_path)
    manifest = load_manifest(manifest_path)
    bootstrap = manifest_path.parent / manifest["runtime_bootstrap"]["file"]
    monkeypatch.setenv("RUNTIME_BOOTSTRAP", str(bootstrap))
    monkeypatch.setenv("RUNTIME_BOOTSTRAP_SHA256", manifest["runtime_bootstrap"]["sha256"])
    monkeypatch.setenv("NANOCHAT_DTYPE", manifest["runtime"]["compute_dtype"])
    calls: dict[str, object] = {}

    def fake_probe(python_executable, *, checkout):
        calls["probe_python"] = python_executable
        calls["probe_checkout"] = checkout
        return {"observed": True}

    def fake_validate(prepared, observed):
        calls["prepared"] = prepared
        calls["observed"] = observed
        return {
            "schema_version": "nanochat-environment-v1",
            "environment_lock": prepared["file_provenance"]["environment_lock"],
            "observed": observed,
            "verified_lockfiles": [],
        }

    monkeypatch.setattr(
        "csc_fast_scaling_law.environment_probe.probe_environment", fake_probe
    )
    monkeypatch.setattr(
        "csc_fast_scaling_law.nanochat_task_runner.validate_environment_observation",
        fake_validate,
    )
    receipt = validate_bound_runtime_environment(
        manifest_path, manifest, gpu_count=FORMAL_CAMPAIGN_GPU_COUNT
    )

    assert calls["observed"] == {"observed": True}
    prepared = calls["prepared"]
    assert prepared["schedule"]["gpu_count"] == 4
    assert prepared["plan_row"] == {
        "attention_backend": "fa3",
        "compute_dtype": "bfloat16",
        "fp8": "false",
    }
    assert receipt["environment_lock_sha256"] == manifest["environment_lock"]["sha256"]
    assert receipt["runtime_bootstrap_sha256"] == manifest["runtime_bootstrap"]["sha256"]
