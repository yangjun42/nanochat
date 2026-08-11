from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from csc_fast_scaling_law.campaign_contract import (
    ContractError,
    finalize_task,
    load_campaign,
    prepare_task,
    reduce_campaign,
    sha256_file,
    validate_complete_task,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_campaign(
    tmp_path: Path,
    *,
    target_states: tuple[str, ...] = ("source", "source"),
) -> tuple[Path, Path]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    checkout = tmp_path / "code_checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Synthetic Test"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "synthetic@example.invalid"],
        check=True,
    )
    (checkout / "code.txt").write_text("synthetic code\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "code.txt"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-q", "-m", "synthetic"], check=True)
    code_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    rows = []
    for index, target_state in enumerate(target_states, start=1):
        rows.append(
            {
                "task_id": f"synthetic-{index}",
                "init_seed": str(100 + index),
                "data_seed": str(200 + index),
                "schedule_id": "synthetic-schedule",
                "gpu_count": "2",
                "target_tokens": "2000",
                "target_state": target_state,
            }
        )
    plan = inputs / "plan.csv"
    with plan.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    named_inputs = {
        "plan": plan,
        "plan_frozen": inputs / "PLAN_FROZEN",
        "data_manifest": inputs / "data.json",
        "validation_manifest": inputs / "validation.json",
        "tokenizer_manifest": inputs / "tokenizer.json",
        "tokenizer_consistency_receipt": inputs / "tokenizer_consistency_receipt.json",
        "environment_lock": inputs / "environment.lock",
        "schedule_manifest": inputs / "schedule.json",
        "decision_rules": inputs / "decision_rules.json",
        "data_preflight_receipt": inputs / "data_preflight_receipt.json",
        "text_overlap_audit": inputs / "text_overlap_audit.json",
        "data_capacity_receipt": inputs / "data_capacity_receipt.json",
    }
    for name, path in named_inputs.items():
        if name != "plan":
            _write_json(path, {"synthetic": True, "kind": name})

    artifact_root = tmp_path / "artifacts"
    campaign_path = tmp_path / "campaign.json"
    campaign = {
        "schema_version": "nanochat-campaign-v1",
        "campaign_id": "synthetic-contract-test",
        "code": {"commit": code_commit, "dirty": False},
        "files": {
            name: {
                "path": str(path.relative_to(tmp_path)),
                "sha256": sha256_file(path),
            }
            for name, path in named_inputs.items()
        },
        "artifacts": {
            "root": str(artifact_root.relative_to(tmp_path)),
            "required": [
                {
                    "name": "checkpoint",
                    "glob": "work/base_checkpoints/{task_id}/model.pt",
                    "min_count": 1,
                    "max_count": 1,
                },
                {
                    "name": "checkpoint_meta",
                    "glob": "work/base_checkpoints/{task_id}/meta.json",
                    "min_count": 1,
                    "max_count": 1,
                },
                {
                    "name": "metric",
                    "glob": "metrics/final.json",
                    "min_count": 1,
                    "max_count": 1,
                },
            ],
            "forbid_globs": ["work/**/optim_*.pt"],
        },
        "reducer": {
            "metric_json": "metrics/final.json",
            "required_fields": ["val_bpb", "N_scaling", "D_actual"],
        },
    }
    _write_json(campaign_path, campaign)
    return campaign_path, artifact_root


def _complete_task(campaign_path: Path, artifact_root: Path, index: int) -> Path:
    prepared = prepare_task(
        campaign_path=campaign_path,
        task_index=index,
        artifact_root=artifact_root,
        gpu_count=2,
        checkout=campaign_path.parent / "code_checkout",
    )
    assert prepared["action"] == "run"
    staging = Path(prepared["staging_dir"])
    task_id = prepared["task_id"]
    checkpoint = staging / "work" / "base_checkpoints" / task_id
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.pt").write_bytes(f"synthetic model {index}".encode())
    _write_json(checkpoint / "meta.json", {"synthetic": True, "step": index})
    _write_json(
        staging / "metrics" / "final.json",
        {
            "schema_version": "nanochat-task-runner-v1",
            "status": "success",
            "task_id": task_id,
            "val_bpb": 1.0 + index / 100,
            "N_scaling": 1000 * index,
            "D_actual": 2000,
            "gpu_seconds_train": 2.5,
        },
    )
    return finalize_task(prepared_path=Path(prepared["prepared"]), exit_code=0)


def test_task_is_promoted_only_after_success_and_hash_validation(tmp_path: Path) -> None:
    campaign_path, artifact_root = _make_campaign(tmp_path, target_states=("source",))
    final_dir = _complete_task(campaign_path, artifact_root, 1)
    campaign = load_campaign(campaign_path)
    validate_complete_task(final_dir, campaign=campaign, row=campaign.plan_rows[0])

    rerun = prepare_task(
        campaign_path=campaign_path,
        task_index=1,
        artifact_root=artifact_root,
        gpu_count=2,
        checkout=campaign_path.parent / "code_checkout",
    )
    assert rerun["action"] == "skip"

    (final_dir / "metrics" / "final.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ContractError, match="inventory or hash mismatch"):
        prepare_task(
            campaign_path=campaign_path,
            task_index=1,
            artifact_root=artifact_root,
            gpu_count=2,
            checkout=campaign_path.parent / "code_checkout",
        )


def test_partial_final_directory_never_counts_as_complete(tmp_path: Path) -> None:
    campaign_path, artifact_root = _make_campaign(tmp_path, target_states=("source",))
    partial_final = artifact_root / "tasks" / "synthetic-1"
    partial_final.mkdir(parents=True)
    (partial_final / "model.pt").write_bytes(b"partial")
    with pytest.raises(ContractError, match="is partial"):
        prepare_task(
            campaign_path=campaign_path,
            task_index=1,
            artifact_root=artifact_root,
            gpu_count=2,
            checkout=campaign_path.parent / "code_checkout",
        )


def test_provisional_or_unsealed_target_is_not_an_executable_campaign(
    tmp_path: Path,
) -> None:
    campaign_path, _ = _make_campaign(tmp_path, target_states=("source",))
    campaign_json = json.loads(campaign_path.read_text(encoding="utf-8"))
    plan_path = campaign_path.parent / campaign_json["files"]["plan"]["path"]
    with plan_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for invalid_state in ("provisional", "unsealed"):
        rows[0]["target_state"] = invalid_state
        with plan_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        campaign_json["files"]["plan"]["sha256"] = sha256_file(plan_path)
        _write_json(campaign_path, campaign_json)
        with pytest.raises(ContractError, match="not executable"):
            load_campaign(campaign_path)


def test_abandoned_partial_attempt_is_retried_not_skipped(tmp_path: Path) -> None:
    campaign_path, artifact_root = _make_campaign(tmp_path, target_states=("source",))
    first = prepare_task(
        campaign_path=campaign_path,
        task_index=1,
        artifact_root=artifact_root,
        gpu_count=2,
        checkout=campaign_path.parent / "code_checkout",
    )
    Path(first["staging_dir"], "interrupted-checkpoint.pt").write_bytes(b"partial")
    second = prepare_task(
        campaign_path=campaign_path,
        task_index=1,
        artifact_root=artifact_root,
        gpu_count=2,
        checkout=campaign_path.parent / "code_checkout",
    )
    assert second["action"] == "run"
    assert second["staging_dir"] != first["staging_dir"]
    assert Path(first["staging_dir"]).is_dir()


def test_prepare_rejects_dirty_or_wrong_compute_checkout(tmp_path: Path) -> None:
    campaign_path, artifact_root = _make_campaign(tmp_path, target_states=("source",))
    checkout = campaign_path.parent / "code_checkout"
    (checkout / "code.txt").write_text("dirty change\n", encoding="utf-8")
    with pytest.raises(ContractError, match="checkout is dirty"):
        prepare_task(
            campaign_path=campaign_path,
            task_index=1,
            artifact_root=artifact_root,
            gpu_count=2,
            checkout=checkout,
        )

    subprocess.run(["git", "-C", str(checkout), "add", "code.txt"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-q", "-m", "wrong head"], check=True)
    with pytest.raises(ContractError, match="checkout HEAD mismatch"):
        prepare_task(
            campaign_path=campaign_path,
            task_index=1,
            artifact_root=artifact_root,
            gpu_count=2,
            checkout=checkout,
        )


def test_reducer_fails_closed_on_missing_task_then_reduces_exact_set(tmp_path: Path) -> None:
    campaign_path, artifact_root = _make_campaign(tmp_path)
    _complete_task(campaign_path, artifact_root, 1)
    reduced_path = tmp_path / "reduced.json"
    with pytest.raises(ContractError, match="task set mismatch"):
        reduce_campaign(
            campaign_path=campaign_path,
            artifact_root=artifact_root,
            out_path=reduced_path,
        )
    assert not reduced_path.exists()

    _complete_task(campaign_path, artifact_root, 2)
    reduce_campaign(
        campaign_path=campaign_path,
        artifact_root=artifact_root,
        out_path=reduced_path,
    )
    reduced = json.loads(reduced_path.read_text(encoding="utf-8"))
    assert reduced["status"] == "complete"
    assert reduced["task_count"] == 2
    assert [row["task_id"] for row in reduced["rows"]] == ["synthetic-1", "synthetic-2"]
    assert all(row["gpu_count"] == "2" for row in reduced["rows"])
    assert all(row["target_tokens"] == "2000" for row in reduced["rows"])


def test_sealed_task_requires_hash_bound_unlock(tmp_path: Path) -> None:
    campaign_path, artifact_root = _make_campaign(tmp_path, target_states=("sealed",))
    with pytest.raises(ContractError, match="requires an unlock"):
        prepare_task(
            campaign_path=campaign_path,
            task_index=1,
            artifact_root=artifact_root,
            gpu_count=2,
            checkout=campaign_path.parent / "code_checkout",
        )

    campaign = load_campaign(campaign_path)
    unlock_path = tmp_path / "unlock.json"
    unlock = {
        "schema_version": "nanochat-unlock-v1",
        "campaign_id": campaign.campaign_id,
        "campaign_manifest_sha256": campaign.sha256,
        "plan_sha256": campaign.plan.sha256,
        "decision_rules_sha256": campaign.files["decision_rules"].sha256,
        "status": "unlocked",
        "task_ids": ["synthetic-1"],
    }
    _write_json(unlock_path, unlock)
    prepared = prepare_task(
        campaign_path=campaign_path,
        task_index=1,
        artifact_root=artifact_root,
        unlock_path=unlock_path,
        gpu_count=2,
        checkout=campaign_path.parent / "code_checkout",
    )
    assert prepared["action"] == "run"

    unlock["plan_sha256"] = "f" * 64
    _write_json(unlock_path, unlock)
    with pytest.raises(ContractError, match="plan_sha256 mismatch"):
        prepare_task(
            campaign_path=campaign_path,
            task_index=1,
            artifact_root=artifact_root,
            unlock_path=unlock_path,
            gpu_count=2,
            checkout=campaign_path.parent / "code_checkout",
        )


def test_forbidden_optimizer_state_prevents_promotion(tmp_path: Path) -> None:
    campaign_path, artifact_root = _make_campaign(tmp_path, target_states=("source",))
    prepared = prepare_task(
        campaign_path=campaign_path,
        task_index=1,
        artifact_root=artifact_root,
        gpu_count=2,
        checkout=campaign_path.parent / "code_checkout",
    )
    staging = Path(prepared["staging_dir"])
    checkpoint = staging / "work" / "base_checkpoints" / "synthetic-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.pt").write_bytes(b"synthetic")
    _write_json(checkpoint / "meta.json", {"synthetic": True})
    (checkpoint / "optim_000001_rank0.pt").write_bytes(b"not allowed")
    _write_json(
        staging / "metrics" / "final.json",
        {"val_bpb": 1.0, "N_scaling": 1000, "D_actual": 2000},
    )
    with pytest.raises(ContractError, match="forbidden artifacts"):
        finalize_task(prepared_path=Path(prepared["prepared"]), exit_code=0)
    assert not Path(prepared["final_dir"]).exists()


def test_failed_attempt_can_never_be_promoted_to_success(tmp_path: Path) -> None:
    campaign_path, artifact_root = _make_campaign(tmp_path, target_states=("source",))
    prepared = prepare_task(
        campaign_path=campaign_path,
        task_index=1,
        artifact_root=artifact_root,
        gpu_count=2,
        checkout=campaign_path.parent / "code_checkout",
    )
    staging = Path(prepared["staging_dir"])
    checkpoint = staging / "work" / "base_checkpoints" / "synthetic-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.pt").write_bytes(b"synthetic")
    _write_json(checkpoint / "meta.json", {"synthetic": True})
    _write_json(
        staging / "metrics" / "final.json",
        {"val_bpb": 1.0, "N_scaling": 1000, "D_actual": 2000},
    )
    _write_json(staging / "FAILURE.json", {"status": "failed"})
    with pytest.raises(ContractError, match="failed attempt may not be promoted"):
        finalize_task(prepared_path=Path(prepared["prepared"]), exit_code=0)
    assert not Path(prepared["final_dir"]).exists()


@pytest.mark.parametrize(
    "metric,error",
    [
        ({}, "missing valid fields"),
        ({"val_bpb": float("nan"), "N_scaling": 1000, "D_actual": 2000}, "missing valid fields"),
        (
            {"val_bpb": 1.0, "N_scaling": 1000, "D_actual": 2000, "gpu_count": 2},
            "conflicts with plan",
        ),
    ],
)
def test_invalid_metric_never_receives_success(
    tmp_path: Path, metric: dict[str, object], error: str
) -> None:
    campaign_path, artifact_root = _make_campaign(tmp_path, target_states=("source",))
    prepared = prepare_task(
        campaign_path=campaign_path,
        task_index=1,
        artifact_root=artifact_root,
        gpu_count=2,
        checkout=campaign_path.parent / "code_checkout",
    )
    staging = Path(prepared["staging_dir"])
    checkpoint = staging / "work" / "base_checkpoints" / "synthetic-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.pt").write_bytes(b"synthetic")
    _write_json(checkpoint / "meta.json", {"synthetic": True})
    _write_json(staging / "metrics" / "final.json", metric)
    with pytest.raises(ContractError, match=error):
        finalize_task(prepared_path=Path(prepared["prepared"]), exit_code=0)
    assert not Path(prepared["final_dir"]).exists()
    assert not (staging / "SUCCESS").exists()
