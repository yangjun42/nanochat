from __future__ import annotations

import json
from pathlib import Path

import pytest

from csc_fast_scaling_law.campaign_contract import ContractError
from csc_fast_scaling_law.submit_contract_array import build_submission
from tests.test_csc_fast_scaling_law_campaign_contract import _make_campaign


def test_submission_parameters_are_explicit_and_identity_free(tmp_path: Path) -> None:
    campaign_path, artifact_root = _make_campaign(tmp_path, target_states=("source",))
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    runner = checkout / "runner.sh"
    runner.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    runner.chmod(0o755)
    job_script = checkout / "job.sbatch"
    job_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    command, environment = build_submission(
        campaign_manifest=campaign_path,
        checkout=checkout,
        runner=runner,
        artifact_root=artifact_root,
        account="synthetic-account",
        project="synthetic-project",
        gpu_count=2,
        gpu_type="synthetic-gpu",
        partition="synthetic-partition",
        time_limit="00:10:00",
        concurrency=3,
        cpus_per_task=4,
        memory="32G",
        job_script=job_script,
    )

    assert "--account=synthetic-account" in command
    assert "--gres=gpu:synthetic-gpu:2" in command
    assert "--array=1-1%3" in command
    assert environment["CSC_PROJECT"] == "synthetic-project"
    assert environment["GPU_COUNT"] == "2"
    assert json.dumps({"command": command, "environment": environment})


def test_submission_caps_total_concurrent_gpus(tmp_path: Path) -> None:
    campaign_path, artifact_root = _make_campaign(tmp_path, target_states=("source",))
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    runner = checkout / "runner.sh"
    runner.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    runner.chmod(0o755)
    job_script = checkout / "job.sbatch"
    job_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    common = {
        "campaign_manifest": campaign_path,
        "checkout": checkout,
        "runner": runner,
        "artifact_root": artifact_root,
        "account": "synthetic-account",
        "project": "synthetic-project",
        "gpu_count": 2,
        "gpu_type": "synthetic-gpu",
        "partition": "synthetic-partition",
        "time_limit": "00:10:00",
        "cpus_per_task": 4,
        "memory": "32G",
        "job_script": job_script,
    }
    command, _ = build_submission(
        **common, concurrency=None, max_concurrent_gpus=40
    )
    assert "--array=1-1%20" in command
    with pytest.raises(ContractError, match="concurrent GPU limit"):
        build_submission(**common, concurrency=21, max_concurrent_gpus=40)


def test_submission_rejects_sealed_task_without_unlock(tmp_path: Path) -> None:
    campaign_path, artifact_root = _make_campaign(tmp_path, target_states=("sealed",))
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    runner = checkout / "runner.sh"
    runner.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    runner.chmod(0o755)
    job_script = checkout / "job.sbatch"
    job_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    with pytest.raises(ContractError, match="requires an unlock"):
        build_submission(
            campaign_manifest=campaign_path,
            checkout=checkout,
            runner=runner,
            artifact_root=artifact_root,
            account="synthetic-account",
            project="synthetic-project",
            gpu_count=2,
            gpu_type="synthetic-gpu",
            partition="synthetic-partition",
            time_limit="00:10:00",
            concurrency=1,
            cpus_per_task=4,
            memory="32G",
            job_script=job_script,
        )
