#!/usr/bin/env python3
"""Submit a manifest-bound task array without embedding site identities.

All account, project, checkout, runner, storage, and GPU values are explicit
arguments (or narrowly named environment variables).  The command is safe to
inspect with the default dry-run mode; ``--submit`` is required to call Slurm.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable

from csc_fast_scaling_law.campaign_contract import (
    ContractError,
    _check_row_unlock,
    load_campaign,
    load_unlock,
)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _absolute_existing(
    path: Path, *, label: str, directory: bool = False, executable: bool = False
) -> Path:
    resolved = path.expanduser().resolve()
    valid = resolved.is_dir() if directory else resolved.is_file()
    if not valid:
        expected = "directory" if directory else "file"
        raise ContractError(f"{label} must be an existing {expected}: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise ContractError(f"{label} must be executable: {resolved}")
    return resolved


def build_submission(
    *,
    campaign_manifest: Path,
    checkout: Path,
    runner: Path,
    artifact_root: Path,
    account: str,
    project: str,
    gpu_count: int,
    gpu_type: str,
    partition: str,
    time_limit: str,
    concurrency: int | None,
    max_concurrent_gpus: int = 40,
    cpus_per_task: int,
    memory: str,
    job_script: Path,
    python_executable: Path | None = None,
    unlock_manifest: Path | None = None,
    job_name: str = "nanochat-contract",
) -> tuple[list[str], dict[str, str]]:
    campaign = load_campaign(campaign_manifest)
    unlock_payload = (
        load_unlock(unlock_manifest, campaign) if unlock_manifest is not None else None
    )
    for row in campaign.plan_rows:
        _check_row_unlock(row, campaign, unlock_payload)
    planned_counts = {int(row["gpu_count"]) for row in campaign.plan_rows}
    if planned_counts != {gpu_count}:
        raise ContractError(
            f"one array must have one GPU count; plan has {sorted(planned_counts)}, requested {gpu_count}"
        )
    if not account or not project:
        raise ContractError("account and project are required")
    if max_concurrent_gpus < 1 or cpus_per_task < 1:
        raise ContractError("max_concurrent_gpus and cpus_per_task must be positive")
    maximum_tasks = max_concurrent_gpus // gpu_count
    if maximum_tasks < 1:
        raise ContractError(
            f"gpu_count={gpu_count} exceeds max_concurrent_gpus={max_concurrent_gpus}"
        )
    effective_concurrency = maximum_tasks if concurrency is None else concurrency
    if effective_concurrency < 1:
        raise ContractError("concurrency must be positive")
    if effective_concurrency * gpu_count > max_concurrent_gpus:
        raise ContractError(
            "array concurrency would exceed the configured concurrent GPU limit: "
            f"{effective_concurrency}*{gpu_count}>{max_concurrent_gpus}"
        )
    checkout = _absolute_existing(checkout, label="checkout", directory=True)
    runner = _absolute_existing(runner, label="runner", executable=True)
    job_script = _absolute_existing(job_script, label="job script")
    campaign_manifest = _absolute_existing(campaign_manifest, label="campaign manifest")
    unlock = (
        _absolute_existing(unlock_manifest, label="unlock manifest")
        if unlock_manifest is not None
        else None
    )
    python_path = (
        _absolute_existing(python_executable, label="Python executable", executable=True)
        if python_executable is not None
        else None
    )
    artifact_root = artifact_root.expanduser().resolve()
    log_root = artifact_root / "slurm_logs"
    array = f"1-{len(campaign.plan_rows)}%{effective_concurrency}"
    command = [
        "sbatch",
        "--parsable",
        f"--account={account}",
        f"--partition={partition}",
        f"--time={time_limit}",
        "--nodes=1",
        f"--ntasks-per-node={gpu_count}",
        f"--cpus-per-task={cpus_per_task}",
        f"--gres=gpu:{gpu_type}:{gpu_count}",
        f"--mem={memory}",
        f"--array={array}",
        f"--job-name={job_name}",
        f"--chdir={checkout}",
        f"--output={log_root}/%x-%A_%a.out",
        "--export=ALL",
        str(job_script),
    ]
    environment = {
        "CAMPAIGN_MANIFEST": str(campaign_manifest),
        "ARTIFACT_ROOT": str(artifact_root),
        "NANOCHAT_CHECKOUT": str(checkout),
        "TASK_RUNNER": str(runner),
        "GPU_COUNT": str(gpu_count),
        "CSC_PROJECT": project,
    }
    runtime_bootstrap = campaign.files.get("runtime_bootstrap")
    if runtime_bootstrap is not None:
        environment["RUNTIME_BOOTSTRAP"] = str(runtime_bootstrap.path)
        environment["RUNTIME_BOOTSTRAP_SHA256"] = runtime_bootstrap.sha256
    if unlock is not None:
        environment["UNLOCK_MANIFEST"] = str(unlock)
    if python_path is not None:
        environment["NANOCHAT_PYTHON"] = str(python_path)
    return command, environment


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--account", default=os.environ.get("CSC_ACCOUNT"))
    parser.add_argument("--project", default=os.environ.get("CSC_PROJECT"))
    parser.add_argument("--gpu-count", type=_positive, required=True)
    parser.add_argument("--gpu-type", default="gh200")
    parser.add_argument("--partition", required=True)
    parser.add_argument("--time", dest="time_limit", required=True)
    parser.add_argument("--concurrency", type=_positive)
    parser.add_argument("--max-concurrent-gpus", type=_positive, default=40)
    parser.add_argument("--cpus-per-task", type=_positive, default=7)
    parser.add_argument("--memory", default="0")
    parser.add_argument("--unlock-manifest", type=Path)
    parser.add_argument("--python", dest="python_executable", type=Path)
    parser.add_argument("--job-name", default="nanochat-contract")
    parser.add_argument(
        "--job-script",
        type=Path,
        default=Path(__file__).with_name("run_contract_task.sbatch"),
    )
    parser.add_argument("--submit", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        command, environment = build_submission(
            campaign_manifest=args.campaign_manifest,
            checkout=args.checkout,
            runner=args.runner,
            artifact_root=args.artifact_root,
            account=args.account or "",
            project=args.project or "",
            gpu_count=args.gpu_count,
            gpu_type=args.gpu_type,
            partition=args.partition,
            time_limit=args.time_limit,
            concurrency=args.concurrency,
            max_concurrent_gpus=args.max_concurrent_gpus,
            cpus_per_task=args.cpus_per_task,
            memory=args.memory,
            python_executable=args.python_executable,
            unlock_manifest=args.unlock_manifest,
            job_name=args.job_name,
            job_script=args.job_script,
        )
    except ContractError as exc:
        print(f"SUBMISSION ERROR: {exc}", file=os.sys.stderr)
        return 2

    preview = {"command": shlex.join(command), "environment": environment}
    if not args.submit:
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0
    Path(environment["ARTIFACT_ROOT"], "slurm_logs").mkdir(parents=True, exist_ok=True)
    submitted = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, **environment},
    )
    print(submitted.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
