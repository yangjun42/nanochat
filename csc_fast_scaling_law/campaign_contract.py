#!/usr/bin/env python3
"""Fail-closed execution and evidence contract for scaling-law campaigns.

The contract deliberately knows nothing about a particular cluster account,
dataset, budget, or experiment.  A private campaign manifest binds those
values to immutable hashes.  Each task is built below ``tasks/.partial`` and
is promoted with a single directory rename only after every required artifact
has been verified.  Reducers accept only complete, hash-consistent task sets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "nanochat-campaign-v1"
UNLOCK_SCHEMA_VERSION = "nanochat-unlock-v1"
TASK_SCHEMA_VERSION = "nanochat-task-v1"
REQUIRED_FILE_REFS = (
    "plan",
    "plan_frozen",
    "data_manifest",
    "validation_manifest",
    "tokenizer_manifest",
    "tokenizer_consistency_receipt",
    "environment_lock",
    "schedule_manifest",
    "decision_rules",
    "data_preflight_receipt",
    "text_overlap_audit",
    "data_capacity_receipt",
)
DEFAULT_PLAN_COLUMNS = (
    "task_id",
    "init_seed",
    "data_seed",
    "schedule_id",
    "gpu_count",
    "target_state",
)
TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")


class ContractError(RuntimeError):
    """Raised whenever evidence is absent, ambiguous, or inconsistent."""


@dataclass(frozen=True)
class FileRef:
    name: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class Campaign:
    path: Path
    sha256: str
    raw: dict[str, Any]
    files: dict[str, FileRef]
    plan_rows: tuple[dict[str, str], ...]

    @property
    def campaign_id(self) -> str:
        return str(self.raw["campaign_id"])

    @property
    def plan(self) -> FileRef:
        return self.files["plan"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"expected a JSON object in {path}")
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _resolve_ref(manifest_dir: Path, raw_path: Any, *, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ContractError(f"{label}.path must be a non-empty string")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _validate_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_plan(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ContractError(f"plan has no header: {path}")
            rows = tuple(dict(row) for row in reader)
    except FileNotFoundError as exc:
        raise ContractError(f"missing plan: {path}") from exc
    if not rows:
        raise ContractError(f"plan has no task rows: {path}")
    return rows


def load_campaign(path: Path) -> Campaign:
    path = path.resolve()
    raw = _load_json(path)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(f"unsupported campaign schema: {raw.get('schema_version')!r}")
    campaign_id = raw.get("campaign_id")
    if not isinstance(campaign_id, str) or not TASK_ID_RE.fullmatch(campaign_id):
        raise ContractError("campaign_id must use only letters, digits, '.', '_' or '-'")

    code = raw.get("code")
    if not isinstance(code, dict):
        raise ContractError("code must be an object")
    commit = code.get("commit")
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        raise ContractError("code.commit must be a 40-64 character lowercase commit digest")
    if code.get("dirty") is not False:
        raise ContractError("code.dirty must be false; uncommitted code is not reproducible")

    raw_files = raw.get("files")
    if not isinstance(raw_files, dict):
        raise ContractError("files must be an object")
    missing_refs = sorted(set(REQUIRED_FILE_REFS) - set(raw_files))
    if missing_refs:
        raise ContractError(f"missing required file references: {', '.join(missing_refs)}")
    files: dict[str, FileRef] = {}
    for name, spec in raw_files.items():
        if not isinstance(spec, dict):
            raise ContractError(f"files.{name} must be an object")
        ref_path = _resolve_ref(path.parent, spec.get("path"), label=f"files.{name}")
        expected = _validate_sha(spec.get("sha256"), label=f"files.{name}.sha256")
        if not ref_path.is_file():
            raise ContractError(f"files.{name} is not a regular file: {ref_path}")
        actual = sha256_file(ref_path)
        if actual != expected:
            raise ContractError(
                f"files.{name} hash mismatch: expected {expected}, observed {actual}"
            )
        files[name] = FileRef(name=name, path=ref_path, sha256=expected)

    rows = _read_plan(files["plan"].path)
    required_columns = raw.get("required_plan_columns", list(DEFAULT_PLAN_COLUMNS))
    if not isinstance(required_columns, list) or not all(
        isinstance(item, str) and item for item in required_columns
    ):
        raise ContractError("required_plan_columns must be a list of non-empty strings")
    task_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        absent = [name for name in required_columns if not row.get(name)]
        if absent:
            raise ContractError(f"plan row {index} has empty required fields: {', '.join(absent)}")
        task_id = row.get("task_id", "")
        if not TASK_ID_RE.fullmatch(task_id):
            raise ContractError(f"invalid task_id in plan row {index}: {task_id!r}")
        if task_id in task_ids:
            raise ContractError(f"duplicate task_id in plan: {task_id}")
        task_ids.add(task_id)
        try:
            int(row["init_seed"])
            int(row["data_seed"])
            gpu_count = int(row["gpu_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"row {index} seeds and gpu_count must be integers") from exc
        if gpu_count < 1:
            raise ContractError(f"row {index} gpu_count must be positive")
        if row.get("target_state") not in {"source", "sealed"}:
            raise ContractError(
                f"row {index} target_state must be source or sealed; provisional/unsealed "
                "targets are not executable plans"
            )

    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ContractError("artifacts must be an object")
    required_artifacts = artifacts.get("required")
    if not isinstance(required_artifacts, list) or not required_artifacts:
        raise ContractError("artifacts.required must be a non-empty list")
    seen_names: set[str] = set()
    for index, requirement in enumerate(required_artifacts):
        if not isinstance(requirement, dict):
            raise ContractError(f"artifacts.required[{index}] must be an object")
        name = requirement.get("name")
        pattern = requirement.get("glob")
        if not isinstance(name, str) or not TASK_ID_RE.fullmatch(name):
            raise ContractError(f"invalid artifact requirement name at index {index}")
        if name in seen_names:
            raise ContractError(f"duplicate artifact requirement name: {name}")
        seen_names.add(name)
        _validate_relative_pattern(pattern, label=f"artifacts.required[{index}].glob")
        for key, default in (("min_count", 1), ("max_count", None)):
            value = requirement.get(key, default)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ContractError(f"{key} for artifact {name} must be a non-negative integer")
        min_count = requirement.get("min_count", 1)
        max_count = requirement.get("max_count")
        if max_count is not None and max_count < min_count:
            raise ContractError(f"artifact {name} max_count is smaller than min_count")
    for index, pattern in enumerate(artifacts.get("forbid_globs", [])):
        _validate_relative_pattern(pattern, label=f"artifacts.forbid_globs[{index}]")

    reducer = raw.get("reducer")
    if not isinstance(reducer, dict):
        raise ContractError("reducer must be an object")
    _validate_relative_pattern(reducer.get("metric_json"), label="reducer.metric_json")
    required_metric_fields = reducer.get("required_fields")
    if not isinstance(required_metric_fields, list) or not required_metric_fields or not all(
        isinstance(item, str) and item for item in required_metric_fields
    ):
        raise ContractError("reducer.required_fields must be a non-empty string list")

    return Campaign(
        path=path,
        sha256=sha256_file(path),
        raw=raw,
        files=files,
        plan_rows=rows,
    )


def _validate_relative_pattern(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty string")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ContractError(f"{label} must stay below the task directory")
    return value


def load_unlock(path: Path, campaign: Campaign) -> dict[str, Any]:
    path = path.resolve()
    raw = _load_json(path)
    if raw.get("schema_version") != UNLOCK_SCHEMA_VERSION:
        raise ContractError(f"unsupported unlock schema: {raw.get('schema_version')!r}")
    expected = {
        "campaign_id": campaign.campaign_id,
        "campaign_manifest_sha256": campaign.sha256,
        "plan_sha256": campaign.plan.sha256,
        "decision_rules_sha256": campaign.files["decision_rules"].sha256,
        "status": "unlocked",
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ContractError(
                f"unlock {key} mismatch: expected {value!r}, observed {raw.get(key)!r}"
            )
    task_ids = raw.get("task_ids")
    if not isinstance(task_ids, list) or not task_ids or not all(
        isinstance(item, str) and TASK_ID_RE.fullmatch(item) for item in task_ids
    ):
        raise ContractError("unlock.task_ids must be a non-empty list of valid task IDs")
    if len(task_ids) != len(set(task_ids)):
        raise ContractError("unlock.task_ids contains duplicates")
    known = {row["task_id"] for row in campaign.plan_rows}
    unknown = sorted(set(task_ids) - known)
    if unknown:
        raise ContractError(f"unlock references unknown tasks: {', '.join(unknown)}")
    return raw


def _check_row_unlock(
    row: Mapping[str, str], campaign: Campaign, unlock: Mapping[str, Any] | None
) -> None:
    if row["target_state"] != "sealed":
        return
    if unlock is None:
        raise ContractError(f"sealed task {row['task_id']} requires an unlock manifest")
    if row["task_id"] not in unlock["task_ids"]:
        raise ContractError(f"unlock manifest does not authorize sealed task {row['task_id']}")


def _artifact_root(campaign: Campaign, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    raw_root = campaign.raw["artifacts"].get("root")
    if not isinstance(raw_root, str) or not raw_root:
        raise ContractError("artifacts.root is absent; pass --artifact-root")
    return _resolve_ref(campaign.path.parent, raw_root, label="artifacts")


def _safe_attempt_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return cleaned[:80] or "manual"


def _format_pattern(pattern: str, row: Mapping[str, str]) -> str:
    try:
        value = pattern.format_map(dict(row))
    except KeyError as exc:
        raise ContractError(f"artifact pattern references missing plan field: {exc.args[0]}") from exc
    return _validate_relative_pattern(value, label="formatted artifact glob")


def _runtime_provenance(gpu_count: int, runner: Path | None) -> dict[str, Any]:
    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NTASKS",
        "SLURM_GPUS_ON_NODE",
    )
    runtime: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "gpu_count": gpu_count,
        "python": sys.version,
        "slurm": {key: os.environ.get(key) for key in slurm_keys},
    }
    if runner is not None:
        runner = runner.expanduser().resolve()
        if not runner.is_file():
            raise ContractError(f"task runner is not a regular file: {runner}")
        runtime["runner"] = {"path": str(runner), "sha256": sha256_file(runner)}
    return runtime


def verify_code_checkout(checkout: Path, campaign: Campaign) -> dict[str, Any]:
    """Verify the compute-node checkout instead of trusting manifest claims."""

    checkout = checkout.expanduser().resolve()
    if not checkout.is_dir():
        raise ContractError(f"code checkout is not a directory: {checkout}")

    def git(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(checkout), *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            stderr = getattr(exc, "stderr", "")
            raise ContractError(
                f"cannot verify git checkout {checkout}: {str(stderr).strip() or exc}"
            ) from exc
        return result.stdout

    head = git("rev-parse", "HEAD").strip()
    expected = campaign.raw["code"]["commit"]
    if head != expected:
        raise ContractError(f"checkout HEAD mismatch: expected {expected}, observed {head}")
    tree = git("rev-parse", "HEAD^{tree}").strip()
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        preview = ", ".join(line.strip() for line in status.splitlines()[:5])
        raise ContractError(f"checkout is dirty; refusing contracted execution: {preview}")
    submodules = git("submodule", "status", "--recursive")
    bad_submodules = [
        line for line in submodules.splitlines() if line and line[0] in {"-", "+", "U"}
    ]
    if bad_submodules:
        raise ContractError(
            "checkout has missing, changed, or conflicted submodules: "
            + ", ".join(bad_submodules)
        )
    return {
        "checkout": str(checkout),
        "head": head,
        "tree": tree,
        "status_sha256": sha256_text(status),
        "submodule_status_sha256": sha256_text(submodules),
    }


def prepare_task(
    *,
    campaign_path: Path,
    task_index: int,
    artifact_root: Path | None = None,
    unlock_path: Path | None = None,
    gpu_count: int | None = None,
    runner: Path | None = None,
    checkout: Path | None = None,
) -> dict[str, Any]:
    campaign = load_campaign(campaign_path)
    if checkout is None:
        raw_checkout = os.environ.get("NANOCHAT_CHECKOUT")
        if not raw_checkout:
            raise ContractError("a compute-node --checkout is required for code verification")
        checkout = Path(raw_checkout)
    observed_code = verify_code_checkout(checkout, campaign)
    if task_index < 1 or task_index > len(campaign.plan_rows):
        raise ContractError(f"task index {task_index} is outside 1..{len(campaign.plan_rows)}")
    row = campaign.plan_rows[task_index - 1]
    unlock = load_unlock(unlock_path, campaign) if unlock_path else None
    _check_row_unlock(row, campaign, unlock)
    unlock_sha256 = sha256_file(unlock_path.resolve()) if unlock_path else None
    planned_gpu_count = int(row["gpu_count"])
    if gpu_count is not None and gpu_count != planned_gpu_count:
        raise ContractError(
            f"task {row['task_id']} planned gpu_count={planned_gpu_count}, got {gpu_count}"
        )
    effective_gpu_count = planned_gpu_count if gpu_count is None else gpu_count
    root = _artifact_root(campaign, artifact_root)
    tasks_root = root / "tasks"
    final_dir = tasks_root / row["task_id"]
    if final_dir.exists():
        validate_complete_task(
            final_dir,
            campaign=campaign,
            row=row,
            unlock_manifest_sha256=unlock_sha256,
        )
        return {
            "action": "skip",
            "task_id": row["task_id"],
            "final_dir": str(final_dir),
            "reason": "validated SUCCESS already exists",
        }

    job_component = _safe_attempt_component(
        os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "manual"
    )
    attempt_id = f"{job_component}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    staging_dir = tasks_root / ".partial" / row["task_id"] / attempt_id
    staging_dir.mkdir(parents=True, exist_ok=False)
    prepared = {
        "schema_version": TASK_SCHEMA_VERSION,
        "status": "prepared",
        "campaign_id": campaign.campaign_id,
        "campaign_manifest": str(campaign.path),
        "campaign_manifest_sha256": campaign.sha256,
        "plan_sha256": campaign.plan.sha256,
        "plan_row_index": task_index,
        "plan_row": row,
        "plan_row_sha256": canonical_hash(row),
        "task_id": row["task_id"],
        "staging_dir": str(staging_dir),
        "final_dir": str(final_dir),
        "code": campaign.raw["code"],
        "file_provenance": {
            name: {"path": str(ref.path), "sha256": ref.sha256}
            for name, ref in sorted(campaign.files.items())
        },
        "seeds": {"init": int(row["init_seed"]), "data": int(row["data_seed"])},
        "schedule": {"id": row["schedule_id"], "gpu_count": effective_gpu_count},
        "unlock_manifest": str(unlock_path.resolve()) if unlock_path else None,
        "unlock_manifest_sha256": unlock_sha256,
        "runtime": _runtime_provenance(effective_gpu_count, runner),
        "observed_code": observed_code,
        "prepared_unix_time": int(time.time()),
    }
    prepared_path = staging_dir / "prepared.json"
    atomic_write_json(prepared_path, prepared)
    return {
        "action": "run",
        "task_id": row["task_id"],
        "prepared": str(prepared_path),
        "staging_dir": str(staging_dir),
        "final_dir": str(final_dir),
    }


def _walk_inventory(root: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    excluded = {"artifact_manifest.json", "SUCCESS", "FAILURE.json"}
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        if path.is_symlink():
            target = os.readlink(path)
            inventory.append(
                {
                    "path": relative,
                    "type": "symlink",
                    "target": target,
                    "sha256": sha256_text(target),
                }
            )
        elif path.is_dir():
            continue
        elif path.is_file():
            inventory.append(
                {
                    "path": relative,
                    "type": "file",
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        else:
            raise ContractError(f"unsupported artifact type: {path}")
    return inventory


def _verify_required_artifacts(
    staging_dir: Path, campaign: Campaign, row: Mapping[str, str]
) -> dict[str, list[str]]:
    required_matches: dict[str, list[str]] = {}
    for requirement in campaign.raw["artifacts"]["required"]:
        pattern = _format_pattern(requirement["glob"], row)
        matches = [
            path
            for path in sorted(staging_dir.glob(pattern), key=lambda value: value.as_posix())
            if path.is_file() and not path.is_symlink()
        ]
        minimum = requirement.get("min_count", 1)
        maximum = requirement.get("max_count")
        if len(matches) < minimum:
            raise ContractError(
                f"artifact {requirement['name']} matched {len(matches)} files; expected at least {minimum}"
            )
        if maximum is not None and len(matches) > maximum:
            raise ContractError(
                f"artifact {requirement['name']} matched {len(matches)} files; expected at most {maximum}"
            )
        required_matches[requirement["name"]] = [
            path.relative_to(staging_dir).as_posix() for path in matches
        ]
    for raw_pattern in campaign.raw["artifacts"].get("forbid_globs", []):
        pattern = _format_pattern(raw_pattern, row)
        forbidden = sorted(staging_dir.glob(pattern), key=lambda value: value.as_posix())
        if forbidden:
            listed = ", ".join(path.relative_to(staging_dir).as_posix() for path in forbidden[:5])
            raise ContractError(f"forbidden artifacts matched {pattern!r}: {listed}")
    return required_matches


def _load_prepared(path: Path) -> dict[str, Any]:
    prepared = _load_json(path.resolve())
    if prepared.get("schema_version") != TASK_SCHEMA_VERSION or prepared.get("status") != "prepared":
        raise ContractError(f"invalid prepared task record: {path}")
    return prepared


def record_failure(prepared_path: Path, exit_code: int, reason: str | None = None) -> Path:
    prepared = _load_prepared(prepared_path)
    staging_dir = Path(prepared["staging_dir"]).resolve()
    failure = {
        "schema_version": TASK_SCHEMA_VERSION,
        "status": "failed",
        "task_id": prepared["task_id"],
        "exit_code": exit_code,
        "reason": reason,
        "failed_unix_time": int(time.time()),
    }
    path = staging_dir / "FAILURE.json"
    atomic_write_json(path, failure)
    return path


def finalize_task(*, prepared_path: Path, exit_code: int) -> Path:
    prepared_path = prepared_path.resolve()
    prepared = _load_prepared(prepared_path)
    staging_dir = Path(prepared["staging_dir"]).resolve()
    final_dir = Path(prepared["final_dir"]).resolve()
    if prepared_path.parent != staging_dir:
        raise ContractError("prepared.json is not directly inside its declared staging directory")
    if (
        staging_dir.parent.name != prepared.get("task_id")
        or staging_dir.parent.parent.name != ".partial"
        or staging_dir.parent.parent.parent.name != "tasks"
    ):
        raise ContractError("staging directory does not follow tasks/.partial/<task>/<attempt>")
    if not staging_dir.is_dir():
        raise ContractError(f"missing staging directory: {staging_dir}")
    if (staging_dir / "FAILURE.json").exists():
        raise ContractError("a failed attempt may not be promoted; start a new partial attempt")
    if exit_code != 0:
        record_failure(prepared_path, exit_code, "task runner returned non-zero")
        raise ContractError(f"task runner failed with exit code {exit_code}")

    campaign = load_campaign(Path(prepared["campaign_manifest"]))
    if campaign.sha256 != prepared.get("campaign_manifest_sha256"):
        raise ContractError("campaign manifest changed after task preparation")
    index = int(prepared["plan_row_index"])
    if index < 1 or index > len(campaign.plan_rows):
        raise ContractError("prepared plan row index is invalid")
    row = campaign.plan_rows[index - 1]
    derived_root = staging_dir.parent.parent.parent.parent
    expected_file_provenance = {
        name: {"path": str(ref.path), "sha256": ref.sha256}
        for name, ref in sorted(campaign.files.items())
    }
    expected_prepared_values = {
        "campaign_id": campaign.campaign_id,
        "plan_sha256": campaign.plan.sha256,
        "plan_row": row,
        "plan_row_sha256": canonical_hash(row),
        "task_id": row["task_id"],
        "code": campaign.raw["code"],
        "final_dir": str(derived_root / "tasks" / row["task_id"]),
        "file_provenance": expected_file_provenance,
        "seeds": {"init": int(row["init_seed"]), "data": int(row["data_seed"])},
        "schedule": {"id": row["schedule_id"], "gpu_count": int(row["gpu_count"])},
    }
    for key, value in expected_prepared_values.items():
        if prepared.get(key) != value:
            raise ContractError(f"prepared {key} no longer matches the campaign")
    raw_observed_code = prepared.get("observed_code")
    if not isinstance(raw_observed_code, dict) or not isinstance(
        raw_observed_code.get("checkout"), str
    ):
        raise ContractError("prepared task lacks observed checkout provenance")
    observed_code = verify_code_checkout(Path(raw_observed_code["checkout"]), campaign)
    if observed_code != raw_observed_code:
        raise ContractError("code checkout changed after task preparation")
    runtime_runner = prepared.get("runtime", {}).get("runner")
    if runtime_runner is not None:
        if not isinstance(runtime_runner, dict) or not isinstance(runtime_runner.get("path"), str):
            raise ContractError("prepared task runner provenance is invalid")
        current_runner = Path(runtime_runner["path"])
        if not current_runner.is_file() or sha256_file(current_runner) != runtime_runner.get("sha256"):
            raise ContractError("task runner changed after task preparation")
    if row["target_state"] == "sealed":
        prepared_unlock_path = prepared.get("unlock_manifest")
        if not isinstance(prepared_unlock_path, str) or not prepared_unlock_path:
            raise ContractError(f"sealed task {row['task_id']} lost its unlock manifest path")
        prepared_unlock = Path(prepared_unlock_path)
        unlock = load_unlock(prepared_unlock, campaign)
        _check_row_unlock(row, campaign, unlock)
        if sha256_file(prepared_unlock) != prepared.get("unlock_manifest_sha256"):
            raise ContractError("unlock manifest changed after task preparation")

    _load_validated_metric(staging_dir, campaign=campaign, row=row)
    required_matches = _verify_required_artifacts(staging_dir, campaign, row)
    inventory = _walk_inventory(staging_dir)
    manifest = {
        "schema_version": TASK_SCHEMA_VERSION,
        "status": "success",
        "campaign_id": campaign.campaign_id,
        "campaign_manifest_sha256": campaign.sha256,
        "plan_sha256": campaign.plan.sha256,
        "plan_row_index": index,
        "plan_row_sha256": canonical_hash(row),
        "task_id": row["task_id"],
        "code": campaign.raw["code"],
        "seeds": prepared["seeds"],
        "schedule": prepared["schedule"],
        "file_provenance": prepared["file_provenance"],
        "unlock_manifest_sha256": prepared["unlock_manifest_sha256"],
        "runtime": prepared["runtime"],
        "observed_code": prepared["observed_code"],
        "required_artifacts": required_matches,
        "inventory": inventory,
        "completed_unix_time": int(time.time()),
    }
    artifact_manifest_path = staging_dir / "artifact_manifest.json"
    atomic_write_json(artifact_manifest_path, manifest)
    success = {
        "schema_version": TASK_SCHEMA_VERSION,
        "status": "success",
        "task_id": row["task_id"],
        "campaign_manifest_sha256": campaign.sha256,
        "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
    }
    atomic_write_json(staging_dir / "SUCCESS", success)
    if final_dir.exists():
        raise ContractError(
            f"final task directory appeared during execution; preserving staging attempt: {final_dir}"
        )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging_dir, final_dir)
    return final_dir


def validate_complete_task(
    task_dir: Path,
    *,
    campaign: Campaign,
    row: Mapping[str, str],
    unlock_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if task_dir.is_symlink():
        raise ContractError(f"completed task directory may not be a symlink: {task_dir}")
    task_dir = task_dir.resolve()
    if not task_dir.is_dir():
        raise ContractError(f"missing completed task directory: {task_dir}")
    success_path = task_dir / "SUCCESS"
    manifest_path = task_dir / "artifact_manifest.json"
    if (task_dir / "FAILURE.json").exists():
        raise ContractError(f"task {row['task_id']} contains both failure and success state")
    if not success_path.is_file() or not manifest_path.is_file():
        raise ContractError(
            f"task {row['task_id']} is partial: SUCCESS and artifact_manifest.json are both required"
        )
    success = _load_json(success_path)
    manifest = _load_json(manifest_path)
    expected = {
        "schema_version": TASK_SCHEMA_VERSION,
        "status": "success",
        "task_id": row["task_id"],
        "campaign_manifest_sha256": campaign.sha256,
    }
    for key, value in expected.items():
        if success.get(key) != value:
            raise ContractError(f"task {row['task_id']} SUCCESS {key} mismatch")
        if manifest.get(key) != value:
            raise ContractError(f"task {row['task_id']} artifact manifest {key} mismatch")
    if success.get("artifact_manifest_sha256") != sha256_file(manifest_path):
        raise ContractError(f"task {row['task_id']} artifact manifest hash mismatch")
    manifest_expected = {
        "plan_sha256": campaign.plan.sha256,
        "plan_row_sha256": canonical_hash(dict(row)),
        "code": campaign.raw["code"],
        "seeds": {"init": int(row["init_seed"]), "data": int(row["data_seed"])},
        "schedule": {"id": row["schedule_id"], "gpu_count": int(row["gpu_count"])},
    }
    if row["target_state"] == "sealed":
        if unlock_manifest_sha256 is None:
            raise ContractError(f"sealed task {row['task_id']} has no validated unlock hash")
        manifest_expected["unlock_manifest_sha256"] = unlock_manifest_sha256
    for key, value in manifest_expected.items():
        if manifest.get(key) != value:
            raise ContractError(f"task {row['task_id']} {key} mismatch")
    observed_code = manifest.get("observed_code")
    if not isinstance(observed_code, dict):
        raise ContractError(f"task {row['task_id']} lacks observed code provenance")
    if observed_code.get("head") != campaign.raw["code"]["commit"]:
        raise ContractError(f"task {row['task_id']} observed checkout HEAD mismatch")
    if observed_code.get("status_sha256") != sha256_text(""):
        raise ContractError(f"task {row['task_id']} was not prepared from a clean checkout")
    current_inventory = _walk_inventory(task_dir)
    if current_inventory != manifest.get("inventory"):
        raise ContractError(f"task {row['task_id']} file inventory or hash mismatch")
    required = _verify_required_artifacts(task_dir, campaign, row)
    if required != manifest.get("required_artifacts"):
        raise ContractError(f"task {row['task_id']} required artifact set changed")
    return manifest


def _metric_value_is_valid(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _load_validated_metric(
    task_dir: Path, *, campaign: Campaign, row: Mapping[str, str]
) -> dict[str, Any]:
    metric_relative = _format_pattern(campaign.raw["reducer"]["metric_json"], row)
    metric = _load_json(task_dir / metric_relative)
    required_fields = campaign.raw["reducer"]["required_fields"]
    absent = [
        field
        for field in required_fields
        if field not in metric or not _metric_value_is_valid(metric[field])
    ]
    if absent:
        raise ContractError(
            f"task {row['task_id']} metric is missing valid fields: {', '.join(absent)}"
        )
    for key, value in metric.items():
        if key in row and row[key] != value:
            raise ContractError(f"metric field {key} conflicts with plan for task {row['task_id']}")
    return metric


def reduce_campaign(
    *,
    campaign_path: Path,
    out_path: Path,
    artifact_root: Path | None = None,
    unlock_path: Path | None = None,
) -> Path:
    campaign = load_campaign(campaign_path)
    unlock = load_unlock(unlock_path, campaign) if unlock_path else None
    unlock_sha256 = sha256_file(unlock_path.resolve()) if unlock_path else None
    root = _artifact_root(campaign, artifact_root)
    tasks_root = root / "tasks"
    expected_ids = {row["task_id"] for row in campaign.plan_rows}
    observed_ids = {
        path.name
        for path in tasks_root.iterdir()
        if path.name != ".partial" and path.is_dir()
    } if tasks_root.is_dir() else set()
    missing = sorted(expected_ids - observed_ids)
    extra = sorted(observed_ids - expected_ids)
    if missing or extra:
        raise ContractError(f"task set mismatch; missing={missing}, extra={extra}")

    rows: list[dict[str, Any]] = []
    task_manifest_hashes: dict[str, str] = {}
    for row in campaign.plan_rows:
        _check_row_unlock(row, campaign, unlock)
        task_dir = tasks_root / row["task_id"]
        validate_complete_task(
            task_dir,
            campaign=campaign,
            row=row,
            unlock_manifest_sha256=unlock_sha256,
        )
        metric = _load_validated_metric(task_dir, campaign=campaign, row=row)
        merged: dict[str, Any] = dict(row)
        for key, value in metric.items():
            merged[key] = value
        rows.append(merged)
        task_manifest_hashes[row["task_id"]] = sha256_file(task_dir / "artifact_manifest.json")

    reduced = {
        "schema_version": TASK_SCHEMA_VERSION,
        "status": "complete",
        "campaign_id": campaign.campaign_id,
        "campaign_manifest_sha256": campaign.sha256,
        "plan_sha256": campaign.plan.sha256,
        "unlock_manifest_sha256": unlock_sha256,
        "task_count": len(rows),
        "task_manifest_sha256": task_manifest_hashes,
        "rows": rows,
    }
    out_path = out_path.expanduser().resolve()
    atomic_write_json(out_path, reduced)
    return out_path


def _shell_assignments(result: Mapping[str, Any]) -> str:
    names = {
        "action": "CONTRACT_ACTION",
        "task_id": "CONTRACT_TASK_ID",
        "prepared": "CONTRACT_PREPARED",
        "staging_dir": "CONTRACT_STAGING_DIR",
        "final_dir": "CONTRACT_FINAL_DIR",
        "reason": "CONTRACT_REASON",
    }
    return "\n".join(
        f"{shell_name}={shlex.quote(str(result.get(key, '')))}"
        for key, shell_name in names.items()
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate all immutable campaign inputs")
    validate.add_argument("--campaign-manifest", type=Path, required=True)
    validate.add_argument("--unlock-manifest", type=Path)

    prepare = subparsers.add_parser("prepare-task", help="create an isolated partial task directory")
    prepare.add_argument("--campaign-manifest", type=Path, required=True)
    prepare.add_argument("--task-index", type=int, required=True)
    prepare.add_argument("--artifact-root", type=Path)
    prepare.add_argument("--unlock-manifest", type=Path)
    prepare.add_argument("--gpu-count", type=int)
    prepare.add_argument("--runner", type=Path)
    prepare.add_argument("--checkout", type=Path, required=True)
    prepare.add_argument("--format", choices=("json", "shell"), default="json")

    finalize = subparsers.add_parser("finalize-task", help="verify and atomically promote a task")
    finalize.add_argument("--prepared", type=Path, required=True)
    finalize.add_argument("--exit-code", type=int, required=True)

    fail = subparsers.add_parser("record-failure", help="record an interrupted task without promotion")
    fail.add_argument("--prepared", type=Path, required=True)
    fail.add_argument("--exit-code", type=int, required=True)
    fail.add_argument("--reason")

    reduce_parser = subparsers.add_parser("reduce", help="strictly reduce a complete campaign")
    reduce_parser.add_argument("--campaign-manifest", type=Path, required=True)
    reduce_parser.add_argument("--artifact-root", type=Path)
    reduce_parser.add_argument("--unlock-manifest", type=Path)
    reduce_parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "validate":
            campaign = load_campaign(args.campaign_manifest)
            unlock = load_unlock(args.unlock_manifest, campaign) if args.unlock_manifest else None
            for row in campaign.plan_rows:
                _check_row_unlock(row, campaign, unlock)
            print(
                json.dumps(
                    {
                        "campaign_id": campaign.campaign_id,
                        "campaign_manifest_sha256": campaign.sha256,
                        "plan_sha256": campaign.plan.sha256,
                        "task_count": len(campaign.plan_rows),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "prepare-task":
            result = prepare_task(
                campaign_path=args.campaign_manifest,
                task_index=args.task_index,
                artifact_root=args.artifact_root,
                unlock_path=args.unlock_manifest,
                gpu_count=args.gpu_count,
                runner=args.runner,
                checkout=args.checkout,
            )
            print(_shell_assignments(result) if args.format == "shell" else json.dumps(result, sort_keys=True))
        elif args.command == "finalize-task":
            print(finalize_task(prepared_path=args.prepared, exit_code=args.exit_code))
        elif args.command == "record-failure":
            print(record_failure(args.prepared, args.exit_code, args.reason))
        elif args.command == "reduce":
            print(
                reduce_campaign(
                    campaign_path=args.campaign_manifest,
                    artifact_root=args.artifact_root,
                    unlock_path=args.unlock_manifest,
                    out_path=args.out,
                )
            )
        return 0
    except ContractError as exc:
        print(f"CONTRACT ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
