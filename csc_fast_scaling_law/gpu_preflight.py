"""World-size numerical-equivalence and throughput preflight for nanochat.

This diagnostic is deliberately independent of the dataset loader.  A single
pre-materialized tensor contains every global token batch.  The 1/2/4-GPU
workers partition that tensor without changing its contents, model seed,
optimizer, or schedule.  Numerical equivalence is a gate; throughput is an
informational measurement and can never make the numerical gate pass or fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


MANIFEST_VERSION = "nanochat_gpu_preflight/v1"
WORKER_RESULT_VERSION = "nanochat_gpu_preflight_worker/v1"
SUMMARY_VERSION = "nanochat_gpu_preflight_summary/v1"
ENVIRONMENT_RECEIPT_VERSION = "nanochat_gpu_preflight_environment/v1"
TOKEN_FORMAT = "nanochat_global_token_batches/v1"
FORMAL_RUNTIME_PROFILE_VERSION = "nanochat_formal_runtime_profile/v1"
PREFLIGHT_WORLD_SIZES = (1, 2, 4)
FORMAL_CAMPAIGN_GPU_COUNT = 4


class PreflightContractError(ValueError):
    """Raised when a preflight artifact violates its frozen contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(_canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    torch.save(value, partial)
    os.replace(partial, path)


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PreflightContractError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise PreflightContractError(f"{label} must be >= {minimum}")
    return value


def _require_number(
    value: Any, label: str, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreflightContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PreflightContractError(f"{label} must be a finite number")
    if minimum is not None and result < minimum:
        raise PreflightContractError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise PreflightContractError(f"{label} must be <= {maximum}")
    return result


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreflightContractError(f"{label} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise PreflightContractError(
            f"{label} keys differ from the contract: missing={missing} extra={extra}"
        )


def _require_relative_file(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PreflightContractError(f"{label} must be a non-empty relative path")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise PreflightContractError(f"{label} must be relative")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise PreflightContractError(f"{label} escapes its artifact directory") from exc
    return resolved


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PreflightContractError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _validate_commit(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PreflightContractError("code_commit must be a full lowercase Git commit")
    return value


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the complete immutable preflight manifest."""

    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "preflight_only",
            "formal_campaign_gpu_count",
            "world_sizes",
            "code_commit",
            "created_utc",
            "init_seed",
            "token_batches",
            "environment_lock",
            "runtime_bootstrap",
            "formal_runtime_profile",
            "model",
            "optimizer",
            "schedule",
            "execution",
            "runtime",
            "tolerances",
        },
        "manifest",
    )

    if manifest.get("schema_version") != MANIFEST_VERSION:
        raise PreflightContractError(f"schema_version must be {MANIFEST_VERSION}")
    if manifest.get("preflight_only") is not True:
        raise PreflightContractError("preflight_only must be true")
    if manifest.get("formal_campaign_gpu_count") != FORMAL_CAMPAIGN_GPU_COUNT:
        raise PreflightContractError(
            "formal_campaign_gpu_count is frozen at 4; this preflight cannot change it"
        )
    if manifest.get("world_sizes") != list(PREFLIGHT_WORLD_SIZES):
        raise PreflightContractError("world_sizes must be exactly [1, 2, 4]")
    _validate_commit(manifest.get("code_commit"))
    if not isinstance(manifest.get("created_utc"), str) or not manifest["created_utc"]:
        raise PreflightContractError("created_utc must be a non-empty string")

    fixed_artifact_names = {
        "environment_lock": "environment_lock.json",
        "runtime_bootstrap": "runtime_bootstrap.sh",
        "formal_runtime_profile": "formal_runtime_profile.json",
    }
    for artifact_name, fixed_filename in fixed_artifact_names.items():
        artifact = _require_mapping(manifest.get(artifact_name), artifact_name)
        _require_exact_keys(artifact, {"file", "sha256"}, artifact_name)
        if artifact.get("file") != fixed_filename:
            raise PreflightContractError(
                f"{artifact_name}.file must be the fixed copied artifact {fixed_filename}"
            )
        _validate_sha256(artifact.get("sha256"), f"{artifact_name}.sha256")

    token = _require_mapping(manifest.get("token_batches"), "token_batches")
    _require_exact_keys(
        token,
        {
            "format",
            "file",
            "sha256",
            "source",
            "num_updates",
            "global_batch_sequences",
            "sequence_length",
            "vocab_size",
        },
        "token_batches",
    )
    if token.get("format") != TOKEN_FORMAT:
        raise PreflightContractError(f"token_batches.format must be {TOKEN_FORMAT}")
    _validate_sha256(token.get("sha256"), "token_batches.sha256")
    if not isinstance(token.get("file"), str) or not token["file"]:
        raise PreflightContractError("token_batches.file must be a relative path")
    if token.get("source") not in ("synthetic_uniform", "provided_tensor"):
        raise PreflightContractError("token_batches.source is invalid")
    updates = _require_int(token.get("num_updates"), "token_batches.num_updates", minimum=1)
    global_sequences = _require_int(
        token.get("global_batch_sequences"),
        "token_batches.global_batch_sequences",
        minimum=1,
    )
    sequence_length = _require_int(
        token.get("sequence_length"), "token_batches.sequence_length", minimum=2
    )
    vocab_size = _require_int(token.get("vocab_size"), "token_batches.vocab_size", minimum=2)

    model = _require_mapping(manifest.get("model"), "model")
    _require_exact_keys(
        model,
        {
            "sequence_len",
            "vocab_size",
            "n_layer",
            "n_head",
            "n_kv_head",
            "n_embd",
            "window_pattern",
        },
        "model",
    )
    if _require_int(model.get("sequence_len"), "model.sequence_len", minimum=2) != sequence_length:
        raise PreflightContractError("model.sequence_len must match token batch sequence_length")
    if _require_int(model.get("vocab_size"), "model.vocab_size", minimum=2) != vocab_size:
        raise PreflightContractError("model.vocab_size must match token batch vocab_size")
    _require_int(model.get("n_layer"), "model.n_layer", minimum=1)
    n_head = _require_int(model.get("n_head"), "model.n_head", minimum=1)
    n_kv_head = _require_int(model.get("n_kv_head"), "model.n_kv_head", minimum=1)
    n_embd = _require_int(model.get("n_embd"), "model.n_embd", minimum=24)
    if (
        n_embd % n_head != 0
        or (n_embd // n_head) % 2 != 0
        or n_kv_head > n_head
        or n_head % n_kv_head != 0
    ):
        raise PreflightContractError("model head geometry is invalid")
    pattern = model.get("window_pattern")
    if not isinstance(pattern, str) or not pattern or any(c not in "SL" for c in pattern.upper()):
        raise PreflightContractError("model.window_pattern must contain only S/L")

    optimizer = _require_mapping(manifest.get("optimizer"), "optimizer")
    _require_exact_keys(
        optimizer,
        {"unembedding_lr", "embedding_lr", "matrix_lr", "scalar_lr", "weight_decay"},
        "optimizer",
    )
    for key in (
        "unembedding_lr",
        "embedding_lr",
        "matrix_lr",
        "scalar_lr",
        "weight_decay",
    ):
        _require_number(optimizer.get(key), f"optimizer.{key}", minimum=0.0)

    schedule = _require_mapping(manifest.get("schedule"), "schedule")
    _require_exact_keys(
        schedule,
        {
            "num_updates",
            "warmup_steps",
            "muon_momentum_warmup_steps",
            "warmdown_ratio",
            "final_lr_fraction",
            "schedule_version",
        },
        "schedule",
    )
    if _require_int(schedule.get("num_updates"), "schedule.num_updates", minimum=1) != updates:
        raise PreflightContractError("schedule.num_updates must match token batch num_updates")
    warmup = _require_int(schedule.get("warmup_steps"), "schedule.warmup_steps", minimum=0)
    muon_warmup = _require_int(
        schedule.get("muon_momentum_warmup_steps"),
        "schedule.muon_momentum_warmup_steps",
        minimum=0,
    )
    warmdown = _require_number(
        schedule.get("warmdown_ratio"), "schedule.warmdown_ratio", minimum=0.0, maximum=1.0
    )
    if warmdown >= 1.0:
        raise PreflightContractError("schedule.warmdown_ratio must be < 1")
    _require_number(
        schedule.get("final_lr_fraction"),
        "schedule.final_lr_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    version = schedule.get("schedule_version")
    if version not in ("legacy_v0", "exact_last_update_v1"):
        raise PreflightContractError("schedule.schedule_version is invalid")
    from nanochat.optimizer_schedule import validate_schedule_geometry

    try:
        validate_schedule_geometry(
            num_iterations=updates,
            warmup_steps=warmup,
            muon_momentum_warmup_steps=muon_warmup,
            warmdown_ratio=warmdown,
            schedule_version=version,
        )
    except ValueError as exc:
        raise PreflightContractError(f"invalid schedule geometry: {exc}") from exc

    execution = _require_mapping(manifest.get("execution"), "execution")
    _require_exact_keys(
        execution,
        {"device_batch_sequences", "timing_warmup_updates"},
        "execution",
    )
    device_batch = _require_int(
        execution.get("device_batch_sequences"),
        "execution.device_batch_sequences",
        minimum=1,
    )
    timing_warmup = _require_int(
        execution.get("timing_warmup_updates"),
        "execution.timing_warmup_updates",
        minimum=0,
    )
    if timing_warmup >= updates:
        raise PreflightContractError("at least one update must remain for throughput measurement")
    for world_size in PREFLIGHT_WORLD_SIZES:
        if global_sequences % (world_size * device_batch) != 0:
            raise PreflightContractError(
                "global_batch_sequences must be divisible by "
                f"world_size*device_batch_sequences for world_size={world_size}"
            )

    runtime = _require_mapping(manifest.get("runtime"), "runtime")
    _require_exact_keys(
        runtime,
        {"compute_dtype", "attention_backend", "compile_model", "fp8_recipe"},
        "runtime",
    )
    if runtime.get("compute_dtype") not in ("bfloat16", "float32"):
        raise PreflightContractError("runtime.compute_dtype must be bfloat16 or float32")
    if runtime.get("attention_backend") not in ("fa3", "sdpa"):
        raise PreflightContractError("runtime.attention_backend must be fa3 or sdpa")
    if (
        runtime.get("attention_backend") == "fa3"
        and runtime.get("compute_dtype") != "bfloat16"
    ):
        raise PreflightContractError("FA3 preflight requires bfloat16 compute")
    if runtime.get("compile_model") is not True:
        raise PreflightContractError("runtime.compile_model must be true")
    fp8_recipe = runtime.get("fp8_recipe")
    if fp8_recipe is not None and fp8_recipe != "tensorwise":
        raise PreflightContractError("runtime.fp8_recipe must be null or tensorwise")
    if fp8_recipe is not None and runtime.get("compute_dtype") != "bfloat16":
        raise PreflightContractError("FP8 preflight requires bfloat16 compute")

    _require_int(manifest.get("init_seed"), "init_seed", minimum=0)
    tolerances = _require_mapping(manifest.get("tolerances"), "tolerances")
    _require_exact_keys(
        tolerances,
        {"loss_atol", "loss_rtol", "parameter_max_abs", "parameter_relative_l2"},
        "tolerances",
    )
    for key in ("loss_atol", "loss_rtol", "parameter_max_abs", "parameter_relative_l2"):
        value = _require_number(tolerances.get(key), f"tolerances.{key}", minimum=0.0)
        if value != 0.0:
            raise PreflightContractError(
                "v1 preflight tolerances are frozen at zero; inspect any difference manually"
            )


def load_manifest(path: Path, *, verify_token_file: bool = True) -> dict[str, Any]:
    path = path.resolve()
    manifest = _load_json(path)
    if not isinstance(manifest, dict):
        raise PreflightContractError("preflight manifest must be a JSON object")
    validate_manifest(manifest)
    bound_artifacts = {
        "token_batches": manifest["token_batches"],
        "environment_lock": manifest["environment_lock"],
        "runtime_bootstrap": manifest["runtime_bootstrap"],
        "formal_runtime_profile": manifest["formal_runtime_profile"],
    }
    for label, artifact in bound_artifacts.items():
        artifact_path = _require_relative_file(path.parent, artifact["file"], f"{label}.file")
        if verify_token_file:
            if not artifact_path.is_file():
                raise PreflightContractError(f"bound artifact is missing: {artifact_path}")
            observed = _sha256_file(artifact_path)
            if observed != artifact["sha256"]:
                raise PreflightContractError(
                    f"{label} SHA-256 mismatch: "
                    f"expected={artifact['sha256']} observed={observed}"
                )
    if verify_token_file:
        profile_spec = manifest["formal_runtime_profile"]
        profile_path = _require_relative_file(
            path.parent, profile_spec["file"], "formal_runtime_profile.file"
        )
        profile = _validate_formal_runtime_profile_source(profile_path)
        for key in (
            "compute_dtype",
            "attention_backend",
            "compile_model",
            "fp8_recipe",
        ):
            if manifest["runtime"][key] != profile[key]:
                raise PreflightContractError(
                    f"manifest runtime.{key} differs from the formal runtime profile"
                )
    return manifest


def load_token_batches(manifest_path: Path, manifest: Mapping[str, Any]) -> torch.Tensor:
    token_spec = manifest["token_batches"]
    token_path = _require_relative_file(
        manifest_path.resolve().parent, token_spec["file"], "token_batches.file"
    )
    payload = torch.load(token_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != TOKEN_FORMAT:
        raise PreflightContractError("token batch file has the wrong format")
    tokens = payload.get("tokens")
    if not isinstance(tokens, torch.Tensor):
        raise PreflightContractError("token batch file does not contain a tensor")
    expected_shape = (
        token_spec["num_updates"],
        token_spec["global_batch_sequences"],
        token_spec["sequence_length"] + 1,
    )
    if tuple(tokens.shape) != expected_shape:
        raise PreflightContractError(
            f"token tensor shape mismatch: expected={expected_shape} observed={tuple(tokens.shape)}"
        )
    if tokens.dtype not in (torch.int32, torch.int64):
        raise PreflightContractError("token tensor must use int32 or int64")
    if tokens.numel() == 0:
        raise PreflightContractError("token tensor is empty")
    minimum = int(tokens.min().item())
    maximum = int(tokens.max().item())
    if minimum < 0 or maximum >= token_spec["vocab_size"]:
        raise PreflightContractError(
            f"token ids outside [0, {token_spec['vocab_size']}): min={minimum} max={maximum}"
        )
    return tokens.contiguous()


def partition_global_batches(
    tokens: torch.Tensor,
    *,
    world_size: int,
    rank: int,
    device_batch_sequences: int,
) -> torch.Tensor:
    """Return one rank's fixed contiguous share of every global batch.

    This function has no RNG and no loader state. Concatenating the rank shares
    in rank order exactly reconstructs the pre-materialized global tensor.
    """

    if tokens.ndim != 3:
        raise PreflightContractError("global token batches must be rank-3")
    if world_size not in PREFLIGHT_WORLD_SIZES:
        raise PreflightContractError("world_size must be one of 1, 2, 4")
    if rank < 0 or rank >= world_size:
        raise PreflightContractError("rank is outside world_size")
    global_sequences = tokens.shape[1]
    if global_sequences % world_size != 0:
        raise PreflightContractError("global sequence count is not divisible by world_size")
    local_sequences = global_sequences // world_size
    if local_sequences % device_batch_sequences != 0:
        raise PreflightContractError(
            "rank-local sequence count is not divisible by device_batch_sequences"
        )
    start = rank * local_sequences
    return tokens[:, start : start + local_sequences, :]


def _load_input_tokens(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        value = value.get("tokens")
    if not isinstance(value, torch.Tensor):
        raise PreflightContractError("--input-token-tensor must contain a tensor or {'tokens': tensor}")
    if value.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise PreflightContractError("--input-token-tensor must use an integer dtype")
    return value.detach().to(device="cpu", dtype=torch.int32).contiguous()


def _detect_git_commit(checkout: Path) -> str:
    command = ["git", "-C", str(checkout), "rev-parse", "HEAD"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return _validate_commit(result.stdout.strip())


def _validate_formal_runtime_profile_source(path: Path) -> dict[str, Any]:
    try:
        profile = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightContractError(f"invalid formal runtime profile: {exc}") from exc
    if not isinstance(profile, dict):
        raise PreflightContractError("formal runtime profile must be a JSON object")
    _require_exact_keys(
        profile,
        {
            "schema_version",
            "gpu_count",
            "compute_dtype",
            "attention_backend",
            "compile_model",
            "fp8_recipe",
        },
        "formal runtime profile",
    )
    if profile.get("schema_version") != FORMAL_RUNTIME_PROFILE_VERSION:
        raise PreflightContractError(
            f"formal runtime profile schema must be {FORMAL_RUNTIME_PROFILE_VERSION}"
        )
    if profile.get("gpu_count") != FORMAL_CAMPAIGN_GPU_COUNT:
        raise PreflightContractError("formal runtime profile gpu_count must be 4")
    if profile.get("compute_dtype") not in ("bfloat16", "float32"):
        raise PreflightContractError("formal runtime profile compute_dtype is invalid")
    if profile.get("attention_backend") not in ("fa3", "sdpa"):
        raise PreflightContractError("formal runtime profile attention_backend is invalid")
    if profile.get("compile_model") is not True:
        raise PreflightContractError(
            "formal runtime profile compile_model must be true to match base_train"
        )
    if profile.get("fp8_recipe") not in (None, "tensorwise"):
        raise PreflightContractError("formal runtime profile fp8_recipe is invalid")
    if profile["attention_backend"] == "fa3" and profile["compute_dtype"] != "bfloat16":
        raise PreflightContractError("formal FA3 runtime profile requires bfloat16")
    if profile["fp8_recipe"] is not None and profile["compute_dtype"] != "bfloat16":
        raise PreflightContractError("formal FP8 runtime profile requires bfloat16")
    return profile


def _validate_environment_lock_source(
    path: Path, *, compute_dtype: str, attention_backend: str
) -> dict[str, Any]:
    """Validate a previously captured formal lock without recording its source path."""

    from csc_fast_scaling_law.environment_probe import EXPECTED_KEYS
    from csc_fast_scaling_law.nanochat_task_runner import ENVIRONMENT_SCHEMA_VERSION

    try:
        lock = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightContractError(f"invalid environment lock: {exc}") from exc
    if not isinstance(lock, dict):
        raise PreflightContractError("environment lock must be a JSON object")
    _require_exact_keys(lock, {"schema_version", "expected", "lockfiles"}, "environment lock")
    if lock.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        raise PreflightContractError(
            f"environment lock schema must be {ENVIRONMENT_SCHEMA_VERSION}"
        )
    expected = _require_mapping(lock.get("expected"), "environment lock expected")
    _require_exact_keys(expected, set(EXPECTED_KEYS), "environment lock expected")
    if expected.get("compute_dtype") != compute_dtype:
        raise PreflightContractError("environment lock compute_dtype differs from preflight")
    if expected.get("attention_backend") != attention_backend:
        raise PreflightContractError("environment lock attention_backend differs from preflight")
    if expected.get("cuda_available") is not True:
        raise PreflightContractError("environment lock must be captured in a CUDA allocation")
    for key in (
        "python_version",
        "python_implementation",
        "python_executable_sha256",
        "torch_version",
        "torch_cuda_version",
        "pyarrow_version",
        "rustbpe_version",
        "tokenizers_version",
        "tiktoken_version",
        "kernels_version",
    ):
        value = expected.get(key)
        if not isinstance(value, str) or not value:
            raise PreflightContractError(f"environment lock expected.{key} is missing")
    _validate_sha256(
        expected.get("python_executable_sha256"),
        "environment lock expected.python_executable_sha256",
    )
    if attention_backend == "fa3":
        requested = expected.get("fa3_requested_revision")
        resolved = expected.get("fa3_resolved_revision")
        if (
            not isinstance(requested, str)
            or re.fullmatch(r"[0-9a-f]{40}", requested) is None
            or resolved != requested
        ):
            raise PreflightContractError(
                "environment lock must bind one identical requested/resolved FA3 commit"
            )
        origin = expected.get("fa3_interface_origin")
        loaded_files = expected.get("fa3_loaded_files")
        if not isinstance(origin, str) or not origin:
            raise PreflightContractError("environment lock lacks the FA3 interface origin")
        if not isinstance(loaded_files, list) or not loaded_files:
            raise PreflightContractError("environment lock lacks loaded FA3 file hashes")
        covered_origin = False
        for index, item in enumerate(loaded_files):
            if not isinstance(item, dict) or set(item) != {"module", "path", "sha256"}:
                raise PreflightContractError(
                    f"environment lock fa3_loaded_files[{index}] is invalid"
                )
            if not isinstance(item.get("module"), str) or not item["module"]:
                raise PreflightContractError(
                    f"environment lock fa3_loaded_files[{index}].module is invalid"
                )
            if not isinstance(item.get("path"), str) or not item["path"]:
                raise PreflightContractError(
                    f"environment lock fa3_loaded_files[{index}].path is invalid"
                )
            _validate_sha256(
                item.get("sha256"),
                f"environment lock fa3_loaded_files[{index}].sha256",
            )
            covered_origin = covered_origin or item["path"] == origin
        if not covered_origin:
            raise PreflightContractError(
                "environment lock FA3 origin is not covered by a loaded-file hash"
            )

    lockfiles = lock.get("lockfiles")
    if not isinstance(lockfiles, list) or not lockfiles:
        raise PreflightContractError("environment lock must bind dependency lockfiles")
    for index, spec in enumerate(lockfiles):
        if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
            raise PreflightContractError(f"environment lock lockfiles[{index}] is invalid")
        raw_path = spec.get("path")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise PreflightContractError(
                "environment lock dependency paths must be absolute before it is copied"
            )
        dependency_path = Path(raw_path).expanduser().resolve()
        digest = _validate_sha256(
            spec.get("sha256"), f"environment lock lockfiles[{index}].sha256"
        )
        if not dependency_path.is_file() or _sha256_file(dependency_path) != digest:
            raise PreflightContractError(
                f"environment lock dependency lockfile is missing or changed: index={index}"
            )
    return lock


def _copy_bound_input(source: Path, destination: Path, *, label: str) -> str:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise PreflightContractError(f"{label} is missing")
    before = _sha256_file(source)
    shutil.copyfile(source, destination)
    after = _sha256_file(destination)
    if after != before:
        raise PreflightContractError(f"{label} changed while it was copied")
    return after


def prepare_preflight(args: argparse.Namespace) -> Path:
    """Materialize tokens and atomically create a frozen preflight directory."""

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise PreflightContractError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(output_dir.name + f".partial.{os.getpid()}")
    if staging.exists():
        raise PreflightContractError(f"staging directory already exists: {staging}")
    staging.mkdir()

    formal_runtime_profile_source = Path(args.formal_runtime_profile).expanduser().resolve()
    formal_runtime_profile = _validate_formal_runtime_profile_source(
        formal_runtime_profile_source
    )
    formal_runtime_profile_path = staging / "formal_runtime_profile.json"
    formal_runtime_profile_sha256 = _copy_bound_input(
        formal_runtime_profile_source,
        formal_runtime_profile_path,
        label="formal runtime profile",
    )
    environment_lock_source = Path(args.environment_lock).expanduser().resolve()
    _validate_environment_lock_source(
        environment_lock_source,
        compute_dtype=formal_runtime_profile["compute_dtype"],
        attention_backend=formal_runtime_profile["attention_backend"],
    )
    environment_lock_path = staging / "environment_lock.json"
    environment_lock_sha256 = _copy_bound_input(
        environment_lock_source, environment_lock_path, label="environment lock"
    )
    runtime_bootstrap_path = staging / "runtime_bootstrap.sh"
    runtime_bootstrap_sha256 = _copy_bound_input(
        Path(args.runtime_bootstrap), runtime_bootstrap_path, label="runtime bootstrap"
    )
    if runtime_bootstrap_path.stat().st_size == 0:
        raise PreflightContractError("runtime bootstrap must not be empty")

    if args.input_token_tensor:
        tokens = _load_input_tokens(Path(args.input_token_tensor))
        source = "provided_tensor"
        if tokens.ndim != 3:
            raise PreflightContractError("input token tensor must have [updates, global_sequences, T+1]")
        updates, global_sequences, token_width = tokens.shape
        sequence_length = token_width - 1
        if args.num_updates is not None and args.num_updates != updates:
            raise PreflightContractError("--num-updates disagrees with input token tensor")
        if args.global_batch_sequences is not None and args.global_batch_sequences != global_sequences:
            raise PreflightContractError("--global-batch-sequences disagrees with input token tensor")
        if args.sequence_length is not None and args.sequence_length != sequence_length:
            raise PreflightContractError("--sequence-length disagrees with input token tensor")
    else:
        source = "synthetic_uniform"
        updates = args.num_updates if args.num_updates is not None else 8
        global_sequences = (
            args.global_batch_sequences if args.global_batch_sequences is not None else 256
        )
        sequence_length = args.sequence_length if args.sequence_length is not None else 2048
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.token_seed)
        tokens = torch.randint(
            0,
            args.vocab_size,
            (updates, global_sequences, sequence_length + 1),
            dtype=torch.int32,
            generator=generator,
        )

    if tokens.numel() == 0:
        raise PreflightContractError("token tensor is empty")
    token_minimum = int(tokens.min().item())
    token_maximum = int(tokens.max().item())
    if token_minimum < 0 or token_maximum >= args.vocab_size:
        raise PreflightContractError(
            f"token ids outside [0, {args.vocab_size}): "
            f"min={token_minimum} max={token_maximum}"
        )

    token_path = staging / "global_token_batches.pt"
    _atomic_torch_save(token_path, {"format": TOKEN_FORMAT, "tokens": tokens})
    code_commit = args.code_commit or _detect_git_commit(Path(args.checkout).resolve())
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_VERSION,
        "preflight_only": True,
        "formal_campaign_gpu_count": FORMAL_CAMPAIGN_GPU_COUNT,
        "world_sizes": list(PREFLIGHT_WORLD_SIZES),
        "code_commit": _validate_commit(code_commit),
        "created_utc": _utc_now(),
        "init_seed": args.init_seed,
        "environment_lock": {
            "file": environment_lock_path.name,
            "sha256": environment_lock_sha256,
        },
        "runtime_bootstrap": {
            "file": runtime_bootstrap_path.name,
            "sha256": runtime_bootstrap_sha256,
        },
        "formal_runtime_profile": {
            "file": formal_runtime_profile_path.name,
            "sha256": formal_runtime_profile_sha256,
        },
        "token_batches": {
            "format": TOKEN_FORMAT,
            "file": token_path.name,
            "sha256": _sha256_file(token_path),
            "source": source,
            "num_updates": int(updates),
            "global_batch_sequences": int(global_sequences),
            "sequence_length": int(sequence_length),
            "vocab_size": args.vocab_size,
        },
        "model": {
            "sequence_len": int(sequence_length),
            "vocab_size": args.vocab_size,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_kv_head": args.n_kv_head,
            "n_embd": args.n_embd,
            "window_pattern": args.window_pattern,
        },
        "optimizer": {
            "unembedding_lr": args.unembedding_lr,
            "embedding_lr": args.embedding_lr,
            "matrix_lr": args.matrix_lr,
            "scalar_lr": args.scalar_lr,
            "weight_decay": args.weight_decay,
        },
        "schedule": {
            "num_updates": int(updates),
            "warmup_steps": args.warmup_steps,
            "muon_momentum_warmup_steps": args.muon_momentum_warmup_steps,
            "warmdown_ratio": args.warmdown_ratio,
            "final_lr_fraction": args.final_lr_fraction,
            "schedule_version": args.schedule_version,
        },
        "execution": {
            "device_batch_sequences": args.device_batch_sequences,
            "timing_warmup_updates": args.timing_warmup_updates,
        },
        "runtime": {
            "compute_dtype": formal_runtime_profile["compute_dtype"],
            "attention_backend": formal_runtime_profile["attention_backend"],
            "compile_model": formal_runtime_profile["compile_model"],
            "fp8_recipe": formal_runtime_profile["fp8_recipe"],
        },
        "tolerances": {
            "loss_atol": 0.0,
            "loss_rtol": 0.0,
            "parameter_max_abs": 0.0,
            "parameter_relative_l2": 0.0,
        },
    }
    validate_manifest(manifest)
    _atomic_json(staging / "preflight_manifest.json", manifest)
    os.replace(staging, output_dir)
    return output_dir / "preflight_manifest.json"


def _parameter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().to(device="cpu").contiguous()
        for name, parameter in sorted(model.named_parameters())
    }


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().to(device="cpu").contiguous()
    return cpu.reshape(-1).view(torch.uint8).numpy().tobytes()


def parameter_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        metadata = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        digest.update(_canonical_json_bytes(metadata))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _state_delta_stats(
    initial: Mapping[str, torch.Tensor], final: Mapping[str, torch.Tensor]
) -> tuple[float, float]:
    sum_squares = 0.0
    maximum = 0.0
    if set(initial) != set(final):
        raise PreflightContractError("initial/final parameter keys differ")
    for name in sorted(initial):
        left, right = initial[name], final[name]
        if left.shape != right.shape:
            raise PreflightContractError(f"parameter shape changed for {name}")
        delta = right.to(torch.float64) - left.to(torch.float64)
        sum_squares += float(torch.sum(delta * delta).item())
        if delta.numel():
            maximum = max(maximum, float(delta.abs().max().item()))
    return math.sqrt(sum_squares), maximum


def _semantic_state_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / "initial_parameters.pt", output_dir / "final_parameters.pt"


def _validate_environment_receipt(
    receipt: Any,
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    gpu_count: int,
) -> str:
    receipt = _require_mapping(receipt, "environment validation receipt")
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "gpu_count",
            "environment_lock_sha256",
            "runtime_bootstrap_sha256",
            "formal_validation",
        },
        "environment validation receipt",
    )
    if receipt.get("schema_version") != ENVIRONMENT_RECEIPT_VERSION:
        raise PreflightContractError("environment validation receipt schema is invalid")
    if receipt.get("gpu_count") != gpu_count:
        raise PreflightContractError("environment validation receipt GPU count is invalid")
    if receipt.get("environment_lock_sha256") != manifest["environment_lock"]["sha256"]:
        raise PreflightContractError("environment validation receipt lock hash mismatch")
    if receipt.get("runtime_bootstrap_sha256") != manifest["runtime_bootstrap"]["sha256"]:
        raise PreflightContractError("environment validation receipt bootstrap hash mismatch")
    formal = _require_mapping(receipt.get("formal_validation"), "formal environment validation")
    if formal.get("schema_version") != "nanochat-environment-v1":
        raise PreflightContractError("formal environment validation schema is invalid")
    formal_lock = _require_mapping(
        formal.get("environment_lock"), "formal environment lock provenance"
    )
    if formal_lock.get("sha256") != manifest["environment_lock"]["sha256"]:
        raise PreflightContractError("formal validation used another environment lock")
    observed = _require_mapping(formal.get("observed"), "formal environment observation")
    lock_path = _require_relative_file(
        manifest_path.resolve().parent,
        manifest["environment_lock"]["file"],
        "environment_lock.file",
    )
    raw_formal_lock_path = formal_lock.get("path")
    if (
        not isinstance(raw_formal_lock_path, str)
        or Path(raw_formal_lock_path).expanduser().resolve() != lock_path
    ):
        raise PreflightContractError("formal validation did not use the copied environment lock")
    lock = _load_json(lock_path)
    expected = _require_mapping(lock.get("expected"), "environment lock expected")
    for key, expected_value in expected.items():
        if observed.get(key) != expected_value:
            raise PreflightContractError(
                f"recorded environment observation differs from lock for {key}"
            )
    if observed.get("cuda_device_count") != gpu_count:
        raise PreflightContractError("recorded visible CUDA device count is invalid")
    return hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest()


def _check_checkout(commit: str) -> None:
    checkout = Path.cwd()
    observed = _detect_git_commit(checkout)
    if observed != commit:
        raise PreflightContractError(
            f"checkout commit mismatch: manifest={commit} observed={observed}"
        )
    status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if status:
        raise PreflightContractError("checkout is dirty; refusing numerical preflight")


def _bind_compute_dtype_environment(manifest: Mapping[str, Any]) -> None:
    expected = manifest["runtime"]["compute_dtype"]
    configured = os.environ.get("NANOCHAT_DTYPE")
    if configured is None:
        os.environ["NANOCHAT_DTYPE"] = expected
    elif configured != expected:
        raise PreflightContractError(
            f"NANOCHAT_DTYPE={configured} disagrees with manifest={expected}"
        )


def _verify_sourced_runtime_bootstrap(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> str:
    """Prove that the sourced bootstrap has the manifest-bound content hash."""

    expected = manifest["runtime_bootstrap"]["sha256"]
    declared = os.environ.get("RUNTIME_BOOTSTRAP_SHA256")
    if declared != expected:
        raise PreflightContractError(
            "RUNTIME_BOOTSTRAP_SHA256 does not match the frozen preflight manifest"
        )
    raw_path = os.environ.get("RUNTIME_BOOTSTRAP")
    if not raw_path:
        raise PreflightContractError("RUNTIME_BOOTSTRAP must name the sourced bootstrap")
    actual_path = Path(raw_path).expanduser().resolve()
    if not actual_path.is_file() or _sha256_file(actual_path) != expected:
        raise PreflightContractError("the sourced RUNTIME_BOOTSTRAP is missing or hash-mismatched")
    copied_path = _require_relative_file(
        manifest_path.resolve().parent,
        manifest["runtime_bootstrap"]["file"],
        "runtime_bootstrap.file",
    )
    if not copied_path.is_file() or _sha256_file(copied_path) != expected:
        raise PreflightContractError("the manifest-copied runtime bootstrap changed")
    return expected


def validate_bound_runtime_environment(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    gpu_count: int,
) -> dict[str, Any]:
    """Run and reuse the formal campaign environment probe and validator."""

    from csc_fast_scaling_law.environment_probe import probe_environment
    from csc_fast_scaling_law.nanochat_task_runner import (
        ENVIRONMENT_SCHEMA_VERSION,
        validate_environment_observation,
    )

    if gpu_count not in PREFLIGHT_WORLD_SIZES:
        raise PreflightContractError("environment validation GPU count must be 1, 2, or 4")
    _bind_compute_dtype_environment(manifest)
    bootstrap_sha256 = _verify_sourced_runtime_bootstrap(manifest_path, manifest)
    lock_spec = manifest["environment_lock"]
    lock_path = _require_relative_file(
        manifest_path.resolve().parent, lock_spec["file"], "environment_lock.file"
    )
    observed = probe_environment(sys.executable, checkout=Path.cwd())
    prepared = {
        "file_provenance": {
            "environment_lock": {
                "path": str(lock_path),
                "sha256": lock_spec["sha256"],
            }
        },
        "schedule": {"gpu_count": gpu_count},
        "plan_row": {
            "attention_backend": manifest["runtime"]["attention_backend"],
            "compute_dtype": manifest["runtime"]["compute_dtype"],
            "fp8": "true" if manifest["runtime"]["fp8_recipe"] is not None else "false",
        },
        "observed_code": {"checkout": str(Path.cwd().resolve())},
    }
    try:
        validated = validate_environment_observation(prepared, observed)
    except Exception as exc:
        raise PreflightContractError(f"formal environment validation failed: {exc}") from exc
    if validated.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        raise PreflightContractError("formal environment validator returned the wrong schema")
    return {
        "schema_version": ENVIRONMENT_RECEIPT_VERSION,
        "gpu_count": gpu_count,
        "environment_lock_sha256": lock_spec["sha256"],
        "runtime_bootstrap_sha256": bootstrap_sha256,
        "formal_validation": validated,
    }


def _convert_fp8(model: torch.nn.Module, recipe: str | None) -> int:
    if recipe is None:
        return 0
    from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training

    config = Float8LinearConfig.from_recipe_name(recipe)

    def module_filter(module: torch.nn.Module, _fqn: str) -> bool:
        if not isinstance(module, torch.nn.Linear):
            return False
        if module.in_features % 16 or module.out_features % 16:
            return False
        return min(module.in_features, module.out_features) >= 128

    convert_to_float8_training(model, config=config, module_filter_fn=module_filter)
    return sum(1 for module in model.modules() if "Float8" in type(module).__name__)


def _all_ranks_same(value: str, world_size: int, dist: Any) -> bool:
    values: list[str | None] = [None] * world_size
    dist.all_gather_object(values, value)
    return all(item == value for item in values)


def run_worker(manifest_path: Path, output_dir: Path) -> None:
    """Run one torchrun world size and write a rank-zero result bundle."""

    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    expected_dtype = manifest["runtime"]["compute_dtype"]
    _bind_compute_dtype_environment(manifest)

    # Imports are deferred until after NANOCHAT_DTYPE is frozen.
    import torch.distributed as dist
    from nanochat.common import COMPUTE_DTYPE, compute_cleanup, compute_init
    from nanochat.flash_attention import USE_FA3
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.optimizer_schedule import (
        learning_rate_multiplier,
        muon_momentum,
        weight_decay_multiplier,
    )

    if not torch.cuda.is_available():
        raise PreflightContractError("GPU preflight requires CUDA")
    ddp, rank, _local_rank, world_size, device = compute_init("cuda")
    if not ddp:
        compute_cleanup()
        raise PreflightContractError("run-worker must be launched with torchrun, including world_size=1")
    try:
        if world_size not in PREFLIGHT_WORLD_SIZES:
            raise PreflightContractError("torchrun world size must be one of 1, 2, 4")
        _check_checkout(manifest["code_commit"])
        environment_box: list[dict[str, Any] | None] = [None]
        if rank == 0:
            try:
                environment_box[0] = validate_bound_runtime_environment(
                    manifest_path,
                    manifest,
                    gpu_count=FORMAL_CAMPAIGN_GPU_COUNT,
                )
            except Exception as exc:
                environment_box[0] = {"error": str(exc)}
        dist.broadcast_object_list(environment_box, src=0)
        environment_receipt = environment_box[0]
        if not isinstance(environment_receipt, dict):
            raise PreflightContractError("rank zero returned no environment validation receipt")
        if "error" in environment_receipt:
            raise PreflightContractError(
                f"world{world_size} environment validation failed: "
                f"{environment_receipt['error']}"
            )
        output_dir = output_dir.resolve()
        error: list[str | None] = [None]
        if rank == 0:
            try:
                if output_dir.exists():
                    raise PreflightContractError(
                        f"refusing to overwrite worker output: {output_dir}"
                    )
                output_dir.mkdir(parents=True)
            except Exception as exc:  # propagate rank-zero filesystem failures
                error[0] = str(exc)
        dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise PreflightContractError(error[0])

        observed_dtype = str(COMPUTE_DTYPE).removeprefix("torch.")
        observed_backend = "fa3" if USE_FA3 else "sdpa"
        if observed_dtype != expected_dtype:
            raise PreflightContractError(
                f"compute dtype mismatch: expected={expected_dtype} observed={observed_dtype}"
            )
        if observed_backend != manifest["runtime"]["attention_backend"]:
            raise PreflightContractError(
                "attention backend mismatch: "
                f"expected={manifest['runtime']['attention_backend']} observed={observed_backend}"
            )

        tokens = load_token_batches(manifest_path, manifest)
        local_tokens = partition_global_batches(
            tokens,
            world_size=world_size,
            rank=rank,
            device_batch_sequences=manifest["execution"]["device_batch_sequences"],
        ).to(device=device, dtype=torch.long)
        del tokens

        init_seed = manifest["init_seed"]
        torch.manual_seed(init_seed)
        torch.cuda.manual_seed_all(init_seed)
        config = GPTConfig(**manifest["model"])
        with torch.device("meta"):
            original_model = GPT(config)
        original_model.to_empty(device=device)
        original_model.init_weights()
        fp8_modules = _convert_fp8(original_model, manifest["runtime"]["fp8_recipe"])

        initial_state = _parameter_state(original_model)
        initial_semantic_hash = parameter_state_sha256(initial_state)
        if not _all_ranks_same(initial_semantic_hash, world_size, dist):
            raise PreflightContractError("initial parameters differ across ranks")

        training_model: torch.nn.Module = original_model
        if manifest["runtime"]["compile_model"]:
            training_model = torch.compile(training_model, dynamic=False)
        optimizer = training_model.setup_optimizer(**manifest["optimizer"])

        schedule = manifest["schedule"]
        device_batch = manifest["execution"]["device_batch_sequences"]
        grad_accum_steps = local_tokens.shape[1] // device_batch
        global_prediction_tokens = (
            manifest["token_batches"]["global_batch_sequences"]
            * manifest["token_batches"]["sequence_length"]
        )
        losses: list[float] = []
        elapsed_seconds: list[float] = []
        for update in range(schedule["num_updates"]):
            optimizer.zero_grad(set_to_none=True)
            local_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
            dist.barrier()
            torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            update_tokens = local_tokens[update]
            for micro_step in range(grad_accum_steps):
                micro = update_tokens[
                    micro_step * device_batch : (micro_step + 1) * device_batch
                ]
                inputs = micro[:, :-1].contiguous()
                targets = micro[:, 1:].contiguous()
                loss = training_model(inputs, targets)
                local_loss_sum += loss.detach().to(torch.float64)
                (loss / grad_accum_steps).backward()

            lr_multiplier = learning_rate_multiplier(
                update,
                num_iterations=schedule["num_updates"],
                warmup_steps=schedule["warmup_steps"],
                warmdown_ratio=schedule["warmdown_ratio"],
                final_lr_fraction=schedule["final_lr_fraction"],
                schedule_version=schedule["schedule_version"],
            )
            momentum = muon_momentum(
                update,
                num_iterations=schedule["num_updates"],
                warmup_steps=schedule["muon_momentum_warmup_steps"],
                warmdown_ratio=schedule["warmdown_ratio"],
                schedule_version=schedule["schedule_version"],
            )
            decay = manifest["optimizer"]["weight_decay"] * weight_decay_multiplier(
                update,
                num_iterations=schedule["num_updates"],
                schedule_version=schedule["schedule_version"],
            )
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lr_multiplier
                if group["kind"] == "muon":
                    group["momentum"] = momentum
                    group["weight_decay"] = decay
            optimizer.step()
            training_model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            local_elapsed = torch.tensor(time.perf_counter() - start_time, device=device)
            dist.all_reduce(local_elapsed, op=dist.ReduceOp.MAX)
            elapsed_seconds.append(float(local_elapsed.item()))

            local_mean = local_loss_sum / grad_accum_steps
            dist.all_reduce(local_mean, op=dist.ReduceOp.AVG)
            losses.append(float(local_mean.item()))

        final_state = _parameter_state(original_model)
        final_semantic_hash = parameter_state_sha256(final_state)
        if not _all_ranks_same(final_semantic_hash, world_size, dist):
            raise PreflightContractError("final parameters differ across ranks")
        delta_l2, delta_max_abs = _state_delta_stats(initial_state, final_state)
        max_memory = torch.tensor(
            torch.cuda.max_memory_allocated(device), dtype=torch.int64, device=device
        )
        dist.all_reduce(max_memory, op=dist.ReduceOp.MAX)

        timing_warmup = manifest["execution"]["timing_warmup_updates"]
        measured_elapsed = elapsed_seconds[timing_warmup:]
        throughput = [global_prediction_tokens / seconds for seconds in measured_elapsed]
        if rank == 0:
            initial_path, final_path = _semantic_state_paths(output_dir)
            _atomic_torch_save(initial_path, initial_state)
            _atomic_torch_save(final_path, final_state)
            result = {
                "schema_version": WORKER_RESULT_VERSION,
                "preflight_only": True,
                "formal_campaign_gpu_count": FORMAL_CAMPAIGN_GPU_COUNT,
                "created_utc": _utc_now(),
                "manifest_sha256": _sha256_file(manifest_path),
                "code_commit": manifest["code_commit"],
                "world_size": world_size,
                "environment_contract": {
                    "environment_lock_sha256": manifest["environment_lock"]["sha256"],
                    "runtime_bootstrap_sha256": manifest["runtime_bootstrap"]["sha256"],
                    "formal_runtime_profile_sha256": manifest[
                        "formal_runtime_profile"
                    ]["sha256"],
                    "validation_receipt_sha256": hashlib.sha256(
                        _canonical_json_bytes(environment_receipt)
                    ).hexdigest(),
                    "validation_receipt": environment_receipt,
                },
                "runtime": {
                    "compute_dtype": observed_dtype,
                    "attention_backend": observed_backend,
                    "compile_model": manifest["runtime"]["compile_model"],
                    "fp8_recipe": manifest["runtime"]["fp8_recipe"],
                    "fp8_module_count": fp8_modules,
                    "device_name": torch.cuda.get_device_name(device),
                    "torch_version": torch.__version__,
                },
                "input_contract": {
                    "token_sha256": manifest["token_batches"]["sha256"],
                    "global_batch_sequences": manifest["token_batches"][
                        "global_batch_sequences"
                    ],
                    "sequence_length": manifest["token_batches"]["sequence_length"],
                    "num_updates": schedule["num_updates"],
                    "device_batch_sequences": device_batch,
                    "gradient_accumulation_steps": grad_accum_steps,
                },
                "numerical": {
                    "losses": losses,
                    "initial_parameter_sha256": initial_semantic_hash,
                    "final_parameter_sha256": final_semantic_hash,
                    "parameter_delta_l2": delta_l2,
                    "parameter_delta_max_abs": delta_max_abs,
                    "rank_parameters_exactly_synchronized": True,
                },
                "throughput": {
                    "gate_role": "informational_only",
                    "timing_warmup_updates": timing_warmup,
                    "measured_update_indices": list(range(timing_warmup, schedule["num_updates"])),
                    "elapsed_seconds": measured_elapsed,
                    "prediction_tokens_per_second": throughput,
                    "median_prediction_tokens_per_second": statistics.median(throughput),
                    "global_prediction_tokens_per_update": global_prediction_tokens,
                    "maximum_memory_allocated_bytes_over_ranks": int(max_memory.item()),
                },
                "artifacts": {
                    "initial_parameters": {
                        "file": initial_path.name,
                        "sha256": _sha256_file(initial_path),
                    },
                    "final_parameters": {
                        "file": final_path.name,
                        "sha256": _sha256_file(final_path),
                    },
                },
            }
            _atomic_json(output_dir / "result.json", result)
            (output_dir / "SUCCESS").write_text("PASS\n", encoding="utf-8")
        dist.barrier()
    finally:
        compute_cleanup()


def _load_worker_result(
    manifest_path: Path, results_root: Path, world_size: int
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    result_dir = results_root / f"world{world_size}"
    if not (result_dir / "SUCCESS").is_file():
        raise PreflightContractError(f"world{world_size} has no SUCCESS marker")
    result = _load_json(result_dir / "result.json")
    if not isinstance(result, dict) or result.get("schema_version") != WORKER_RESULT_VERSION:
        raise PreflightContractError(f"world{world_size} result has wrong schema")
    if result.get("world_size") != world_size:
        raise PreflightContractError(f"world{world_size} result declares wrong world size")
    if result.get("preflight_only") is not True:
        raise PreflightContractError("worker result must remain preflight-only")
    if result.get("formal_campaign_gpu_count") != FORMAL_CAMPAIGN_GPU_COUNT:
        raise PreflightContractError("worker result changed formal campaign GPU count")
    if result.get("manifest_sha256") != _sha256_file(manifest_path):
        raise PreflightContractError(f"world{world_size} is bound to another manifest")
    manifest = load_manifest(manifest_path, verify_token_file=False)
    if result.get("code_commit") != manifest["code_commit"]:
        raise PreflightContractError(f"world{world_size} code commit mismatch")
    environment_contract = _require_mapping(
        result.get("environment_contract"), "environment_contract"
    )
    _require_exact_keys(
        environment_contract,
        {
            "environment_lock_sha256",
            "runtime_bootstrap_sha256",
            "formal_runtime_profile_sha256",
            "validation_receipt_sha256",
            "validation_receipt",
        },
        "environment_contract",
    )
    expected_environment_hashes = {
        "environment_lock_sha256": manifest["environment_lock"]["sha256"],
        "runtime_bootstrap_sha256": manifest["runtime_bootstrap"]["sha256"],
        "formal_runtime_profile_sha256": manifest["formal_runtime_profile"]["sha256"],
    }
    for key, expected in expected_environment_hashes.items():
        if environment_contract.get(key) != expected:
            raise PreflightContractError(f"world{world_size} {key} mismatch")
    receipt_sha256 = _validate_environment_receipt(
        environment_contract.get("validation_receipt"),
        manifest_path=manifest_path,
        manifest=manifest,
        gpu_count=FORMAL_CAMPAIGN_GPU_COUNT,
    )
    if environment_contract.get("validation_receipt_sha256") != receipt_sha256:
        raise PreflightContractError(
            f"world{world_size} environment validation receipt hash mismatch"
        )
    runtime = _require_mapping(result.get("runtime"), "runtime")
    for key in ("compute_dtype", "attention_backend", "compile_model", "fp8_recipe"):
        if runtime.get(key) != manifest["runtime"][key]:
            raise PreflightContractError(
                f"world{world_size} runtime.{key} differs from the manifest"
            )
    input_contract = _require_mapping(result.get("input_contract"), "input_contract")
    if input_contract.get("token_sha256") != manifest["token_batches"]["sha256"]:
        raise PreflightContractError(f"world{world_size} consumed another token tensor")
    expected_input_values = {
        "global_batch_sequences": manifest["token_batches"]["global_batch_sequences"],
        "sequence_length": manifest["token_batches"]["sequence_length"],
        "num_updates": manifest["token_batches"]["num_updates"],
        "device_batch_sequences": manifest["execution"]["device_batch_sequences"],
        "gradient_accumulation_steps": manifest["token_batches"][
            "global_batch_sequences"
        ]
        // (world_size * manifest["execution"]["device_batch_sequences"]),
    }
    for key, expected in expected_input_values.items():
        if input_contract.get(key) != expected:
            raise PreflightContractError(
                f"world{world_size} input_contract.{key} differs from the manifest"
            )
    numerical = _require_mapping(result.get("numerical"), "numerical")
    if numerical.get("rank_parameters_exactly_synchronized") is not True:
        raise PreflightContractError(f"world{world_size} ranks did not synchronize parameters")

    states: list[dict[str, torch.Tensor]] = []
    for key in ("initial_parameters", "final_parameters"):
        artifact = _require_mapping(result.get("artifacts", {}).get(key), f"artifacts.{key}")
        state_path = _require_relative_file(result_dir, artifact.get("file"), f"artifacts.{key}.file")
        expected_hash = _validate_sha256(artifact.get("sha256"), f"artifacts.{key}.sha256")
        if not state_path.is_file() or _sha256_file(state_path) != expected_hash:
            raise PreflightContractError(f"world{world_size} {key} artifact failed hash check")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in state.items()
        ):
            raise PreflightContractError(f"world{world_size} {key} state is invalid")
        states.append(state)
    if parameter_state_sha256(states[0]) != numerical.get("initial_parameter_sha256"):
        raise PreflightContractError(f"world{world_size} initial parameter semantic hash mismatch")
    if parameter_state_sha256(states[1]) != numerical.get("final_parameter_sha256"):
        raise PreflightContractError(f"world{world_size} final parameter semantic hash mismatch")
    delta_l2, delta_max_abs = _state_delta_stats(states[0], states[1])
    recorded_delta_l2 = _require_number(
        numerical.get("parameter_delta_l2"), f"world{world_size}.parameter_delta_l2", minimum=0.0
    )
    recorded_delta_max = _require_number(
        numerical.get("parameter_delta_max_abs"),
        f"world{world_size}.parameter_delta_max_abs",
        minimum=0.0,
    )
    if not math.isclose(delta_l2, recorded_delta_l2, rel_tol=1e-12, abs_tol=1e-15):
        raise PreflightContractError(f"world{world_size} parameter_delta_l2 is inconsistent")
    if not math.isclose(delta_max_abs, recorded_delta_max, rel_tol=1e-12, abs_tol=1e-15):
        raise PreflightContractError(
            f"world{world_size} parameter_delta_max_abs is inconsistent"
        )
    losses = numerical.get("losses")
    if not isinstance(losses, list) or not losses:
        raise PreflightContractError(f"world{world_size} losses must be a non-empty list")
    if len(losses) != manifest["token_batches"]["num_updates"]:
        raise PreflightContractError(
            f"world{world_size} loss count does not match manifest num_updates"
        )
    for update, loss in enumerate(losses):
        _require_number(loss, f"world{world_size}.losses[{update}]")
    return result, states[0], states[1]


def _compare_final_states(
    reference_initial: Mapping[str, torch.Tensor],
    reference_final: Mapping[str, torch.Tensor],
    candidate_final: Mapping[str, torch.Tensor],
) -> tuple[float, float, float]:
    if set(reference_initial) != set(reference_final) or set(reference_final) != set(candidate_final):
        raise PreflightContractError("parameter state keys differ across world sizes")
    reference_delta_sq = 0.0
    difference_sq = 0.0
    maximum = 0.0
    for name in sorted(reference_final):
        initial = reference_initial[name]
        reference = reference_final[name]
        candidate = candidate_final[name]
        if initial.shape != reference.shape or reference.shape != candidate.shape:
            raise PreflightContractError(f"parameter shape differs across world sizes: {name}")
        ref_delta = reference.to(torch.float64) - initial.to(torch.float64)
        difference = candidate.to(torch.float64) - reference.to(torch.float64)
        reference_delta_sq += float(torch.sum(ref_delta * ref_delta).item())
        difference_sq += float(torch.sum(difference * difference).item())
        if difference.numel():
            maximum = max(maximum, float(difference.abs().max().item()))
    difference_l2 = math.sqrt(difference_sq)
    reference_delta_l2 = math.sqrt(reference_delta_sq)
    relative_l2 = difference_l2 / max(reference_delta_l2, 1e-30)
    return maximum, difference_l2, relative_l2


def aggregate_preflight(
    manifest_path: Path, results_root: Path, summary_path: Path
) -> dict[str, Any]:
    """Aggregate 1/2/4-GPU results; only numerical checks affect status."""

    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    results_root = results_root.resolve()
    loaded = {
        world: _load_worker_result(manifest_path, results_root, world)
        for world in PREFLIGHT_WORLD_SIZES
    }
    orchestrator_environment_path = results_root / "orchestrator_environment.json"
    if not orchestrator_environment_path.is_file():
        raise PreflightContractError("orchestrator environment validation receipt is missing")
    orchestrator_environment = _load_json(orchestrator_environment_path)
    orchestrator_environment_semantic_sha256 = _validate_environment_receipt(
        orchestrator_environment,
        manifest_path=manifest_path,
        manifest=manifest,
        gpu_count=FORMAL_CAMPAIGN_GPU_COUNT,
    )
    reference_result, reference_initial, reference_final = loaded[1]
    reference_initial_hash = parameter_state_sha256(reference_initial)
    if reference_initial_hash != reference_result["numerical"]["initial_parameter_sha256"]:
        raise PreflightContractError("reference initial semantic hash mismatch")

    tolerances = manifest["tolerances"]
    numerical_comparisons: list[dict[str, Any]] = []
    numerical_pass = True
    reference_losses = reference_result["numerical"]["losses"]
    for world_size in (2, 4):
        result, initial, final = loaded[world_size]
        if parameter_state_sha256(initial) != reference_initial_hash:
            raise PreflightContractError(f"world{world_size} initial parameters differ from world1")
        losses = result["numerical"]["losses"]
        if len(losses) != len(reference_losses):
            raise PreflightContractError(f"world{world_size} loss count differs from world1")
        loss_rows: list[dict[str, Any]] = []
        losses_pass = True
        for update, (reference, candidate) in enumerate(zip(reference_losses, losses)):
            reference_f = _require_number(reference, f"world1.losses[{update}]")
            candidate_f = _require_number(candidate, f"world{world_size}.losses[{update}]")
            absolute = abs(candidate_f - reference_f)
            allowed = tolerances["loss_atol"] + tolerances["loss_rtol"] * abs(reference_f)
            passed = absolute <= allowed
            losses_pass = losses_pass and passed
            loss_rows.append(
                {
                    "update": update,
                    "reference": reference_f,
                    "candidate": candidate_f,
                    "absolute_difference": absolute,
                    "allowed_difference": allowed,
                    "pass": passed,
                }
            )
        maximum, difference_l2, relative_l2 = _compare_final_states(
            reference_initial, reference_final, final
        )
        parameters_pass = (
            maximum <= tolerances["parameter_max_abs"]
            and relative_l2 <= tolerances["parameter_relative_l2"]
        )
        comparison_pass = losses_pass and parameters_pass
        numerical_pass = numerical_pass and comparison_pass
        numerical_comparisons.append(
            {
                "candidate_world_size": world_size,
                "reference_world_size": 1,
                "pass": comparison_pass,
                "loss": {
                    "pass": losses_pass,
                    "maximum_absolute_difference": max(
                        row["absolute_difference"] for row in loss_rows
                    ),
                    "updates": loss_rows,
                },
                "parameter_delta": {
                    "pass": parameters_pass,
                    "maximum_absolute_difference": maximum,
                    "difference_l2": difference_l2,
                    "relative_l2_to_reference_update": relative_l2,
                    "maximum_absolute_tolerance": tolerances["parameter_max_abs"],
                    "relative_l2_tolerance": tolerances["parameter_relative_l2"],
                },
            }
        )

    base_throughput = _require_number(
        reference_result["throughput"]["median_prediction_tokens_per_second"],
        "world1 median throughput",
        minimum=0.0,
    )
    if base_throughput <= 0.0:
        raise PreflightContractError("world1 median throughput must be positive")
    throughput_rows: list[dict[str, Any]] = []
    for world_size in PREFLIGHT_WORLD_SIZES:
        median = _require_number(
            loaded[world_size][0]["throughput"]["median_prediction_tokens_per_second"],
            f"world{world_size} median throughput",
            minimum=0.0,
        )
        if median <= 0.0:
            raise PreflightContractError(f"world{world_size} median throughput must be positive")
        throughput_rows.append(
            {
                "world_size": world_size,
                "median_prediction_tokens_per_second": median,
                "speedup_vs_1_gpu": median / base_throughput,
                "parallel_efficiency_vs_1_gpu": median / (base_throughput * world_size),
            }
        )

    summary = {
        "schema_version": SUMMARY_VERSION,
        "preflight_only": True,
        "formal_campaign_gpu_count": FORMAL_CAMPAIGN_GPU_COUNT,
        "created_utc": _utc_now(),
        "manifest_sha256": _sha256_file(manifest_path),
        "overall_status": "PASS" if numerical_pass else "FAIL",
        "environment_contract": {
            "environment_lock_sha256": manifest["environment_lock"]["sha256"],
            "runtime_bootstrap_sha256": manifest["runtime_bootstrap"]["sha256"],
            "formal_runtime_profile_sha256": manifest["formal_runtime_profile"][
                "sha256"
            ],
            "orchestrator_validation_receipt_file_sha256": _sha256_file(
                orchestrator_environment_path
            ),
            "orchestrator_validation_receipt_semantic_sha256": (
                orchestrator_environment_semantic_sha256
            ),
            "worker_validation_receipt_sha256_by_world_size": {
                str(world_size): loaded[world_size][0]["environment_contract"][
                    "validation_receipt_sha256"
                ]
                for world_size in PREFLIGHT_WORLD_SIZES
            },
            "status": "PASS",
        },
        "numerical_equivalence": {
            "gate_role": "required",
            "status": "PASS" if numerical_pass else "FAIL",
            "tolerances": tolerances,
            "comparisons": numerical_comparisons,
        },
        "throughput": {
            "gate_role": "informational_only",
            "status": "MEASURED_NOT_GATED",
            "rows": throughput_rows,
        },
    }
    _atomic_json(summary_path.resolve(), summary)
    return summary


def orchestrate_preflight(manifest_path: Path, output_root: Path) -> dict[str, Any]:
    """Sequentially launch torchrun for 1/2/4 GPUs, then aggregate."""

    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    _check_checkout(manifest["code_commit"])
    _bind_compute_dtype_environment(manifest)
    orchestrator_environment = validate_bound_runtime_environment(
        manifest_path, manifest, gpu_count=FORMAL_CAMPAIGN_GPU_COUNT
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() < FORMAL_CAMPAIGN_GPU_COUNT:
        raise PreflightContractError("orchestrator requires one node with at least four visible GPUs")
    output_root = output_root.resolve()
    if output_root.exists():
        raise PreflightContractError(f"refusing to overwrite preflight results: {output_root}")
    output_root.mkdir(parents=True)
    _atomic_json(output_root / "orchestrator_environment.json", orchestrator_environment)
    environment = os.environ.copy()
    environment["NANOCHAT_DTYPE"] = manifest["runtime"]["compute_dtype"]
    for world_size in PREFLIGHT_WORLD_SIZES:
        staging = output_root / f"world{world_size}.partial"
        final = output_root / f"world{world_size}"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world_size}",
            "-m",
            "csc_fast_scaling_law.gpu_preflight",
            "run-worker",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(staging),
        ]
        subprocess.run(command, check=True, env=environment)
        if not (staging / "SUCCESS").is_file():
            raise PreflightContractError(f"world{world_size} worker returned without SUCCESS")
        os.replace(staging, final)
    return aggregate_preflight(manifest_path, output_root, output_root / "summary.json")


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkout", default=".")
    parser.add_argument("--code-commit")
    parser.add_argument("--environment-lock", required=True)
    parser.add_argument("--runtime-bootstrap", required=True)
    parser.add_argument("--formal-runtime-profile", required=True)
    parser.add_argument("--input-token-tensor")
    parser.add_argument("--token-seed", type=int, default=1729)
    parser.add_argument("--num-updates", type=int)
    parser.add_argument("--global-batch-sequences", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--init-seed", type=int, default=42)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-kv-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--window-pattern", default="L")
    parser.add_argument("--unembedding-lr", type=float, default=0.008)
    parser.add_argument("--embedding-lr", type=float, default=0.3)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.28)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--muon-momentum-warmup-steps", type=int, default=2)
    parser.add_argument("--warmdown-ratio", type=float, default=0.5)
    parser.add_argument("--final-lr-fraction", type=float, default=0.05)
    parser.add_argument(
        "--schedule-version",
        choices=("legacy_v0", "exact_last_update_v1"),
        default="exact_last_update_v1",
    )
    parser.add_argument("--device-batch-sequences", type=int, default=16)
    parser.add_argument("--timing-warmup-updates", type=int, default=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="materialize immutable global token batches")
    _add_prepare_arguments(prepare)
    worker = subparsers.add_parser("run-worker", help="run one torchrun world size")
    worker.add_argument("--manifest", required=True)
    worker.add_argument("--output-dir", required=True)
    aggregate = subparsers.add_parser("aggregate", help="compare completed 1/2/4-GPU runs")
    aggregate.add_argument("--manifest", required=True)
    aggregate.add_argument("--results-root", required=True)
    aggregate.add_argument("--summary", required=True)
    orchestrate = subparsers.add_parser("orchestrate", help="launch 1/2/4 GPU runs and aggregate")
    orchestrate.add_argument("--manifest", required=True)
    orchestrate.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        print(prepare_preflight(args))
        return 0
    if args.command == "run-worker":
        run_worker(Path(args.manifest), Path(args.output_dir))
        return 0
    if args.command == "aggregate":
        summary = aggregate_preflight(
            Path(args.manifest), Path(args.results_root), Path(args.summary)
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["overall_status"] == "PASS" else 1
    if args.command == "orchestrate":
        summary = orchestrate_preflight(Path(args.manifest), Path(args.output_root))
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["overall_status"] == "PASS" else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
