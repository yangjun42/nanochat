#!/usr/bin/env bash
# Generic manifest-bound task wrapper. Resource flags are supplied by
# submit_contract_array.py so this file contains no account or machine identity.
#SBATCH --output=%x-%A_%a.out

set -uo pipefail

: "${CAMPAIGN_MANIFEST:?Set CAMPAIGN_MANIFEST}"
: "${ARTIFACT_ROOT:?Set ARTIFACT_ROOT}"
: "${NANOCHAT_CHECKOUT:?Set NANOCHAT_CHECKOUT}"
: "${TASK_RUNNER:?Set TASK_RUNNER}"
: "${GPU_COUNT:?Set GPU_COUNT}"

if [ -n "${RUNTIME_BOOTSTRAP:-}" ]; then
  : "${RUNTIME_BOOTSTRAP_SHA256:?Set RUNTIME_BOOTSTRAP_SHA256 with RUNTIME_BOOTSTRAP}"
  if command -v sha256sum >/dev/null 2>&1; then
    read -r observed_bootstrap_sha _ < <(sha256sum -- "${RUNTIME_BOOTSTRAP}")
  elif command -v shasum >/dev/null 2>&1; then
    read -r observed_bootstrap_sha _ < <(shasum -a 256 -- "${RUNTIME_BOOTSTRAP}")
  else
    echo "No SHA-256 utility is available to verify RUNTIME_BOOTSTRAP" >&2
    exit 2
  fi
  if [ "${observed_bootstrap_sha}" != "${RUNTIME_BOOTSTRAP_SHA256}" ]; then
    echo "RUNTIME_BOOTSTRAP hash mismatch; refusing to source it" >&2
    exit 2
  fi
  # shellcheck source=/dev/null
  source "${RUNTIME_BOOTSTRAP}"
fi

PYTHON_BIN="${NANOCHAT_PYTHON:-python3}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-1}"
cd "${NANOCHAT_CHECKOUT}" || exit 2

prepare_command=("${PYTHON_BIN}" -m csc_fast_scaling_law.campaign_contract prepare-task \
  --campaign-manifest "${CAMPAIGN_MANIFEST}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --task-index "${TASK_INDEX}" \
  --gpu-count "${GPU_COUNT}" \
  --runner "${TASK_RUNNER}" \
  --checkout "${NANOCHAT_CHECKOUT}" \
  --format shell)
if [ -n "${UNLOCK_MANIFEST:-}" ]; then
  prepare_command+=(--unlock-manifest "${UNLOCK_MANIFEST}")
fi

prepare_output="$("${prepare_command[@]}")" || exit "$?"
eval "${prepare_output}"

if [ "${CONTRACT_ACTION}" = "skip" ]; then
  echo "Validated complete task; skipping ${CONTRACT_TASK_ID}: ${CONTRACT_REASON}"
  exit 0
fi

export CONTRACT_PREPARED CONTRACT_STAGING_DIR CONTRACT_FINAL_DIR CONTRACT_TASK_ID
export NANOCHAT_BASE_DIR="${CONTRACT_STAGING_DIR}/work"
export CSC_METRICS_DIR="${CONTRACT_STAGING_DIR}/metrics"
export CSC_LOG_DIR="${CONTRACT_STAGING_DIR}/logs"
mkdir -p "${NANOCHAT_BASE_DIR}" "${CSC_METRICS_DIR}" "${CSC_LOG_DIR}"

record_signal_failure() {
  local signal_name="$1"
  "${PYTHON_BIN}" -m csc_fast_scaling_law.campaign_contract record-failure \
    --prepared "${CONTRACT_PREPARED}" \
    --exit-code 143 \
    --reason "received ${signal_name}" >/dev/null 2>&1 || true
}
trap 'record_signal_failure TERM' TERM
trap 'record_signal_failure INT' INT

set +e
"${TASK_RUNNER}" "${CONTRACT_PREPARED}"
runner_rc="$?"
set -e

if [ "${runner_rc}" -ne 0 ]; then
  "${PYTHON_BIN}" -m csc_fast_scaling_law.campaign_contract record-failure \
    --prepared "${CONTRACT_PREPARED}" \
    --exit-code "${runner_rc}" \
    --reason "task runner returned non-zero"
  exit "${runner_rc}"
fi

"${PYTHON_BIN}" -m csc_fast_scaling_law.campaign_contract finalize-task \
  --prepared "${CONTRACT_PREPARED}" \
  --exit-code 0
