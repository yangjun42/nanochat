#!/usr/bin/env bash
set -euo pipefail

: "${CAMPAIGN_ID:?Set CAMPAIGN_ID to an isolated campaign name.}"
: "${PLAN_CSV:?Set PLAN_CSV to the absolute path of the plan CSV.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${NANOCHAT_CHECKOUT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT="${CSC_PROJECT:-project_2017828}"
USER_NAME="${CSC_USER:-${USER}}"
CACHE_ROOT="${NANOCHAT_CACHE_ROOT:-/scratch/${PROJECT}/${USER_NAME}/nanochat-cache}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${CACHE_ROOT}/${CAMPAIGN_ID}}"
ROWS_PER_JOB="${ROWS_PER_JOB:-30}"
JOB_LABEL="${JOB_LABEL:-fsl-plan}"
PARTITION="${PARTITION:-gpumedium}"
TRAIN_TIME="${TRAIN_TIME:-06:00:00}"
EVAL_TIME="${EVAL_TIME:-03:00:00}"
LOG_DIR="${LOG_DIR:-${NANOCHAT_BASE_DIR}/submit_logs}"

if [[ ! -s "${PLAN_CSV}" ]]; then
  echo "ERROR: missing PLAN_CSV=${PLAN_CSV}" >&2
  exit 2
fi
if [[ ! "${ROWS_PER_JOB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: ROWS_PER_JOB must be a positive integer" >&2
  exit 2
fi

plan_rows="$(python - "${PLAN_CSV}" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)"

mkdir -p "${LOG_DIR}"
train_script="${REPO_DIR}/csc_fast_scaling_law/run_train_sequence.sbatch"
eval_script="${REPO_DIR}/csc_fast_scaling_law/run_eval_sequence.sbatch"

printf 'campaign\trows\ttrain_job\teval_job\n'
for ((start = 1; start <= plan_rows; start += ROWS_PER_JOB)); do
  end=$((start + ROWS_PER_JOB - 1))
  if ((end > plan_rows)); then
    end="${plan_rows}"
  fi

  exports="ALL,CAMPAIGN_ID=${CAMPAIGN_ID},PLAN_CSV=${PLAN_CSV},NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR},NANOCHAT_CHECKOUT=${REPO_DIR},ROW_START=${start},ROW_END=${end},CONTINUE_ON_ERROR=0"
  train_job="$(
    sbatch --parsable \
      --account="${PROJECT}" \
      --partition="${PARTITION}" \
      --time="${TRAIN_TIME}" \
      --job-name="${JOB_LABEL}-t-${start}-${end}" \
      --chdir="${REPO_DIR}" \
      --output="${LOG_DIR}/%x-%j.out" \
      --export="${exports}" \
      "${train_script}"
  )"
  eval_job="$(
    sbatch --parsable \
      --account="${PROJECT}" \
      --partition="${PARTITION}" \
      --time="${EVAL_TIME}" \
      --dependency="afterok:${train_job}" \
      --job-name="${JOB_LABEL}-e-${start}-${end}" \
      --chdir="${REPO_DIR}" \
      --output="${LOG_DIR}/%x-%j.out" \
      --export="${exports}" \
      "${eval_script}"
  )"
  printf '%s\t%s-%s\t%s\t%s\n' \
    "${CAMPAIGN_ID}" "${start}" "${end}" "${train_job}" "${eval_job}"
done
