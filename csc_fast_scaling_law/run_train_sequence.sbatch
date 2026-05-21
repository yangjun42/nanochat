#!/usr/bin/env bash
# Sequential fallback for small grids when Slurm array submission hits job limits.
# Submit with the same exports as run_train_array.sbatch, but without --array.
# Optional: ROW_START=1 ROW_END=9 CONTINUE_ON_ERROR=0
#SBATCH --job-name=nanochat-fsl-train-seq
#SBATCH --account=project_2017828
#SBATCH --partition=gpumedium
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=4
#SBATCH --mem=480G
#SBATCH --output=%x-%j.out

set -uo pipefail

PROJECT="${CSC_PROJECT:-project_2017828}"
USER_NAME="${CSC_USER:-${USER:-yangjun1}}"
REPO_DIR="${NANOCHAT_CHECKOUT:-/scratch/${PROJECT}/${USER_NAME}/nanochat}"
PLAN_CSV="${PLAN_CSV:-${REPO_DIR}/csc_fast_scaling_law/inputs/smoke_plan.csv}"
ROW_START="${ROW_START:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

if [ ! -s "$PLAN_CSV" ]; then
  echo "ERROR: missing PLAN_CSV=$PLAN_CSV" >&2
  exit 2
fi

plan_rows="$(
python - "$PLAN_CSV" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
)"
ROW_END="${ROW_END:-$plan_rows}"

if [ "$ROW_START" -lt 1 ] || [ "$ROW_END" -lt "$ROW_START" ] || [ "$ROW_END" -gt "$plan_rows" ]; then
  echo "ERROR: invalid ROW_START=$ROW_START ROW_END=$ROW_END for $plan_rows plan rows" >&2
  exit 2
fi

export SLURM_ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
overall_rc=0

for task_id in $(seq "$ROW_START" "$ROW_END"); do
  export SLURM_ARRAY_TASK_ID="$task_id"
  echo "===== sequential train row ${task_id}/${plan_rows} START ====="
  bash "$REPO_DIR/csc_fast_scaling_law/run_train_array.sbatch"
  rc="$?"
  echo "===== sequential train row ${task_id}/${plan_rows} END rc=${rc} ====="
  if [ "$rc" -ne 0 ]; then
    overall_rc="$rc"
    if [ "$CONTINUE_ON_ERROR" != "1" ]; then
      exit "$rc"
    fi
  fi
done

exit "$overall_rc"
