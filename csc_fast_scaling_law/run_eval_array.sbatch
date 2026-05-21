#!/usr/bin/env bash
# Submit after training, with the same CAMPAIGN_ID/NANOCHAT_BASE_DIR.
#SBATCH --job-name=nanochat-fsl-bpb
#SBATCH --account=project_2017828
#SBATCH --partition=gpumedium
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=4
#SBATCH --mem=480G
#SBATCH --output=%x-%A_%a.out

set -uo pipefail

PROJECT="${CSC_PROJECT:-project_2017828}"
USER_NAME="${CSC_USER:-${USER:-yangjun1}}"
REPO_DIR="${NANOCHAT_CHECKOUT:-/scratch/${PROJECT}/${USER_NAME}/nanochat}"
CACHE_ROOT="${NANOCHAT_CACHE_ROOT:-/scratch/${PROJECT}/${USER_NAME}/nanochat-cache}"
VENV_DIR="${NANOCHAT_VENV:-/scratch/${PROJECT}/venv/nanochat-pytorch210}"
CAMPAIGN_ID="${CAMPAIGN_ID:-fsl-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${CACHE_ROOT}/${CAMPAIGN_ID}}"
PLAN_CSV="${PLAN_CSV:-${REPO_DIR}/csc_fast_scaling_law/inputs/smoke_plan.csv}"
TRAIN_METRICS_DIR="${TRAIN_METRICS_DIR:-${NANOCHAT_BASE_DIR}/train_metrics}"
export CSC_METRICS_DIR="${CSC_METRICS_DIR:-${NANOCHAT_BASE_DIR}/eval_metrics}"
export CSC_LOG_DIR="${CSC_LOG_DIR:-${NANOCHAT_BASE_DIR}/eval_logs}"
export CSC_RUN_KIND="${CSC_RUN_KIND:-fsl_bpb_eval}"
export GPU_SAMPLE_INTERVAL_SECONDS="${GPU_SAMPLE_INTERVAL_SECONDS:-5}"
SPLIT_TOKENS="${SPLIT_TOKENS:-33554432}"
EVAL_DEVICE_BATCH_SIZE="${EVAL_DEVICE_BATCH_SIZE:-16}"
unset LD_PRELOAD

mkdir -p "$CSC_METRICS_DIR" "$CSC_LOG_DIR"
cd "$REPO_DIR" || exit 2

export LMOD_MODULEPATH_INIT="/appl/modulefiles/.modulepath_init_$(uname -m)"
source /usr/share/lmod/9.1.2/init/profile
module --initial_load --no_redirect restore >/dev/null 2>&1 || true
module purge
module load python-pytorch/2.10

if [ ! -d "$VENV_DIR" ]; then
  echo "ERROR: missing venv $VENV_DIR" >&2
  exit 2
fi
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR/csc_fast_scaling_law/wandb_stub:$REPO_DIR:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

. "$REPO_DIR/csc_fast_scaling_law/run_stage.sh"

if [ ! -s "$PLAN_CSV" ]; then
  echo "ERROR: missing PLAN_CSV=$PLAN_CSV" >&2
  exit 2
fi

task_id="${SLURM_ARRAY_TASK_ID:-1}"
eval "$(
python - "$PLAN_CSV" "$task_id" <<'PY'
import csv
import shlex
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
idx = int(sys.argv[2]) - 1
if idx < 0 or idx >= len(rows):
    raise SystemExit(f"array index {idx+1} out of range 1..{len(rows)}")
for key, val in rows[idx].items():
    print(f"{key.upper()}={shlex.quote(str(val))}")
PY
)"

export CSC_DEPTH="$DEPTH"
export CSC_RECIPE_BAND="$RECIPE_BAND"
checkpoint_dir="${NANOCHAT_BASE_DIR}/base_checkpoints/${MODEL_TAG}"
if [ ! -d "$checkpoint_dir" ]; then
  echo "ERROR: missing checkpoint $checkpoint_dir" >&2
  exit 3
fi

eval_stage="bpb_${STAGE}"
set +e
run_stage "$eval_stage" \
  torchrun --standalone --nproc_per_node=4 -m scripts.base_eval \
    --eval=bpb \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$EVAL_DEVICE_BATCH_SIZE" \
    --split-tokens="$SPLIT_TOKENS"
rc="$?"
set -e

python -m csc_fast_scaling_law.collect_metrics \
  --plan "$PLAN_CSV" \
  --train-metrics-dir "$TRAIN_METRICS_DIR" \
  --eval-metrics-dir "$CSC_METRICS_DIR" \
  --out-dir "${NANOCHAT_BASE_DIR}/merged_metrics" || true
exit "$rc"
