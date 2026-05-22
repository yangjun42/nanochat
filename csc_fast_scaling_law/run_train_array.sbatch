#!/usr/bin/env bash
# Submit with: sbatch --array=1-2%1 csc_fast_scaling_law/run_train_array.sbatch
#SBATCH --job-name=nanochat-fsl-train
#SBATCH --account=project_2017828
#SBATCH --partition=gpumedium
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=4
#SBATCH --mem=480G
#SBATCH --output=%x-%A_%a.out

set -uo pipefail

PROJECT="${CSC_PROJECT:-project_2017828}"
USER_NAME="${CSC_USER:-${USER:-yangjun1}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SUBMIT_REPO_DIR="${SLURM_SUBMIT_DIR:-}"
if [ -n "$SUBMIT_REPO_DIR" ] && [ -d "${SUBMIT_REPO_DIR}/csc_fast_scaling_law" ]; then
  DEFAULT_REPO_DIR="$SUBMIT_REPO_DIR"
fi
REPO_DIR="${NANOCHAT_CHECKOUT:-${DEFAULT_REPO_DIR}}"
CACHE_ROOT="${NANOCHAT_CACHE_ROOT:-/scratch/${PROJECT}/${USER_NAME}/nanochat-cache}"
VENV_DIR="${NANOCHAT_VENV:-/scratch/${PROJECT}/venv/nanochat-pytorch210}"
CAMPAIGN_ID="${CAMPAIGN_ID:-fsl-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"
SHARED_DATA_DIR="${SHARED_DATA_DIR:-${CACHE_ROOT}/pretrain/base_data_climbmix}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${CACHE_ROOT}/${CAMPAIGN_ID}}"
export CSC_METRICS_DIR="${CSC_METRICS_DIR:-${NANOCHAT_BASE_DIR}/train_metrics}"
export CSC_LOG_DIR="${CSC_LOG_DIR:-${NANOCHAT_BASE_DIR}/train_logs}"
export CSC_RUN_KIND="${CSC_RUN_KIND:-fsl_train}"
export GPU_SAMPLE_INTERVAL_SECONDS="${GPU_SAMPLE_INTERVAL_SECONDS:-5}"
PLAN_CSV="${PLAN_CSV:-${REPO_DIR}/csc_fast_scaling_law/inputs/smoke_plan.csv}"
unset LD_PRELOAD

mkdir -p "$NANOCHAT_BASE_DIR" "$CSC_METRICS_DIR" "$CSC_LOG_DIR"
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

find_tokenizer_dir() {
  find "$CACHE_ROOT" -path "*/tokenizer/tokenizer.pkl" -print 2>/dev/null | sort -r | head -n 1 | sed 's#/tokenizer.pkl$##'
}

TOKENIZER_SOURCE_DIR="${TOKENIZER_SOURCE_DIR:-$(find_tokenizer_dir)}"
if [ -z "$TOKENIZER_SOURCE_DIR" ] || [ ! -s "${TOKENIZER_SOURCE_DIR}/tokenizer.pkl" ]; then
  echo "ERROR: no staged tokenizer found under $CACHE_ROOT" >&2
  exit 2
fi
if [ ! -d "$SHARED_DATA_DIR" ]; then
  echo "ERROR: missing staged dataset $SHARED_DATA_DIR" >&2
  exit 2
fi
if [ ! -e "${NANOCHAT_BASE_DIR}/base_data_climbmix" ]; then
  ln -s "$SHARED_DATA_DIR" "${NANOCHAT_BASE_DIR}/base_data_climbmix"
fi
if [ ! -e "${NANOCHAT_BASE_DIR}/tokenizer" ]; then
  ln -s "$TOKENIZER_SOURCE_DIR" "${NANOCHAT_BASE_DIR}/tokenizer"
fi

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
ASPECT_RATIO="${ASPECT_RATIO:-64}"
HEAD_DIM="${HEAD_DIM:-128}"
MODEL_DIM="${MODEL_DIM:-}"
checkpoint_dir="${NANOCHAT_BASE_DIR}/base_checkpoints/${MODEL_TAG}"
if [ -d "$checkpoint_dir" ] && find "$checkpoint_dir" -maxdepth 1 -type f | grep -q .; then
  echo "Checkpoint appears to exist, skipping: $checkpoint_dir"
  exit 0
fi

seed_args=()
if grep -q 'parser.add_argument("--seed"' scripts/base_train.py; then
  seed_args=(--seed "$SEED")
fi
fp8_args=()
if [ "${FP8:-0}" = "1" ]; then
  fp8_args=(--fp8)
fi
model_dim_args=()
if [ -n "${MODEL_DIM:-}" ] && [ "$MODEL_DIM" != "0" ] && [ "$MODEL_DIM" != "-1" ]; then
  model_dim_args=(--model-dim "$MODEL_DIM")
fi

set +e
run_stage "$STAGE" \
  torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    "${fp8_args[@]}" \
    --run=dummy \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --max-seq-len="$MAX_SEQ_LEN" \
    --window-pattern="$WINDOW_PATTERN" \
    --warmup-steps="$WARMUP_STEPS" \
    --eval-every=-1 \
    --sample-every=-1 \
    --core-metric-every=-1 \
    --save-every=-1 \
    --num-iterations="$NUM_ITERATIONS" \
    --depth="$DEPTH" \
    --aspect-ratio="$ASPECT_RATIO" \
    --head-dim="$HEAD_DIM" \
    "${model_dim_args[@]}" \
    "${seed_args[@]}"
rc="$?"
set -e

python -m csc_fast_scaling_law.collect_metrics \
  --plan "$PLAN_CSV" \
  --train-metrics-dir "$CSC_METRICS_DIR" \
  --eval-metrics-dir "${NANOCHAT_BASE_DIR}/eval_metrics" \
  --out-dir "${NANOCHAT_BASE_DIR}/merged_metrics" || true
exit "$rc"
