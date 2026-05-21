#!/usr/bin/env bash
# Source this file and call: run_stage <stage> <command...>

run_stage() {
  local stage="$1"
  shift
  mkdir -p "${CSC_METRICS_DIR:?}" "${CSC_LOG_DIR:?}"
  local log_path="${CSC_LOG_DIR}/${stage}.log"
  local jsonl="${CSC_METRICS_DIR}/stage_metrics.jsonl"
  local gpu_csv="${CSC_METRICS_DIR}/gpu_metrics.csv"
  local host
  host="$(hostname)"

  if [ ! -s "$gpu_csv" ]; then
    printf 'stage,job_id,hostname,timestamp,gpu_index,gpu_name,util_gpu_pct,mem_used_mib,mem_total_mib,util_mem_pct,power_w,clock_sm_mhz\n' > "$gpu_csv"
  fi

  local start_ts end_ts start_s end_s rc sampler_pid command_str
  command_str="$*"
  start_ts="$(date -Is)"
  start_s="$(date +%s)"

  (
    while true; do
      if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,utilization.memory,power.draw,clocks.sm \
          --format=csv,noheader,nounits 2>/dev/null | \
        awk -v st="$stage" -v jid="${SLURM_JOB_ID:-manual}" -v hn="$host" 'BEGIN{FS=", *"; OFS=","} {print st,jid,hn,$1,$2,$3,$4,$5,$6,$7,$8,$9}' >> "$gpu_csv" || true
      fi
      sleep "${GPU_SAMPLE_INTERVAL_SECONDS:-5}"
    done
  ) &
  sampler_pid="$!"

  echo "===== ${stage} START ${start_ts} =====" | tee "$log_path"
  echo "COMMAND: $*" | tee -a "$log_path"
  "$@" >> "$log_path" 2>&1
  rc="$?"
  end_ts="$(date -Is)"
  end_s="$(date +%s)"
  kill "$sampler_pid" >/dev/null 2>&1 || true
  wait "$sampler_pid" >/dev/null 2>&1 || true
  echo "===== ${stage} END ${end_ts} rc=${rc} =====" | tee -a "$log_path"

  RUN_STAGE="$stage" RUN_STAGE_HOST="$host" RUN_STAGE_START="$start_ts" RUN_STAGE_END="$end_ts" RUN_STAGE_LOG="$log_path" RUN_STAGE_COMMAND="$command_str" RUN_STAGE_RC="$rc" RUN_STAGE_START_S="$start_s" RUN_STAGE_END_S="$end_s" python - "$jsonl" <<'PY'
import json
import os
import sys

row = {
    "stage": os.environ["RUN_STAGE"],
    "job_id": os.environ.get("SLURM_JOB_ID") or "manual",
    "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
    "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "partition": os.environ.get("SLURM_JOB_PARTITION"),
    "hostname": os.environ["RUN_STAGE_HOST"],
    "start_time": os.environ["RUN_STAGE_START"],
    "end_time": os.environ["RUN_STAGE_END"],
    "elapsed_seconds": int(os.environ["RUN_STAGE_END_S"]) - int(os.environ["RUN_STAGE_START_S"]),
    "exit_code": int(os.environ["RUN_STAGE_RC"]),
    "command": os.environ["RUN_STAGE_COMMAND"],
    "log_path": os.environ["RUN_STAGE_LOG"],
    "run_kind": os.environ.get("CSC_RUN_KIND"),
    "nanochat_base_dir": os.environ.get("NANOCHAT_BASE_DIR"),
    "depth": os.environ.get("CSC_DEPTH"),
    "recipe_band": os.environ.get("CSC_RECIPE_BAND"),
}
with open(sys.argv[1], "a", encoding="utf-8") as f:
    f.write(json.dumps(row, sort_keys=True) + "\n")
PY
  return "$rc"
}

