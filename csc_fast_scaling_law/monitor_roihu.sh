#!/usr/bin/env bash
set -euo pipefail

REMOTE="${ROIHU_HOST:-roihu-gpu}"
PROJECT="${CSC_PROJECT:-project_2017828}"
USER_NAME="${CSC_USER:-yangjun1}"
CACHE_ROOT="${NANOCHAT_CACHE_ROOT:-/scratch/${PROJECT}/${USER_NAME}/nanochat-cache}"
CAMPAIGN_ID="${CAMPAIGN_ID:-}"

ssh "$REMOTE" "echo '== queue =='; squeue -u \"\$USER\" -o '%.18i %.12P %.35j %.2t %.10M %.10l %.6D %R'; echo '== partitions =='; sinfo -o '%.14P %.8a %.10l %.8D %.8t %.20G' | sed -n '1,30p'; echo '== recent sacct =='; sacct -u \"\$USER\" --starttime now-24hours --format=JobID,JobName%30,Partition,State,Elapsed,ExitCode -P | tail -40"

if [ -n "$CAMPAIGN_ID" ]; then
  ssh "$REMOTE" "base='${CACHE_ROOT}/${CAMPAIGN_ID}/merged_metrics'; if [ -s \"\$base/metrics.csv\" ]; then echo '== metrics tail =='; tail -5 \"\$base/metrics.csv\"; else echo 'No merged metrics yet at' \"\$base\"; fi"
fi

