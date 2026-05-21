#!/usr/bin/env bash
set -euo pipefail

REMOTE="${ROIHU_HOST:-roihu-gpu}"
PROJECT="${CSC_PROJECT:-project_2017828}"
USER_NAME="${CSC_USER:-yangjun1}"
CAMPAIGN_ID="${CAMPAIGN_ID:?Set CAMPAIGN_ID to sync}"
CACHE_ROOT="${NANOCHAT_CACHE_ROOT:-/scratch/${PROJECT}/${USER_NAME}/nanochat-cache}"
LOCAL_OUT="${LOCAL_OUT:-/Users/yangju/Library/CloudStorage/OneDrive-UniversityofHelsinki/Projects/jun/fast-scaling-law-fitting-experiments/synthetic_poc/outputs/roihu/${CAMPAIGN_ID}}"

mkdir -p "$LOCAL_OUT"
rsync -av \
  "${REMOTE}:${CACHE_ROOT}/${CAMPAIGN_ID}/merged_metrics/" \
  "${LOCAL_OUT}/merged_metrics/"
rsync -av \
  "${REMOTE}:${CACHE_ROOT}/${CAMPAIGN_ID}/train_metrics/stage_metrics.jsonl" \
  "${REMOTE}:${CACHE_ROOT}/${CAMPAIGN_ID}/eval_metrics/stage_metrics.jsonl" \
  "${LOCAL_OUT}/" || true

echo "Synced ${CAMPAIGN_ID} to ${LOCAL_OUT}"
