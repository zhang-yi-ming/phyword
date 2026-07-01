#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${1:?Usage: $0 <owner/physicalword-assets> [hf_token]}"
HF_TOKEN_ARG="${2:-${HF_TOKEN:-}}"

DATABASE_ROOT="/mnt/nas/zhangyiming/database"
PROJECT_ROOT="/mnt/nas/zhangyiming/last05_beta/last05_mot2_action"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

COMMON_ARGS=(--repo-type dataset)
CREATE_ARGS=(--repo-type dataset --exist-ok)
if [[ -n "$HF_TOKEN_ARG" ]]; then
  COMMON_ARGS+=(--token "$HF_TOKEN_ARG")
  CREATE_ARGS+=(--token "$HF_TOKEN_ARG")
fi

hf repo create "$REPO_ID" "${CREATE_ARGS[@]}"

hf upload "$REPO_ID" "$PROJECT_ROOT/HF_ASSETS_README.md" README.md \
  "${COMMON_ARGS[@]}" \
  --commit-message "Add asset README"

hf upload "$REPO_ID" \
  "$DATABASE_ROOT/ckpt/pretrained/LaST0_Pretrain_AE_chunk16/tfmr" \
  "ckpt/pretrained/LaST0_Pretrain_AE_chunk16/tfmr" \
  "${COMMON_ARGS[@]}" \
  --commit-message "Upload action expert"

hf upload "$REPO_ID" \
  "$DATABASE_ROOT/data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual" \
  "data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual" \
  "${COMMON_ARGS[@]}" \
  --commit-message "Upload LIBERO spatial training data"

hf upload "$REPO_ID" "$DATABASE_ROOT/rlbench" "rlbench" \
  "${COMMON_ARGS[@]}" \
  --commit-message "Upload RLBench training data"

echo "Uploaded to: https://huggingface.co/datasets/$REPO_ID"
