#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${1:?Usage: $0 <owner/physicalword-assets> [hf_token]}"
HF_TOKEN_ARG="${2:-${HF_TOKEN:-}}"

DATABASE_ROOT="/mnt/nas/zhangyiming/database"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [[ "${HF_USE_PROXY:-0}" == "1" ]]; then
  export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:10808}"
  export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:10808}"
  export http_proxy="${http_proxy:-$HTTP_PROXY}"
  export https_proxy="${https_proxy:-$HTTPS_PROXY}"
  unset ALL_PROXY all_proxy
else
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

retry() {
  local attempt=1
  local max_attempts="${HF_UPLOAD_RETRIES:-100}"
  local sleep_seconds="${HF_UPLOAD_RETRY_SLEEP:-60}"
  until "$@"; do
    local status=$?
    if (( attempt >= max_attempts )); then
      return "$status"
    fi
    echo "[WARN] upload attempt ${attempt}/${max_attempts} failed with exit code ${status}; retrying in ${sleep_seconds}s..."
    sleep "$sleep_seconds"
    attempt=$((attempt + 1))
  done
}

COMMON_ARGS=(--repo-type dataset)
if [[ -n "$HF_TOKEN_ARG" ]]; then
  COMMON_ARGS+=(--token "$HF_TOKEN_ARG")
fi

retry hf upload-large-folder "$REPO_ID" "$DATABASE_ROOT" \
  "${COMMON_ARGS[@]}" \
  --include "rlbench/**" \
  --exclude "rlbench/train/json/cosmos_text_cache_rlbench_keyframe/**" \
  --num-workers "${HF_UPLOAD_WORKERS:-8}" \
  --no-bars

retry hf upload-large-folder "$REPO_ID" "$DATABASE_ROOT" \
  "${COMMON_ARGS[@]}" \
  --include "rlbench/train/json/cosmos_text_cache_rlbench_keyframe/**" \
  --num-workers "${HF_UPLOAD_WORKERS:-8}" \
  --no-bars

echo "RLBench upload resumed/completed for: https://huggingface.co/datasets/$REPO_ID"
