#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/test_libero_new_batch_worker.sh"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-40}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-4}"
JOB_POLL_SECONDS="${JOB_POLL_SECONDS:-5}"
DRY_RUN="${DRY_RUN:-false}"

# Edit these rows to define the batch. CHECKPOINTS can contain checkpoint
# directories or .pt files accepted by test_libero_new_batch_worker.sh.
# Leave an entry empty to let the worker select the latest checkpoint from RUN_NAME.
CHECKPOINTS=(
  ""
)

# Arrays below support either one value (broadcast to every checkpoint) or
# exactly one value per CHECKPOINTS entry.
RUN_NAME_VALUES=("cosmos2B_action1B_mot2_libero_spatial")
CHECKPOINT_NAME_VALUES=("")
NUM_TRIALS_PER_TASK_VALUES=(10)
ACTION_SELF_CAUSAL_IN_BRIDGE_VALUES=("true")
STATE_ENCODING_MODE_VALUES=("mlp")
USE_VALUE_PREDICTION_VALUES=("false")
USE_ACTION_VALUE_PREDICTION_VALUES=("false")
ACTION_VALUE_LOSS_WEIGHT_VALUES=("1.0")
VALUE_TOKEN_MASK_VIDEO_TO_VALUE_VALUES=("false")
VALUE_TOKEN_MASK_NONVALUE_TO_VALUE_VALUES=("false")
BRIDGE_POS_SCHEME_VALUES=("llama1d")
DECOSMOS_VALUES=("false")
COSMOS_SELF_ONLY_BRIDGE_VALUES=("false")
REWRITE_EVAL_PROMPT_VALUES=("false")
COSMOS_TEXT_CACHE_PATH_VALUES=("/mnt/nas/zhangyiming/database/data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual/cosmos_text_cache_raw_full_concat")
COSMOS_DENOISE_STEPS_VALUES=(2)
NUM_OPEN_LOOP_STEPS_VALUES=(8)
ROBOT_STATE_VALUES=(0)
SEED_VALUES=(0)
STATE_PLACEHOLDER_TOKENS_VALUES=(8)
ACTION_INTERMEDIATE_SIZE_VALUES=(0)
ACTION_USE_LATENT_PREFIX_VALUES=("true")
IMG_LATENTS_PER_FUTURE_VALUES=(0)
STATE_LATENTS_PER_FUTURE_VALUES=(0)
NUM_FUTURE_FRAMES_VALUES=(0)
FUTURE_FRAME_STRIDE_VALUES=(8)
VIDEO_FRAMES_VALUES=(1)
NUM_COND_INPUT_FRAMES_VALUES=(1)
ACTION_CHUNK_VALUES=(16)
FPS_VALUES=(20)
CONTROL_FREQ_VALUES=(0)
ACTION_REPEAT_VALUES=(1)
TASK_SUITE_NAME_VALUES=("libero_spatial")

RUN_COUNT="${#CHECKPOINTS[@]}"
if [[ "$RUN_COUNT" -le 0 ]]; then
  echo "[ERROR] CHECKPOINTS must contain at least one entry."
  exit 1
fi

is_true() {
  case "$1" in
    1|true|TRUE|True|yes|YES|Yes|y|Y|on|ON|On) return 0 ;;
    *) return 1 ;;
  esac
}

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -lt 1 ]]; then
    echo "[ERROR] ${name} must be a positive integer, got: ${value}"
    exit 1
  fi
}

array_len() {
  local array_name="$1"
  eval "printf '%s' \"\${#${array_name}[@]}\""
}

array_value() {
  local array_name="$1"
  local idx="$2"
  local len
  len="$(array_len "$array_name")"
  if [[ "$len" -eq 1 ]]; then
    eval "printf '%s' \"\${${array_name}[0]}\""
  else
    eval "printf '%s' \"\${${array_name}[$idx]}\""
  fi
}

validate_broadcast_array() {
  local array_name="$1"
  local len
  len="$(array_len "$array_name")"
  if [[ "$len" -ne 1 && "$len" -ne "$RUN_COUNT" ]]; then
    echo "[ERROR] ${array_name} must contain either 1 entry or exactly ${RUN_COUNT} entries, got ${len}."
    exit 1
  fi
}

require_positive_int "MAX_PARALLEL_JOBS" "$MAX_PARALLEL_JOBS"
require_positive_int "CPU_THREADS_PER_JOB" "$CPU_THREADS_PER_JOB"
require_positive_int "JOB_POLL_SECONDS" "$JOB_POLL_SECONDS"

IFS=',' read -r -a AVAILABLE_GPU_IDS <<< "$GPU_IDS"
if [[ "${#AVAILABLE_GPU_IDS[@]}" -le 0 ]]; then
  echo "[ERROR] GPU_IDS must contain at least one GPU id, got: ${GPU_IDS}"
  exit 1
fi

for gpu_idx in "${!AVAILABLE_GPU_IDS[@]}"; do
  gpu_id="${AVAILABLE_GPU_IDS[$gpu_idx]//[[:space:]]/}"
  if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] GPU_IDS must be a comma-separated list of numeric GPU ids, got: ${GPU_IDS}"
    exit 1
  fi
  AVAILABLE_GPU_IDS[$gpu_idx]="$gpu_id"
done

if [[ "$MAX_PARALLEL_JOBS" -gt "${#AVAILABLE_GPU_IDS[@]}" ]]; then
  echo "[WARN] MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS} is larger than GPU count=${#AVAILABLE_GPU_IDS[@]}; some GPUs may run multiple eval jobs."
fi

for array_name in \
  CHECKPOINTS \
  RUN_NAME_VALUES \
  CHECKPOINT_NAME_VALUES \
  NUM_TRIALS_PER_TASK_VALUES \
  ACTION_SELF_CAUSAL_IN_BRIDGE_VALUES \
  STATE_ENCODING_MODE_VALUES \
  USE_VALUE_PREDICTION_VALUES \
  USE_ACTION_VALUE_PREDICTION_VALUES \
  ACTION_VALUE_LOSS_WEIGHT_VALUES \
  VALUE_TOKEN_MASK_VIDEO_TO_VALUE_VALUES \
  VALUE_TOKEN_MASK_NONVALUE_TO_VALUE_VALUES \
  BRIDGE_POS_SCHEME_VALUES \
  DECOSMOS_VALUES \
  COSMOS_SELF_ONLY_BRIDGE_VALUES \
  REWRITE_EVAL_PROMPT_VALUES \
  COSMOS_TEXT_CACHE_PATH_VALUES \
  COSMOS_DENOISE_STEPS_VALUES \
  NUM_OPEN_LOOP_STEPS_VALUES \
  ROBOT_STATE_VALUES \
  SEED_VALUES \
  STATE_PLACEHOLDER_TOKENS_VALUES \
  ACTION_INTERMEDIATE_SIZE_VALUES \
  ACTION_USE_LATENT_PREFIX_VALUES \
  IMG_LATENTS_PER_FUTURE_VALUES \
  STATE_LATENTS_PER_FUTURE_VALUES \
  NUM_FUTURE_FRAMES_VALUES \
  FUTURE_FRAME_STRIDE_VALUES \
  VIDEO_FRAMES_VALUES \
  NUM_COND_INPUT_FRAMES_VALUES \
  ACTION_CHUNK_VALUES \
  FPS_VALUES \
  CONTROL_FREQ_VALUES \
  ACTION_REPEAT_VALUES \
  TASK_SUITE_NAME_VALUES
do
  validate_broadcast_array "$array_name"
done

if [[ ! -f "$TEST_SCRIPT" ]]; then
  echo "[ERROR] Cannot find worker script: ${TEST_SCRIPT}"
  exit 1
fi

failed=0
RUNNING_PIDS=()
RUNNING_GPUS=()

cleanup_finished_jobs() {
  local kept_pids=()
  local kept_gpus=()
  local idx pid gpu

  for idx in "${!RUNNING_PIDS[@]}"; do
    pid="${RUNNING_PIDS[$idx]}"
    gpu="${RUNNING_GPUS[$idx]}"
    if kill -0 "$pid" 2>/dev/null; then
      kept_pids+=("$pid")
      kept_gpus+=("$gpu")
    else
      if ! wait "$pid"; then
        failed=1
      fi
    fi
  done

  if [[ "${#kept_pids[@]}" -gt 0 ]]; then
    RUNNING_PIDS=("${kept_pids[@]}")
    RUNNING_GPUS=("${kept_gpus[@]}")
  else
    RUNNING_PIDS=()
    RUNNING_GPUS=()
  fi
}

gpu_active_count() {
  local target_gpu="$1"
  local count=0
  local running_gpu

  if [[ "${#RUNNING_GPUS[@]}" -eq 0 ]]; then
    echo 0
    return
  fi

  for running_gpu in "${RUNNING_GPUS[@]}"; do
    if [[ "$running_gpu" == "$target_gpu" ]]; then
      count=$((count + 1))
    fi
  done

  echo "$count"
}

select_gpu() {
  local best_gpu="${AVAILABLE_GPU_IDS[0]}"
  local best_count
  local gpu count

  best_count="$(gpu_active_count "$best_gpu")"
  for gpu in "${AVAILABLE_GPU_IDS[@]}"; do
    count="$(gpu_active_count "$gpu")"
    if [[ "$count" -lt "$best_count" ]]; then
      best_gpu="$gpu"
      best_count="$count"
    fi
  done

  echo "$best_gpu"
}

wait_for_available_slot() {
  while true; do
    cleanup_finished_jobs
    if [[ "${#RUNNING_PIDS[@]}" -lt "$MAX_PARALLEL_JOBS" ]]; then
      break
    fi
    sleep "$JOB_POLL_SECONDS"
  done
}

validate_run_config() {
  local idx="$1"
  local state_encoding_mode bridge_pos_scheme use_value_prediction decosmos num_trials cosmos_steps open_loop_steps video_frames num_cond_input_frames

  state_encoding_mode="$(array_value STATE_ENCODING_MODE_VALUES "$idx")"
  if [[ "$state_encoding_mode" != "token" && "$state_encoding_mode" != "mlp" ]]; then
    echo "[ERROR] STATE_ENCODING_MODE_VALUES[$idx] must be token or mlp, got: ${state_encoding_mode}"
    exit 1
  fi

  bridge_pos_scheme="$(array_value BRIDGE_POS_SCHEME_VALUES "$idx")"
  case "$bridge_pos_scheme" in
    mrope|mrope_interleave|llama1d|local|last0) ;;
    *)
      echo "[ERROR] BRIDGE_POS_SCHEME_VALUES[$idx] must be mrope, mrope_interleave, llama1d, local, or last0; got: ${bridge_pos_scheme}"
      exit 1
      ;;
  esac

  use_value_prediction="$(array_value USE_VALUE_PREDICTION_VALUES "$idx")"
  decosmos="$(array_value DECOSMOS_VALUES "$idx")"
  if is_true "$use_value_prediction" && is_true "$decosmos"; then
    echo "[ERROR] USE_VALUE_PREDICTION_VALUES[$idx]=true requires DECOSMOS_VALUES[$idx]=false."
    exit 1
  fi

  num_trials="$(array_value NUM_TRIALS_PER_TASK_VALUES "$idx")"
  cosmos_steps="$(array_value COSMOS_DENOISE_STEPS_VALUES "$idx")"
  open_loop_steps="$(array_value NUM_OPEN_LOOP_STEPS_VALUES "$idx")"
  video_frames="$(array_value VIDEO_FRAMES_VALUES "$idx")"
  num_cond_input_frames="$(array_value NUM_COND_INPUT_FRAMES_VALUES "$idx")"
  require_positive_int "NUM_TRIALS_PER_TASK_VALUES[$idx]" "$num_trials"
  require_positive_int "COSMOS_DENOISE_STEPS_VALUES[$idx]" "$cosmos_steps"
  require_positive_int "NUM_OPEN_LOOP_STEPS_VALUES[$idx]" "$open_loop_steps"
  require_positive_int "VIDEO_FRAMES_VALUES[$idx]" "$video_frames"
  require_positive_int "NUM_COND_INPUT_FRAMES_VALUES[$idx]" "$num_cond_input_frames"
  if [[ "$num_cond_input_frames" -gt "$video_frames" ]]; then
    echo "[ERROR] NUM_COND_INPUT_FRAMES_VALUES[$idx] must be <= VIDEO_FRAMES_VALUES[$idx], got ${num_cond_input_frames} > ${video_frames}."
    exit 1
  fi
}

print_run_config() {
  local idx="$1"
  local run_id="$2"
  local assigned_gpu="$3"

  echo "============================================================"
  echo "[INFO] Starting LIBERO eval run ${run_id}/${RUN_COUNT} (max_parallel=${MAX_PARALLEL_JOBS}, gpu=${assigned_gpu}, dry_run=${DRY_RUN})"
  echo "[INFO] checkpoint=$(array_value CHECKPOINTS "$idx")"
  echo "[INFO] run_name=$(array_value RUN_NAME_VALUES "$idx")"
  echo "[INFO] num_trials_per_task=$(array_value NUM_TRIALS_PER_TASK_VALUES "$idx")"
  echo "[INFO] action_self_causal_in_bridge=$(array_value ACTION_SELF_CAUSAL_IN_BRIDGE_VALUES "$idx")"
  echo "[INFO] state_encoding_mode=$(array_value STATE_ENCODING_MODE_VALUES "$idx")"
  echo "[INFO] use_value_prediction=$(array_value USE_VALUE_PREDICTION_VALUES "$idx")"
  echo "[INFO] use_action_value_prediction=$(array_value USE_ACTION_VALUE_PREDICTION_VALUES "$idx")"
  echo "[INFO] action_value_loss_weight=$(array_value ACTION_VALUE_LOSS_WEIGHT_VALUES "$idx")"
  echo "[INFO] value_token_mask_video_to_value=$(array_value VALUE_TOKEN_MASK_VIDEO_TO_VALUE_VALUES "$idx")"
  echo "[INFO] value_token_mask_nonvalue_to_value=$(array_value VALUE_TOKEN_MASK_NONVALUE_TO_VALUE_VALUES "$idx")"
  echo "[INFO] bridge_pos_scheme=$(array_value BRIDGE_POS_SCHEME_VALUES "$idx")"
  echo "[INFO] decosmos=$(array_value DECOSMOS_VALUES "$idx")"
  echo "[INFO] cosmos_denoise_steps=$(array_value COSMOS_DENOISE_STEPS_VALUES "$idx")"
  echo "[INFO] num_open_loop_steps=$(array_value NUM_OPEN_LOOP_STEPS_VALUES "$idx")"
  echo "[INFO] video_frames=$(array_value VIDEO_FRAMES_VALUES "$idx")"
  echo "[INFO] num_cond_input_frames=$(array_value NUM_COND_INPUT_FRAMES_VALUES "$idx")"
  echo "[INFO] robot_state=$(array_value ROBOT_STATE_VALUES "$idx")"
  echo "[INFO] seed=$(array_value SEED_VALUES "$idx")"
  echo "[INFO] CUDA_VISIBLE_DEVICES=${assigned_gpu}"
  echo "[INFO] cpu_threads_per_job=${CPU_THREADS_PER_JOB}"
  echo "============================================================"
}

print_dry_run_env() {
  local idx="$1"
  local assigned_gpu="$2"
  local run_stamp="$3"

  echo "[DRY_RUN] Worker environment:"
  echo "  CUDA_VISIBLE_DEVICES=${assigned_gpu}"
  echo "  EGL_DEVICE_ID=${assigned_gpu}"
  echo "  CUDA_DEVICE=0"
  echo "  CPU_THREADS_PER_JOB=${CPU_THREADS_PER_JOB}"
  echo "  PRETRAINED_CHECKPOINT=$(array_value CHECKPOINTS "$idx")"
  echo "  RUN_NAME=$(array_value RUN_NAME_VALUES "$idx")"
  echo "  CHECKPOINT_NAME=$(array_value CHECKPOINT_NAME_VALUES "$idx")"
  echo "  NUM_TRIALS_PER_TASK=$(array_value NUM_TRIALS_PER_TASK_VALUES "$idx")"
  echo "  ACTION_SELF_CAUSAL_IN_BRIDGE=$(array_value ACTION_SELF_CAUSAL_IN_BRIDGE_VALUES "$idx")"
  echo "  STATE_ENCODING_MODE=$(array_value STATE_ENCODING_MODE_VALUES "$idx")"
  echo "  USE_VALUE_PREDICTION=$(array_value USE_VALUE_PREDICTION_VALUES "$idx")"
  echo "  USE_ACTION_VALUE_PREDICTION=$(array_value USE_ACTION_VALUE_PREDICTION_VALUES "$idx")"
  echo "  ACTION_VALUE_LOSS_WEIGHT=$(array_value ACTION_VALUE_LOSS_WEIGHT_VALUES "$idx")"
  echo "  VALUE_TOKEN_MASK_VIDEO_TO_VALUE=$(array_value VALUE_TOKEN_MASK_VIDEO_TO_VALUE_VALUES "$idx")"
  echo "  VALUE_TOKEN_MASK_NONVALUE_TO_VALUE=$(array_value VALUE_TOKEN_MASK_NONVALUE_TO_VALUE_VALUES "$idx")"
  echo "  BRIDGE_POS_SCHEME=$(array_value BRIDGE_POS_SCHEME_VALUES "$idx")"
  echo "  DECOSMOS=$(array_value DECOSMOS_VALUES "$idx")"
  echo "  COSMOS_SELF_ONLY_BRIDGE=$(array_value COSMOS_SELF_ONLY_BRIDGE_VALUES "$idx")"
  echo "  REWRITE_EVAL_PROMPT=$(array_value REWRITE_EVAL_PROMPT_VALUES "$idx")"
  echo "  COSMOS_TEXT_CACHE_PATH=$(array_value COSMOS_TEXT_CACHE_PATH_VALUES "$idx")"
  echo "  COSMOS_DENOISE_STEPS=$(array_value COSMOS_DENOISE_STEPS_VALUES "$idx")"
  echo "  NUM_OPEN_LOOP_STEPS=$(array_value NUM_OPEN_LOOP_STEPS_VALUES "$idx")"
  echo "  VIDEO_FRAMES=$(array_value VIDEO_FRAMES_VALUES "$idx")"
  echo "  NUM_COND_INPUT_FRAMES=$(array_value NUM_COND_INPUT_FRAMES_VALUES "$idx")"
  echo "  ROBOT_STATE=$(array_value ROBOT_STATE_VALUES "$idx")"
  echo "  SEED=$(array_value SEED_VALUES "$idx")"
  echo "  RUN_STAMP=${run_stamp}"
  echo "  RUN_ID_NOTE=${run_stamp}"
}

trap 'echo "[ERROR] Interrupted; stopping running LIBERO eval jobs."; for pid in "${RUNNING_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; exit 130' INT TERM

for idx in $(seq 0 $((RUN_COUNT - 1))); do
  validate_run_config "$idx"
done

for idx in $(seq 0 $((RUN_COUNT - 1))); do
  wait_for_available_slot

  run_id=$((idx + 1))
  assigned_gpu="$(select_gpu)"
  run_stamp="batch_run_${run_id}_$(date +%Y_%m_%d-%H_%M_%S)_pid$$_job${idx}"
  print_run_config "$idx" "$run_id" "$assigned_gpu"

  if is_true "$DRY_RUN"; then
    echo "[DRY_RUN] Would launch ${TEST_SCRIPT} with RUN_STAMP=${run_stamp}"
    print_dry_run_env "$idx" "$assigned_gpu" "$run_stamp"
    continue
  fi

  CUDA_VISIBLE_DEVICES="$assigned_gpu" \
  EGL_DEVICE_ID="$assigned_gpu" \
  CUDA_DEVICE="0" \
  CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
  PRETRAINED_CHECKPOINT="$(array_value CHECKPOINTS "$idx")" \
  RUN_NAME="$(array_value RUN_NAME_VALUES "$idx")" \
  CHECKPOINT_NAME="$(array_value CHECKPOINT_NAME_VALUES "$idx")" \
  NUM_TRIALS_PER_TASK="$(array_value NUM_TRIALS_PER_TASK_VALUES "$idx")" \
  ACTION_SELF_CAUSAL_IN_BRIDGE="$(array_value ACTION_SELF_CAUSAL_IN_BRIDGE_VALUES "$idx")" \
  STATE_ENCODING_MODE="$(array_value STATE_ENCODING_MODE_VALUES "$idx")" \
  USE_VALUE_PREDICTION="$(array_value USE_VALUE_PREDICTION_VALUES "$idx")" \
  USE_ACTION_VALUE_PREDICTION="$(array_value USE_ACTION_VALUE_PREDICTION_VALUES "$idx")" \
  ACTION_VALUE_LOSS_WEIGHT="$(array_value ACTION_VALUE_LOSS_WEIGHT_VALUES "$idx")" \
  VALUE_TOKEN_MASK_VIDEO_TO_VALUE="$(array_value VALUE_TOKEN_MASK_VIDEO_TO_VALUE_VALUES "$idx")" \
  VALUE_TOKEN_MASK_NONVALUE_TO_VALUE="$(array_value VALUE_TOKEN_MASK_NONVALUE_TO_VALUE_VALUES "$idx")" \
  BRIDGE_POS_SCHEME="$(array_value BRIDGE_POS_SCHEME_VALUES "$idx")" \
  DECOSMOS="$(array_value DECOSMOS_VALUES "$idx")" \
  COSMOS_SELF_ONLY_BRIDGE="$(array_value COSMOS_SELF_ONLY_BRIDGE_VALUES "$idx")" \
  REWRITE_EVAL_PROMPT="$(array_value REWRITE_EVAL_PROMPT_VALUES "$idx")" \
  COSMOS_TEXT_CACHE_PATH="$(array_value COSMOS_TEXT_CACHE_PATH_VALUES "$idx")" \
  COSMOS_DENOISE_STEPS="$(array_value COSMOS_DENOISE_STEPS_VALUES "$idx")" \
  NUM_OPEN_LOOP_STEPS="$(array_value NUM_OPEN_LOOP_STEPS_VALUES "$idx")" \
  ROBOT_STATE="$(array_value ROBOT_STATE_VALUES "$idx")" \
  SEED="$(array_value SEED_VALUES "$idx")" \
  STATE_PLACEHOLDER_TOKENS="$(array_value STATE_PLACEHOLDER_TOKENS_VALUES "$idx")" \
  ACTION_INTERMEDIATE_SIZE="$(array_value ACTION_INTERMEDIATE_SIZE_VALUES "$idx")" \
  ACTION_USE_LATENT_PREFIX="$(array_value ACTION_USE_LATENT_PREFIX_VALUES "$idx")" \
  IMG_LATENTS_PER_FUTURE="$(array_value IMG_LATENTS_PER_FUTURE_VALUES "$idx")" \
  STATE_LATENTS_PER_FUTURE="$(array_value STATE_LATENTS_PER_FUTURE_VALUES "$idx")" \
  NUM_FUTURE_FRAMES="$(array_value NUM_FUTURE_FRAMES_VALUES "$idx")" \
  FUTURE_FRAME_STRIDE="$(array_value FUTURE_FRAME_STRIDE_VALUES "$idx")" \
  VIDEO_FRAMES="$(array_value VIDEO_FRAMES_VALUES "$idx")" \
  NUM_COND_INPUT_FRAMES="$(array_value NUM_COND_INPUT_FRAMES_VALUES "$idx")" \
  ACTION_CHUNK="$(array_value ACTION_CHUNK_VALUES "$idx")" \
  FPS="$(array_value FPS_VALUES "$idx")" \
  CONTROL_FREQ="$(array_value CONTROL_FREQ_VALUES "$idx")" \
  ACTION_REPEAT="$(array_value ACTION_REPEAT_VALUES "$idx")" \
  TASK_SUITE_NAME="$(array_value TASK_SUITE_NAME_VALUES "$idx")" \
  RUN_STAMP="$run_stamp" \
  RUN_ID_NOTE="$run_stamp" \
    bash "$TEST_SCRIPT" &
  pid="$!"
  RUNNING_PIDS+=("$pid")
  RUNNING_GPUS+=("$assigned_gpu")
  echo "[INFO] Launched run ${run_id}/${RUN_COUNT} as pid=${pid} on gpu=${assigned_gpu}."
done

while [[ "${#RUNNING_PIDS[@]}" -gt 0 ]]; do
  cleanup_finished_jobs
  if [[ "${#RUNNING_PIDS[@]}" -gt 0 ]]; then
    sleep "$JOB_POLL_SECONDS"
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "[ERROR] One or more LIBERO eval runs failed."
  exit 1
fi

if is_true "$DRY_RUN"; then
  echo "[INFO] Dry-run finished for ${RUN_COUNT} LIBERO eval runs."
else
  echo "[INFO] Finished ${RUN_COUNT} LIBERO eval runs."
fi
