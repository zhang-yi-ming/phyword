#!/usr/bin/env bash
set -Eeuo pipefail

LAST05_ROOT="${LAST05_ROOT:-/mnt/nas/zhangyiming/last05_beta/last05_mot2_action}"
COSMOS_ROOT="${COSMOS_ROOT:-/mnt/nas/zhangyiming/experiments}"
LIBERO_ROOT="${LIBERO_ROOT:-/mnt/nas/zhangxuheng/LIBERO}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-/mnt/nas/zhangyiming/last05_beta/experiments_new}"

LOG_DIR="${EXPERIMENTS_ROOT}/shell"
mkdir -p "$LOG_DIR"
RUN_STAMP="${RUN_STAMP:-$(date +%Y_%m_%d-%H_%M_%S)-single-interactive}"
SHELL_LOG="$LOG_DIR/test_libero_single_interactive_${RUN_STAMP}.log"

OUTPUT_ROOT_DIR="${OUTPUT_ROOT_DIR:-${LAST05_ROOT}/exp_mot2_action_spatial}"
RUN_NAME="${RUN_NAME:-cosmos2B_action1B_mot2_libero_spatial}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-}"
RUN_DIR="${OUTPUT_ROOT_DIR}/${RUN_NAME}"
if [[ -n "${PRETRAINED_CHECKPOINT:-}" ]]; then
  PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT}"
elif [[ -n "$CHECKPOINT_NAME" ]]; then
  PRETRAINED_CHECKPOINT="${RUN_DIR}/${CHECKPOINT_NAME}"
else
  PRETRAINED_CHECKPOINT="$(find "$RUN_DIR" -maxdepth 1 -type d -name 'checkpoint-epoch-*-step-*' | sort -V | tail -n 1)"
fi

JANUS_MODEL_PATH="${JANUS_MODEL_PATH:-/mnt/nas/zhangyiming/database/ckpt/pretrained/LaST0_Pretrain_AE_chunk16/tfmr}"
ACTION_MODEL_PATH="${ACTION_MODEL_PATH:-/mnt/nas/zhangyiming/database/ckpt/pretrained/LaST0_Pretrain_AE_chunk16/tfmr}"
COSMOS_MODEL_PATH="${COSMOS_MODEL_PATH:-/mnt/nas/zhangyiming/database/ckpt/pretrained/Cosmos-Predict2.5-2B/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt}"
COSMOS_EXPERIMENT_NAME="${COSMOS_EXPERIMENT_NAME:-Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only}"
COSMOS_TEXT_CACHE_PATH="${COSMOS_TEXT_CACHE_PATH:-/mnt/nas/zhangyiming/database/data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual/cosmos_text_cache_raw_full_concat}"

REWRITE_EVAL_PROMPT="${REWRITE_EVAL_PROMPT:-false}"
USE_VALUE_PREDICTION="${USE_VALUE_PREDICTION:-false}"
USE_ACTION_VALUE_PREDICTION="${USE_ACTION_VALUE_PREDICTION:-false}"
VALUE_TOKEN_MASK_VIDEO_TO_VALUE="${VALUE_TOKEN_MASK_VIDEO_TO_VALUE:-false}"
VALUE_TOKEN_MASK_NONVALUE_TO_VALUE="${VALUE_TOKEN_MASK_NONVALUE_TO_VALUE:-false}"
USE_HISTORY_TRAJECTORY_JANUS_IMAGE="${USE_HISTORY_TRAJECTORY_JANUS_IMAGE:-false}"
HISTORY_TRAJECTORY_CAMERA_CONFIG_PATH="${HISTORY_TRAJECTORY_CAMERA_CONFIG_PATH:-${LAST05_ROOT}/experiments/robot/libero/libero_camera_params.yaml}"

BRIDGE_POS_SCHEME="${BRIDGE_POS_SCHEME:-llama1d}"
IMG_LATENTS_PER_FUTURE="${IMG_LATENTS_PER_FUTURE:-0}"
STATE_LATENTS_PER_FUTURE="${STATE_LATENTS_PER_FUTURE:-0}"
NUM_FUTURE_FRAMES="${NUM_FUTURE_FRAMES:-0}"
TOTAL_LATENT_TOKENS="${TOTAL_LATENT_TOKENS:-1}"
FUTURE_FRAME_STRIDE="${FUTURE_FRAME_STRIDE:-8}"
ROBOT_STATE="${ROBOT_STATE:-0}"
STATE_PLACEHOLDER_TOKENS="${STATE_PLACEHOLDER_TOKENS:-8}"
STATE_ENCODING_MODE="${STATE_ENCODING_MODE:-mlp}"
ACTION_INTERMEDIATE_SIZE="${ACTION_INTERMEDIATE_SIZE:-0}"
ACTION_USE_LATENT_PREFIX="${ACTION_USE_LATENT_PREFIX:-true}"
DECOSMOS="${DECOSMOS:-false}"
COSMOS_SELF_ONLY_BRIDGE="${COSMOS_SELF_ONLY_BRIDGE:-false}"
ACTION_SELF_CAUSAL_IN_BRIDGE="${ACTION_SELF_CAUSAL_IN_BRIDGE:-true}"

TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_spatial}"
INITIAL_STATES_PATH="${INITIAL_STATES_PATH:-DEFAULT}"
VIDEO_FRAMES="${VIDEO_FRAMES:-1}"
NUM_COND_INPUT_FRAMES="${NUM_COND_INPUT_FRAMES:-1}"
ACTION_CHUNK="${ACTION_CHUNK:-16}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SEED="${SEED:-0}"
FPS="${FPS:-20}"
COSMOS_DENOISE_STEPS="${COSMOS_DENOISE_STEPS:-2}"
ACTION_DENOISE_STEPS="${ACTION_DENOISE_STEPS:-10}"
NUM_OPEN_LOOP_STEPS="${NUM_OPEN_LOOP_STEPS:-8}"
CONTROL_FREQ="${CONTROL_FREQ:-0}"
ACTION_REPEAT="${ACTION_REPEAT:-1}"
SINGLE_OUTPUT_DIR="${SINGLE_OUTPUT_DIR:-${EXPERIMENTS_ROOT}/single_interactive/${RUN_STAMP}_${RUN_NAME}}"
PRINT_CONFIG="${PRINT_CONFIG:-false}"
COMMAND="${COMMAND:-}"
REPL_AFTER_COMMAND="${REPL_AFTER_COMMAND:-false}"

exec > >(tee -a "$SHELL_LOG") 2>&1
set -x

trap 'rc=$?; echo "[ERROR] test_libero_single_interactive.sh failed with exit code ${rc}"' ERR

cd "$LAST05_ROOT"
source /root/miniconda3/bin/activate /root/miniconda3/envs/last05
export PATH=/root/miniconda3/envs/last05/bin:$PATH
export PYTHONPATH="${LAST05_ROOT}:${LIBERO_ROOT}:${COSMOS_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

unset LD_PRELOAD
export MUJOCO_GL=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYOPENGL_PLATFORM=egl

if [[ ! -e "$PRETRAINED_CHECKPOINT" ]]; then
  echo "[ERROR] checkpoint does not exist: $PRETRAINED_CHECKPOINT"
  exit 1
fi

echo "[INFO] shell log: $SHELL_LOG"
echo "[INFO] checkpoint: $PRETRAINED_CHECKPOINT"
echo "[INFO] single output dir: $SINGLE_OUTPUT_DIR"
echo "[INFO] python: $(which python)"

EXTRA_ARGS=()
if [[ -n "$COMMAND" ]]; then
  EXTRA_ARGS+=(--command "$COMMAND")
fi
if [[ "$REPL_AFTER_COMMAND" == "true" || "$REPL_AFTER_COMMAND" == "1" ]]; then
  EXTRA_ARGS+=(--repl_after_command)
fi
if [[ "$PRINT_CONFIG" == "true" || "$PRINT_CONFIG" == "1" ]]; then
  EXTRA_ARGS+=(--print_config)
fi

python -u "$LAST05_ROOT/experiments/robot/libero/run_libero_single_interactive.py" \
  --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
  --cosmos_experiment_name "$COSMOS_EXPERIMENT_NAME" \
  --cosmos_model_path "$COSMOS_MODEL_PATH" \
  --cosmos_text_cache_path "$COSMOS_TEXT_CACHE_PATH" \
  --model_path "$JANUS_MODEL_PATH" \
  --action_model_path "$ACTION_MODEL_PATH" \
  --task_suite_name "$TASK_SUITE_NAME" \
  --initial_states_path "$INITIAL_STATES_PATH" \
  --video_frames "$VIDEO_FRAMES" \
  --num_cond_input_frames "$NUM_COND_INPUT_FRAMES" \
  --action_chunk "$ACTION_CHUNK" \
  --total_latent_tokens "$TOTAL_LATENT_TOKENS" \
  --img_latents_per_future "$IMG_LATENTS_PER_FUTURE" \
  --state_latents_per_future "$STATE_LATENTS_PER_FUTURE" \
  --num_future_frames "$NUM_FUTURE_FRAMES" \
  --future_frame_stride "$FUTURE_FRAME_STRIDE" \
  --robot_state "$ROBOT_STATE" \
  --state_placeholder_tokens "$STATE_PLACEHOLDER_TOKENS" \
  --state_encoding_mode "$STATE_ENCODING_MODE" \
  --cuda "$CUDA_DEVICE" \
  --seed "$SEED" \
  --action_intermediate_size "$ACTION_INTERMEDIATE_SIZE" \
  --action_use_latent_prefix "$ACTION_USE_LATENT_PREFIX" \
  --decosmos "$DECOSMOS" \
  --use_value_prediction "$USE_VALUE_PREDICTION" \
  --use_action_value_prediction "$USE_ACTION_VALUE_PREDICTION" \
  --value_token_mask_video_to_value "$VALUE_TOKEN_MASK_VIDEO_TO_VALUE" \
  --value_token_mask_nonvalue_to_value "$VALUE_TOKEN_MASK_NONVALUE_TO_VALUE" \
  --fps "$FPS" \
  --cosmos_denoise_steps "$COSMOS_DENOISE_STEPS" \
  --action_denoise_steps "$ACTION_DENOISE_STEPS" \
  --num_open_loop_steps "$NUM_OPEN_LOOP_STEPS" \
  --control_freq "$CONTROL_FREQ" \
  --action_repeat "$ACTION_REPEAT" \
  --cosmos_self_only_bridge "$COSMOS_SELF_ONLY_BRIDGE" \
  --bridge_pos_scheme "$BRIDGE_POS_SCHEME" \
  --rewrite_eval_prompt "$REWRITE_EVAL_PROMPT" \
  --use_history_trajectory_janus_image "$USE_HISTORY_TRAJECTORY_JANUS_IMAGE" \
  --history_trajectory_camera_config_path "$HISTORY_TRAJECTORY_CAMERA_CONFIG_PATH" \
  --action_self_causal_in_bridge "$ACTION_SELF_CAUSAL_IN_BRIDGE" \
  --single_output_dir "$SINGLE_OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
