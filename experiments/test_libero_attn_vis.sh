#!/usr/bin/env bash
set -Eeuo pipefail


LAST05_ROOT="/mnt/nas/zhangyiming/last05_beta/last05"
COSMOS_ROOT="/mnt/nas/zhangyiming/experiments"
LIBERO_ROOT="/mnt/nas/zhangxuheng/LIBERO"
EXPERIMENTS_ROOT="/mnt/nas/zhangyiming/last05_beta/experiments"

LOG_DIR="${EXPERIMENTS_ROOT}/shell"
mkdir -p "$LOG_DIR"
EVAL_TIMESTAMP="$(date +%Y_%m_%d-%H_%M_%S)"
SHELL_LOG="$LOG_DIR/test_libero_attn_vis_shell_${EVAL_TIMESTAMP}.log"

OUTPUT_ROOT_DIR="${LAST05_ROOT}/exp_cosmos_vla_3expert"
RUN_NAME="${RUN_NAME:-cosmos2B_janus1B_2expert_spatial_token_no_tr}"
RUN_NAME_PREFIX="cosmos2B_janus1B_3expert"
EVAL_RUN_NAME="$RUN_NAME"
if [[ "$EVAL_RUN_NAME" == "$RUN_NAME_PREFIX"* ]]; then
  EVAL_RUN_NAME="${EVAL_RUN_NAME#"$RUN_NAME_PREFIX"}"
fi
EVAL_RUN_NAME="${EVAL_RUN_NAME#_}"
EVAL_RUN_NAME="${EVAL_RUN_NAME#-}"
if [[ -z "$EVAL_RUN_NAME" ]]; then
  EVAL_RUN_NAME="run"
fi
EVAL_ARTIFACT_NAME="${EVAL_TIMESTAMP}_${EVAL_RUN_NAME}"
PRED_VIDEO_DIR="${EXPERIMENTS_ROOT}/predict/${EVAL_ARTIFACT_NAME}"
ROLLOUT_VIDEO_DIR="${EXPERIMENTS_ROOT}/rollouts/${EVAL_ARTIFACT_NAME}"
VALUE_VIS_DIR="${EXPERIMENTS_ROOT}/value_visualizations/${EVAL_ARTIFACT_NAME}"
ATTN_VIS_DIR="${EXPERIMENTS_ROOT}/attention_visualizations/${EVAL_ARTIFACT_NAME}"
BASH_HPARAMS_FILE="${LOG_DIR}/test_libero_attn_vis_hparams_${EVAL_ARTIFACT_NAME}.env"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint-epoch-39-step-33960}"
COSMOS_TEXT_CACHE_PATH="${COSMOS_TEXT_CACHE_PATH:-}"


REWRITE_EVAL_PROMPT="${REWRITE_EVAL_PROMPT:-false}"
USE_VALUE_PREDICTION="${USE_VALUE_PREDICTION:-false}"
USE_ACTION_VALUE_PREDICTION="${USE_ACTION_VALUE_PREDICTION:-false}"
ACTION_VALUE_LOSS_WEIGHT="${ACTION_VALUE_LOSS_WEIGHT:-1.0}"
VALUE_TOKEN_MASK_VIDEO_TO_VALUE="${VALUE_TOKEN_MASK_VIDEO_TO_VALUE:-false}"
VALUE_TOKEN_MASK_NONVALUE_TO_VALUE="${VALUE_TOKEN_MASK_NONVALUE_TO_VALUE:-false}"
USE_HISTORY_TRAJECTORY_JANUS_IMAGE="${USE_HISTORY_TRAJECTORY_JANUS_IMAGE:-false}"
HISTORY_TRAJECTORY_CAMERA_CONFIG_PATH="${HISTORY_TRAJECTORY_CAMERA_CONFIG_PATH:-${LAST05_ROOT}/experiments/robot/libero/libero_camera_params.yaml}"



RUN_DIR="${OUTPUT_ROOT_DIR}/${RUN_NAME}"
if [[ -n "${PRETRAINED_CHECKPOINT:-}" ]]; then
  PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT}"
elif [[ -n "$CHECKPOINT_NAME" ]]; then
  PRETRAINED_CHECKPOINT="${RUN_DIR}/${CHECKPOINT_NAME}"
else
  if [[ ! -d "$RUN_DIR" ]]; then
    echo "[ERROR] Run directory does not exist: ${RUN_DIR}. Set PRETRAINED_CHECKPOINT or CHECKPOINT_NAME."
    exit 1
  fi
  PRETRAINED_CHECKPOINT="$(find "$RUN_DIR" -maxdepth 1 -type d -name 'checkpoint-epoch-*-step-*' | sort -V | tail -n 1)"
  if [[ -z "$PRETRAINED_CHECKPOINT" ]]; then
    echo "[ERROR] No checkpoint found under ${RUN_DIR}. Set PRETRAINED_CHECKPOINT or CHECKPOINT_NAME."
    exit 1
  fi
fi
JANUS_MODEL_PATH="/mnt/nas/zhangyiming/database/ckpt/pretrained/Janus-Pro-1B"
ACTION_MODEL_PATH="/mnt/nas/zhangyiming/database/ckpt/pretrained/LaST0_Pretrain_AE_chunk16/tfmr"
COSMOS_MODEL_PATH="/mnt/nas/zhangyiming/database/ckpt/pretrained/Cosmos-Predict2.5-2B/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt"
COSMOS_EXPERIMENT_NAME="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only"

# Bridge position schemes: mrope, mrope_interleave, llama1d
BRIDGE_POS_SCHEME="llama1d"
IMG_LATENTS_PER_FUTURE=0
STATE_LATENTS_PER_FUTURE=0
NUM_FUTURE_FRAMES=0
TOTAL_LATENT_TOKENS="${TOTAL_LATENT_TOKENS:-1}"
FUTURE_FRAME_STRIDE=8
ROBOT_STATE=0
STATE_PLACEHOLDER_TOKENS=8
STATE_ENCODING_MODE="mlp"
ACTION_INTERMEDIATE_SIZE=0
ACTION_USE_LATENT_PREFIX="true"
DECOSMOS="true"

COSMOS_SELF_ONLY_BRIDGE="true"
ACTION_SELF_CAUSAL_IN_BRIDGE="true"

TASK_SUITE_NAME="libero_spatial"
VIDEO_FRAMES=1
NUM_COND_INPUT_FRAMES=1
ACTION_CHUNK=16
CUDA_DEVICE=0
SEED=0
FPS=20
COSMOS_DENOISE_STEPS=2
NUM_OPEN_LOOP_STEPS=8
CONTROL_FREQ=0
ACTION_REPEAT=1
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-3}"


exec > >(tee -a "$SHELL_LOG") 2>&1
set -x

trap 'rc=$?; echo "[ERROR] test_libero.sh failed with exit code ${rc}"' ERR

write_hparam() {
  local key="$1"
  printf '%s=%s\n' "$key" "${!key}"
}

{
  for key in \
    EVAL_TIMESTAMP RUN_NAME RUN_NAME_PREFIX EVAL_RUN_NAME EVAL_ARTIFACT_NAME \
    LAST05_ROOT COSMOS_ROOT LIBERO_ROOT EXPERIMENTS_ROOT LOG_DIR SHELL_LOG BASH_HPARAMS_FILE \
    OUTPUT_ROOT_DIR CHECKPOINT_NAME RUN_DIR PRETRAINED_CHECKPOINT \
    PRED_VIDEO_DIR ROLLOUT_VIDEO_DIR VALUE_VIS_DIR ATTN_VIS_DIR \
    JANUS_MODEL_PATH ACTION_MODEL_PATH COSMOS_MODEL_PATH COSMOS_EXPERIMENT_NAME COSMOS_TEXT_CACHE_PATH \
    REWRITE_EVAL_PROMPT USE_VALUE_PREDICTION USE_ACTION_VALUE_PREDICTION ACTION_VALUE_LOSS_WEIGHT VALUE_TOKEN_MASK_VIDEO_TO_VALUE VALUE_TOKEN_MASK_NONVALUE_TO_VALUE \
    USE_HISTORY_TRAJECTORY_JANUS_IMAGE HISTORY_TRAJECTORY_CAMERA_CONFIG_PATH \
    BRIDGE_POS_SCHEME IMG_LATENTS_PER_FUTURE STATE_LATENTS_PER_FUTURE NUM_FUTURE_FRAMES \
    TOTAL_LATENT_TOKENS FUTURE_FRAME_STRIDE ROBOT_STATE STATE_PLACEHOLDER_TOKENS STATE_ENCODING_MODE \
    ACTION_INTERMEDIATE_SIZE ACTION_USE_LATENT_PREFIX DECOSMOS COSMOS_SELF_ONLY_BRIDGE ACTION_SELF_CAUSAL_IN_BRIDGE \
    TASK_SUITE_NAME VIDEO_FRAMES NUM_COND_INPUT_FRAMES ACTION_CHUNK CUDA_DEVICE SEED FPS COSMOS_DENOISE_STEPS NUM_OPEN_LOOP_STEPS CONTROL_FREQ ACTION_REPEAT
  do
    write_hparam "$key"
  done
} > "$BASH_HPARAMS_FILE"

cd "$LAST05_ROOT"
source /root/miniconda3/bin/activate /root/miniconda3/envs/last05
export PATH=/root/miniconda3/envs/last05/bin:$PATH
#export HF_HOME=/media/huggingFace
export PYTHONPATH="${LAST05_ROOT}:${LIBERO_ROOT}:${COSMOS_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE=offline

unset LD_PRELOAD
export MUJOCO_GL=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYOPENGL_PLATFORM=egl

echo "[INFO] shell log: $SHELL_LOG"
echo "[INFO] bash hparams file: $BASH_HPARAMS_FILE"
echo "[INFO] eval artifact name: $EVAL_ARTIFACT_NAME"
echo "[INFO] predicted video dir: $PRED_VIDEO_DIR"
echo "[INFO] rollout video dir: $ROLLOUT_VIDEO_DIR"
echo "[INFO] value visualization dir: $VALUE_VIS_DIR"
echo "[INFO] attention visualization dir: $ATTN_VIS_DIR"
echo "[INFO] checkpoint: $PRETRAINED_CHECKPOINT"
echo "[INFO] working dir: $(pwd)"
echo "[INFO] repo root: $LAST05_ROOT"
echo "[INFO] mujoco gl backend: $MUJOCO_GL"
echo "[INFO] egl device id: $EGL_DEVICE_ID"
echo "[INFO] python: $(which python)"
echo "[INFO] bash hparams begin"
sed -n '1,200p' "$BASH_HPARAMS_FILE"
echo "[INFO] bash hparams end"
python -V
python -c "import sys; print('[INFO] sys.executable:', sys.executable)"
python -c "import libero; print('[INFO] libero:', libero.__file__)"

# Launch LIBERO-Spatial evals
python -u "$LAST05_ROOT/experiments/robot/libero/run_libero_eval_attn_vis.py" \
  --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
  --cosmos_experiment_name "$COSMOS_EXPERIMENT_NAME" \
  --cosmos_model_path "$COSMOS_MODEL_PATH" \
  --cosmos_text_cache_path "$COSMOS_TEXT_CACHE_PATH" \
  --model_path "$JANUS_MODEL_PATH" \
  --action_model_path "$ACTION_MODEL_PATH" \
  --task_suite_name "$TASK_SUITE_NAME" \
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
  --predicted_video_save_dir "$PRED_VIDEO_DIR" \
  --rollout_video_save_dir "$ROLLOUT_VIDEO_DIR" \
  --value_visualization_dir "$VALUE_VIS_DIR" \
  --attention_visualization_dir "$ATTN_VIS_DIR" \
  --bash_hparams_path "$BASH_HPARAMS_FILE" \
  --eval_artifact_name "$EVAL_ARTIFACT_NAME" \
  --fps "$FPS" \
  --cosmos_denoise_steps "$COSMOS_DENOISE_STEPS" \
  --num_open_loop_steps "$NUM_OPEN_LOOP_STEPS" \
  --control_freq "$CONTROL_FREQ" \
  --action_repeat "$ACTION_REPEAT" \
  --cosmos_self_only_bridge "$COSMOS_SELF_ONLY_BRIDGE" \
  --bridge_pos_scheme "$BRIDGE_POS_SCHEME" \
  --rewrite_eval_prompt "$REWRITE_EVAL_PROMPT" \
  --use_history_trajectory_janus_image "$USE_HISTORY_TRAJECTORY_JANUS_IMAGE" \
  --history_trajectory_camera_config_path "$HISTORY_TRAJECTORY_CAMERA_CONFIG_PATH" \
  --num_trials_per_task "$NUM_TRIALS_PER_TASK" \
  --action_self_causal_in_bridge "$ACTION_SELF_CAUSAL_IN_BRIDGE"
