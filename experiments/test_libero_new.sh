#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAST05_ROOT="${LAST05_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LAST05_BETA_ROOT="${LAST05_BETA_ROOT:-$(cd "${LAST05_ROOT}/.." && pwd)}"

for _last05_env in "${LAST05_LOCAL_ENV:-}" "${LAST05_BETA_ROOT}/last05_local_env.sh" "${LAST05_ROOT}/last05_local_env.sh"; do
  if [[ -n "${_last05_env}" && -f "${_last05_env}" ]]; then
    # shellcheck disable=SC1090
    source "${_last05_env}"
    break
  fi
done

DATABASE_ROOT="${DATABASE_ROOT:-/mnt/nas/zhangyiming/database}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-${DATABASE_ROOT}/ckpt/pretrained}"
REQUIRES_ROOT="${REQUIRES_ROOT:-/mnt/nas/zhangyiming/requires}"
CONDA_BASE="${CONDA_BASE:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-last05}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${CONDA_BASE}/envs/${CONDA_ENV_NAME}}"
EXPERIMENTS_LIBERO_ROOT="${EXPERIMENTS_LIBERO_ROOT:-${LAST05_BETA_ROOT}/experiments}"
EXPERIMENTS_RLBENCH_ROOT="${EXPERIMENTS_RLBENCH_ROOT:-${LAST05_BETA_ROOT}/experiments_rlbench}"
COSMOS_ROOT="${COSMOS_ROOT:-/mnt/nas/zhangyiming/experiments}"
LIBERO_ROOT="${LIBERO_ROOT:-/mnt/nas/zhangxuheng/LIBERO}"
PYREP_PYTHON_PATH="${PYREP_PYTHON_PATH:-/mnt/nas/zhangyawen/zhangyiming/python_pkgs}"
LIFT3D_ROOT="${LIFT3D_ROOT:-${REQUIRES_ROOT}/LIFT3D}"
RLBENCH_ROOT="${RLBENCH_ROOT:-${LIFT3D_ROOT}/third_party/RLBench}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-${EXPERIMENTS_LIBERO_ROOT:-${LAST05_BETA_ROOT}/experiments}}"

RUN_NAME_FOR_LOG="${RUN_NAME:-cosmos2B_action1B_mot2_libero_spatial}"
LOG_DIR="${EXPERIMENTS_ROOT}/shell"
mkdir -p "$LOG_DIR"
EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y_%m_%d-%H_%M_%S)}"
SHELL_LOG="${SHELL_LOG:-$LOG_DIR/test_libero_shell_${EVAL_TIMESTAMP}_${RUN_NAME_FOR_LOG}.log}"

OUTPUT_ROOT_DIR="${OUTPUT_ROOT_DIR:-${LAST05_ROOT}/exp_mot2_action_spatial}"
RUN_NAME="${RUN_NAME:-cosmos2B_action1B_mot2_libero_spatial}"
RUN_NAME_PREFIX="cosmos2B_action1B_mot2"
EVAL_RUN_NAME="$RUN_NAME"
if [[ "$EVAL_RUN_NAME" == "$RUN_NAME_PREFIX"* ]]; then
  EVAL_RUN_NAME="${EVAL_RUN_NAME#"$RUN_NAME_PREFIX"}"
fi
EVAL_RUN_NAME="${EVAL_RUN_NAME#_}"
EVAL_RUN_NAME="${EVAL_RUN_NAME#-}"
if [[ -z "$EVAL_RUN_NAME" ]]; then
  EVAL_RUN_NAME="run"
fi
EVAL_ARTIFACT_NAME="${EVAL_ARTIFACT_NAME:-${EVAL_TIMESTAMP}_${EVAL_RUN_NAME}}"
PRED_VIDEO_DIR="${EXPERIMENTS_ROOT}/predict/${EVAL_ARTIFACT_NAME}"
ROLLOUT_VIDEO_DIR="${EXPERIMENTS_ROOT}/rollouts/${EVAL_ARTIFACT_NAME}"
VALUE_VIS_DIR="${EXPERIMENTS_ROOT}/value_visualizations/${EVAL_ARTIFACT_NAME}"
BASH_HPARAMS_FILE="${LOG_DIR}/test_libero_hparams_${EVAL_ARTIFACT_NAME}.env"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-}"
COSMOS_TEXT_CACHE_PATH="${COSMOS_TEXT_CACHE_PATH:-${DATABASE_ROOT}/data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual/cosmos_text_cache_raw_full_concat}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${EXPERIMENTS_ROOT}/logs}"
RUN_ID_NOTE="${RUN_ID_NOTE:-${EVAL_ARTIFACT_NAME}}"


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
JANUS_MODEL_PATH="${JANUS_MODEL_PATH:-${PRETRAINED_ROOT}/LaST0_Pretrain_AE_chunk16/tfmr}"
ACTION_MODEL_PATH="${ACTION_MODEL_PATH:-${PRETRAINED_ROOT}/LaST0_Pretrain_AE_chunk16/tfmr}"
COSMOS_MODEL_PATH="${COSMOS_MODEL_PATH:-${PRETRAINED_ROOT}/Cosmos-Predict2.5-2B/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt}"
COSMOS_EXPERIMENT_NAME="${COSMOS_EXPERIMENT_NAME:-Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only}"

# Bridge position schemes: mrope, mrope_interleave, llama1d
BRIDGE_POS_SCHEME="${BRIDGE_POS_SCHEME:-mrope}"
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
DECOSMOS="false"

COSMOS_SELF_ONLY_BRIDGE="false"
ACTION_SELF_CAUSAL_IN_BRIDGE="true"

TASK_SUITE_NAME="libero_spatial"
VIDEO_H="${VIDEO_H:-256}"
VIDEO_W="${VIDEO_W:-256}"
VIDEO_FRAMES=17
NUM_COND_INPUT_FRAMES=5
ACTION_DIM=7
ACTION_CHUNK=16
CUDA_DEVICE=0
SEED=0
FPS=20
COSMOS_DENOISE_STEPS=2
NUM_OPEN_LOOP_STEPS=8
CONTROL_FREQ=0
ACTION_REPEAT=1

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
    PRED_VIDEO_DIR ROLLOUT_VIDEO_DIR VALUE_VIS_DIR LOCAL_LOG_DIR RUN_ID_NOTE \
    JANUS_MODEL_PATH ACTION_MODEL_PATH COSMOS_MODEL_PATH COSMOS_EXPERIMENT_NAME COSMOS_TEXT_CACHE_PATH \
    REWRITE_EVAL_PROMPT USE_VALUE_PREDICTION USE_ACTION_VALUE_PREDICTION ACTION_VALUE_LOSS_WEIGHT VALUE_TOKEN_MASK_VIDEO_TO_VALUE VALUE_TOKEN_MASK_NONVALUE_TO_VALUE \
    USE_HISTORY_TRAJECTORY_JANUS_IMAGE HISTORY_TRAJECTORY_CAMERA_CONFIG_PATH \
    BRIDGE_POS_SCHEME IMG_LATENTS_PER_FUTURE STATE_LATENTS_PER_FUTURE NUM_FUTURE_FRAMES \
    TOTAL_LATENT_TOKENS FUTURE_FRAME_STRIDE ROBOT_STATE STATE_PLACEHOLDER_TOKENS STATE_ENCODING_MODE \
    ACTION_INTERMEDIATE_SIZE ACTION_USE_LATENT_PREFIX DECOSMOS COSMOS_SELF_ONLY_BRIDGE ACTION_SELF_CAUSAL_IN_BRIDGE \
    TASK_SUITE_NAME VIDEO_H VIDEO_W VIDEO_FRAMES NUM_COND_INPUT_FRAMES ACTION_DIM ACTION_CHUNK CUDA_DEVICE SEED FPS COSMOS_DENOISE_STEPS NUM_OPEN_LOOP_STEPS CONTROL_FREQ ACTION_REPEAT
  do
    write_hparam "$key"
  done
} > "$BASH_HPARAMS_FILE"

cd "$LAST05_ROOT"
source "${CONDA_BASE}/bin/activate" "${CONDA_ENV_PATH}"
export PATH="${CONDA_ENV_PATH}/bin:$PATH"
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
python -u "$LAST05_ROOT/experiments/robot/libero/run_libero_eval_new.py" \
  --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
  --cosmos_experiment_name "$COSMOS_EXPERIMENT_NAME" \
  --cosmos_model_path "$COSMOS_MODEL_PATH" \
  --cosmos_text_cache_path "$COSMOS_TEXT_CACHE_PATH" \
  --model_path "$JANUS_MODEL_PATH" \
  --action_model_path "$ACTION_MODEL_PATH" \
  --task_suite_name "$TASK_SUITE_NAME" \
  --video_h "$VIDEO_H" \
  --video_w "$VIDEO_W" \
  --video_frames "$VIDEO_FRAMES" \
  --num_cond_input_frames "$NUM_COND_INPUT_FRAMES" \
  --action_dim "$ACTION_DIM" \
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
  --local_log_dir "$LOCAL_LOG_DIR" \
  --run_id_note "$RUN_ID_NOTE" \
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
  --action_self_causal_in_bridge "$ACTION_SELF_CAUSAL_IN_BRIDGE"
