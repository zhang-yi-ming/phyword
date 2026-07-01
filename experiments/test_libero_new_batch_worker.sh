#!/usr/bin/env bash
set -Eeuo pipefail


LAST05_ROOT="${LAST05_ROOT:-/mnt/nas/zhangyiming/last05_beta/last05_mot2_action}"
COSMOS_ROOT="${COSMOS_ROOT:-/mnt/nas/zhangyiming/experiments}"
LIBERO_ROOT="${LIBERO_ROOT:-/mnt/nas/zhangxuheng/LIBERO}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-/mnt/nas/zhangyiming/last05_beta/experiments_new}"

LOG_DIR="${EXPERIMENTS_ROOT}/shell"
mkdir -p "$LOG_DIR"
RUN_STAMP="${RUN_STAMP:-$(date +%Y_%m_%d-%H_%M_%S)-pid$$}"
RUN_ID_NOTE="${RUN_ID_NOTE:-}"
EVAL_TIMESTAMP="$RUN_STAMP"
SHELL_LOG="$LOG_DIR/test_libero_shell_${EVAL_TIMESTAMP}.log"

OUTPUT_ROOT_DIR="${OUTPUT_ROOT_DIR:-${LAST05_ROOT}/exp_mot2_action_spatial}"
RUN_NAME="${RUN_NAME:-cosmos2B_action1B_mot2_libero_spatial}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-cosmos2B_action1B_mot2}"
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
PRED_VIDEO_DIR="${PRED_VIDEO_DIR:-${EXPERIMENTS_ROOT}/predict/${EVAL_ARTIFACT_NAME}}"
ROLLOUT_VIDEO_DIR="${ROLLOUT_VIDEO_DIR:-${EXPERIMENTS_ROOT}/rollouts/${EVAL_ARTIFACT_NAME}}"
VALUE_VIS_DIR="${VALUE_VIS_DIR:-${EXPERIMENTS_ROOT}/value_visualizations/${EVAL_ARTIFACT_NAME}}"
BASH_HPARAMS_FILE="${LOG_DIR}/test_libero_hparams_${EVAL_ARTIFACT_NAME}.env"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-}"
COSMOS_TEXT_CACHE_PATH="${COSMOS_TEXT_CACHE_PATH:-/mnt/nas/zhangyiming/database/data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual/cosmos_text_cache_raw_full_concat}"
#COSMOS_TEXT_CACHE_PATH="${COSMOS_TEXT_CACHE_PATH:-/mnt/nas/zhangyiming/database/data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual/cosmos_text_cache}"
#COSMOS_TEXT_CACHE_PATH="${COSMOS_TEXT_CACHE_PATH:-/mnt/nas/zhangyiming/database/data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual/cosmos_text_cache_rewritten_prompts_raw_full_concat}"

REWRITE_EVAL_PROMPT="${REWRITE_EVAL_PROMPT:-false}"
USE_VALUE_PREDICTION="${USE_VALUE_PREDICTION:-false}"
USE_ACTION_VALUE_PREDICTION="${USE_ACTION_VALUE_PREDICTION:-false}"
ACTION_VALUE_LOSS_WEIGHT="${ACTION_VALUE_LOSS_WEIGHT:-1.0}"
VALUE_TOKEN_MASK_VIDEO_TO_VALUE="${VALUE_TOKEN_MASK_VIDEO_TO_VALUE:-false}"
VALUE_TOKEN_MASK_NONVALUE_TO_VALUE="${VALUE_TOKEN_MASK_NONVALUE_TO_VALUE:-false}"




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
JANUS_MODEL_PATH="${JANUS_MODEL_PATH:-/mnt/nas/zhangyiming/database/ckpt/pretrained/LaST0_Pretrain_AE_chunk16/tfmr}"
ACTION_MODEL_PATH="${ACTION_MODEL_PATH:-/mnt/nas/zhangyiming/database/ckpt/pretrained/LaST0_Pretrain_AE_chunk16/tfmr}"
COSMOS_MODEL_PATH="${COSMOS_MODEL_PATH:-/mnt/nas/zhangyiming/database/ckpt/pretrained/Cosmos-Predict2.5-2B/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt}"
COSMOS_EXPERIMENT_NAME="${COSMOS_EXPERIMENT_NAME:-Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only}"

# Bridge position schemes: mrope, mrope_interleave, llama1d
BRIDGE_POS_SCHEME="${BRIDGE_POS_SCHEME:-mrope}"
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
VIDEO_H="${VIDEO_H:-256}"
VIDEO_W="${VIDEO_W:-256}"
VIDEO_FRAMES="${VIDEO_FRAMES:-1}"
NUM_COND_INPUT_FRAMES="${NUM_COND_INPUT_FRAMES:-1}"
ACTION_CHUNK="${ACTION_CHUNK:-16}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SEED="${SEED:-0}"
FPS="${FPS:-20}"
COSMOS_DENOISE_STEPS="${COSMOS_DENOISE_STEPS:-2}"
NUM_OPEN_LOOP_STEPS="${NUM_OPEN_LOOP_STEPS:-8}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
CONTROL_FREQ="${CONTROL_FREQ:-0}"
ACTION_REPEAT="${ACTION_REPEAT:-1}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-4}"
EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"

exec > >(tee -a "$SHELL_LOG") 2>&1
set -x

trap 'rc=$?; echo "[ERROR] test_libero.sh failed with exit code ${rc}"' ERR

write_hparam() {
  local key="$1"
  printf '%s=%s\n' "$key" "${!key-}"
}

{
  for key in \
    RUN_STAMP RUN_ID_NOTE EVAL_TIMESTAMP RUN_NAME RUN_NAME_PREFIX EVAL_RUN_NAME EVAL_ARTIFACT_NAME \
    LAST05_ROOT COSMOS_ROOT LIBERO_ROOT EXPERIMENTS_ROOT LOG_DIR SHELL_LOG BASH_HPARAMS_FILE \
    OUTPUT_ROOT_DIR CHECKPOINT_NAME RUN_DIR PRETRAINED_CHECKPOINT \
    PRED_VIDEO_DIR ROLLOUT_VIDEO_DIR VALUE_VIS_DIR \
    JANUS_MODEL_PATH ACTION_MODEL_PATH COSMOS_MODEL_PATH COSMOS_EXPERIMENT_NAME COSMOS_TEXT_CACHE_PATH \
    REWRITE_EVAL_PROMPT USE_VALUE_PREDICTION USE_ACTION_VALUE_PREDICTION ACTION_VALUE_LOSS_WEIGHT VALUE_TOKEN_MASK_VIDEO_TO_VALUE VALUE_TOKEN_MASK_NONVALUE_TO_VALUE \
    BRIDGE_POS_SCHEME IMG_LATENTS_PER_FUTURE STATE_LATENTS_PER_FUTURE NUM_FUTURE_FRAMES \
    TOTAL_LATENT_TOKENS FUTURE_FRAME_STRIDE ROBOT_STATE STATE_PLACEHOLDER_TOKENS STATE_ENCODING_MODE \
    ACTION_INTERMEDIATE_SIZE ACTION_USE_LATENT_PREFIX DECOSMOS COSMOS_SELF_ONLY_BRIDGE ACTION_SELF_CAUSAL_IN_BRIDGE \
    TASK_SUITE_NAME VIDEO_H VIDEO_W VIDEO_FRAMES NUM_COND_INPUT_FRAMES ACTION_CHUNK CUDA_DEVICE SEED FPS COSMOS_DENOISE_STEPS NUM_OPEN_LOOP_STEPS \
    NUM_TRIALS_PER_TASK CONTROL_FREQ ACTION_REPEAT CPU_THREADS_PER_JOB CUDA_VISIBLE_DEVICES EGL_DEVICE_ID
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS_PER_JOB}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS_PER_JOB}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_THREADS_PER_JOB}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$CPU_THREADS_PER_JOB}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

unset LD_PRELOAD
export MUJOCO_GL=egl
export EGL_DEVICE_ID="$EGL_DEVICE_ID"
export PYOPENGL_PLATFORM=egl

RUN_ID_NOTE_ARGS=()
if [[ -n "$RUN_ID_NOTE" ]]; then
  RUN_ID_NOTE_ARGS=(--run_id_note "$RUN_ID_NOTE")
fi

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
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] cuda arg: $CUDA_DEVICE"
echo "[INFO] cpu threads per job: $CPU_THREADS_PER_JOB"
echo "[INFO] num trials per task: $NUM_TRIALS_PER_TASK"
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
  --bash_hparams_path "$BASH_HPARAMS_FILE" \
  --eval_artifact_name "$EVAL_ARTIFACT_NAME" \
  --fps "$FPS" \
  --cosmos_denoise_steps "$COSMOS_DENOISE_STEPS" \
  --num_open_loop_steps "$NUM_OPEN_LOOP_STEPS" \
  --num_trials_per_task "$NUM_TRIALS_PER_TASK" \
  --control_freq "$CONTROL_FREQ" \
  --action_repeat "$ACTION_REPEAT" \
  "${RUN_ID_NOTE_ARGS[@]}" \
  --cosmos_self_only_bridge "$COSMOS_SELF_ONLY_BRIDGE" \
  --bridge_pos_scheme "$BRIDGE_POS_SCHEME" \
  --rewrite_eval_prompt "$REWRITE_EVAL_PROMPT" \
  --action_self_causal_in_bridge "$ACTION_SELF_CAUSAL_IN_BRIDGE"
