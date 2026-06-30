#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAST05_ROOT="${LAST05_ROOT:-/mnt/nas/zhangyiming/last05_beta/last05}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-/mnt/nas/zhangyiming/last05_beta/experiments_rlbench}"
OUTPUT_ROOT_DIR="${OUTPUT_ROOT_DIR:-${LAST05_ROOT}/exp_cosmos_vla_3expert_rlbench_keyframe}"

RUN_NAME="${RUN_NAME:-cosmos2B_janus1B_2expert_rlbench_keyframe_tokenlatent_hidden_300ep_1chunk}"
DEFAULT_PRETRAINED_CHECKPOINT="${DEFAULT_PRETRAINED_CHECKPOINT:-/mnt/nas/zhangyiming/last05_beta/last05/exp_cosmos_vla_3expert_rlbench_keyframe/cosmos2B_janus1B_2expert_rlbench_keyframe_tokenlatent_100ep_1chunk/checkpoint-epoch-99-step-9400}"
DATA_JSON="${DATA_JSON:-/mnt/nas/zhangyiming/database/rlbench/train/json/train_action_chunk1_sumpos_lastrot.json}"
JANUS_MODEL_PATH="${JANUS_MODEL_PATH:-/mnt/nas/zhangyiming/database/ckpt/pretrained/Janus-Pro-1B}"
ACTION_MODEL_PATH="${ACTION_MODEL_PATH:-/mnt/nas/zhangyiming/database/ckpt/pretrained/LaST0_Pretrain_AE_chunk16/tfmr}"
COSMOS_MODEL_PATH="${COSMOS_MODEL_PATH:-/mnt/nas/zhangyiming/database/ckpt/pretrained/Cosmos-Predict2.5-2B/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt}"
COSMOS_EXPERIMENT_NAME="${COSMOS_EXPERIMENT_NAME:-Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only}"
COSMOS_TEXT_CACHE_PATH="${COSMOS_TEXT_CACHE_PATH:-}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

CHECKPOINT_NAME="${CHECKPOINT_NAME:-}"
RUN_DIR="${OUTPUT_ROOT_DIR}/${RUN_NAME}"
if [[ -n "${PRETRAINED_CHECKPOINT:-}" ]]; then
  PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT}"
elif [[ -n "$CHECKPOINT_NAME" ]]; then
  PRETRAINED_CHECKPOINT="${RUN_DIR}/${CHECKPOINT_NAME}"
else
  PRETRAINED_CHECKPOINT="$DEFAULT_PRETRAINED_CHECKPOINT"
fi

if [[ ! -e "$PRETRAINED_CHECKPOINT" ]]; then
  echo "[ERROR] PRETRAINED_CHECKPOINT does not exist: ${PRETRAINED_CHECKPOINT}"
  exit 1
fi

EVAL_TIMESTAMP="$(date +%Y_%m_%d-%H_%M_%S)"
EVAL_ARTIFACT_NAME="${EVAL_ARTIFACT_NAME:-${EVAL_TIMESTAMP}_${RUN_NAME}}"
if [[ -n "${ATTN_VIS_DIR:-}" ]]; then
  ATTENTION_VISUALIZATION_DIR="$ATTN_VIS_DIR"
else
  ATTENTION_VISUALIZATION_DIR="${ATTENTION_VISUALIZATION_DIR:-${EXPERIMENTS_ROOT}/attn_visualizations/${EVAL_ARTIFACT_NAME}}"
fi
LOG_DIR="${LOG_DIR:-${ATTENTION_VISUALIZATION_DIR}/shell}"
mkdir -p "$LOG_DIR" "$ATTENTION_VISUALIZATION_DIR"
SHELL_LOG="${LOG_DIR}/test_rlbench_trainset_attn_vis_shell_${EVAL_ARTIFACT_NAME}.log"
BASH_HPARAMS_FILE="${ATTENTION_VISUALIZATION_DIR}/test_rlbench_trainset_attn_vis_hparams_${EVAL_ARTIFACT_NAME}.env"

ACTION_DIM="${ACTION_DIM:-7}"
ACTION_CHUNK="${ACTION_CHUNK:-1}"
VIDEO_FRAMES="${VIDEO_FRAMES:-5}"
NUM_COND_INPUT_FRAMES="${NUM_COND_INPUT_FRAMES:-1}"
NUM_FUTURE_FRAMES="${NUM_FUTURE_FRAMES:-0}"
IMG_LATENTS_PER_FUTURE="${IMG_LATENTS_PER_FUTURE:-0}"
STATE_LATENTS_PER_FUTURE="${STATE_LATENTS_PER_FUTURE:-0}"
TOTAL_LATENT_TOKENS="${TOTAL_LATENT_TOKENS:-1}"
LATENT_TOKEN_MODE_RAW="${TOTAL_LATENT_TOKENS}"
LATENT_TOKEN_MODE="$(printf '%s' "${LATENT_TOKEN_MODE_RAW}" | tr '[:upper:]' '[:lower:]')"
case "${LATENT_TOKEN_MODE}" in
  1|v)
    LATENT_TOKEN_MODE="v"
    TOTAL_LATENT_TOKEN_COUNT=1
    ;;
  n)
    LATENT_TOKEN_MODE="n"
    TOTAL_LATENT_TOKEN_COUNT=1
    ;;
  2|vn)
    LATENT_TOKEN_MODE="vn"
    TOTAL_LATENT_TOKEN_COUNT=2
    ;;
  *)
    echo "ERROR: TOTAL_LATENT_TOKENS must be one of v, n, vn, 1, or 2; got '${LATENT_TOKEN_MODE_RAW}'." >&2
    exit 1
    ;;
esac
EXTRA_SPECIAL_TOKENS="${EXTRA_SPECIAL_TOKENS:-}"
FUTURE_FRAME_STRIDE="${FUTURE_FRAME_STRIDE:-1}"
ROBOT_STATE="${ROBOT_STATE:-0}"
STATE_PLACEHOLDER_TOKENS="${STATE_PLACEHOLDER_TOKENS:-1}"
STATE_DIM="${STATE_DIM:-7}"
STATE_ENCODING_MODE="${STATE_ENCODING_MODE:-mlp}"
ACTION_INTERMEDIATE_SIZE="${ACTION_INTERMEDIATE_SIZE:-5632}"
VIDEO_LOSS_WEIGHT="${VIDEO_LOSS_WEIGHT:-0}"
LATENT_LOSS_WEIGHT="${LATENT_LOSS_WEIGHT:-1.0}"
USE_LATENT_HIDDEN_SIM_LOSS="${USE_LATENT_HIDDEN_SIM_LOSS:-1}"
LATENT_HIDDEN_SIM_LOSS_WEIGHT="${LATENT_HIDDEN_SIM_LOSS_WEIGHT:-1.0}"
DECOSMOS="${DECOSMOS:-1}"
TRAIN_EMBED_TOKENS="${TRAIN_EMBED_TOKENS:-1}"
ACTION_USE_LATENT_PREFIX="${ACTION_USE_LATENT_PREFIX:-1}"
COSMOS_SELF_ONLY_BRIDGE="${COSMOS_SELF_ONLY_BRIDGE:-1}"
BRIDGE_POS_SCHEME="${BRIDGE_POS_SCHEME:-llama1d}"
ACTION_SELF_CAUSAL_IN_BRIDGE="${ACTION_SELF_CAUSAL_IN_BRIDGE:-1}"
VIDEO_H="${VIDEO_H:-32}"
VIDEO_W="${VIDEO_W:-32}"
FPS="${FPS:-20}"
ACTION_DENOISE_STEPS="${ACTION_DENOISE_STEPS:-10}"
COSMOS_DENOISE_STEPS="${COSMOS_DENOISE_STEPS:-2}"
USE_VALUE_PREDICTION="${USE_VALUE_PREDICTION:-0}"
USE_ACTION_VALUE_PREDICTION="${USE_ACTION_VALUE_PREDICTION:-0}"
VALUE_TOKEN_MASK_VIDEO_TO_VALUE="${VALUE_TOKEN_MASK_VIDEO_TO_VALUE:-0}"
VALUE_TOKEN_MASK_NONVALUE_TO_VALUE="${VALUE_TOKEN_MASK_NONVALUE_TO_VALUE:-0}"

NUM_TRAJECTORIES_PER_TASK="${NUM_TRAJECTORIES_PER_TASK:-1}"
TASK_NAMES="${TASK_NAMES:-}"
MAX_RECORDS_PER_EPISODE="${MAX_RECORDS_PER_EPISODE:-0}"
MAX_TOTAL_RECORDS="${MAX_TOTAL_RECORDS:-0}"
ATTN_VIS_TILE_SIZE="${ATTN_VIS_TILE_SIZE:-256}"
ATTN_VIS_ALPHA="${ATTN_VIS_ALPHA:-0.45}"
ATTN_VIS_CAPTURE_MODE="${ATTN_VIS_CAPTURE_MODE:-all}"
ATTN_VIS_TOP_RATIO="${ATTN_VIS_TOP_RATIO:-}"
ATTN_VIS_TOP_SOFTNESS="${ATTN_VIS_TOP_SOFTNESS:-0.05}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SEED="${SEED:-0}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-10}"

exec > >(tee -a "$SHELL_LOG") 2>&1
set -x
trap 'rc=$?; echo "[ERROR] test_rlbench_trainset_attn_vis.sh failed with exit code ${rc}"' ERR

write_hparam() {
  local key="$1"
  printf '%s=%s\n' "$key" "${!key}"
}

{
  for key in \
    SCRIPT_DIR LAST05_ROOT EXPERIMENTS_ROOT OUTPUT_ROOT_DIR RUN_NAME RUN_DIR CHECKPOINT_NAME DEFAULT_PRETRAINED_CHECKPOINT PRETRAINED_CHECKPOINT \
    EVAL_TIMESTAMP EVAL_ARTIFACT_NAME ATTENTION_VISUALIZATION_DIR LOG_DIR SHELL_LOG BASH_HPARAMS_FILE \
    OMP_NUM_THREADS HF_HUB_OFFLINE \
    DATA_JSON JANUS_MODEL_PATH ACTION_MODEL_PATH COSMOS_MODEL_PATH COSMOS_EXPERIMENT_NAME COSMOS_TEXT_CACHE_PATH \
    ACTION_DIM ACTION_CHUNK VIDEO_FRAMES NUM_COND_INPUT_FRAMES NUM_FUTURE_FRAMES IMG_LATENTS_PER_FUTURE STATE_LATENTS_PER_FUTURE TOTAL_LATENT_TOKENS LATENT_TOKEN_MODE_RAW LATENT_TOKEN_MODE TOTAL_LATENT_TOKEN_COUNT EXTRA_SPECIAL_TOKENS FUTURE_FRAME_STRIDE \
    ROBOT_STATE STATE_PLACEHOLDER_TOKENS STATE_DIM STATE_ENCODING_MODE ACTION_INTERMEDIATE_SIZE \
    VIDEO_LOSS_WEIGHT LATENT_LOSS_WEIGHT USE_LATENT_HIDDEN_SIM_LOSS LATENT_HIDDEN_SIM_LOSS_WEIGHT \
    DECOSMOS TRAIN_EMBED_TOKENS ACTION_USE_LATENT_PREFIX COSMOS_SELF_ONLY_BRIDGE BRIDGE_POS_SCHEME ACTION_SELF_CAUSAL_IN_BRIDGE \
    VIDEO_H VIDEO_W FPS ACTION_DENOISE_STEPS COSMOS_DENOISE_STEPS USE_VALUE_PREDICTION USE_ACTION_VALUE_PREDICTION \
    VALUE_TOKEN_MASK_VIDEO_TO_VALUE VALUE_TOKEN_MASK_NONVALUE_TO_VALUE \
    NUM_TRAJECTORIES_PER_TASK TASK_NAMES MAX_RECORDS_PER_EPISODE MAX_TOTAL_RECORDS \
    ATTN_VIS_TILE_SIZE ATTN_VIS_ALPHA ATTN_VIS_CAPTURE_MODE ATTN_VIS_TOP_RATIO ATTN_VIS_TOP_SOFTNESS CUDA_DEVICE SEED EMPTY_CACHE_EVERY
  do
    write_hparam "$key"
  done
} > "$BASH_HPARAMS_FILE"

cd "$LAST05_ROOT"
source /root/miniconda3/bin/activate /root/miniconda3/envs/last05
export PATH=/root/miniconda3/envs/last05/bin:$PATH
export PYTHONPATH="${LAST05_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS
export HF_HUB_OFFLINE
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
unset LD_PRELOAD

echo "[INFO] shell log: $SHELL_LOG"
echo "[INFO] bash hparams file: $BASH_HPARAMS_FILE"
echo "[INFO] eval artifact name: $EVAL_ARTIFACT_NAME"
echo "[INFO] attention visualization dir: $ATTENTION_VISUALIZATION_DIR"
echo "[INFO] checkpoint: $PRETRAINED_CHECKPOINT"
echo "[INFO] latent token mode: ${LATENT_TOKEN_MODE} (count=${TOTAL_LATENT_TOKEN_COUNT}, input=${LATENT_TOKEN_MODE_RAW})"
echo "[INFO] working dir: $(pwd)"
echo "[INFO] python: $(which python)"
echo "[INFO] bash hparams begin"
sed -n '1,220p' "$BASH_HPARAMS_FILE"
echo "[INFO] bash hparams end"
python -V
python -c "import sys; print('[INFO] sys.executable:', sys.executable)"

python -u "${SCRIPT_DIR}/run_rlbench_trainset_attn_vis.py" \
  --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
  --model_path "$JANUS_MODEL_PATH" \
  --action_model_path "$ACTION_MODEL_PATH" \
  --cosmos_model_path "$COSMOS_MODEL_PATH" \
  --cosmos_experiment_name "$COSMOS_EXPERIMENT_NAME" \
  --cosmos_text_cache_path "$COSMOS_TEXT_CACHE_PATH" \
  --data_path "$DATA_JSON" \
  --attention_visualization_dir "$ATTENTION_VISUALIZATION_DIR" \
  --attention_visualization_tile_size "$ATTN_VIS_TILE_SIZE" \
  --attention_visualization_alpha "$ATTN_VIS_ALPHA" \
  --attention_visualization_capture_mode "$ATTN_VIS_CAPTURE_MODE" \
  --attention_visualization_top_ratio "$ATTN_VIS_TOP_RATIO" \
  --attention_visualization_top_softness "$ATTN_VIS_TOP_SOFTNESS" \
  --bash_hparams_path "$BASH_HPARAMS_FILE" \
  --eval_artifact_name "$EVAL_ARTIFACT_NAME" \
  --task_names "$TASK_NAMES" \
  --num_trajectories_per_task "$NUM_TRAJECTORIES_PER_TASK" \
  --max_records_per_episode "$MAX_RECORDS_PER_EPISODE" \
  --max_total_records "$MAX_TOTAL_RECORDS" \
  --cuda "$CUDA_DEVICE" \
  --seed "$SEED" \
  --video_h "$VIDEO_H" \
  --video_w "$VIDEO_W" \
  --video_frames "$VIDEO_FRAMES" \
  --num_cond_input_frames "$NUM_COND_INPUT_FRAMES" \
  --action_dim "$ACTION_DIM" \
  --action_chunk "$ACTION_CHUNK" \
  --robot_state "$ROBOT_STATE" \
  --state_placeholder_tokens "$STATE_PLACEHOLDER_TOKENS" \
  --state_dim "$STATE_DIM" \
  --state_encoding_mode "$STATE_ENCODING_MODE" \
  --action_intermediate_size "$ACTION_INTERMEDIATE_SIZE" \
  --total_latent_tokens "$TOTAL_LATENT_TOKEN_COUNT" \
  --latent_token_mode "$LATENT_TOKEN_MODE" \
  --extra_special_tokens "$EXTRA_SPECIAL_TOKENS" \
  --img_latents_per_future "$IMG_LATENTS_PER_FUTURE" \
  --state_latents_per_future "$STATE_LATENTS_PER_FUTURE" \
  --num_future_frames "$NUM_FUTURE_FRAMES" \
  --future_frame_stride "$FUTURE_FRAME_STRIDE" \
  --video_loss_weight "$VIDEO_LOSS_WEIGHT" \
  --latent_loss_weight "$LATENT_LOSS_WEIGHT" \
  --use_latent_hidden_sim_loss "$USE_LATENT_HIDDEN_SIM_LOSS" \
  --latent_hidden_sim_loss_weight "$LATENT_HIDDEN_SIM_LOSS_WEIGHT" \
  --cosmos_self_only_bridge "$COSMOS_SELF_ONLY_BRIDGE" \
  --train_embed_tokens "$TRAIN_EMBED_TOKENS" \
  --decosmos "$DECOSMOS" \
  --use_value_prediction "$USE_VALUE_PREDICTION" \
  --use_action_value_prediction "$USE_ACTION_VALUE_PREDICTION" \
  --value_token_mask_video_to_value "$VALUE_TOKEN_MASK_VIDEO_TO_VALUE" \
  --value_token_mask_nonvalue_to_value "$VALUE_TOKEN_MASK_NONVALUE_TO_VALUE" \
  --bridge_pos_scheme "$BRIDGE_POS_SCHEME" \
  --action_use_latent_prefix "$ACTION_USE_LATENT_PREFIX" \
  --action_self_causal_in_bridge "$ACTION_SELF_CAUSAL_IN_BRIDGE" \
  --action_denoise_steps "$ACTION_DENOISE_STEPS" \
  --cosmos_denoise_steps "$COSMOS_DENOISE_STEPS" \
  --fps "$FPS" \
  --empty_cache_every "$EMPTY_CACHE_EVERY"
