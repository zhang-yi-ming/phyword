#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAST05_ROOT="${LAST05_ROOT:-/mnt/amlfs-07/shared/physicalword/last05_beta_git/last05_mot2_action_fis}"
LAST05_BETA_ROOT="${LAST05_BETA_ROOT:-$(cd "${LAST05_ROOT}/.." && pwd)}"
DATABASE_ROOT="${DATABASE_ROOT:-/mnt/amlfs-07/shared/physicalword}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-${DATABASE_ROOT}/ckpt/pretrained}"
COSMOS_ROOT="${COSMOS_ROOT:-/mnt/nas/zhangyiming/experiments}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-${LAST05_BETA_ROOT}/exp}"
OUTPUT_ROOT_DIR="${OUTPUT_ROOT_DIR:-${LAST05_ROOT}/exp_mot2_action_fis_spatial}"
PYREP_PYTHON_PATH="${PYREP_PYTHON_PATH:-/mnt/nas/zhangyawen/zhangyiming/python_pkgs}"
LIFT3D_ROOT="${LIFT3D_ROOT:-/mnt/nas/zhangyiming/requires/LIFT3D}"
RLBENCH_ROOT="${RLBENCH_ROOT:-${LIFT3D_ROOT}/third_party/RLBench}"

RUN_NAME="${RUN_NAME:-cosmos2B_action1B_mot2_fis_libero_spatial}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-}"
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
if [[ ! -e "$PRETRAINED_CHECKPOINT" ]]; then
  echo "[ERROR] PRETRAINED_CHECKPOINT does not exist: ${PRETRAINED_CHECKPOINT}"
  exit 1
fi

EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y_%m_%d-%H_%M_%S)}"
EVAL_ARTIFACT_NAME="${EVAL_ARTIFACT_NAME:-${EVAL_TIMESTAMP}_${RUN_NAME}}"
if [[ -n "${ATTN_VIS_DIR:-}" ]]; then
  ATTENTION_VISUALIZATION_DIR="$ATTN_VIS_DIR"
else
  ATTENTION_VISUALIZATION_DIR="${ATTENTION_VISUALIZATION_DIR:-${EXPERIMENTS_ROOT}/attn_visualizations/${EVAL_ARTIFACT_NAME}}"
fi
LOG_DIR="${LOG_DIR:-${ATTENTION_VISUALIZATION_DIR}/shell}"
mkdir -p "$LOG_DIR" "$ATTENTION_VISUALIZATION_DIR"
SHELL_LOG="${LOG_DIR}/test_libero_trainset_attn_vis_shell_${EVAL_ARTIFACT_NAME}.log"
BASH_HPARAMS_FILE="${ATTENTION_VISUALIZATION_DIR}/test_libero_trainset_attn_vis_hparams_${EVAL_ARTIFACT_NAME}.env"

DATA_JSON="${DATA_JSON:-/mnt/amlfs-07/shared/physicalword/data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual/train_with_atomic_action_shared.json}"
ACTION_EXPERT_PATH="${ACTION_EXPERT_PATH:-${PRETRAINED_ROOT}/LaST0_Pretrain_AE_chunk16/tfmr}"
JANUS_INIT_MODE="${JANUS_INIT_MODE:-janus_pro_ae_flow}"
JANUS_PRO_MODEL_PATH="${JANUS_PRO_MODEL_PATH:-${PRETRAINED_ROOT}/Janus-Pro-1B}"
JANUS_MODEL_PATH="${JANUS_MODEL_PATH:-${JANUS_PRO_MODEL_PATH}}"
ACTION_MODEL_PATH="${ACTION_MODEL_PATH:-${ACTION_EXPERT_PATH}}"
COSMOS_MODEL_PATH="${COSMOS_MODEL_PATH:-${PRETRAINED_ROOT}/Cosmos-Predict2.5-2B/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt}"
COSMOS_EXPERIMENT_NAME="${COSMOS_EXPERIMENT_NAME:-Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only}"
COSMOS_TEXT_CACHE_PATH="${COSMOS_TEXT_CACHE_PATH:-${DATABASE_ROOT}/data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual/cosmos_text_cache_raw_full_concat}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-/mnt/amlfs-07/shared/physicalword/miniforge3/bin/activate}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/mnt/amlfs-07/shared/physicalword/envs/last05_packed}"

ACTION_DIM="${ACTION_DIM:-7}"
ACTION_CHUNK="${ACTION_CHUNK:-16}"
VIDEO_H="${VIDEO_H:-256}"
VIDEO_W="${VIDEO_W:-256}"
VIDEO_FRAMES="${VIDEO_FRAMES:-17}"
NUM_COND_INPUT_FRAMES="${NUM_COND_INPUT_FRAMES:-5}"
IMG_LATENTS_PER_FUTURE=0
STATE_LATENTS_PER_FUTURE=0
NUM_FUTURE_FRAMES=0
FUTURE_FRAME_STRIDE="${FUTURE_FRAME_STRIDE:-8}"
ROBOT_STATE="${ROBOT_STATE:-0}"
STATE_PLACEHOLDER_TOKENS="${STATE_PLACEHOLDER_TOKENS:-8}"
STATE_DIM="${STATE_DIM:-8}"
STATE_ENCODING_MODE="${STATE_ENCODING_MODE:-mlp}"
BRIDGE_POS_SCHEME="${BRIDGE_POS_SCHEME:-mrope}"
ACTION_SELF_CAUSAL_IN_BRIDGE="${ACTION_SELF_CAUSAL_IN_BRIDGE:-1}"
ACTION_USE_LATENT_PREFIX="${ACTION_USE_LATENT_PREFIX:-1}"
COSMOS_SELF_ONLY_BRIDGE="${COSMOS_SELF_ONLY_BRIDGE:-0}"
DECOSMOS="${DECOSMOS:-0}"
FPS="${FPS:-10}"
ACTION_DENOISE_STEPS="${ACTION_DENOISE_STEPS:-10}"
COSMOS_DENOISE_STEPS="${COSMOS_DENOISE_STEPS:-2}"
ACTION_DETACH_SLOW_PREFIX="${ACTION_DETACH_SLOW_PREFIX:-0}"
ACTION_DETACH_VIDEO_BRANCH="${ACTION_DETACH_VIDEO_BRANCH:-0}"
ACTION_DETACH_SLOW_VIDEO_BRANCH="${ACTION_DETACH_SLOW_VIDEO_BRANCH:-0}"

SPATIAL_TOKEN_MODE_RAW="${SPATIAL_TOKEN_MODE:-${TOTAL_LATENT_TOKENS:-v}}"
SPATIAL_TOKEN_MODE="$(printf '%s' "${SPATIAL_TOKEN_MODE_RAW}" | tr '[:upper:]' '[:lower:]')"
case "${SPATIAL_TOKEN_MODE}" in
  1|v) SPATIAL_TOKEN_MODE="v"; TOTAL_SPATIAL_TOKEN_COUNT=1 ;;
  n) SPATIAL_TOKEN_MODE="n"; TOTAL_SPATIAL_TOKEN_COUNT=1 ;;
  2|vn) SPATIAL_TOKEN_MODE="vn"; TOTAL_SPATIAL_TOKEN_COUNT=2 ;;
  *)
    echo "ERROR: SPATIAL_TOKEN_MODE must be one of v, n, vn, 1, or 2; got '${SPATIAL_TOKEN_MODE_RAW}'." >&2
    exit 1
    ;;
esac

DEFAULT_EXTRA_SPECIAL_TOKENS="</PAD>,</MOVE>,</PICK>,</PLACE>,</ROTATE>,</PULL>,</PUSH>,</NONE>,</box>,</broom>,</charger>,</frame>,</fridge>,</lamp>,</laptop>,</phone>,</toilet>,</umbrella>,</watering_can>,</wine>"
EXTRA_SPECIAL_TOKENS="${EXTRA_SPECIAL_TOKENS:-${DEFAULT_EXTRA_SPECIAL_TOKENS}}"
USE_SPATIAL_HIDDEN_SIM_LOSS="${USE_SPATIAL_HIDDEN_SIM_LOSS:-1}"
SPATIAL_HIDDEN_SIM_LOSS_MODE="${SPATIAL_HIDDEN_SIM_LOSS_MODE:-siglip}"
SPATIAL_HIDDEN_SIM_LOSS_WEIGHT="${SPATIAL_HIDDEN_SIM_LOSS_WEIGHT:-1.0}"
USE_SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS="${USE_SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS:-0}"
SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS_WEIGHT="${SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS_WEIGHT:-1.0}"
WAN21_VAE_PATH="${WAN21_VAE_PATH:-${PRETRAINED_ROOT}/wan2.1_vae/original/Wan2.1_VAE.pth}"

TASK_NAMES="${TASK_NAMES:-}"
NUM_TRAJECTORIES_PER_TASK="${NUM_TRAJECTORIES_PER_TASK:-1}"
MAX_RECORDS_PER_EPISODE="${MAX_RECORDS_PER_EPISODE:-0}"
MAX_TOTAL_RECORDS="${MAX_TOTAL_RECORDS:-0}"
ATTN_VIS_TILE_SIZE="${ATTN_VIS_TILE_SIZE:-160}"
ATTN_VIS_ALPHA="${ATTN_VIS_ALPHA:-0.45}"
ATTN_VIS_CAPTURE_MODE="${ATTN_VIS_CAPTURE_MODE:-last}"
ATTN_VIS_TOP_RATIO="${ATTN_VIS_TOP_RATIO:-}"
ATTN_VIS_TOP_SOFTNESS="${ATTN_VIS_TOP_SOFTNESS:-0.05}"
ATTN_VIS_GAP="${ATTN_VIS_GAP:-8}"
ATTN_VIS_HEADER_HEIGHT="${ATTN_VIS_HEADER_HEIGHT:-24}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SEED="${SEED:-0}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-10}"

exec > >(tee -a "$SHELL_LOG") 2>&1
set -x
trap 'rc=$?; echo "[ERROR] test_libero_trainset_attn_vis_mot2_action.sh failed with exit code ${rc}"' ERR

write_hparam() {
  local key="$1"
  printf '%s=%s\n' "$key" "${!key}"
}

{
  for key in \
    SCRIPT_DIR LAST05_ROOT LAST05_BETA_ROOT DATABASE_ROOT PRETRAINED_ROOT COSMOS_ROOT EXPERIMENTS_ROOT OUTPUT_ROOT_DIR RUN_NAME RUN_DIR CHECKPOINT_NAME PRETRAINED_CHECKPOINT \
    EVAL_TIMESTAMP EVAL_ARTIFACT_NAME ATTENTION_VISUALIZATION_DIR LOG_DIR SHELL_LOG BASH_HPARAMS_FILE \
    OMP_NUM_THREADS HF_HUB_OFFLINE CONDA_ACTIVATE CONDA_ENV_PATH \
    DATA_JSON ACTION_EXPERT_PATH JANUS_INIT_MODE JANUS_PRO_MODEL_PATH JANUS_MODEL_PATH ACTION_MODEL_PATH COSMOS_MODEL_PATH COSMOS_EXPERIMENT_NAME COSMOS_TEXT_CACHE_PATH \
    ACTION_DIM ACTION_CHUNK VIDEO_H VIDEO_W VIDEO_FRAMES NUM_COND_INPUT_FRAMES IMG_LATENTS_PER_FUTURE STATE_LATENTS_PER_FUTURE NUM_FUTURE_FRAMES FUTURE_FRAME_STRIDE \
    ROBOT_STATE STATE_PLACEHOLDER_TOKENS STATE_DIM STATE_ENCODING_MODE \
    BRIDGE_POS_SCHEME ACTION_SELF_CAUSAL_IN_BRIDGE ACTION_USE_LATENT_PREFIX COSMOS_SELF_ONLY_BRIDGE DECOSMOS \
    ACTION_DETACH_SLOW_PREFIX ACTION_DETACH_VIDEO_BRANCH ACTION_DETACH_SLOW_VIDEO_BRANCH \
    FPS ACTION_DENOISE_STEPS COSMOS_DENOISE_STEPS \
    SPATIAL_TOKEN_MODE_RAW SPATIAL_TOKEN_MODE TOTAL_SPATIAL_TOKEN_COUNT EXTRA_SPECIAL_TOKENS \
    USE_SPATIAL_HIDDEN_SIM_LOSS SPATIAL_HIDDEN_SIM_LOSS_MODE SPATIAL_HIDDEN_SIM_LOSS_WEIGHT \
    USE_SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS_WEIGHT WAN21_VAE_PATH \
    TASK_NAMES NUM_TRAJECTORIES_PER_TASK MAX_RECORDS_PER_EPISODE MAX_TOTAL_RECORDS \
    ATTN_VIS_TILE_SIZE ATTN_VIS_ALPHA ATTN_VIS_CAPTURE_MODE ATTN_VIS_TOP_RATIO ATTN_VIS_TOP_SOFTNESS ATTN_VIS_GAP ATTN_VIS_HEADER_HEIGHT CUDA_DEVICE SEED EMPTY_CACHE_EVERY \
    PYREP_PYTHON_PATH LIFT3D_ROOT RLBENCH_ROOT
  do
    write_hparam "$key"
  done
} > "$BASH_HPARAMS_FILE"

cd "$LAST05_ROOT"
source "$CONDA_ACTIVATE" "$CONDA_ENV_PATH"
export PATH="${CONDA_ENV_PATH}/bin:$PATH"
export PYTHONPATH="${PYREP_PYTHON_PATH}:${LIFT3D_ROOT}:${RLBENCH_ROOT}:${COSMOS_ROOT}:${LAST05_ROOT}:${PYTHONPATH:-}"
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
echo "[INFO] spatial token mode: ${SPATIAL_TOKEN_MODE} (count=${TOTAL_SPATIAL_TOKEN_COUNT}, input=${SPATIAL_TOKEN_MODE_RAW})"
echo "[INFO] working dir: $(pwd)"
echo "[INFO] python: $(which python)"
echo "[INFO] bash hparams begin"
sed -n '1,260p' "$BASH_HPARAMS_FILE"
echo "[INFO] bash hparams end"
python -V
python -c "import sys; print('[INFO] sys.executable:', sys.executable)"

python -u "${SCRIPT_DIR}/run_libero_trainset_attn_vis_mot2_action.py" \
  --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
  --model_path "$JANUS_MODEL_PATH" \
  --action_model_path "$ACTION_MODEL_PATH" \
  --action_expert_path "$ACTION_EXPERT_PATH" \
  --janus_init_mode "$JANUS_INIT_MODE" \
  --janus_pro_model_path "$JANUS_PRO_MODEL_PATH" \
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
  --attention_visualization_gap "$ATTN_VIS_GAP" \
  --attention_visualization_header_height "$ATTN_VIS_HEADER_HEIGHT" \
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
  --total_latent_tokens "$TOTAL_SPATIAL_TOKEN_COUNT" \
  --latent_token_mode "$SPATIAL_TOKEN_MODE" \
  --extra_special_tokens "$EXTRA_SPECIAL_TOKENS" \
  --img_latents_per_future "$IMG_LATENTS_PER_FUTURE" \
  --state_latents_per_future "$STATE_LATENTS_PER_FUTURE" \
  --num_future_frames "$NUM_FUTURE_FRAMES" \
  --future_frame_stride "$FUTURE_FRAME_STRIDE" \
  --use_latent_hidden_sim_loss "$USE_SPATIAL_HIDDEN_SIM_LOSS" \
  --latent_hidden_sim_loss_mode "$SPATIAL_HIDDEN_SIM_LOSS_MODE" \
  --latent_hidden_sim_loss_weight "$SPATIAL_HIDDEN_SIM_LOSS_WEIGHT" \
  --use_latent_hidden_wan_downsample_sim_loss "$USE_SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS" \
  --latent_hidden_wan_downsample_sim_loss_weight "$SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS_WEIGHT" \
  --wan21_vae_path "$WAN21_VAE_PATH" \
  --cosmos_self_only_bridge "$COSMOS_SELF_ONLY_BRIDGE" \
  --decosmos "$DECOSMOS" \
  --bridge_pos_scheme "$BRIDGE_POS_SCHEME" \
  --action_use_latent_prefix "$ACTION_USE_LATENT_PREFIX" \
  --action_self_causal_in_bridge "$ACTION_SELF_CAUSAL_IN_BRIDGE" \
  --action_detach_slow_prefix "$ACTION_DETACH_SLOW_PREFIX" \
  --action_detach_video_branch "$ACTION_DETACH_VIDEO_BRANCH" \
  --action_detach_slow_video_branch "$ACTION_DETACH_SLOW_VIDEO_BRANCH" \
  --action_denoise_steps "$ACTION_DENOISE_STEPS" \
  --cosmos_denoise_steps "$COSMOS_DENOISE_STEPS" \
  --fps "$FPS" \
  --empty_cache_every "$EMPTY_CACHE_EVERY"
