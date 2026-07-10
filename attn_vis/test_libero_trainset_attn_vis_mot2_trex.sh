#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAST05_ROOT="${LAST05_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LAST05_BETA_ROOT="${LAST05_BETA_ROOT:-$(cd "${LAST05_ROOT}/.." && pwd)}"
REQUESTED_CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
REQUESTED_CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"

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
CONDA_ENV_NAME="${REQUESTED_CONDA_ENV_NAME:-last05_packed}"
CONDA_ENV_PATH="${REQUESTED_CONDA_ENV_PATH:-${DATABASE_ROOT}/envs/${CONDA_ENV_NAME}}"
COSMOS_ROOT="${COSMOS_ROOT:-/mnt/nas/zhangyiming/experiments}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-${LAST05_BETA_ROOT}/experiments}"
OUTPUT_ROOT_DIR="${OUTPUT_ROOT_DIR:-${LAST05_ROOT}/exp_mot2_trex_action_spatial}"
PYREP_PYTHON_PATH="${PYREP_PYTHON_PATH:-/mnt/nas/zhangyawen/zhangyiming/python_pkgs}"
LIFT3D_ROOT="${LIFT3D_ROOT:-${REQUIRES_ROOT}/LIFT3D}"
RLBENCH_ROOT="${RLBENCH_ROOT:-${LIFT3D_ROOT}/third_party/RLBench}"
COSMOS_CUDA_SHIM_ROOT="${COSMOS_CUDA_SHIM_ROOT:-${LAST05_BETA_ROOT}/last05_mot2_action_fis}"
COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-${REQUIRES_ROOT}/CoppeliaSim}"

RUN_NAME="${RUN_NAME:-cosmos2B_trex2B_libero_spatial}"
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
SHELL_LOG="${LOG_DIR}/test_libero_trainset_attn_vis_mot2_trex_shell_${EVAL_ARTIFACT_NAME}.log"
BASH_HPARAMS_FILE="${ATTENTION_VISUALIZATION_DIR}/test_libero_trainset_attn_vis_mot2_trex_hparams_${EVAL_ARTIFACT_NAME}.env"

DATA_JSON="${DATA_JSON:-${DATABASE_ROOT}/data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual/train_with_atomic_action_shared.json}"
JANUS_MODEL_PATH="${JANUS_MODEL_PATH:-${PRETRAINED_ROOT}/T-Rex_pretrain_mecka22k_epoch1}"
ACTION_MODEL_PATH="${ACTION_MODEL_PATH:-${PRETRAINED_ROOT}/T-Rex_pretrain_mecka22k_epoch1}"
COSMOS_MODEL_PATH="${COSMOS_MODEL_PATH:-${PRETRAINED_ROOT}/Cosmos-Predict2.5-2B/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt}"
COSMOS_EXPERIMENT_NAME="${COSMOS_EXPERIMENT_NAME:-Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only}"
COSMOS_TEXT_CACHE_PATH="${COSMOS_TEXT_CACHE_PATH:-${DATABASE_ROOT}/data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual/cosmos_text_cache_raw_full_concat}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

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
ACTION_SELF_CAUSAL_IN_BRIDGE="${ACTION_SELF_CAUSAL_IN_BRIDGE:-true}"
ACTION_USE_LATENT_PREFIX="${ACTION_USE_LATENT_PREFIX:-true}"
ACTION_INSERT_LAYER="${ACTION_INSERT_LAYER:-0}"
DETACH_ACTION_COSMOS_KV="${DETACH_ACTION_COSMOS_KV:-0}"
COSMOS_SELF_ONLY_BRIDGE="${COSMOS_SELF_ONLY_BRIDGE:-false}"
DECOSMOS="${DECOSMOS:-false}"
FPS="${FPS:-20}"
ACTION_DENOISE_STEPS="${ACTION_DENOISE_STEPS:-10}"
COSMOS_DENOISE_STEPS="${COSMOS_DENOISE_STEPS:-2}"

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

DEFAULT_SPECIAL_TOKEN_VOCAB="</MOVE>,</BOWWL>,</PICK>,</PLACE>,</APPROACH>,</bowl>"
SPECIAL_TOKEN_VOCAB="${SPECIAL_TOKEN_VOCAB:-${DEFAULT_SPECIAL_TOKEN_VOCAB}}"
USE_SPATIAL_HIDDEN_SIM_LOSS="${USE_SPATIAL_HIDDEN_SIM_LOSS:-1}"
SPATIAL_HIDDEN_SIM_LOSS_MODE="${SPATIAL_HIDDEN_SIM_LOSS_MODE:-siglip}"
SPATIAL_HIDDEN_SIM_POOL_MODE="${SPATIAL_HIDDEN_SIM_POOL_MODE:-pool}"
SPATIAL_HIDDEN_SIM_LOSS_WEIGHT="${SPATIAL_HIDDEN_SIM_LOSS_WEIGHT:-1}"
USE_SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS="${USE_SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS:-0}"
SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS_WEIGHT="${SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS_WEIGHT:-1.0}"
WAN21_VAE_PATH="${WAN21_VAE_PATH:-${PRETRAINED_ROOT}/wan2.1_vae/original/Wan2.1_VAE.pth}"

TASK_NAMES="${TASK_NAMES:-}"
NUM_TRAJECTORIES_PER_TASK="${NUM_TRAJECTORIES_PER_TASK:-2}"
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
trap 'rc=$?; echo "[ERROR] test_libero_trainset_attn_vis_mot2_trex.sh failed with exit code ${rc}"' ERR

write_hparam() {
  local key="$1"
  printf '%s=%s\n' "$key" "${!key}"
}

{
  for key in \
    SCRIPT_DIR LAST05_ROOT LAST05_BETA_ROOT DATABASE_ROOT PRETRAINED_ROOT COSMOS_ROOT EXPERIMENTS_ROOT OUTPUT_ROOT_DIR RUN_NAME RUN_DIR CHECKPOINT_NAME PRETRAINED_CHECKPOINT \
    EVAL_TIMESTAMP EVAL_ARTIFACT_NAME ATTENTION_VISUALIZATION_DIR LOG_DIR SHELL_LOG BASH_HPARAMS_FILE \
    OMP_NUM_THREADS HF_HUB_OFFLINE CONDA_BASE CONDA_ENV_NAME CONDA_ENV_PATH COSMOS_CUDA_SHIM_ROOT \
    DATA_JSON JANUS_MODEL_PATH ACTION_MODEL_PATH COSMOS_MODEL_PATH COSMOS_EXPERIMENT_NAME COSMOS_TEXT_CACHE_PATH \
    ACTION_DIM ACTION_CHUNK VIDEO_H VIDEO_W VIDEO_FRAMES NUM_COND_INPUT_FRAMES IMG_LATENTS_PER_FUTURE STATE_LATENTS_PER_FUTURE NUM_FUTURE_FRAMES FUTURE_FRAME_STRIDE \
    ROBOT_STATE STATE_PLACEHOLDER_TOKENS STATE_DIM STATE_ENCODING_MODE \
    BRIDGE_POS_SCHEME ACTION_SELF_CAUSAL_IN_BRIDGE ACTION_USE_LATENT_PREFIX ACTION_INSERT_LAYER DETACH_ACTION_COSMOS_KV COSMOS_SELF_ONLY_BRIDGE DECOSMOS \
    FPS ACTION_DENOISE_STEPS COSMOS_DENOISE_STEPS \
    SPATIAL_TOKEN_MODE_RAW SPATIAL_TOKEN_MODE TOTAL_SPATIAL_TOKEN_COUNT SPECIAL_TOKEN_VOCAB \
    USE_SPATIAL_HIDDEN_SIM_LOSS SPATIAL_HIDDEN_SIM_LOSS_MODE SPATIAL_HIDDEN_SIM_POOL_MODE SPATIAL_HIDDEN_SIM_LOSS_WEIGHT \
    USE_SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS_WEIGHT WAN21_VAE_PATH \
    TASK_NAMES NUM_TRAJECTORIES_PER_TASK MAX_RECORDS_PER_EPISODE MAX_TOTAL_RECORDS \
    ATTN_VIS_TILE_SIZE ATTN_VIS_ALPHA ATTN_VIS_CAPTURE_MODE ATTN_VIS_TOP_RATIO ATTN_VIS_TOP_SOFTNESS ATTN_VIS_GAP ATTN_VIS_HEADER_HEIGHT CUDA_DEVICE SEED EMPTY_CACHE_EVERY \
    PYREP_PYTHON_PATH LIFT3D_ROOT RLBENCH_ROOT COPPELIASIM_ROOT
  do
    write_hparam "$key"
  done
} > "$BASH_HPARAMS_FILE"

cd "$LAST05_ROOT"
source "${CONDA_BASE}/bin/activate" "${CONDA_ENV_PATH}"
export PATH="${CONDA_ENV_PATH}/bin:$PATH"
if [[ -f "${COSMOS_CUDA_SHIM_ROOT}/cosmos_cuda.py" ]]; then
  export PYTHONPATH="${PYREP_PYTHON_PATH}:${LIFT3D_ROOT}:${RLBENCH_ROOT}:${COSMOS_ROOT}:${LAST05_ROOT}:${COSMOS_CUDA_SHIM_ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${PYREP_PYTHON_PATH}:${LIFT3D_ROOT}:${RLBENCH_ROOT}:${COSMOS_ROOT}:${LAST05_ROOT}:${PYTHONPATH:-}"
fi
export OMP_NUM_THREADS
export HF_HUB_OFFLINE
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
unset LD_PRELOAD
export COPPELIASIM_ROOT
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${COPPELIASIM_ROOT}:${COPPELIASIM_ROOT}/platforms:${CONDA_ENV_PATH}/lib"

echo "[INFO] shell log: $SHELL_LOG"
echo "[INFO] bash hparams file: $BASH_HPARAMS_FILE"
echo "[INFO] eval artifact name: $EVAL_ARTIFACT_NAME"
echo "[INFO] attention visualization dir: $ATTENTION_VISUALIZATION_DIR"
echo "[INFO] checkpoint: $PRETRAINED_CHECKPOINT"
echo "[INFO] spatial token mode: ${SPATIAL_TOKEN_MODE} (count=${TOTAL_SPATIAL_TOKEN_COUNT}, input=${SPATIAL_TOKEN_MODE_RAW})"
echo "[INFO] action insert layer: ${ACTION_INSERT_LAYER}"
echo "[INFO] working dir: $(pwd)"
echo "[INFO] python: $(which python)"
echo "[INFO] bash hparams begin"
sed -n '1,260p' "$BASH_HPARAMS_FILE"
echo "[INFO] bash hparams end"
python -V
python -c "import sys; print('[INFO] sys.executable:', sys.executable)"

python -u "${SCRIPT_DIR}/run_libero_trainset_attn_vis_mot2_trex.py" \
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
  --special_token_vocab "$SPECIAL_TOKEN_VOCAB" \
  --img_latents_per_future "$IMG_LATENTS_PER_FUTURE" \
  --state_latents_per_future "$STATE_LATENTS_PER_FUTURE" \
  --num_future_frames "$NUM_FUTURE_FRAMES" \
  --future_frame_stride "$FUTURE_FRAME_STRIDE" \
  --use_latent_hidden_sim_loss "$USE_SPATIAL_HIDDEN_SIM_LOSS" \
  --latent_hidden_sim_loss_mode "$SPATIAL_HIDDEN_SIM_LOSS_MODE" \
  --latent_hidden_sim_pool_mode "$SPATIAL_HIDDEN_SIM_POOL_MODE" \
  --latent_hidden_sim_loss_weight "$SPATIAL_HIDDEN_SIM_LOSS_WEIGHT" \
  --use_latent_hidden_wan_downsample_sim_loss "$USE_SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS" \
  --latent_hidden_wan_downsample_sim_loss_weight "$SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS_WEIGHT" \
  --wan21_vae_path "$WAN21_VAE_PATH" \
  --cosmos_self_only_bridge "$COSMOS_SELF_ONLY_BRIDGE" \
  --decosmos "$DECOSMOS" \
  --bridge_pos_scheme "$BRIDGE_POS_SCHEME" \
  --action_use_latent_prefix "$ACTION_USE_LATENT_PREFIX" \
  --action_self_causal_in_bridge "$ACTION_SELF_CAUSAL_IN_BRIDGE" \
  --action_insert_layer "$ACTION_INSERT_LAYER" \
  --detach_action_cosmos_kv "$DETACH_ACTION_COSMOS_KV" \
  --action_denoise_steps "$ACTION_DENOISE_STEPS" \
  --cosmos_denoise_steps "$COSMOS_DENOISE_STEPS" \
  --fps "$FPS" \
  --empty_cache_every "$EMPTY_CACHE_EVERY"
