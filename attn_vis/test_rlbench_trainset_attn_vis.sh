#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAST05_ROOT="${LAST05_ROOT:-/mnt/nas/zhangyiming/last05_beta/last05_mot2_trex_action}"
COSMOS_ROOT="${COSMOS_ROOT:-/mnt/nas/zhangyiming/experiments}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-/mnt/nas/zhangyiming/last05_beta/experiments_rlbench}"
OUTPUT_ROOT_DIR="${OUTPUT_ROOT_DIR:-${LAST05_ROOT}/exp_mot2_trex_action_spatial_rlbench_keyframe}"
PYREP_PYTHON_PATH="${PYREP_PYTHON_PATH:-/mnt/nas/zhangyawen/zhangyiming/python_pkgs}"
LIFT3D_ROOT="${LIFT3D_ROOT:-/mnt/nas/zhangyiming/requires/LIFT3D}"
RLBENCH_ROOT="${RLBENCH_ROOT:-${LIFT3D_ROOT}/third_party/RLBench}"

RUN_NAME="${RUN_NAME:-cosmos2B_trex2B_mot2_rlbench_keyframe_spatial_v_new}"
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
SHELL_LOG="${LOG_DIR}/test_rlbench_trainset_attn_vis_shell_${EVAL_ARTIFACT_NAME}.log"
BASH_HPARAMS_FILE="${ATTENTION_VISUALIZATION_DIR}/test_rlbench_trainset_attn_vis_hparams_${EVAL_ARTIFACT_NAME}.env"

DATA_JSON="${DATA_JSON:-/mnt/nas/zhangyiming/database/rlbench/train/json/train_action_chunk1_sumpos_lastrot.json}"
QWEN3VL2B_MODEL_PATH="${QWEN3VL2B_MODEL_PATH:-/mnt/amlfs-07/shared/physicalword/ckpt/pretraine/Qwen3-VL-2B-Instruct}"
JANUS_MODEL_PATH="${JANUS_MODEL_PATH:-${QWEN3VL2B_MODEL_PATH}}"
ACTION_MODEL_PATH="${ACTION_MODEL_PATH:-/mnt/nas/zhangyiming/database/ckpt/pretrained/T-Rex_pretrain_mecka22k_epoch1}"
COSMOS_MODEL_PATH="${COSMOS_MODEL_PATH:-/mnt/nas/zhangyiming/database/ckpt/pretrained/Cosmos-Predict2.5-2B/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt}"
COSMOS_EXPERIMENT_NAME="${COSMOS_EXPERIMENT_NAME:-Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only}"
COSMOS_TEXT_CACHE_PATH="${COSMOS_TEXT_CACHE_PATH:-/mnt/nas/zhangyiming/database/rlbench/train/json/cosmos_text_cache_rlbench_keyframe}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

ACTION_DIM="${ACTION_DIM:-7}"
ACTION_CHUNK="${ACTION_CHUNK:-1}"
VIDEO_H="${VIDEO_H:-256}"
VIDEO_W="${VIDEO_W:-256}"
VIDEO_FRAMES="${VIDEO_FRAMES:-9}"
NUM_COND_INPUT_FRAMES="${NUM_COND_INPUT_FRAMES:-5}"
IMG_LATENTS_PER_FUTURE=0
STATE_LATENTS_PER_FUTURE=0
NUM_FUTURE_FRAMES=0
FUTURE_FRAME_STRIDE="${FUTURE_FRAME_STRIDE:-1}"
ROBOT_STATE="${ROBOT_STATE:-0}"
STATE_PLACEHOLDER_TOKENS="${STATE_PLACEHOLDER_TOKENS:-1}"
STATE_DIM="${STATE_DIM:-7}"
STATE_ENCODING_MODE="${STATE_ENCODING_MODE:-mlp}"
BRIDGE_POS_SCHEME="${BRIDGE_POS_SCHEME:-mrope}"
ACTION_SELF_CAUSAL_IN_BRIDGE="${ACTION_SELF_CAUSAL_IN_BRIDGE:-true}"
ACTION_USE_LATENT_PREFIX="${ACTION_USE_LATENT_PREFIX:-true}"
RIGHT_SINGLE_ATTN_POSITION="${RIGHT_SINGLE_ATTN_POSITION:-last4}"
COSMOS_SELF_ONLY_BRIDGE="${COSMOS_SELF_ONLY_BRIDGE:-false}"
DECOSMOS="${DECOSMOS:-false}"
FPS="${FPS:-10}"
ACTION_DENOISE_STEPS="${ACTION_DENOISE_STEPS:-10}"
COSMOS_DENOISE_STEPS="${COSMOS_DENOISE_STEPS:-2}"

SPATIAL_TOKEN_MODE_RAW="${SPATIAL_TOKEN_MODE:-${TOTAL_LATENT_TOKENS:-v}}"
SPATIAL_TOKEN_MODE="$(printf '%s' "${SPATIAL_TOKEN_MODE_RAW}" | tr '[:upper:]' '[:lower:]')"
case "${SPATIAL_TOKEN_MODE}" in
  1|v)
    SPATIAL_TOKEN_MODE="v"
    TOTAL_SPATIAL_TOKEN_COUNT=1
    ;;
  n)
    SPATIAL_TOKEN_MODE="n"
    TOTAL_SPATIAL_TOKEN_COUNT=1
    ;;
  2|vn)
    SPATIAL_TOKEN_MODE="vn"
    TOTAL_SPATIAL_TOKEN_COUNT=2
    ;;
  *)
    echo "ERROR: SPATIAL_TOKEN_MODE must be one of v, n, vn, 1, or 2; got '${SPATIAL_TOKEN_MODE_RAW}'." >&2
    exit 1
    ;;
esac

DEFAULT_SPECIAL_TOKEN_VOCAB="</PAD>,</MOVE>,</PICK>,</PLACE>,</ROTATE>,</PULL>,</PUSH>,</NONE>,</box>,</broom>,</charger>,</frame>,</fridge>,</lamp>,</laptop>,</phone>,</toilet>,</umbrella>,</watering_can>,</wine>"
SPECIAL_TOKEN_VOCAB="${SPECIAL_TOKEN_VOCAB:-${DEFAULT_SPECIAL_TOKEN_VOCAB}}"
USE_SPATIAL_HIDDEN_SIM_LOSS="${USE_SPATIAL_HIDDEN_SIM_LOSS:-1}"
SPATIAL_HIDDEN_SIM_LOSS_MODE="${SPATIAL_HIDDEN_SIM_LOSS_MODE:-siglip}"

if [[ -z "$QWEN3VL2B_MODEL_PATH" ]]; then
  echo "[ERROR] QWEN3VL2B_MODEL_PATH must be set for the 32-layer right-branch architecture."
  exit 1
fi

DEFAULT_TASK_NAMES="close_box,close_laptop_lid,sweep_to_dustpan,phone_on_base,toilet_seat_down,close_fridge,place_wine_at_rack_location,water_plants,take_umbrella_out_of_umbrella_stand,take_frame_off_hanger"
TASK_IDS="${TASK_IDS:-}"
if [[ -n "${TASK_NAMES:-}" ]]; then
  TASK_NAMES="${TASK_NAMES}"
elif [[ -n "$TASK_IDS" ]]; then
  IFS=',' read -r -a DEFAULT_TASK_ARRAY <<< "$DEFAULT_TASK_NAMES"
  TASK_NAMES=""
  IFS=',' read -r -a TASK_ID_ARRAY <<< "$TASK_IDS"
  for raw_task_id in "${TASK_ID_ARRAY[@]}"; do
    task_id="${raw_task_id//[[:space:]]/}"
    if [[ ! "$task_id" =~ ^[0-9]+$ ]]; then
      echo "[ERROR] Invalid TASK_IDS entry: ${raw_task_id}. Use comma-separated integers in [0, 9]."
      exit 1
    fi
    if (( task_id < 0 || task_id >= ${#DEFAULT_TASK_ARRAY[@]} )); then
      echo "[ERROR] TASK_IDS entry out of range: ${task_id}. Valid range is 0-9."
      exit 1
    fi
    if [[ -n "$TASK_NAMES" ]]; then
      TASK_NAMES+=","
    fi
    TASK_NAMES+="${DEFAULT_TASK_ARRAY[$task_id]}"
  done
else
  TASK_NAMES=""
fi

NUM_TRAJECTORIES_PER_TASK="${NUM_TRAJECTORIES_PER_TASK:-1}"
MAX_RECORDS_PER_EPISODE="${MAX_RECORDS_PER_EPISODE:-0}"
MAX_TOTAL_RECORDS="${MAX_TOTAL_RECORDS:-0}"
ATTN_VIS_TILE_SIZE="${ATTN_VIS_TILE_SIZE:-256}"
ATTN_VIS_ALPHA="${ATTN_VIS_ALPHA:-0.45}"
ATTN_VIS_CAPTURE_MODE="${ATTN_VIS_CAPTURE_MODE:-last}"
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
    SCRIPT_DIR LAST05_ROOT COSMOS_ROOT EXPERIMENTS_ROOT OUTPUT_ROOT_DIR RUN_NAME RUN_DIR CHECKPOINT_NAME PRETRAINED_CHECKPOINT \
    EVAL_TIMESTAMP EVAL_ARTIFACT_NAME ATTENTION_VISUALIZATION_DIR LOG_DIR SHELL_LOG BASH_HPARAMS_FILE \
    OMP_NUM_THREADS HF_HUB_OFFLINE \
    DATA_JSON JANUS_MODEL_PATH ACTION_MODEL_PATH QWEN3VL2B_MODEL_PATH COSMOS_MODEL_PATH COSMOS_EXPERIMENT_NAME COSMOS_TEXT_CACHE_PATH \
    ACTION_DIM ACTION_CHUNK VIDEO_H VIDEO_W VIDEO_FRAMES NUM_COND_INPUT_FRAMES IMG_LATENTS_PER_FUTURE STATE_LATENTS_PER_FUTURE NUM_FUTURE_FRAMES FUTURE_FRAME_STRIDE \
    ROBOT_STATE STATE_PLACEHOLDER_TOKENS STATE_DIM STATE_ENCODING_MODE \
    BRIDGE_POS_SCHEME ACTION_SELF_CAUSAL_IN_BRIDGE ACTION_USE_LATENT_PREFIX RIGHT_SINGLE_ATTN_POSITION COSMOS_SELF_ONLY_BRIDGE DECOSMOS \
    FPS ACTION_DENOISE_STEPS COSMOS_DENOISE_STEPS \
    SPATIAL_TOKEN_MODE_RAW SPATIAL_TOKEN_MODE TOTAL_SPATIAL_TOKEN_COUNT SPECIAL_TOKEN_VOCAB USE_SPATIAL_HIDDEN_SIM_LOSS SPATIAL_HIDDEN_SIM_LOSS_MODE \
    DEFAULT_TASK_NAMES TASK_IDS TASK_NAMES NUM_TRAJECTORIES_PER_TASK MAX_RECORDS_PER_EPISODE MAX_TOTAL_RECORDS \
    ATTN_VIS_TILE_SIZE ATTN_VIS_ALPHA ATTN_VIS_CAPTURE_MODE ATTN_VIS_TOP_RATIO ATTN_VIS_TOP_SOFTNESS CUDA_DEVICE SEED EMPTY_CACHE_EVERY \
    PYREP_PYTHON_PATH LIFT3D_ROOT RLBENCH_ROOT
  do
    write_hparam "$key"
  done
} > "$BASH_HPARAMS_FILE"

cd "$LAST05_ROOT"
source /root/miniconda3/bin/activate /root/miniconda3/envs/last05
export PATH=/root/miniconda3/envs/last05/bin:$PATH
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
sed -n '1,240p' "$BASH_HPARAMS_FILE"
echo "[INFO] bash hparams end"
python -V
python -c "import sys; print('[INFO] sys.executable:', sys.executable)"

python -u "${SCRIPT_DIR}/run_rlbench_trainset_attn_vis_mot2_trex.py" \
  --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
  --model_path "$JANUS_MODEL_PATH" \
  --action_model_path "$ACTION_MODEL_PATH" \
  --qwen3vl2b_model_path "$QWEN3VL2B_MODEL_PATH" \
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
  --total_latent_tokens "$TOTAL_SPATIAL_TOKEN_COUNT" \
  --latent_token_mode "$SPATIAL_TOKEN_MODE" \
  --special_token_vocab "$SPECIAL_TOKEN_VOCAB" \
  --img_latents_per_future "$IMG_LATENTS_PER_FUTURE" \
  --state_latents_per_future "$STATE_LATENTS_PER_FUTURE" \
  --num_future_frames "$NUM_FUTURE_FRAMES" \
  --future_frame_stride "$FUTURE_FRAME_STRIDE" \
  --use_latent_hidden_sim_loss "$USE_SPATIAL_HIDDEN_SIM_LOSS" \
  --latent_hidden_sim_loss_mode "$SPATIAL_HIDDEN_SIM_LOSS_MODE" \
  --cosmos_self_only_bridge "$COSMOS_SELF_ONLY_BRIDGE" \
  --decosmos "$DECOSMOS" \
  --bridge_pos_scheme "$BRIDGE_POS_SCHEME" \
  --action_use_latent_prefix "$ACTION_USE_LATENT_PREFIX" \
  --action_self_causal_in_bridge "$ACTION_SELF_CAUSAL_IN_BRIDGE" \
  --right_single_attn_position "$RIGHT_SINGLE_ATTN_POSITION" \
  --action_denoise_steps "$ACTION_DENOISE_STEPS" \
  --cosmos_denoise_steps "$COSMOS_DENOISE_STEPS" \
  --fps "$FPS" \
  --empty_cache_every "$EMPTY_CACHE_EVERY"
