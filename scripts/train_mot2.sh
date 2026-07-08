#!/bin/bash
set -e

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
CONDA_ENV_NAME="${CONDA_ENV_NAME:-last05_qwen3vl}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${CONDA_BASE}/envs/${CONDA_ENV_NAME}}"
EXPERIMENTS_LIBERO_ROOT="${EXPERIMENTS_LIBERO_ROOT:-${LAST05_BETA_ROOT}/experiments}"
EXPERIMENTS_RLBENCH_ROOT="${EXPERIMENTS_RLBENCH_ROOT:-${LAST05_BETA_ROOT}/experiments_rlbench}"
COSMOS_ROOT="${COSMOS_ROOT:-/mnt/nas/zhangyiming/experiments}"
LIBERO_ROOT="${LIBERO_ROOT:-/mnt/nas/zhangxuheng/LIBERO}"
PYREP_PYTHON_PATH="${PYREP_PYTHON_PATH:-/mnt/nas/zhangyawen/zhangyiming/python_pkgs}"
LIFT3D_ROOT="${LIFT3D_ROOT:-${REQUIRES_ROOT}/LIFT3D}"
RLBENCH_ROOT="${RLBENCH_ROOT:-${LIFT3D_ROOT}/third_party/RLBench}"

cd "${LAST05_ROOT}/scripts"
source "${CONDA_BASE}/bin/activate" "${CONDA_ENV_PATH}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PATH="${CONDA_ENV_PATH}/bin:$PATH"
export PYTHONPATH="${LAST05_ROOT}:${PYTHONPATH:-}"
LEGACY_LAST05_1_BIN="${LEGACY_LAST05_1_BIN:-/media/miniconda3/envs/last05.1/bin}"
if [[ -d "${LEGACY_LAST05_1_BIN}" ]]; then
  export PATH="${LEGACY_LAST05_1_BIN}:$PATH"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-cosmos_janus_mot2_fis_libero_spatial}"
RUN_NAME="${RUN_NAME:-cosmos2B_action1B_mot2_fis_libero_spatial}"
OUTPUT_ROOT_DIR="${OUTPUT_ROOT_DIR:-${LAST05_ROOT}/exp_mot2_action_fis_spatial}"

DATA_JSON="${DATA_JSON:-/mnt/amlfs-07/shared/physicalword/data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual/train_with_atomic_action_shared.json}"
ACTION_EXPERT_PATH="${ACTION_EXPERT_PATH:-${PRETRAINED_ROOT}/LaST0_Pretrain_AE_chunk16/tfmr}"
JANUS_INIT_MODE="${JANUS_INIT_MODE:-janus_pro_ae_flow}"
JANUS_PRO_MODEL_PATH="${JANUS_PRO_MODEL_PATH:-${PRETRAINED_ROOT}/Janus-Pro-1B}"
COSMOS_PT_PATH="${COSMOS_PT_PATH:-${PRETRAINED_ROOT}/Cosmos-Predict2.5-2B/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt}"
COSMOS_EXP_NAME="${COSMOS_EXP_NAME:-Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only}"

COSMOS_TEXT_CACHE_PATH="${COSMOS_TEXT_CACHE_PATH:-${DATABASE_ROOT}/data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual/cosmos_text_cache_raw_full_concat}"

NUM_PROCESSES="${NUM_PROCESSES:-8}"
TRAIN_BSZ="${TRAIN_BSZ:-6}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1e-4}"
COSMOS_CORE_LR_RATIO="${COSMOS_CORE_LR_RATIO:-0.02}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-8}"

VIDEO_H="${VIDEO_H:-256}"
VIDEO_W="${VIDEO_W:-256}"
VIDEO_FRAMES="${VIDEO_FRAMES:-17}"
NUM_COND_INPUT_FRAMES="${NUM_COND_INPUT_FRAMES:-5}"
ACTION_DIM="${ACTION_DIM:-7}"
ACTION_CHUNK="${ACTION_CHUNK:-16}"
ROBOT_STATE="${ROBOT_STATE:-0}"
STATE_PLACEHOLDER_TOKENS="${STATE_PLACEHOLDER_TOKENS:-8}"
STATE_DIM="${STATE_DIM:-8}"
STATE_ENCODING_MODE="${STATE_ENCODING_MODE:-mlp}"

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

DEFAULT_SPECIAL_TOKEN_VOCAB="</PAD>,</MOVE>,</PICK>,</PLACE>,</ROTATE>,</PULL>,</PUSH>,</NONE>,</box>,</broom>,</charger>,</frame>,</fridge>,</lamp>,</laptop>,</phone>,</toilet>,</umbrella>,</watering_can>,</wine>"
SPECIAL_TOKEN_VOCAB="${SPECIAL_TOKEN_VOCAB:-${DEFAULT_SPECIAL_TOKEN_VOCAB}}"

VIDEO_LOSS_WEIGHT="${VIDEO_LOSS_WEIGHT:-1}"
SPATIAL_LOSS_WEIGHT="${SPATIAL_LOSS_WEIGHT:-1.0}"
USE_SPATIAL_HIDDEN_SIM_LOSS="${USE_SPATIAL_HIDDEN_SIM_LOSS:-1}"
SPATIAL_HIDDEN_SIM_LOSS_MODE="${SPATIAL_HIDDEN_SIM_LOSS_MODE:-siglip}"
SPATIAL_HIDDEN_SIM_POOL_MODE="${SPATIAL_HIDDEN_SIM_POOL_MODE:-pool}"
SPATIAL_HIDDEN_SIM_LOSS_WEIGHT="${SPATIAL_HIDDEN_SIM_LOSS_WEIGHT:-1.0}"
USE_SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS="${USE_SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS:-0}"
SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS_WEIGHT="${SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS_WEIGHT:-1.0}"
WAN21_VAE_PATH="${WAN21_VAE_PATH:-${PRETRAINED_ROOT}/wan2.1_vae/original/Wan2.1_VAE.pth}"
FUTURE_FRAME_STRIDE="${FUTURE_FRAME_STRIDE:-8}"
BRIDGE_POS_SCHEME="${BRIDGE_POS_SCHEME:-mrope}"
ACTION_DETACH_SLOW_PREFIX="${ACTION_DETACH_SLOW_PREFIX:-0}"
ACTION_DETACH_VIDEO_BRANCH="${ACTION_DETACH_VIDEO_BRANCH:-0}"
ACTION_DETACH_SLOW_VIDEO_BRANCH="${ACTION_DETACH_SLOW_VIDEO_BRANCH:-0}"

echo ">>> Starting FiS 2-MoT Libero training: ${RUN_NAME}"
echo ">>> Spatial token mode: ${SPATIAL_TOKEN_MODE} count=${TOTAL_SPATIAL_TOKEN_COUNT}"
echo ">>> Janus init mode: ${JANUS_INIT_MODE}"
echo ">>> action_detach_slow_prefix=${ACTION_DETACH_SLOW_PREFIX}"
echo ">>> action_detach_video_branch=${ACTION_DETACH_VIDEO_BRANCH}"
echo ">>> action_detach_slow_video_branch=${ACTION_DETACH_SLOW_VIDEO_BRANCH}"

accelerate launch --config_file ../config/sft.yaml \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  --machine_rank 0 \
  --deepspeed_multinode_launcher standard train_mot2.py \
  --experiment_name "${EXPERIMENT_NAME}" \
  --run_name "${RUN_NAME}" \
  --action_expert_path "${ACTION_EXPERT_PATH}" \
  --janus_init_mode "${JANUS_INIT_MODE}" \
  --janus_pro_model_path "${JANUS_PRO_MODEL_PATH}" \
  --cosmos_model_path "${COSMOS_PT_PATH}" \
  --cosmos_experiment_name "${COSMOS_EXP_NAME}" \
  --cosmos_text_cache_path "${COSMOS_TEXT_CACHE_PATH}" \
  --data_path "${DATA_JSON}" \
  --output_dir "${OUTPUT_ROOT_DIR}" \
  --log_dir "${OUTPUT_ROOT_DIR}" \
  --video_h "${VIDEO_H}" \
  --video_w "${VIDEO_W}" \
  --video_frames "${VIDEO_FRAMES}" \
  --num_cond_input_frames "${NUM_COND_INPUT_FRAMES}" \
  --fps 20 \
  --action_dim "${ACTION_DIM}" \
  --action_chunk "${ACTION_CHUNK}" \
  --n_epochs "${N_EPOCHS:-100}" \
  --save_freq "${SAVE_FREQ:-10}" \
  --train_bsz_per_gpu "${TRAIN_BSZ}" \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --pin_memory 1 \
  --persistent_workers 1 \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LR}" \
  --cosmos_core_lr_ratio "${COSMOS_CORE_LR_RATIO}" \
  --min_lr_ratio 0 \
  --warmup_rates 0 \
  --weight_decay 0 \
  --total_latent_tokens "${TOTAL_SPATIAL_TOKEN_COUNT}" \
  --latent_token_mode "${SPATIAL_TOKEN_MODE}" \
  --special_token_vocab "${SPECIAL_TOKEN_VOCAB}" \
  --future_frame_stride "${FUTURE_FRAME_STRIDE}" \
  --video_loss_weight "${VIDEO_LOSS_WEIGHT}" \
  --latent_loss_weight "${SPATIAL_LOSS_WEIGHT}" \
  --use_latent_hidden_sim_loss "${USE_SPATIAL_HIDDEN_SIM_LOSS}" \
  --latent_hidden_sim_loss_mode "${SPATIAL_HIDDEN_SIM_LOSS_MODE}" \
  --latent_hidden_sim_pool_mode "${SPATIAL_HIDDEN_SIM_POOL_MODE}" \
  --latent_hidden_sim_loss_weight "${SPATIAL_HIDDEN_SIM_LOSS_WEIGHT}" \
  --use_latent_hidden_wan_downsample_sim_loss "${USE_SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS}" \
  --latent_hidden_wan_downsample_sim_loss_weight "${SPATIAL_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS_WEIGHT}" \
  --wan21_vae_path "${WAN21_VAE_PATH}" \
  --bridge_pos_scheme "${BRIDGE_POS_SCHEME}" \
  --action_detach_slow_prefix "${ACTION_DETACH_SLOW_PREFIX}" \
  --action_detach_video_branch "${ACTION_DETACH_VIDEO_BRANCH}" \
  --action_detach_slow_video_branch "${ACTION_DETACH_SLOW_VIDEO_BRANCH}" \
  --freeze_video_after "${FREEZE_VIDEO_AFTER:-100}" \
  --robot_state "${ROBOT_STATE}" \
  --state_placeholder_tokens "${STATE_PLACEHOLDER_TOKENS}" \
  --state_dim "${STATE_DIM}" \
  --state_encoding_mode "${STATE_ENCODING_MODE}"

echo ">>> Training Finished."
