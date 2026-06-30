#!/bin/bash
set -e

# export http_proxy=http://192.168.32.28:18000 && export https_proxy=http://192.168.32.28:18000

LAST05_ROOT="/mnt/nas/zhangyiming/last05_beta/last05"

cd "${LAST05_ROOT}/scripts"
source /root/miniconda3/bin/activate /root/miniconda3/envs/last05
export WANDB_API_KEY="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"
export PATH=/root/miniconda3/envs/last05/bin:$PATH
# export HF_HOME=/media/huggingFace
export PYTHONPATH="${LAST05_ROOT}:/mnt/nas/zhangyiming/last05_beta/last05:${PYTHONPATH:-}"
export PATH=/media/miniconda3/envs/last05.1/bin:$PATH
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

EXPERIMENT_NAME="cosmos_janus_2expert_rlbench_keyframe"

RUN_NAME="cosmos2B_janus1B_2expert_rlbench_keyframe_tokenlatent_hidden_nlatent"

OUTPUT_ROOT_DIR="${LAST05_ROOT}/exp_cosmos_vla_3expert_rlbench_keyframe"

DATA_JSON="/mnt/nas/zhangyiming/database/rlbench/train/json/train_action_chunk1_sumpos_lastrot.json"

JANUS_MODEL_PATH="/mnt/nas/zhangyiming/database/ckpt/pretrained/Janus-Pro-1B"
ACTION_EXPERT_PATH="/mnt/nas/zhangyiming/database/ckpt/pretrained/LaST0_Pretrain_AE_chunk16/tfmr"

COSMOS_PT_PATH="/mnt/nas/zhangyiming/database/ckpt/pretrained/Cosmos-Predict2.5-2B/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt"
COSMOS_EXP_NAME="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only"
COSMOS_TEXT_CACHE_PATH="${COSMOS_TEXT_CACHE_PATH:-}"



NUM_GPUS=8
NUM_PROCESSES=8
TRAIN_BSZ=8
GRAD_ACCUM=1
LR=1e-4
COSMOS_CORE_LR_RATIO=0.02
NUM_WORKERS=4
PIN_MEMORY=1
PERSISTENT_WORKERS=1


ACTION_INTERMEDIATE_SIZE=5632
ACTION_DIM=7
ACTION_CHUNK=1
VIDEO_FRAMES=5
NUM_COND_INPUT_FRAMES=1


ROBOT_STATE=0
STATE_PLACEHOLDER_TOKENS=1
STATE_DIM=7
STATE_ENCODING_MODE="mlp"

NUM_FUTURE_FRAMES=0
IMG_LATENTS_PER_FUTURE=0
STATE_LATENTS_PER_FUTURE=0
FUTURE_FRAME_STRIDE=1
TOTAL_LATENT_TOKENS="${TOTAL_LATENT_TOKENS:-n}"
LATENT_TOKEN_MODE_RAW="${TOTAL_LATENT_TOKENS}"
LATENT_TOKEN_MODE="$(printf '%s' "${LATENT_TOKEN_MODE_RAW}" | tr '[:upper:]' '[:lower:]')"
case "${LATENT_TOKEN_MODE}" in
    1|v)
        LATENT_TOKEN_MODE="v"
        TOTAL_LATENT_TOKEN_COUNT=1
        ;;
    2|vn)
        LATENT_TOKEN_MODE="vn"
        TOTAL_LATENT_TOKEN_COUNT=2
        ;;
    n)
        LATENT_TOKEN_MODE="n"
        TOTAL_LATENT_TOKEN_COUNT=1
        ;;
    *)
        echo "ERROR: TOTAL_LATENT_TOKENS must be one of v, n, vn, 1, or 2; got '${LATENT_TOKEN_MODE_RAW}'." >&2
        exit 1
        ;;
esac
DEFAULT_EXTRA_SPECIAL_TOKENS="</PAD>,</box>,</broom>,</charger>,</frame>,</fridge>,</lamp>,</laptop>,</phone>,</toilet>,</umbrella>,</watering_can>,</wine>"
EXTRA_SPECIAL_TOKENS="${EXTRA_SPECIAL_TOKENS:-${DEFAULT_EXTRA_SPECIAL_TOKENS}}"

LATENT_LOSS_WEIGHT=1.0
VIDEO_LOSS_WEIGHT=0
USE_LATENT_HIDDEN_SIM_LOSS="${USE_LATENT_HIDDEN_SIM_LOSS:-1}"
LATENT_HIDDEN_SIM_LOSS_MODE="${LATENT_HIDDEN_SIM_LOSS_MODE:-siglip}"
LATENT_HIDDEN_SIM_LOSS_WEIGHT="${LATENT_HIDDEN_SIM_LOSS_WEIGHT:-1}"
USE_LATENT_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS="${USE_LATENT_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS:-0}"
LATENT_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS_WEIGHT="${LATENT_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS_WEIGHT:-1.0}"
WAN21_VAE_PATH="${WAN21_VAE_PATH:-/mnt/nas/zhangyiming/database/ckpt/pretrained/wan2.1_vae/original/Wan2.1_VAE.pth}"

USE_VALUE_PREDICTION="${USE_VALUE_PREDICTION:-0}"
USE_ACTION_VALUE_PREDICTION="${USE_ACTION_VALUE_PREDICTION:-0}"

VALUE_TOKEN_MASK_VIDEO_TO_VALUE="${VALUE_TOKEN_MASK_VIDEO_TO_VALUE:-0}"
VALUE_TOKEN_MASK_NONVALUE_TO_VALUE="${VALUE_TOKEN_MASK_NONVALUE_TO_VALUE:-0}"

VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-1.0}"
ACTION_VALUE_LOSS_WEIGHT="${ACTION_VALUE_LOSS_WEIGHT:-1.0}"

ACTION_GT_LATENT_AFTER_EPOCH=0
COSMOS_SELF_ONLY_BRIDGE=1
TRAIN_EMBED_TOKENS=1
NO_DETACH_LATENT_INPUT="${NO_DETACH_LATENT_INPUT:-1}"
DECOSMOS=1
ACTION_USE_LATENT_PREFIX=1
# Bridge position schemes: mrope, mrope_interleave, llama1d
BRIDGE_POS_SCHEME="llama1d"

echo ">>> Starting 2-Expert RLBench Keyframe CoT Training: ${RUN_NAME}"
echo ">>> Latent token mode: ${LATENT_TOKEN_MODE} (count=${TOTAL_LATENT_TOKEN_COUNT}, input=${LATENT_TOKEN_MODE_RAW})"


accelerate launch --config_file ../config/sft.yaml \
    --num_processes ${NUM_PROCESSES}  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard train_cot_rlbench_keyframe.py \
    --experiment_name ${EXPERIMENT_NAME} \
    --run_name ${RUN_NAME} \
    --model_path ${JANUS_MODEL_PATH} \
    --action_expert_path ${ACTION_EXPERT_PATH} \
    --cosmos_model_path ${COSMOS_PT_PATH} \
    --cosmos_experiment_name ${COSMOS_EXP_NAME} \
    --cosmos_text_cache_path "${COSMOS_TEXT_CACHE_PATH}" \
    --data_path ${DATA_JSON} \
    --output_dir ${OUTPUT_ROOT_DIR} \
    --log_dir ${OUTPUT_ROOT_DIR} \
    --video_h 32 \
    --video_w 32 \
    --video_frames ${VIDEO_FRAMES} \
    --num_cond_input_frames ${NUM_COND_INPUT_FRAMES} \
    --fps 20 \
    --action_dim ${ACTION_DIM} \
    --action_chunk ${ACTION_CHUNK} \
    --n_epochs 300 \
    --save_freq 150 \
    --train_bsz_per_gpu ${TRAIN_BSZ} \
    --num_workers ${NUM_WORKERS} \
    --pin_memory ${PIN_MEMORY} \
    --persistent_workers ${PERSISTENT_WORKERS} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --cosmos_core_lr_ratio ${COSMOS_CORE_LR_RATIO} \
    --min_lr_ratio 0 \
    --warmup_rates 0 \
    --weight_decay 0 \
    --action_intermediate_size ${ACTION_INTERMEDIATE_SIZE} \
    --total_latent_tokens ${TOTAL_LATENT_TOKEN_COUNT} \
    --latent_token_mode "${LATENT_TOKEN_MODE}" \
    --extra_special_tokens "${EXTRA_SPECIAL_TOKENS}" \
    --img_latents_per_future ${IMG_LATENTS_PER_FUTURE} \
    --state_latents_per_future ${STATE_LATENTS_PER_FUTURE} \
    --num_future_frames ${NUM_FUTURE_FRAMES} \
    --future_frame_stride ${FUTURE_FRAME_STRIDE} \
    --video_loss_weight ${VIDEO_LOSS_WEIGHT} \
    --latent_loss_weight ${LATENT_LOSS_WEIGHT} \
    --use_latent_hidden_sim_loss ${USE_LATENT_HIDDEN_SIM_LOSS} \
    --latent_hidden_sim_loss_mode "${LATENT_HIDDEN_SIM_LOSS_MODE}" \
    --latent_hidden_sim_loss_weight ${LATENT_HIDDEN_SIM_LOSS_WEIGHT} \
    --use_latent_hidden_wan_downsample_sim_loss ${USE_LATENT_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS} \
    --latent_hidden_wan_downsample_sim_loss_weight ${LATENT_HIDDEN_WAN_DOWNSAMPLE_SIM_LOSS_WEIGHT} \
    --wan21_vae_path "${WAN21_VAE_PATH}" \
    --use_value_prediction ${USE_VALUE_PREDICTION} \
    --use_action_value_prediction ${USE_ACTION_VALUE_PREDICTION} \
    --value_token_mask_video_to_value ${VALUE_TOKEN_MASK_VIDEO_TO_VALUE} \
    --value_token_mask_nonvalue_to_value ${VALUE_TOKEN_MASK_NONVALUE_TO_VALUE} \
    --value_loss_weight ${VALUE_LOSS_WEIGHT} \
    --action_value_loss_weight ${ACTION_VALUE_LOSS_WEIGHT} \
    --action_gt_latent_after_epoch ${ACTION_GT_LATENT_AFTER_EPOCH} \
    --cosmos_self_only_bridge ${COSMOS_SELF_ONLY_BRIDGE} \
    --train_embed_tokens ${TRAIN_EMBED_TOKENS} \
    --no_detach_latent_input ${NO_DETACH_LATENT_INPUT} \
    --decosmos ${DECOSMOS} \
    --action_use_latent_prefix ${ACTION_USE_LATENT_PREFIX} \
    --bridge_pos_scheme "${BRIDGE_POS_SCHEME}" \
    --freeze_video_after 300 \
    --robot_state ${ROBOT_STATE} \
    --state_placeholder_tokens ${STATE_PLACEHOLDER_TOKENS} \
    --state_dim ${STATE_DIM} \
    --state_encoding_mode ${STATE_ENCODING_MODE} \
    --action_self_causal_in_bridge 1


echo ">>> Training Finished."
