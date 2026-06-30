#!/bin/bash
set -e

# export http_proxy=http://192.168.32.28:18000 && export https_proxy=http://192.168.32.28:18000

LAST05_ROOT="/mnt/nas/zhangyiming/last05_beta/last05_mot2_action"

cd "${LAST05_ROOT}/scripts"
source /root/miniconda3/bin/activate /root/miniconda3/envs/last05
export WANDB_API_KEY="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"
export PATH=/root/miniconda3/envs/last05/bin:$PATH
# export HF_HOME=/media/huggingFace
export PYTHONPATH="${LAST05_ROOT}:${PYTHONPATH:-}"
export PATH=/media/miniconda3/envs/last05.1/bin:$PATH
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

EXPERIMENT_NAME="cosmos_janus_3expert_libero_spatial_baseline"

RUN_NAME="cosmos2B_janus1B_3expert_baseline"

OUTPUT_ROOT_DIR="../exp_cosmos_vla_3expert"

DATA_JSON="/mnt/nas/zhangyiming/database/data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual/train.json"

JANUS_MODEL_PATH="/mnt/nas/zhangyiming/database/ckpt/pretrained/Janus-Pro-1B"
ACTION_EXPERT_PATH="/mnt/nas/zhangyiming/database/ckpt/pretrained/LaST0_Pretrain_AE_chunk16/tfmr"

COSMOS_PT_PATH="/mnt/nas/zhangyiming/database/ckpt/pretrained/Cosmos-Predict2.5-2B/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt"
COSMOS_EXP_NAME="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only"
COSMOS_TEXT_CACHE_PATH="${COSMOS_TEXT_CACHE_PATH:-/mnt/nas/zhangyiming/database/data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual/cosmos_text_cache_raw_full_concat}"



NUM_GPUS=8
NUM_PROCESSES=8
TRAIN_BSZ=8
GRAD_ACCUM=1
LR=1e-4
COSMOS_CORE_LR_RATIO=0.02
NUM_WORKERS=4
PIN_MEMORY=1
PERSISTENT_WORKERS=1


ACTION_INTERMEDIATE_SIZE=1408
VIDEO_FRAMES=17
NUM_COND_INPUT_FRAMES=5

IMG_LATENTS_PER_FUTURE=0
STATE_LATENTS_PER_FUTURE=0
ROBOT_STATE=0
STATE_PLACEHOLDER_TOKENS=8
STATE_ENCODING_MODE="mlp"

NUM_FUTURE_FRAMES=0
FUTURE_FRAME_STRIDE=8
TOTAL_LATENT_TOKENS="${TOTAL_LATENT_TOKENS:-1}"

LATENT_LOSS_WEIGHT=1.0
VIDEO_LOSS_WEIGHT=1.0
USE_LATENT_HIDDEN_SIM_LOSS="${USE_LATENT_HIDDEN_SIM_LOSS:-0}"
LATENT_HIDDEN_SIM_LOSS_MODE="${LATENT_HIDDEN_SIM_LOSS_MODE:-siglip}"
LATENT_HIDDEN_SIM_LOSS_WEIGHT="${LATENT_HIDDEN_SIM_LOSS_WEIGHT:-1.0}"
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
DECOSMOS=0
ACTION_USE_LATENT_PREFIX=1
USE_HISTORY_TRAJECTORY_JANUS_IMAGE="${USE_HISTORY_TRAJECTORY_JANUS_IMAGE:-0}"
# Bridge position schemes: mrope, mrope_interleave, llama1d
BRIDGE_POS_SCHEME="mrope"

echo ">>> Starting 3-Expert CoT Training: ${RUN_NAME}"


accelerate launch --config_file ../config/sft.yaml \
    --num_processes ${NUM_PROCESSES}  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard train_cot.py \
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
    --use_history_trajectory_janus_image ${USE_HISTORY_TRAJECTORY_JANUS_IMAGE} \
    --video_h 256 \
    --video_w 256 \
    --video_frames ${VIDEO_FRAMES} \
    --num_cond_input_frames ${NUM_COND_INPUT_FRAMES} \
    --fps 20 \
    --action_dim 7 \
    --action_chunk 16 \
    --n_epochs 100 \
    --save_freq 10 \
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
    --total_latent_tokens ${TOTAL_LATENT_TOKENS} \
    --img_latents_per_future ${IMG_LATENTS_PER_FUTURE} \
    --state_latents_per_future ${STATE_LATENTS_PER_FUTURE} \
    --num_future_frames ${NUM_FUTURE_FRAMES} \
    --future_frame_stride ${FUTURE_FRAME_STRIDE} \
    --video_loss_weight ${VIDEO_LOSS_WEIGHT} \
    --latent_loss_weight ${LATENT_LOSS_WEIGHT} \
    --use_latent_hidden_sim_loss ${USE_LATENT_HIDDEN_SIM_LOSS} \
    --latent_hidden_sim_loss_mode "${LATENT_HIDDEN_SIM_LOSS_MODE}" \
    --latent_hidden_sim_loss_weight ${LATENT_HIDDEN_SIM_LOSS_WEIGHT} \
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
    --decosmos ${DECOSMOS} \
    --action_use_latent_prefix ${ACTION_USE_LATENT_PREFIX} \
    --bridge_pos_scheme "${BRIDGE_POS_SCHEME}" \
    --freeze_video_after 100 \
    --robot_state ${ROBOT_STATE} \
    --state_placeholder_tokens ${STATE_PLACEHOLDER_TOKENS} \
    --state_encoding_mode ${STATE_ENCODING_MODE} \
    --action_self_causal_in_bridge 0


echo ">>> Training Finished."
