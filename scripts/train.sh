#!/bin/bash
set -e

# export http_proxy=http://192.168.32.28:18000 && export https_proxy=http://192.168.32.28:18000

cd /media/liuzhuoyang/last05_0405/scripts
source /media/miniconda3/bin/activate /media/miniconda3/envs/last05

export PATH=/media/miniconda3/envs/last05/bin:$PATH
export HF_HOME=/media/huggingFace
export HF_HUB_OFFLINE=1
export PYTHONPATH=/media/liuzhuoyang/last05_0405:$PYTHONPATH
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

EXPERIMENT_NAME="cosmos_janus_libero_spatial"
RUN_NAME="cosmos2B_janus1B_mot_spatial_full_fft_bs8_lr1e-4_0405"
OUTPUT_ROOT_DIR="../exp_cosmos_vla"

DATA_JSON="/media/liuzhuoyang/cosmos_mot/training_data/libero_cosmos_janus/train.json"

JANUS_MODEL_PATH="/media/liuzhuoyang/LCoT_VLA/Janus-Pro-1B"
ACTION_MODEL_PATH="/media/liuzhuoyang/LCoT_VLA/exp_pretrain/action_only_flow/janus_pro_siglip_encoder_1B_no_state_lr_2e-5_flow_1217/checkpoint-4-5530345/tfmr"

COSMOS_PT_PATH="/media/liuzhuoyang/cosmos_mot/ckpts/Cosmos-Predict2.5-2B/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt"
COSMOS_EXP_NAME="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only"

NUM_PROCESSES=8
TRAIN_BSZ=8
GRAD_ACCUM=4
LR=1e-4
ACTION_INTERMEDIATE_SIZE=1408
SHARE_VIDEO_ACTION_TIMESTEP=0

echo ">>> Starting Training: ${RUN_NAME}"

accelerate launch --config_file ../config/sft.yaml \
    --num_processes ${NUM_PROCESSES}  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard train.py \
    --experiment_name ${EXPERIMENT_NAME} \
    --run_name ${RUN_NAME} \
    --model_path ${JANUS_MODEL_PATH} \
    --action_model_path ${ACTION_MODEL_PATH} \
    --cosmos_model_path ${COSMOS_PT_PATH} \
    --cosmos_experiment_name ${COSMOS_EXP_NAME} \
    --data_path ${DATA_JSON} \
    --output_dir ${OUTPUT_ROOT_DIR} \
    --log_dir ${OUTPUT_ROOT_DIR} \
    --video_h 256 \
    --video_w 256 \
    --video_frames 16 \
    --fps 10 \
    --action_dim 7 \
    --action_chunk 16 \
    --n_epochs 100 \
    --save_freq 10 \
    --train_bsz_per_gpu ${TRAIN_BSZ} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --min_lr_ratio 0.05 \
    --warmup_rates 0.05 \
    --weight_decay 0.01 \
    --action_intermediate_size ${ACTION_INTERMEDIATE_SIZE} \
    --share_video_action_timestep ${SHARE_VIDEO_ACTION_TIMESTEP} \
    --freeze_video_after 20

echo ">>> Training Finished."
