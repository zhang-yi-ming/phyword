#!/bin/bash

set -e

export http_proxy=http://192.168.32.28:18000 && export https_proxy=http://192.168.32.28:18000

cd /media/liuzhuoyang/LCoT_VLA_MOT/scripts
source /media/miniconda3/bin/activate /media/miniconda3/envs/janus_cot
export PATH=/media/miniconda3/envs/janus_cot/bin:$PATH
export HF_HOME=/media/huggingFace
export PYTHONPATH=/media/liuzhuoyang/LCoT_VLA_MOT:/media/liuzhuoyang/LCoT_VLA_MOT/transformers:$PYTHONPATH
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


export MASTER_ADDR=10.20.19.220
export MASTER_PORT=29500

export WORLD_SIZE=2
export NPROC_PER_NODE=8

export RANK=1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

NUM_PROCESSES=$((WORLD_SIZE * NPROC_PER_NODE))

echo "--------------------------------------------------------"
echo "Distributed Training Config:"
echo "  MASTER_ADDR: ${MASTER_ADDR}"
echo "  WORLD_SIZE:  ${WORLD_SIZE}"
echo "  RANK:        ${RANK} (Current Machine)"
echo "  TOTAL PROCS: ${NUM_PROCESSES}"
echo "--------------------------------------------------------"


MANUAL_S1_CKPT=""

BASE_RUN_NAME="janus_pro_siglip_uni3d_1B_1e-4_mot_pretrain1220e1_bridgev2_view1_sparse_slow1_fast4_state_8_0101"
EXPERIMENT_NAME="latent_cot_mot_flow"
OUTPUT_ROOT_DIR="../exp"

DATA_JSON="/media/liuzhuoyang/LCoT_VLA_MOT/training_data/bridge_single_json/bridgev2_view1_chunk4_fast4_sparse_fastslow_train.json"
ORIGIN_MODEL_PATH="/media/liuzhuoyang/LCoT_VLA_MOT/exp/latent_cot_mot_flow/janus_pro_siglip_uni3d_1B_1e-4_mot_pretrain1220e1_bridgev2_view1_sparse_slow1_fast4_state_8_0101/stage1/checkpoint-1-130172/tfmr"
ACTION_MODEL_PATH="/media/liuzhuoyang/LCoT_VLA/exp_pretrain/action_only_flow/janus_pro_siglip_encoder_1B_no_state_lr_2e-5_flow_1217/checkpoint-4-5530345/tfmr"
PC_EMBEDDER_CKPT="/media/liuzhuoyang/Uni3D/checkpoints/modelzoo/uni3d-b/model.pt"

TRAIN_BSZ=4
LR=1e-4

if [ -z "$MANUAL_S1_CKPT" ]; then
    # ---------------- Case A: From Scratch (Stage 1 -> Stage 2) ----------------

    STAGE1_SUB_NAME="${BASE_RUN_NAME}/stage1"
    
    echo "=========================================================="
    echo "NO Stage 1 CKPT provided. Starting from scratch."
    echo ">>> Executing Stage 1: ${STAGE1_SUB_NAME}"
    echo "=========================================================="

    accelerate launch --config_file ../config/sft.yaml \
        --num_processes ${NUM_PROCESSES}  \
        --num_machines ${WORLD_SIZE} \
        --machine_rank ${RANK} \
        --main_process_ip ${MASTER_ADDR} \
        --main_process_port ${MASTER_PORT} \
        --deepspeed_multinode_launcher standard train_janus_siglip_flow_fastslow_bridge.py \
        --model_path ${ORIGIN_MODEL_PATH} \
        --action_model_path ${ACTION_MODEL_PATH} \
        --data_path ${DATA_JSON} \
        --data_root "" \
        --n_epochs 10 \
        --save_freq 2 \
        --action_dim 7 \
        --train_bsz_per_gpu ${TRAIN_BSZ} \
        --learning_rate ${LR} \
        --min_lr_ratio 0 \
        --weight_decay 0 \
        --gradient_accumulation_steps 1 \
        --output_dir ${OUTPUT_ROOT_DIR} \
        --log_dir ${OUTPUT_ROOT_DIR} \
        --experiment_name ${EXPERIMENT_NAME} \
        --load_action_from_latent 0 \
        --load_action_from_pretrain 1 \
        --freeze_latent 0 \
        --use_latent 1 \
        --latent_size 8 \
        --compress_strategy average \
        --pointcloud_embedder_ckpt_path ${PC_EMBEDDER_CKPT} \
        --run_name ${STAGE1_SUB_NAME}

    echo ">>> Stage 1 Finished."

    S1_OUTPUT_DIR="${OUTPUT_ROOT_DIR}/${EXPERIMENT_NAME}/${STAGE1_SUB_NAME}"
    LAST_CKPT_DIR=$(ls -d ${S1_OUTPUT_DIR}/checkpoint-* | sort -V | tail -n 1)
    if [ -z "${LAST_CKPT_DIR}" ]; then
        echo "Error: Could not find any checkpoint in ${S1_OUTPUT_DIR}"
        exit 1
    fi

    STAGE2_INPUT_MODEL="${LAST_CKPT_DIR}/tfmr"
    echo ">>> Auto-detected Stage 1 Checkpoint: ${STAGE2_INPUT_MODEL}"

else
    # ---------------- Case B: Load S1 CKPT (Skip Stage 1 -> Stage 2) ----------------
    
    echo "=========================================================="
    echo "Stage 1 CKPT provided: ${MANUAL_S1_CKPT}"
    echo ">>> Skipping Stage 1, jumping directly to Stage 2..."
    echo "=========================================================="
    
    STAGE2_INPUT_MODEL="${MANUAL_S1_CKPT}"
fi

# STAGE2_SUB_NAME="${BASE_RUN_NAME}/stage2"

# echo "=========================================================="
# echo "Starting Stage 2 Training: ${STAGE2_SUB_NAME}"
# echo "Loading Model from: ${STAGE2_INPUT_MODEL}"
# echo "Config: Freeze Latent = 1, Epochs = 400"
# echo "=========================================================="

# accelerate launch --config_file ../config/sft.yaml \
#     --num_processes ${NUM_PROCESSES}  \
#     --num_machines 1 \
#     --machine_rank 0 \
#     --deepspeed_multinode_launcher standard train_janus_siglip_flow_fastslow_bridge.py \
#     --model_path ${STAGE2_INPUT_MODEL} \
#     --action_model_path ${ACTION_MODEL_PATH} \
#     --data_path ${DATA_JSON} \
#     --data_root "" \
#     --n_epochs 40 \
#     --save_freq 1 \
#     --action_dim 7 \
#     --train_bsz_per_gpu ${TRAIN_BSZ} \
#     --learning_rate ${LR} \
#     --min_lr_ratio 0 \
#     --weight_decay 0 \
#     --gradient_accumulation_steps 1 \
#     --output_dir ${OUTPUT_ROOT_DIR} \
#     --log_dir ${OUTPUT_ROOT_DIR} \
#     --experiment_name ${EXPERIMENT_NAME} \
#     --load_action_from_latent 0 \
#     --load_action_from_pretrain 1 \
#     --freeze_latent 1 \
#     --use_latent 1 \
#     --latent_size 8 \
#     --compress_strategy average \
#     --pointcloud_embedder_ckpt_path ${PC_EMBEDDER_CKPT} \
#     --run_name ${STAGE2_SUB_NAME}

# echo "=========================================================="
# echo "All Done!"
# echo "Stage 2 Artifacts saved to: ${OUTPUT_ROOT_DIR}/${EXPERIMENT_NAME}/${STAGE2_SUB_NAME}"
# echo "=========================================================="

