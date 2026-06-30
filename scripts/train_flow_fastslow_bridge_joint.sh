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

# export MASTER_ADDR=$(hostname -I | awk '{print $1}') # for rank0
export MASTER_ADDR=10.20.0.138
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

BASE_RUN_NAME="janus_pro_siglip_uni3d_1B_1e-4_mot_pretrain1220e5_bridge_view1_joint_sparse_slow1_fast4_state_8_0115"
EXPERIMENT_NAME="latent_cot_mot_flow"
OUTPUT_ROOT_DIR="../exp"

DATA_JSON="/media/liuzhuoyang/LCoT_VLA_MOT/training_data/bridge_single_json/bridgev2_view1_chunk4_fast4_sparse_fastslow_train.json"
ORIGIN_MODEL_PATH="/media/liuzhuoyang/LCoT_VLA/Janus-Pro-1B"
ACTION_MODEL_PATH="/media/liuzhuoyang/LCoT_VLA/exp_pretrain/action_only_flow/janus_pro_siglip_encoder_1B_no_state_lr_2e-5_flow_1217/checkpoint-4-5530345/tfmr"
PC_EMBEDDER_CKPT="/media/liuzhuoyang/Uni3D/checkpoints/modelzoo/uni3d-b/model.pt"

TRAIN_BSZ=8
LR=1e-4

STAGE1_SUB_NAME="${BASE_RUN_NAME}/stage3"

accelerate launch --config_file ../config/sft_multi.yaml \
    --num_processes ${NUM_PROCESSES}  \
    --num_machines ${WORLD_SIZE} \
    --machine_rank ${RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --deepspeed_multinode_launcher standard train_janus_siglip_flow_fastslow_bridge_joint.py \
    --model_path ${ORIGIN_MODEL_PATH} \
    --action_model_path ${ACTION_MODEL_PATH} \
    --data_path ${DATA_JSON} \
    --data_root "" \
    --n_epochs 40 \
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

echo ">>> Finished."
