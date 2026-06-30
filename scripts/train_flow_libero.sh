#!/bin/bash
set -e

export http_proxy=http://192.168.32.28:18000 && export https_proxy=http://192.168.32.28:18000
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


########## Stage 1 ##########

cd /media/liuzhuoyang/LCoT_VLA_MOT/scripts
source /media/miniconda3/bin/activate /media/miniconda3/envs/janus_cot
export PATH=/media/miniconda3/envs/janus_cot/bin:$PATH
export HF_HOME=/media/huggingFace
export PYTHONPATH=/media/liuzhuoyang/LCoT_VLA_MOT:/media/liuzhuoyang/LCoT_VLA_MOT/transformers:$PYTHONPATH

# RUN_NAME="stage1_libero_spatial"
# EXPERIMENT_NAME="latent_cot_mot_flow"
# OUTPUT_ROOT_DIR="/mnt/data/chenhao_save/ckp/lcot_doublerl_vla"

# DATA_JSON="/mnt/cpfs/chenhao/libero/libero_spatial_no_noops_image/libero_spatial.json"
# ORIGIN_MODEL_PATH=deepseek-ai/Janus-Pro-1B
# ACTION_PRETRAIN_PATH="/mnt/data/chenhao_save/ckp/janus_pro_siglip_encoder_1B_no_state_lr_2e-5_flow_1217_pretrain/checkpoint-0-1106069/tfmr/model.safetensors"

# NUM_PROCESSES=8
# TRAIN_BSZ=16
# LR=1e-4

# accelerate launch --config_file ../config/sft.yaml \
#     --num_processes ${NUM_PROCESSES}  \
#     --num_machines 1 \
#     --machine_rank 0 \
#     --deepspeed_multinode_launcher standard train_janus_siglip_flow.py \
#     --model_path ${ORIGIN_MODEL_PATH} \
#     --data_path ${DATA_JSON} \
#     --data_root "" \
#     --n_epochs 10 \
#     --save_freq 1000 \
#     --action_dim 7 \
#     --action_chunks 8 \
#     --train_bsz_per_gpu ${TRAIN_BSZ} \
#     --learning_rate ${LR} \
#     --min_lr_ratio 0 \
#     --weight_decay 0 \
#     --gradient_accumulation_steps 1 \
#     --output_dir ${OUTPUT_ROOT_DIR} \
#     --log_dir ${OUTPUT_ROOT_DIR} \
#     --experiment_name ${EXPERIMENT_NAME} \
#     --load_action_from_pretrain 1 \
#     --freeze_latent 0 \
#     --use_latent 1 \
#     --use_latent_robot_state 0 \
#     --latent_size 8 \
#     --pointcloud_embedder_ckpt_path "" \
#     --run_name ${RUN_NAME} \
#     --action_pretrain_path ${ACTION_PRETRAIN_PATH} \
#     --freeze_action 1 \
#     --robot_state 0 \
#     --use_pointcloud 0 \
#     --latent_action_same_time_train 0




# ########## Stage 2 ##########


# cd /mnt/cpfs/chenhao/lcot_doublerl/scripts
# source /root/miniconda3/bin/activate /root/miniconda3/envs/double_rl
# export PATH=/root/miniconda3/envs/double_rl/bin:$PATH
# export HF_HOME=/mnt/cpfs/chenhao/huggingface
# export PYTHONPATH=/mnt/cpfs/chenhao/lcot_doublerl:/mnt/cpfs/chenhao/lcot_doublerl/transformers:$PYTHONPATH


# RUN_NAME="stage2_libero_spatial"
# EXPERIMENT_NAME="latent_cot_mot_flow"
# OUTPUT_ROOT_DIR="/mnt/cpfs/chenhao/lcot_doublerl/ckp_dir"

# DATA_JSON="/mnt/cpfs/chenhao/libero/libero_spatial_no_noops_image/libero_spatial.json"
# ORIGIN_MODEL_PATH=/mnt/cpfs/chenhao/ckp/stage1_libero_spatial/checkpoint-9-4260/tfmr
# ACTION_PRETRAIN_PATH="/mnt/cpfs/chenhao/ckp/janus_pro_siglip_encoder_1B_no_state_lr_2e-5_flow_1217_pretrain/checkpoint-0-1106069/tfmr/model.safetensors"

# NUM_PROCESSES=8
# TRAIN_BSZ=14
# LR=1e-4

# accelerate launch --config_file ../config/sft.yaml \
#     --num_processes ${NUM_PROCESSES}  \
#     --num_machines 1 \
#     --machine_rank 0 \
#     --deepspeed_multinode_launcher standard train_janus_siglip_flow.py \
#     --model_path ${ORIGIN_MODEL_PATH} \
#     --data_path ${DATA_JSON} \
#     --data_root "" \
#     --n_epochs 200 \
#     --save_freq 1000 \
#     --action_dim 7 \
#     --action_chunks 8 \
#     --train_bsz_per_gpu ${TRAIN_BSZ} \
#     --learning_rate ${LR} \
#     --min_lr_ratio 0 \
#     --weight_decay 0 \
#     --gradient_accumulation_steps 1 \
#     --output_dir ${OUTPUT_ROOT_DIR} \
#     --log_dir ${OUTPUT_ROOT_DIR} \
#     --experiment_name ${EXPERIMENT_NAME} \
#     --load_action_from_pretrain 1 \
#     --freeze_latent 1 \
#     --use_latent 1 \
#     --use_latent_robot_state 0 \
#     --latent_size 8 \
#     --pointcloud_embedder_ckpt_path "" \
#     --run_name ${RUN_NAME} \
#     --action_pretrain_path ${ACTION_PRETRAIN_PATH} \
#     --freeze_action 0 \
#     --robot_state 0 \
#     --use_pointcloud 0 \
#     --single_branch 0 \
#     --action_branch_image 'main' \
#     --latent_action_same_time_train 0







# ########## Stage 1-2 Joint Training ##########



cd /mnt/cpfs/chenhao/lcot_doublerl/scripts
source /root/miniconda3/bin/activate /root/miniconda3/envs/double_rl
export PATH=/root/miniconda3/envs/double_rl/bin:$PATH
export HF_HOME=/mnt/cpfs/chenhao/huggingface
export PYTHONPATH=/mnt/cpfs/chenhao/lcot_doublerl:/mnt/cpfs/chenhao/lcot_doublerl/transformers:$PYTHONPATH


EXPERIMENT_NAME="latent_cot_mot_flow"
OUTPUT_ROOT_DIR="/mnt/cpfs/chenhao/lcot_doublerl/ckp_dir"

DATA_JSON="/mnt/cpfs/chenhao/libero/libero_spatial_no_noops_image/libero_spatial_latent_8_interval_4.json"
ORIGIN_MODEL_PATH=deepseek-ai/Janus-Pro-1B
ACTION_PRETRAIN_PATH="/mnt/cpfs/chenhao/ckp/janus_pro_siglip_encoder_1B_no_state_lr_2e-5_flow_1217_pretrain/checkpoint-0-1106069/tfmr/model.safetensors"

NUM_PROCESSES=8
TRAIN_BSZ=16
LR=1e-4

RUN_NAME="stage_1_2_joint_libero_spatial_action_branch_main_latent_8_num_4_interval_4"

latent_size=8
latent_tokens_num=1

accelerate launch --config_file ../config/sft.yaml \
    --num_processes ${NUM_PROCESSES}  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard train_janus_siglip_flow.py \
    --model_path ${ORIGIN_MODEL_PATH} \
    --data_path ${DATA_JSON} \
    --data_root "" \
    --n_epochs 200 \
    --save_freq 50 \
    --action_dim 7 \
    --action_chunks 8 \
    --train_bsz_per_gpu ${TRAIN_BSZ} \
    --learning_rate ${LR} \
    --min_lr_ratio 0 \
    --weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --output_dir ${OUTPUT_ROOT_DIR} \
    --log_dir ${OUTPUT_ROOT_DIR} \
    --experiment_name ${EXPERIMENT_NAME} \
    --load_action_from_pretrain 1 \
    --freeze_latent 0 \
    --use_latent 1 \
    --use_latent_robot_state 0 \
    --latent_size $((latent_size * latent_tokens_num)) \
    --pointcloud_embedder_ckpt_path "" \
    --run_name ${RUN_NAME} \
    --action_pretrain_path ${ACTION_PRETRAIN_PATH} \
    --freeze_action 0 \
    --robot_state 0 \
    --use_pointcloud 0 \
    --latent_action_same_time_train 1 \
    --single_branch 0 \
    --compressed_imgs_tokens ${latent_tokens_num} \
    --action_branch_image 'main' ## main, wrist, none


