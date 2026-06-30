#!/usr/bin/env bash
# LOCAL smoke launcher — do NOT commit. Delete after last0-alignment debugging is done.
set -Eeuo pipefail

LAST05_ROOT="/mnt/nas/zhangxuheng/last05_develop"
LIBERO_ROOT="/mnt/nas/zhangxuheng/LIBERO"
LAST0_SFT_ROOT="/mnt/data/zhangxuheng/ckpt/pretrained/LaST0_SFT_LIBERO_spatial"
LAST0_SFT_TFMR="${LAST0_SFT_ROOT}/tfmr"
COSMOS_PT="/mnt/nas/zhangyiming/database/ckpt/pretrained/Cosmos-Predict2.5-2B/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt"
COSMOS_EXP="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only"

LOG_DIR="/tmp/last0_sft_libero_smoke"
mkdir -p "$LOG_DIR"
SHELL_LOG="$LOG_DIR/shell_$(date +%Y_%m_%d-%H_%M_%S).log"
PRED_VIDEO_DIR="$LOG_DIR/videos_$(date +%Y_%m_%d-%H_%M_%S)"

exec > >(tee -a "$SHELL_LOG") 2>&1
set -x

trap 'rc=$?; echo "[ERROR] smoke launcher failed with exit code ${rc}"' ERR

cd "$LAST05_ROOT"
source /root/miniconda3/bin/activate /root/miniconda3/envs/last05
export PATH=/root/miniconda3/envs/last05/bin:$PATH
export PYTHONPATH="${LAST05_ROOT}:${LIBERO_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE=offline

unset LD_PRELOAD || true
export MUJOCO_GL=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYOPENGL_PLATFORM=egl

echo "[INFO] shell log: $SHELL_LOG"
echo "[INFO] predicted video dir: $PRED_VIDEO_DIR"
echo "[INFO] repo root: $LAST05_ROOT"

python -u "$LAST05_ROOT/experiments/robot/libero/run_libero_eval_new.py" \
  --model_path "$LAST0_SFT_TFMR" \
  --action_model_path "$LAST0_SFT_TFMR" \
  --cosmos_model_path "$COSMOS_PT" \
  --cosmos_experiment_name "$COSMOS_EXP" \
  --skip_pretrained_checkpoint true \
  --no_video_kv true \
  --model_variant "three_expert_cot" \
  --bridge_pos_scheme "llama1d" \
  --action_intermediate_size 5632 \
  --cosmos_self_only_bridge true \
  --action_self_causal_in_bridge true \
  --task_suite_name "libero_spatial" \
  --video_frames 16 \
  --action_chunk 16 \
  --total_latent_tokens 8 \
  --robot_state 0 \
  --cuda "0" \
  --seed 0 \
  --num_trials_per_task 5 \
  --num_open_loop_steps 8 \
  --action_repeat 1 \
  --cosmos_denoise_steps 2 \
  --fps 10 \
  --control_freq 0 \
  --predicted_video_save_dir "$PRED_VIDEO_DIR"
