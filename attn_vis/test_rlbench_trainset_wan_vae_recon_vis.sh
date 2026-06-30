#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAST05_ROOT="${LAST05_ROOT:-/mnt/nas/zhangyiming/last05_beta/last05}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-/mnt/nas/zhangyiming/last05_beta/experiments_rlbench}"

DATA_JSON="${DATA_JSON:-/mnt/nas/zhangyiming/database/rlbench/train/json/train_action_chunk1_sumpos_lastrot.json}"
WAN21_VAE_PATH="${WAN21_VAE_PATH:-/mnt/nas/zhangyiming/database/ckpt/pretrained/wan2.1_vae/original/Wan2.1_VAE.pth}"

VIS_MODE="${VIS_MODE:-both}"  # image, video, or both
NUM_TRAJECTORIES_PER_TASK="${NUM_TRAJECTORIES_PER_TASK:-1}"
TASK_NAMES="${TASK_NAMES:-}"
MAX_TASKS="${MAX_TASKS:-0}"
MAX_FRAMES_PER_EPISODE="${MAX_FRAMES_PER_EPISODE:-0}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-4}"
FPS="${FPS:-10}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DTYPE="${DTYPE:-float32}"
VIDEO_CODEC="${VIDEO_CODEC:-}"
DRY_RUN="${DRY_RUN:-0}"

TIMESTAMP="$(date +%Y_%m_%d-%H_%M_%S)"
EVAL_ARTIFACT_NAME="${EVAL_ARTIFACT_NAME:-${TIMESTAMP}_wan_vae_recon_${VIS_MODE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENTS_ROOT}/wan_vae_recon_vis/${EVAL_ARTIFACT_NAME}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/shell}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
SHELL_LOG="${LOG_DIR}/test_rlbench_trainset_wan_vae_recon_vis_${EVAL_ARTIFACT_NAME}.log"
BASH_HPARAMS_FILE="${OUTPUT_DIR}/test_rlbench_trainset_wan_vae_recon_vis_hparams_${EVAL_ARTIFACT_NAME}.env"

if [[ "${CUDA_DEVICE}" == "cpu" ]]; then
  DEVICE="cpu"
else
  DEVICE="${DEVICE:-cuda:${CUDA_DEVICE}}"
fi

exec > >(tee -a "$SHELL_LOG") 2>&1
set -x
trap 'rc=$?; echo "[ERROR] test_rlbench_trainset_wan_vae_recon_vis.sh failed with exit code ${rc}"' ERR

write_hparam() {
  local key="$1"
  printf '%s=%s\n' "$key" "${!key}"
}

{
  for key in \
    SCRIPT_DIR LAST05_ROOT EXPERIMENTS_ROOT DATA_JSON WAN21_VAE_PATH \
    VIS_MODE NUM_TRAJECTORIES_PER_TASK TASK_NAMES MAX_TASKS MAX_FRAMES_PER_EPISODE FRAME_STRIDE \
    IMAGE_SIZE IMAGE_BATCH_SIZE FPS CUDA_DEVICE DEVICE DTYPE VIDEO_CODEC DRY_RUN \
    TIMESTAMP EVAL_ARTIFACT_NAME OUTPUT_DIR LOG_DIR SHELL_LOG BASH_HPARAMS_FILE
  do
    write_hparam "$key"
  done
} > "$BASH_HPARAMS_FILE"

cd "$LAST05_ROOT"
if [[ -f /root/miniconda3/bin/activate ]]; then
  source /root/miniconda3/bin/activate /root/miniconda3/envs/last05
  export PATH=/root/miniconda3/envs/last05/bin:$PATH
fi

export PYTHONPATH="${LAST05_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
unset LD_PRELOAD

echo "[INFO] shell log: $SHELL_LOG"
echo "[INFO] bash hparams file: $BASH_HPARAMS_FILE"
echo "[INFO] output dir: $OUTPUT_DIR"
echo "[INFO] data json: $DATA_JSON"
echo "[INFO] Wan VAE: $WAN21_VAE_PATH"
echo "[INFO] mode: $VIS_MODE"
echo "[INFO] python: $(which python)"
python -V

DRY_RUN_ARGS=()
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "True" ]]; then
  DRY_RUN_ARGS=(--dry_run)
fi

python -u "${SCRIPT_DIR}/run_rlbench_trainset_wan_vae_recon_vis.py" \
  --data_path "$DATA_JSON" \
  --output_dir "$OUTPUT_DIR" \
  --vae_path "$WAN21_VAE_PATH" \
  --mode "$VIS_MODE" \
  --task_names "$TASK_NAMES" \
  --num_trajectories_per_task "$NUM_TRAJECTORIES_PER_TASK" \
  --max_tasks "$MAX_TASKS" \
  --max_frames_per_episode "$MAX_FRAMES_PER_EPISODE" \
  --frame_stride "$FRAME_STRIDE" \
  --image_size "$IMAGE_SIZE" \
  --image_batch_size "$IMAGE_BATCH_SIZE" \
  --fps "$FPS" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --video_codec "$VIDEO_CODEC" \
  "${DRY_RUN_ARGS[@]}"

echo "[INFO] Wan VAE reconstruction visualization finished: $OUTPUT_DIR"
