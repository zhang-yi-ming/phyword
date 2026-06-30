
cd /media/liuzhuoyang/cosmos_mot
source /media/miniconda3/bin/activate /media/miniconda3/envs/cosmos_mot
export PATH=/media/miniconda3/envs/cosmos_mot/bin:$PATH
export HF_HOME=/media/huggingFace
export PYTHONPATH=/media/liuzhuoyang/cosmos_mot/LIBERO:$PYTHONPATH
export PYTHONPATH=/media/liuzhuoyang/cosmos_mot:$PYTHONPATH
export WANDB_MODE=offline
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# export MUJOCO_GL=egl
# export EGL_DEVICE_ID=0
export MUJOCO_GL=osmesa

N=3
Xvfb :$N -screen 0 1024x768x24 &
export DISPLAY=:$N

# Launch LIBERO-Spatial evals
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /media/liuzhuoyang/cosmos_mot/exp_cosmos_vla/cosmos_janus_libero_spatial/checkpoint-epoch-0-step-0 \
  --cosmos_experiment_name "Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only" \
  --cosmos_model_path /media/liuzhuoyang/cosmos_mot/ckpts/Cosmos-Predict2.5-2B/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt \
  --model_path /media/liuzhuoyang/LCoT_VLA/Janus-Pro-1B \
  --task_suite_name libero_spatial \
  --cuda "0" \
  --seed 0

