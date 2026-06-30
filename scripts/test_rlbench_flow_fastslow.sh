cd /media/liuzhuoyang/LCoT_VLA_MOT/scripts
source /media/miniconda3/bin/activate /media/miniconda3/envs/janus_cot
export COPPELIASIM_ROOT=/media/Programs/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

export PATH=/media/miniconda3/envs/janus_cot/bin:$PATH
export HF_HOME=/media/huggingFace
export PYTHONPATH=/media/liuzhuoyang/LCoT_VLA_MOT:$PYTHONPATH

# export CUDA_VISIBLE_DEVICES=4,5,6,7

N=3
Xvfb :$N -screen 0 1024x768x24 &
export DISPLAY=:$N

models=("/media/liuzhuoyang/LCoT_VLA_MOT/exp/latent_cot_mot_flow/janus_pro_siglip_uni3d_1B_1e-4_mot_pretrain1220e5_12tasks_view1_action_latent_sparse_slow1_fast4_pc_state_12_0107/stage3/checkpoint-249-37500/tfmr")
# tasks=("close_box" "close_laptop_lid")
# tasks=("toilet_seat_down" "sweep_to_dustpan")
# tasks=("close_fridge" "place_wine_at_rack_location")
# tasks=("water_plants" "phone_on_base")
# tasks=("take_umbrella_out_of_umbrella_stand" "take_frame_off_hanger")

tasks=("close_box" "close_laptop_lid" "sweep_to_dustpan" "phone_on_base" "toilet_seat_down" "close_fridge" "place_wine_at_rack_location" "water_plants" "take_umbrella_out_of_umbrella_stand" "take_frame_off_hanger")
# tasks=( "toilet_seat_down" "close_fridge" "place_wine_at_rack_location" "water_plants" "take_umbrella_out_of_umbrella_stand" "take_frame_off_hanger")

# tasks=("close_box" "close_laptop_lid" "sweep_to_dustpan" "phone_on_base")

# tasks=("phone_on_base")

for model in "${models[@]}"; do
  exp_name=$(echo "$model" | awk -F'/' '{print $(NF-3)"_"$(NF-2)"_"$(NF-1)}')
  for task in "${tasks[@]}"; do
    python /media/liuzhuoyang/LCoT_VLA_MOT/scripts/test_rlbench_siglip_flow_fastslow.py \
      --model-path ${model} \
      --task-name ${task} \
      --exp-name ${exp_name} \
      --result-dir ${model} \
      --cuda $N \
      --use_robot_state 0 \
      --max-steps 10 \
      --num-episodes 20 \
      --load-pointcloud 0 \
      --dataset-name 'rlbench' \
      --use_latent 1 \
      --latent_size 12 \
      --compress_strategy average \
      --fs_ratio 4 \
      --result-dir /media/liuzhuoyang/LCoT_VLA_MOT/test_results/test_0112_mot_4 \
      --action-chunk 1
  done
done

