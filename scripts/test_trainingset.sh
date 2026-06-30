
cd /media/liuzhuoyang/cosmos_mot/scripts
source /media/miniconda3/bin/activate /media/miniconda3/envs/cosmos_mot
export PYTHONPATH=/media/liuzhuoyang/cosmos_mot:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

python test_trainingset.py \
    --ckpt_dir /media/liuzhuoyang/cosmos_mot/exp_cosmos_vla/cosmos_janus_libero_spatial/checkpoint-epoch-29-step-31050