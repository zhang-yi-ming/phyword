import os
import json
import torch
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM
from decord import VideoReader, cpu
from janus.models import VLChatProcessor, ActionTokenizer
from models.cosmos_janus import CosmosJanusMoT
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint


def attach_cosmos_inference_runtime(model, cosmos_wrapper):
    """Attach inference-only Cosmos sampling state to a loaded MoT model."""
    model.set_cosmos_inference_runtime_from_wrapper(cosmos_wrapper)
    return model


def load_inference_model(ckpt_dir, config):
    print(">>> 1. Loading Processor...")
    vl_chat_processor = VLChatProcessor.from_pretrained(config.model_path, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(vl_chat_processor.tokenizer, need_to_sub=3)

    print(">>> 2. Loading Janus Action Base...")
    janus_model = AutoModelForCausalLM.from_pretrained(
        config.action_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        flow=True, action_dim=config.action_dim, ignore_mismatched_sizes=True
    )
    
    print(">>> 3. Loading Cosmos Video Base (Monkey Patching TextEncoder)...")
    import cosmos_predict2._src.predict2.models.text2world_model_rectified_flow as t2w_module
    import torch.nn as nn
    class DummyTextEncoder(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
    t2w_module.TextEncoder = DummyTextEncoder
    
    experiment_opts = ["data_train=mock", "data_val=mock"]
    cosmos_wrapper, _ = load_model_from_checkpoint(
        experiment_name=config.cosmos_experiment_name,
        s3_checkpoint_dir=config.cosmos_model_path,
        config_file="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        load_ema_to_reg=True,
        to_device="cpu",
        experiment_opts=experiment_opts
    )
    
    print(">>> 4. Building Joint MoT Architecture & Loading Weights...")
    model = CosmosJanusMoT(cosmos_wrapper.net, cosmos_wrapper.tokenizer, janus_model, config)
    attach_cosmos_inference_runtime(model, cosmos_wrapper)
    
    ckpt_path = os.path.join(ckpt_dir, "cosmos_janus_mot.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, "mot_action_weights.pt")
        
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(torch.bfloat16).cuda().eval()
    
    print(">>> 5. Loading Statistics...")
    stats_path = os.path.join(ckpt_dir, "train_statistics.json")
    with open(stats_path, 'r') as f:
        stats_data = json.load(f)
        
    dataset_name = next(iter(stats_data))
    statistic = {
        'action_mask': np.array(stats_data[dataset_name]['action']['mask'], dtype=bool),
        'action_q01': np.array(stats_data[dataset_name]['action']['q01'], dtype=np.float32),
        'action_q99': np.array(stats_data[dataset_name]['action']['q99'], dtype=np.float32),
        'state_mask': np.array(stats_data[dataset_name]['state']['mask'], dtype=bool),
        'state_q01': np.array(stats_data[dataset_name]['state']['q01'], dtype=np.float32),
        'state_q99': np.array(stats_data[dataset_name]['state']['q99'], dtype=np.float32),
    }

    return model, vl_chat_processor, action_tokenizer, statistic

def get_random_episode(config):
    """把散落的 json 数据按 video_path 聚合成轨迹，并随机抽取一条"""
    with open(config.data_path, 'r') as f:
        data = json.load(f)

    episodes = {}
    for item in data:
        vid = item['video_path']
        if vid not in episodes:
            episodes[vid] = []
        episodes[vid].append(item)
    
    selected_vid = random.choice(list(episodes.keys()))
    episode_sequence = episodes[selected_vid]
    episode_sequence.sort(key=lambda x: x.get('frame_index', 0))
    
    print(f"\n[Dataset] Selected Episode: {selected_vid}")
    print(f"[Dataset] Total frames available in this episode: {len(episode_sequence)}")
    
    return episode_sequence

def extract_chunk_data(sample, config, statistic):
    """从单个样本字典中提取图像、动作和视频GT"""
    video_path = sample['video_path']
    start_frame = sample.get('frame_index', 0)
    
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.arange(start_frame, start_frame + config.video_frames)
    indices = np.clip(indices, 0, total_frames - 1)
    frames = vr.get_batch(indices).asnumpy()

    first_frame_img = frames[0]

    action_gt = np.array(sample['action'], dtype=np.float32).reshape(-1, config.action_dim)
    if action_gt.shape[0] < config.action_chunk:
        pad_len = config.action_chunk - action_gt.shape[0]
        action_gt = np.concatenate([action_gt, np.repeat(action_gt[-1:], pad_len, axis=0)], axis=0)
    elif action_gt.shape[0] > config.action_chunk:
        action_gt = action_gt[:config.action_chunk]

    state_gt = np.array(sample['state'], dtype=np.float32) if 'state' in sample else None
    
    return sample['input_prompt'], first_frame_img, action_gt, state_gt, frames


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Path to your saved checkpoint directory')
    parser.add_argument('--data_path', type=str, default="/media/liuzhuoyang/cosmos_mot/training_data/libero_cosmos_janus/train.json")
    parser.add_argument('--model_path', type=str, default="/media/liuzhuoyang/LCoT_VLA/Janus-Pro-1B")
    parser.add_argument('--action_model_path', type=str, default="/media/liuzhuoyang/LCoT_VLA/exp_pretrain/action_only_flow/janus_pro_siglip_encoder_1B_no_state_lr_2e-5_flow_1217/checkpoint-4-5530345/tfmr")
    parser.add_argument('--cosmos_model_path', type=str, default="/media/liuzhuoyang/cosmos_mot/ckpts/Cosmos-Predict2.5-2B/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt")
    parser.add_argument('--cosmos_experiment_name', type=str, default="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only")
    parser.add_argument('--video_h', type=int, default=256)
    parser.add_argument('--video_w', type=int, default=256)
    parser.add_argument('--video_frames', type=int, default=16)
    parser.add_argument('--action_dim', type=int, default=7)
    parser.add_argument('--action_chunk', type=int, default=16)
    parser.add_argument('--robot_state', type=int, default=0)
    parser.add_argument('--action_intermediate_size', type=int, default=0,
                        help='If >0, use a slim MLP with this intermediate size for the action expert bridges')

    # 连续预测的 Chunk 数量
    parser.add_argument('--num_eval_chunks', type=int, default=4)
    # 评估结果统一保存目录
    parser.add_argument('--save_dir', type=str, default="./test_output", help='Directory to save plots and videos')
    config = parser.parse_args()

    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)

    model, vl_chat_processor, action_tokenizer, statistic = load_inference_model(config.ckpt_dir, config)
    
    episode_sequence = get_random_episode(config)
    
    device = torch.device('cuda')
    dtype = torch.bfloat16
    video_transform = transforms.Compose([
        transforms.Resize(min(config.video_h, config.video_w), antialias=True),
        transforms.CenterCrop((config.video_h, config.video_w)),
    ])
    
    all_pred_videos = []
    all_gt_videos = []
    all_pred_actions = []
    all_gt_actions = []
    
    stride = config.action_chunk  
    max_steps = min(config.num_eval_chunks * stride, len(episode_sequence))
    
    print(f"\n>>> Starting Sequential Evaluation: {max_steps//stride} chunks, Stride={stride} frames.")

    for step_idx in range(0, max_steps, stride):
        sample = episode_sequence[step_idx]
        print(f"--- Processing Chunk at Frame {sample.get('frame_index', step_idx)} ---")
        
        prompt, first_frame_np, action_gt, state_gt, video_gt_np = extract_chunk_data(sample, config, statistic)
        
        state_tokens_str = ""
        if config.robot_state and state_gt is not None:
            norm_state = np.where(statistic['state_mask'], np.clip(2 * (state_gt - statistic['state_q01']) / (statistic['state_q99'] - statistic['state_q01'] + 1e-8) - 1.0, -1.0, 1.0), state_gt)
            state_tokens_str = action_tokenizer(norm_state)

        conversation = [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{prompt}{state_tokens_str}"},
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        first_frame_pil = Image.fromarray(first_frame_np.astype(np.uint8))
        janus_inputs = vl_chat_processor(conversations=conversation, images=[first_frame_pil], return_tensors="pt")

        janus_input_ids = janus_inputs.input_ids.to(device)
        janus_pixel_values = janus_inputs.pixel_values.to(device).to(dtype)
        janus_images_seq_mask = janus_inputs.images_seq_mask.to(device)
        janus_images_emb_mask = janus_inputs.images_emb_mask.to(device)

        first_frame_tensor = torch.from_numpy(first_frame_np).permute(2, 0, 1).float() / 255.0
        first_frame_tensor = video_transform(first_frame_tensor).unsqueeze(0).to(device).to(dtype)

        with torch.inference_mode():
            fps_tensor = torch.tensor([10.0], device=device, dtype=dtype)
            pred_video, pred_action = model.forward_flow_joint_inference(
                janus_input_ids=janus_input_ids,
                janus_pixel_values=janus_pixel_values,
                janus_images_seq_mask=janus_images_seq_mask,
                janus_images_emb_mask=janus_images_emb_mask,
                first_frame=first_frame_tensor,
                action_denoise_steps=10,
                fps=fps_tensor
            )

        # 反归一化 Action
        normalized_actions = pred_action.squeeze(0).cpu().float().numpy()
        if normalized_actions.shape[1] in [7, 14]:
            normalized_actions[:, 6] = (normalized_actions[:, 6] >= 0.5).astype(int)
            
        action_pred = np.where(
            statistic['action_mask'],
            0.5 * (normalized_actions + 1.0) * (statistic['action_q99'] - statistic['action_q01']) + statistic['action_q01'],
            normalized_actions
        )
        
        # 收集结果
        all_pred_actions.append(action_pred)
        all_gt_actions.append(action_gt)
        
        pred_vid_np = pred_video.squeeze(0).cpu().float() 
        pred_vid_np = torch.clamp((pred_vid_np + 1.0) / 2.0, 0, 1) * 255.0
        pred_vid_np = pred_vid_np.permute(1, 2, 3, 0).to(torch.uint8)
        all_pred_videos.append(pred_vid_np)
        
        gt_tensor = torch.from_numpy(video_gt_np).permute(0, 3, 1, 2).float() / 255.0
        gt_tensor = video_transform(gt_tensor)
        gt_tensor = (gt_tensor * 255.0).permute(0, 2, 3, 1).to(torch.uint8)
        all_gt_videos.append(gt_tensor)
        
        # 【极其重要】：释放单轮降噪累积的显存，防止 OOM
        torch.cuda.empty_cache()

    # =================================================================
    # 评估与结果保存 (Plotting & Stitching)
    # =================================================================
    print("\n================== FULL TRAJECTORY ACTION ERROR ==================")
    full_action_pred = np.concatenate(all_pred_actions, axis=0)
    full_action_gt = np.concatenate(all_gt_actions, axis=0)
    
    # 计算整个序列的平均绝对误差 (MAE)
    mae_error = np.abs(full_action_pred - full_action_gt).mean(axis=0)
    print(f"Mean Absolute Error per Dimension over {len(full_action_pred)} steps:")
    print(np.round(mae_error, 4))
    print("==================================================================")

    # 1. 绘制并保存每一维 Action 的时序对比图
    T, act_dim = full_action_pred.shape
    xs = np.arange(T)
    print(f"\n>>> Saving {act_dim} individual action trajectory plots to {config.save_dir} ...")

    for idx in range(act_dim):
        plt.figure(figsize=(10, 4))
        plt.plot(xs, full_action_pred[:, idx], label="Predicted", linestyle="--", color="blue")
        plt.plot(xs, full_action_gt[:, idx], label="Ground Truth", alpha=0.6, color="orange")
        plt.title(f"Predicted vs GT - Action Dimension {idx}", fontsize=14)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Action Value", fontsize=12)
        plt.legend(loc="upper right")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(config.save_dir, f"action_dim_{idx}.png")
        plt.savefig(plot_path)
        plt.close()

    npz_path = os.path.join(config.save_dir, "action_trajectory.npz")
    np.savez(npz_path, pred=full_action_pred, gt=full_action_gt)

    print(f">>> Stitching and Saving Long Videos to {config.save_dir} ...")
    final_pred_video = torch.cat(all_pred_videos, dim=0)
    final_gt_video = torch.cat(all_gt_videos, dim=0)
    
    pred_vid_path = os.path.join(config.save_dir, "test_sequence_pred.mp4")
    gt_vid_path = os.path.join(config.save_dir, "test_sequence_gt.mp4")
    
    torchvision.io.write_video(pred_vid_path, final_pred_video, fps=10)
    torchvision.io.write_video(gt_vid_path, final_gt_video, fps=10)
    
    print(f"Done! All assets (Plots, NPZ, and MP4s) have been saved to '{config.save_dir}'.")

if __name__ == "__main__":
    main()
