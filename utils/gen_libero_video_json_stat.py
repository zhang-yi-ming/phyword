import argparse
import json
import os

import numpy as np
from PIL import Image


DEFAULT_DATA_ROOT = "/mnt/nas/zhangyiming/database/data/libero_npy_20hz_224"
DEFAULT_SAVE_ROOT = "/mnt/nas/zhangyiming/database/data/libero_training_data_last05_lastest/libero_spatial_20hz_224_dual"
DEFAULT_TASK_LISTS = [
    "libero_spatial_no_noops",
    # "libero_goal_no_noops",
    # "libero_object_no_noops",
    # "libero_10_no_noops",
]

def parse_video_size(value):
    if value is None or str(value).lower() in ("none", "original"):
        return None
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("--video_size must have width and height")
        return int(value[0]), int(value[1])
    parts = str(value).lower().replace("x", ",").split(",")
    parts = [part.strip() for part in parts if part.strip()]
    if len(parts) != 2:
        raise ValueError("--video_size must look like 224,224 or none")
    return int(parts[0]), int(parts[1])


def normalize_image_array(img_array):
    img_array = np.asarray(img_array)
    while img_array.ndim > 3:
        img_array = img_array[0]
    if img_array.ndim != 3:
        raise ValueError(f"image must be HWC after squeeze, got shape {img_array.shape}")
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    if img_array.shape[-1] != 3:
        raise ValueError(f"image must have 3 channels, got shape {img_array.shape}")
    return img_array.astype(np.uint8)


def resize_frame_if_needed(img_array, video_size):
    if video_size is None:
        return img_array
    if img_array.shape[1] == video_size[0] and img_array.shape[0] == video_size[1]:
        return img_array
    return np.array(Image.fromarray(img_array).resize(video_size, Image.BILINEAR))


def episode_to_frames(episode, view_key, video_size):
    frames = []
    for step_idx, step in enumerate(episode):
        if view_key not in step:
            raise KeyError(f"episode step {step_idx} has no view key {view_key!r}")
        img_array = normalize_image_array(step[view_key])
        frames.append(resize_frame_if_needed(img_array, video_size))
    return frames


def write_video(video_path, frames, fps):
    import imageio

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    imageio.mimwrite(video_path, frames, fps=fps, macro_block_size=1)
    return video_path


def npy_to_video_and_json(args):
    video_size = parse_video_size(args.video_size)
    video_save_dir = os.path.join(args.save_root, "videos")
    json_file = os.path.join(args.save_root, "train.json")
    os.makedirs(video_save_dir, exist_ok=True)

    all_samples = []
    num_episodes = 0

    for task in args.tasks:
        print(f'========== Processing task: {task} ==========')
        task_video_dir = os.path.join(video_save_dir, task)
        os.makedirs(task_video_dir, exist_ok=True)
        
        task_dir = os.path.join(args.data_root, task)
        if not os.path.exists(task_dir):
            print(f"[warn] task dir not found, skipping: {task_dir}")
            continue

        for file in sorted(os.listdir(task_dir)):
            if not file.endswith('.npy'):
                continue
            
            file_base_name = file.replace('.npy', '')
            print(f'Processing episode: {file_base_name}')
            
            episode = np.load(os.path.join(task_dir, file), allow_pickle=True)
            episode_length = len(episode)
            if episode_length == 0:
                print(f"[warn] empty episode, skipping: {file}")
                continue

            primary_frames = episode_to_frames(episode, args.primary_view, video_size)
            action_frames = episode_to_frames(episode, args.action_view, video_size)

            video_path = os.path.abspath(
                os.path.join(task_video_dir, f"{file_base_name}_{args.primary_suffix}.mp4")
            )
            video_action_path = os.path.abspath(
                os.path.join(task_video_dir, f"{file_base_name}_{args.action_suffix}.mp4")
            )
            write_video(video_path, primary_frames, args.fps)
            write_video(video_action_path, action_frames, args.fps)
            num_episodes += 1
            
            for i in range(episode_length):
                step = episode[i]
                current_state = step['state'].copy()
                instruction = step['language_instruction']
                
                action_chunk_list = []
                for k in range(args.action_chunk):
                    future_idx = i + k
                    if future_idx < episode_length:
                        act = episode[future_idx]['action'].copy()
                    else:
                        act = episode[-1]['action'].copy()
                        
                    if isinstance(act, np.ndarray):
                        act = act.tolist()
                    action_chunk_list.append(act)
                
                sample = {
                    "input_prompt": instruction,
                    "video_path": video_path, 
                    "video_action_path": video_action_path,
                    "frame_index": i, 
                    "state": current_state.tolist(),
                    "action": action_chunk_list
                }
                
                all_samples.append(sample)

    print(f"Total samples generated: {len(all_samples)}")
    with open(json_file, 'w') as f:
        json.dump(all_samples, f, indent=2)
        
    return all_samples, num_episodes


def calculate_and_save_stats(all_samples, num_episodes, save_root):
    stats_file = os.path.join(save_root, "train_statistics.json")
    if not all_samples:
        raise SystemExit("no samples generated; check --data_root, --tasks, and view keys")

    print("Calculating statistics for actions and states...")
    actions = []
    states = []
    
    for sample in all_samples:
        actions.append(sample['action'])
        states.append(sample['state'])
        
    actions = np.array(actions)
    states = np.array(states)
    
    if len(actions.shape) == 3:
        actions_flat = actions.reshape(-1, actions.shape[-1])
    else:
        actions_flat = actions

    def get_stats(data, mask):
        return {
            'mean': np.mean(data, axis=0).tolist(),
            'std': np.std(data, axis=0).tolist(),
            'max': np.max(data, axis=0).tolist(),
            'min': np.min(data, axis=0).tolist(),
            'q01': np.quantile(data, 0.01, axis=0).tolist(),
            'q99': np.quantile(data, 0.99, axis=0).tolist(),
            'mask': mask,
        }

    action_mask = [True, True, True, True, True, True, False]
    state_mask = [True] * states.shape[1]
    if states.shape[1] >= 8:
        state_mask[-2:] = [False, False]

    action_stats = get_stats(actions_flat, action_mask)
    state_stats = get_stats(states, state_mask)

    result = {
        "rlbench": { 
            "action": action_stats,
            "state": state_stats,
            "num_transitions": len(actions),
            "num_trajectories": num_episodes,
        }
    }

    with open(stats_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Statistics successfully saved to {stats_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LIBERO .npy episodes into paired primary/action-view videos and train.json."
    )
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--save_root", type=str, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASK_LISTS)
    parser.add_argument("--primary_view", type=str, default="image_primary")
    parser.add_argument("--action_view", type=str, default="image_wrist")
    parser.add_argument("--primary_suffix", type=str, default="primary")
    parser.add_argument("--action_suffix", type=str, default="wrist")
    parser.add_argument("--action_chunk", type=int, default=16)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--video_size",
        type=str,
        default="224,224",
        help="Output size as WIDTH,HEIGHT, e.g. 224,224. Use 'none' to keep source size.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    samples, num_episodes = npy_to_video_and_json(args)
    calculate_and_save_stats(samples, num_episodes, args.save_root)
