import numpy as np
from PIL import Image
import json
import os
import re
from scipy.spatial.transform import Rotation as R

IMAGE_VIEWS_SAVE = ['image_third', 'image_wrist_right', 'image_wrist_left'] 
IMAGE_VIEWS_SLOW = ['image_third'] 
IMAGE_VIEWS_FAST = ['image_third', 'image_wrist_right', 'image_wrist_left'] 
FUTURE_STEPS = 4
FAST_SLOW_RATIO = 4

def unique_euler_xyz_rad(angles, range_style="2pi"):
    rot = R.from_euler('xyz', angles, degrees=False)
    euler = rot.as_euler('xyz', degrees=False)
    euler = (euler + np.pi) % (2 * np.pi) - np.pi
    if euler[1] > np.pi/2:
        euler[1] = np.pi - euler[1]
        euler[0] += np.pi
        euler[2] += np.pi
    elif euler[1] < -np.pi/2:
        euler[1] = -np.pi - euler[1]
        euler[0] += np.pi
        euler[2] += np.pi
    euler = (euler + np.pi) % (2 * np.pi) - np.pi
    if range_style == "2pi":
        euler = euler % (2 * np.pi)
    return euler

def create_padding_assets(save_dir, sample_step):
    padding_info = {}
    
    # 1. Create Black Images
    for view in IMAGE_VIEWS_SAVE:
        if view in sample_step:
            img_shape = sample_step[view].shape # (H, W, C)
            black_img = np.zeros(img_shape, dtype=np.uint8)
            img_obj = Image.fromarray(black_img)
            save_path = f'{save_dir}/padding_black_{view}.png'
            img_obj.save(save_path)
            padding_info[f'image_{view}'] = save_path

    # 2. Create Zero Point Cloud
    if 'point_cloud' in sample_step:
        pc_shape = sample_step['point_cloud'].shape
        zero_pc = np.zeros(pc_shape, dtype=np.float32)
        save_path = f'{save_dir}/padding_zero_pc.npy'
        np.save(save_path, zero_pc)
        padding_info['pc'] = save_path

    # 3. Create Zero State
    if 'state' in sample_step:
        state_dim = len(sample_step['state'])
        padding_info['state'] = [0.0] * state_dim

    return padding_info

def npy_2_jsonl(data_root, img_save_root, jsonl_filename, task_lists):
    with open(jsonl_filename, 'w') as f:
        
        for task in task_lists:
            print(f'Processing task: {task}')

            if not os.path.exists(f'{img_save_root}/{task}'):
                os.makedirs(f'{img_save_root}/{task}', exist_ok=True)

            for file in os.listdir(f'{data_root}/{task}'):
                if not file.endswith('.npy'): 
                    continue

                print('generating:', file, end=' ')

                episode = np.load(f'{data_root}/{task}/{file}', allow_pickle=True)
                file_base_name = file.replace('.npy', '')
                episode_length = len(episode)
                print('episode_length:', episode_length)

                save_dir = f'{img_save_root}/{task}/{file_base_name}'
                
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)

                    for i in range(episode_length):
                        step = episode[i]
                        # Save Images
                        for view in IMAGE_VIEWS_SAVE:
                            if view in step:
                                image_array = step[view]
                                image = Image.fromarray(image_array)
                                image.save(f'{save_dir}/image{i}_{view}.png')
                        # Save PC
                        if 'point_cloud' in step:
                            pc_array = step['point_cloud']
                            np.save(f'{save_dir}/pc{i}.npy', pc_array)
                        else:
                            print(f'Warning: No point_cloud found for {task} step {i}')

                padding_assets = create_padding_assets(save_dir, episode[0])

                for i in range(episode_length):
                    step = episode[i]
                    
                    slow_idx = (i // FAST_SLOW_RATIO) * FAST_SLOW_RATIO
                    
                    image_fast_list = []
                    for view in IMAGE_VIEWS_FAST:
                        image_fast_list.append(f'{save_dir}/image{i}_{view}.png')
                    
                    safe_slow_idx = min(slow_idx, episode_length - 1)
                    image_slow_list = []
                    for view in IMAGE_VIEWS_SLOW:
                        image_slow_list.append(f'{save_dir}/image{safe_slow_idx}_{view}.png')

                    pc_current = f'{save_dir}/pc{i}.npy'
                    
                    output_images_list = [] # List of Lists: [T][View]
                    output_pc_list = []     # List: [T]
                    output_state_list = []  # List: [T]

                    for k in range(1, FUTURE_STEPS + 1):
                        tgt_idx = slow_idx + k
                        
                        if tgt_idx < episode_length:
                            # --- Real Data ---
                            tgt_step = episode[tgt_idx]
                            
                            # Images
                            for view in IMAGE_VIEWS_SLOW:
                                output_images_list.append(f'{save_dir}/image{tgt_idx}_{view}.png')
                            
                            # PC
                            output_pc_list.append(f'{save_dir}/pc{tgt_idx}.npy')
                            
                            # State
                            state_val = tgt_step['state'].copy() # Copy to avoid modifying original
                            if len(state_val) >= 12:
                                state_val[3:6] = unique_euler_xyz_rad(state_val[3:6])
                                state_val[10:13] = unique_euler_xyz_rad(state_val[10:13])
                            if isinstance(state_val, np.ndarray):
                                state_val = state_val.tolist()
                            output_state_list.append(state_val)
                            
                        else:
                            for view in IMAGE_VIEWS_SLOW:
                                output_images_list.append(padding_assets[f'image_{view}'])
                            output_pc_list.append(padding_assets.get('pc', '')) # Empty string if no PC
                            output_state_list.append(padding_assets.get('state', []))

                    # 5. Current Label Processing
                    action_14d = step['action'].copy()
                    action_14d[3:6] = unique_euler_xyz_rad(action_14d[3:6])
                    action_14d[10:13] = unique_euler_xyz_rad(action_14d[10:13])
                    
                    current_state = step['state'].copy()
                    if len(current_state) >= 12:
                         current_state[3:6] = unique_euler_xyz_rad(current_state[3:6])
                         current_state[10:13] = unique_euler_xyz_rad(current_state[10:13])

                    # 6. Construct Dictionary
                    episode_data = {
                        'input_images_fast': image_fast_list,
                        'input_images_slow': image_slow_list,
                        'input_pointcloud': pc_current,
                        
                        'output_images': output_images_list, 
                        'output_pointcloud': output_pc_list,
                        'output_state': output_state_list,
                        
                        'action': action_14d.tolist(),
                        'state': current_state.tolist(),
                        # 'language_instruction': step['language_instruction'],
                        'language_instruction': 'use right arm to lift the lid of the pot and keeps it lifted, then use left arm to pick the yellow corn out of the pot and put it on the plate and use right arm put the lid back',
                    }

                    if 'language_subgoals' in step:
                        episode_data['language_subgoals'] = step['language_subgoals']

                    f.write(json.dumps(episode_data) + '\n')


def jsonl_2_json(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    output_data = []

    for line in lines:
        item = json.loads(line)
        
        new_item = {
            "input_prompt": item["language_instruction"],
            # --- Fast System ---
            "input_image_fast": item["input_images_fast"],
            "input_image_fast_resolution": [384, 384],
            # --- Slow System ---
            "input_image_slow": item["input_images_slow"],
            "input_image_slow_resolution": [384, 384],
            # --- Common Input ---
            "input_pointcloud": [item["input_pointcloud"]],
            # --- Output (Latent) ---
            # output_image 是 [T, Views] 的结构
            "output_image": item["output_images"], 
            "output_pointcloud": item["output_pointcloud"],
            # Output Resolution: [T, Views, 2]
            "output_image_resolution": [384, 384],
            "output_state": item["output_state"],
            # --- Labels ---
            "action": item["action"],
            "state": item["state"]
        }
        
        output_data.append(new_item)
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)


def cal_stats(jsonl_filename):
    actions = []
    states = []
    episode_numbers = set()

    with open(jsonl_filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            actions.append(data['action'])
            states.append(data['state'])
            
            match = re.search(r'episode(\d+)', data['input_images_fast'][0])
            if match:
                episode_numbers.add(int(match.group(1)))

    actions = np.array(actions)
    states = np.array(states)

    def calculate_stats(data, mask=None):
        if mask is None:
            mask = [True] * data.shape[1]
        
        stats = {
            'mean': np.mean(data, axis=0).tolist(),
            'std': np.std(data, axis=0).tolist(),
            'max': np.max(data, axis=0).tolist(),
            'min': np.min(data, axis=0).tolist(),
            'q01': np.quantile(data, 0.01, axis=0).tolist(),
            'q99': np.quantile(data, 0.99, axis=0).tolist(),
            'mask': mask,
        }
        return stats

    action_mask = [True, True, True, True, True, True, False, True, True, True, True, True, True, False]
    state_mask = [True, True, True, True, True, True, False, True, True, True, True, True, True, False]

    action_stats = calculate_stats(actions, action_mask)
    state_stats = calculate_stats(states, state_mask)

    result = {
        "rlbench": {
            "action": action_stats,
            "state": state_stats,
            "num_transitions": len(actions),
            "num_trajectories": max(episode_numbers) + 1 if episode_numbers else 0
        }
    }

    output_path = jsonl_filename.replace("train.jsonl", "train_statistics.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Statistics have been saved to {output_path}")


######## ---------main---------- #########

data_root = "/media/liuzhuoyang/data/franka_dual/npy"
img_save_root = "/media/liuzhuoyang/LCoT_VLA_MOT/training_data/franka_dual_sparse_fastslow"
json_save_root = "/media/liuzhuoyang/LCoT_VLA_MOT/training_data/franka_dual_json"

jsonl_filename = f'{json_save_root}/pick_corn_view1+3_chunk4_fast4_sparse_fastslow_train.jsonl'
json_file = f'{json_save_root}/pick_corn_view1+3_chunk4_fast4_sparse_fastslow_train.json'

task_lists = [
  'pick_corn_0904_keyframe',
#   'scoop_popcorn_0901_keyframe',
#   'scoop_popcorn_0902_keyframe',
#   'scoop_popcorn_0903_keyframe',
#   'scoop_popcorn_0904_keyframe',
#   'scoop_popcorn_0905_keyframe',
]

if not os.path.exists(json_save_root):
    os.makedirs(json_save_root, exist_ok=True)

npy_2_jsonl(data_root, img_save_root, jsonl_filename, task_lists)
cal_stats(jsonl_filename)
jsonl_2_json(jsonl_filename, json_file)