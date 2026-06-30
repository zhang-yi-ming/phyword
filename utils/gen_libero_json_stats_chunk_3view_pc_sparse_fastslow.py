import numpy as np
from PIL import Image
import json
import os
import re
from scipy.spatial.transform import Rotation as R

IMAGE_VIEWS_SAVE = ['image_primary', 'image_wrist'] 
IMAGE_VIEWS_SLOW = ['image_primary'] 
IMAGE_VIEWS_FAST = ['image_wrist'] 
FUTURE_STEPS = 4
FAST_SLOW_RATIO = 1
LATENT_STRIDE = 8
ACTION_CHUNK = 32


def create_padding_assets(save_dir, sample_step):
    padding_info = {}
    
    for view in IMAGE_VIEWS_SAVE:
        if view in sample_step:
            image_array = sample_step[view]
            while image_array.ndim > 3:
                image_array = image_array[0]

            img_shape = image_array.shape

            black_img = np.zeros(img_shape, dtype=np.uint8)
            img_obj = Image.fromarray(black_img)
            save_path = f'{save_dir}/padding_black_{view}.png'
            img_obj.save(save_path)
            padding_info[f'image_{view}'] = save_path

    # Create Zero State
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
                                if image_array.ndim == 4:
                                    image_array = image_array[0]
                                image = Image.fromarray(image_array)
                                image.save(f'{save_dir}/image{i}_{view}.png')

                padding_assets = create_padding_assets(save_dir, episode[0])

                for i in range(episode_length):
                    step = episode[i]
                    
                    slow_idx = (i // FAST_SLOW_RATIO) * FAST_SLOW_RATIO
                    
                    # --- Fast System Input ---
                    image_fast_list = []
                    for view in IMAGE_VIEWS_FAST:
                        image_fast_list.append(f'{save_dir}/image{i}_{view}.png')
                    
                    # --- Slow System Input ---
                    safe_slow_idx = min(slow_idx, episode_length - 1)
                    image_slow_list = []
                    for view in IMAGE_VIEWS_SLOW:
                        image_slow_list.append(f'{save_dir}/image{safe_slow_idx}_{view}.png')
                    
                    output_images_list = [] # List of Lists: [T][View]
                    output_state_list = []  # List: [T]

                    for k in range(1, FUTURE_STEPS + 1):
                        tgt_idx = slow_idx + (k * LATENT_STRIDE)
                        
                        if tgt_idx < episode_length:
                            tgt_step = episode[tgt_idx]
                            
                            # Images
                            for view in IMAGE_VIEWS_SLOW:
                                output_images_list.append(f'{save_dir}/image{tgt_idx}_{view}.png')
                            
                            # State
                            state_val = tgt_step['state'].copy() 
                            if isinstance(state_val, np.ndarray):
                                state_val = state_val.tolist()
                            output_state_list.append(state_val)
                            
                        else:
                            for view in IMAGE_VIEWS_SLOW:
                                output_images_list.append(padding_assets[f'image_{view}'])
                            
                            output_state_list.append(padding_assets.get('state', []))

                    # 5. Current Label Processing
                    current_state = step['state'].copy()

                    ##### MODIFIED: Action Chunking Logic #####
                    # 获取未来 ACTION_CHUNK 帧的 action
                    # 如果超出 episode 长度，则重复最后一帧的 action (Padding)
                    action_chunk_list = []
                    
                    for k in range(ACTION_CHUNK):
                        future_idx = i + k
                        
                        if future_idx < episode_length:
                            act = episode[future_idx]['action'].copy()
                        else:
                            act = episode[episode_length - 1]['action'].copy()
                        
                        if isinstance(act, np.ndarray):
                            act = act.tolist()
                        
                        action_chunk_list.append(act)
                    
                    # 现在的 action_chunk_list 形状应为 [ACTION_CHUNK, 7]
                    ###########################################

                    # 6. Construct Dictionary
                    episode_data = {
                        'input_images_fast': image_fast_list,
                        'input_images_slow': image_slow_list,
                        
                        'output_images': output_images_list, 
                        'output_state': output_state_list,
                        
                        'action': action_chunk_list,
                        'state': current_state.tolist(),
                        'language_instruction': step['language_instruction'],
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
            # --- Output (Latent) ---
            "output_image": item["output_images"], 
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

    if len(actions.shape) == 3:
        actions_flat = actions.reshape(-1, actions.shape[-1])
    else:
        actions_flat = actions

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

    action_mask = [True, True, True, True, True, True, False]
    state_mask = [True, True, True, True, True, True, False, False]

    action_stats = calculate_stats(actions_flat, action_mask)
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

data_root = "/media/liuzhuoyang/data/libero/npy"
img_save_root = "/media/liuzhuoyang/LCoT_VLA_MOT/training_data/libero_two_sparse_fastslow"
json_save_root = "/media/liuzhuoyang/LCoT_VLA_MOT/training_data/libero_two_json"

jsonl_filename = f'{json_save_root}/libero_10_no_noops_view2_chunk4_32_stride8_fast1_sparse_fastslow_train.jsonl'
json_file = f'{json_save_root}/libero_10_no_noops_view2_chunk4_32_stride8_fast1_sparse_fastslow_train.json'
task_lists = [
  'libero_10_no_noops',
#   'libero_goal_no_noops',
#   'libero_object_no_noops',
#   'libero_spatial_no_noops'
]

if not os.path.exists(json_save_root):
    os.makedirs(json_save_root, exist_ok=True)

npy_2_jsonl(data_root, img_save_root, jsonl_filename, task_lists)
cal_stats(jsonl_filename)
jsonl_2_json(jsonl_filename, json_file)

