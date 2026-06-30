import numpy as np
from PIL import Image
import json
import os
import re
from scipy.spatial.transform import Rotation as R

IMAGE_VIEWS = ['front_image'] 
FUTURE_STEPS = 4
FAST_SLOW_RATIO = 1

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
    for view in IMAGE_VIEWS:
        if view in sample_step:
            img_shape = sample_step[view].shape # (H, W, C)
            black_img = np.zeros(img_shape, dtype=np.uint8)
            img_obj = Image.fromarray(black_img)
            save_path = f'{save_dir}/padding_black_{view}.png'
            img_obj.save(save_path)
            padding_info[f'image_{view}'] = save_path

    # 2. Create Zero Point Cloud
    if 'pointcloud' in sample_step:
        pc_shape = sample_step['pointcloud'].shape
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

                    # --- Pass 1: 保存所有帧的图像和点云 ---
                    for i in range(episode_length):
                        step = episode[i]
                        # Save Images
                        for view in IMAGE_VIEWS:
                            if view in step:
                                image_array = step[view]
                                image = Image.fromarray(image_array)
                                image.save(f'{save_dir}/image{i}_{view}.png')
                        # Save PC
                        if 'pointcloud' in step:
                            pc_array = step['pointcloud']
                            np.save(f'{save_dir}/pc{i}.npy', pc_array)
                        else:
                            print(f'Warning: No pointcloud found for {task} step {i}')

                # --- 准备 Padding 资源 ---
                # 使用第一帧作为模板来生成 padding 文件
                padding_assets = create_padding_assets(save_dir, episode[0])

                # --- Pass 2: 生成 JSONL 数据条目 ---
                for i in range(episode_length):
                    step = episode[i]
                    
                    # 1. 计算 Slow Index (每 FAST_SLOW_RATIO 更新一次)
                    # 例如 FAST_SLOW_RATIO=4: 0,1,2,3 -> 0; 4,5,6,7 -> 4
                    slow_idx = (i // FAST_SLOW_RATIO) * FAST_SLOW_RATIO
                    
                    # 2. Input Images (Fast) - 当前帧
                    image_fast_list = []
                    for view in IMAGE_VIEWS:
                        image_fast_list.append(f'{save_dir}/image{i}_{view}.png')
                    
                    # 3. Input Images (Slow) - 对应的 Slow 关键帧
                    # 如果 slow_idx 超过了数据范围(理论上不会，因为是向下取整)，做个保护
                    safe_slow_idx = min(slow_idx, episode_length - 1)
                    image_slow_list = []
                    for view in IMAGE_VIEWS:
                        image_slow_list.append(f'{save_dir}/image{safe_slow_idx}_{view}.png')

                    pc_current = f'{save_dir}/pc{i}.npy'

                    # 4. Outputs (Future Prediction)
                    # 逻辑: 取 [slow_idx+1, slow_idx+2, slow_idx+3, slow_idx+4]
                    # 如果越界，使用 Padding
                    
                    output_images_list = [] # List of Lists: [T][View]
                    output_pc_list = []     # List: [T]
                    output_state_list = []  # List: [T]

                    for k in range(1, FUTURE_STEPS + 1):
                        tgt_idx = slow_idx + k
                        
                        if tgt_idx < episode_length:
                            # --- Real Data ---
                            tgt_step = episode[tgt_idx]
                            
                            # Images
                            for view in IMAGE_VIEWS:
                                output_images_list.append(f'{save_dir}/image{tgt_idx}_{view}.png')
                            
                            # PC
                            output_pc_list.append(f'{save_dir}/pc{tgt_idx}.npy')
                            
                            # State
                            state_val = tgt_step['state'].copy() # Copy to avoid modifying original
                            if len(state_val) >= 6:
                                state_val[3:6] = unique_euler_xyz_rad(state_val[3:6])
                            if isinstance(state_val, np.ndarray):
                                state_val = state_val.tolist()
                            output_state_list.append(state_val)
                            
                        else:
                            # --- Padding Data ---
                            # Images
                            for view in IMAGE_VIEWS:
                                output_images_list.append(padding_assets[f'image_{view}'])
                            
                            # PC
                            output_pc_list.append(padding_assets.get('pc', '')) # Empty string if no PC
                            
                            # State
                            output_state_list.append(padding_assets.get('state', []))

                    # 5. Current Label Processing
                    action_7d = step['action'].copy()
                    action_7d[3:6] = unique_euler_xyz_rad(action_7d[3:6])
                    
                    current_state = step['state'].copy()
                    if len(current_state) >= 6:
                         current_state[3:6] = unique_euler_xyz_rad(current_state[3:6])

                    # 6. Construct Dictionary
                    episode_data = {
                        'input_images_fast': image_fast_list,
                        'input_images_slow': image_slow_list,
                        'input_pointcloud': pc_current,
                        
                        'output_images': output_images_list, 
                        'output_pointcloud': output_pc_list,
                        'output_state': output_state_list,
                        
                        'action': action_7d.tolist(),
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
        
        # 构造最终的 JSON Item
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

    action_mask = [True, True, True, True, True, True, False]
    state_mask = [True, True, True, True, True, True, False]

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

data_root = "/media/liuzhuoyang/data/rlbench/data_rlbench_npy/keyframe_delta_position_abs_euler_1024_4view_lcot_chunk4_img+pc+state_1209/for_rlds"
img_save_root = "/media/liuzhuoyang/LCoT_VLA_MOT/training_data/rlbench_1view_sparse_fastslow"
json_save_root = "/media/liuzhuoyang/LCoT_VLA_MOT/training_data/rlbench_json"

jsonl_filename = f'{json_save_root}/12tasks_1view_chunk1_fast4_sparse_fastslow_train.jsonl'
json_file = f'{json_save_root}/12tasks_1view_chunk1_fast4_sparse_fastslow_train.json'

task_lists = [
  'close_box',
  'close_fridge',
  'close_laptop_lid',
  'phone_on_base',
  'place_wine_at_rack_location',
  'sweep_to_dustpan',
  'take_frame_off_hanger',
  'take_umbrella_out_of_umbrella_stand',
  'toilet_seat_down',
  'water_plants',
  'lamp_on',
  'unplug_charger',
]

if not os.path.exists(json_save_root):
    os.makedirs(json_save_root, exist_ok=True)

npy_2_jsonl(data_root, img_save_root, jsonl_filename, task_lists)
cal_stats(jsonl_filename)
jsonl_2_json(jsonl_filename, json_file)