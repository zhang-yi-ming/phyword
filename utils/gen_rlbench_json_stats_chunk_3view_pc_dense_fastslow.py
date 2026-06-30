import numpy as np
from PIL import Image
import json
import os
import re
from scipy.spatial.transform import Rotation as R

IMAGE_VIEWS = ['front_image'] # 'left_shoulder_image', 'right_shoulder_image'
FUTURE_STEPS = 4

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
                        
                        # 1. 保存当前帧图像
                        for view in IMAGE_VIEWS:
                            if view in step:
                                image_array = step[view]
                                image = Image.fromarray(image_array)
                                image.save(f'{save_dir}/image{i}_{view}.png')

                        # 2. 保存当前帧点云
                        if 'pointcloud' in step:
                            pc_array = step['pointcloud']
                            np.save(f'{save_dir}/pc{i}.npy', pc_array)
                        else:
                            print('No pointcloud found for episode:', task, 'step:', i)
                        
                        # [修改点 1] 删除原来的 "3. 保存未来帧..." 代码块
                        # 因为第 i+k 帧会在它自己的循环轮次中作为当前帧被保存为 image{i+k}.png
                        # 我们不需要重复保存文件，只需要在下面生成 JSON 时指向那些文件即可。

                # 构建映射表: keypoint -> index
                kp_to_idx = {step['keypoint']: idx for idx, step in enumerate(episode)}

                for i in range(episode_length):
                    step = episode[i]
                    
                    # --- Fast System Input (Current Frame) ---
                    image_fast_list = []
                    for view in IMAGE_VIEWS:
                        image_fast_list.append(f'{save_dir}/image{i}_{view}.png')
                    
                    # --- Slow System Input (Sparse Keyframe) ---
                    sparse_kp = step['sparse_keypoint']
                    if sparse_kp in kp_to_idx:
                        slow_idx = kp_to_idx[sparse_kp]
                    else:
                        slow_idx = i
                    
                    image_slow_list = []
                    for view in IMAGE_VIEWS:
                        image_slow_list.append(f'{save_dir}/image{slow_idx}_{view}.png')

                    pc_old = f'{save_dir}/pc{i}.npy'

                    # --- Future Output (Helper for Latent) ---
                    # [修改点 2] 改为通过索引获取未来帧数据
                    image_new_list = []
                    pc_new_list = []
                    state_new_list = []

                    for k in range(FUTURE_STEPS):
                        future_idx = min(i + k + 1, episode_length - 1)
                        future_step = episode[future_idx]

                        image_new_list.append(f'{save_dir}/image{future_idx}_front_image.png')
                        
                        # Output PC: 引用未来那一帧的点云路径
                        # 注意：这里我们假设未来帧也保存了 pc{future_idx}.npy
                        pc_new_list.append(f'{save_dir}/pc{future_idx}.npy')
                        
                        # Output State: 从 future_step 中读取 'state'
                        # 注意：原始npy里的 key 叫 'state'，不是 'state_next_k'
                        state_val = future_step['state'].copy()
                        if len(state_val) >= 6:
                            state_val[3:6] = unique_euler_xyz_rad(state_val[3:6])
                        if isinstance(state_val, np.ndarray):
                            state_val = state_val.tolist()
                        state_new_list.append(state_val)

                    action_7d = step['action'].copy()
                    action_7d[3:6] = unique_euler_xyz_rad(action_7d[3:6])
                    
                    fast_state = step['state'].copy()
                    if len(fast_state) >= 6:
                         fast_state[3:6] = unique_euler_xyz_rad(fast_state[3:6])
                    if i % FUTURE_STEPS == 0:
                        slow_state = fast_state.copy()

                    episode_data = {
                        'input_images_fast': image_fast_list,
                        'input_images_slow': image_slow_list,
                        'input_pointcloud': pc_old,
                        'output_images': image_new_list, 
                        'output_pointcloud': pc_new_list,
                        'output_state': state_new_list,
                        'action': action_7d.tolist(),
                        'state_fast': fast_state.tolist(),
                        'state_slow': slow_state.tolist(),
                        'language_instruction': step['language_instruction'],
                    }

                    if 'language_subgoals' in step:
                        episode_data['language_subgoals'] = step['language_subgoals']

                    f.write(json.dumps(episode_data) + '\n')


def jsonl_2_json(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    output_data = []
    resolution = [384, 384]
    
    for line in lines:
        item = json.loads(line)
        
        # JSON Item
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
            "output_image": item["output_images"], # [[v1,v2]...[v1,v2]]
            "output_pointcloud": item["output_pointcloud"],
            "output_image_resolution": [384, 384],
            "output_state": item["output_state"],
            # --- Labels ---
            "action": item["action"],
            "state_fast": item["state_fast"],
            "state_slow": item["state_slow"],
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
            states.append(data['state_fast'])
            
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

data_root = "/media/liuzhuoyang/data/rlbench/data_rlbench_npy/keyframe_delta_position_abs_euler_1024_4view_lcot_fastslow_chunk4_img+pc+state_1212/for_rlds"
img_save_root = "/media/liuzhuoyang/LCoT_VLA_MOT/training_data/rlbench_1view_fastslow"
json_save_root = "/media/liuzhuoyang/LCoT_VLA_MOT/training_data/json"

jsonl_filename = f'{json_save_root}/4tasks_1view_chunk4_fastslow_train.jsonl'
json_file = f'{json_save_root}/4tasks_1view_chunk4_fastslow_train.json'

task_lists = [
  'close_box',
#   'close_fridge',
  'close_laptop_lid',
  'phone_on_base',
#   'place_wine_at_rack_location',
  'sweep_to_dustpan',
#   'take_frame_off_hanger',
#   'take_umbrella_out_of_umbrella_stand',
#   'toilet_seat_down',
#   'water_plants',
#   'lamp_on',
#   'unplug_charger',
]

if not os.path.exists(json_save_root):
    os.makedirs(json_save_root, exist_ok=True)

npy_2_jsonl(data_root, img_save_root, jsonl_filename, task_lists)
cal_stats(jsonl_filename)
jsonl_2_json(jsonl_filename, json_file)