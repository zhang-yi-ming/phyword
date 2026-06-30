import numpy as np
from PIL import Image
import json
import os
import re
from scipy.spatial.transform import Rotation as R

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
                os.makedirs(f'{img_save_root}/{task}', exist_ok=True) # Modified: 使用 makedirs 并允许存在

            for file in os.listdir(f'{data_root}/{task}'):
                if not file.endswith('.npy'): 
                    continue

                print('generating:', file, end=' ')

                episode = np.load(f'{data_root}/{task}/{file}', allow_pickle=True)

                file_base_name = file.replace('.npy', '') # Modified: 避免变量名混淆
                episode_length = len(episode)
                print('episode_length:', episode_length)

                save_dir = f'{img_save_root}/{task}/{file_base_name}'
                # print(save_dir)
                # input()
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)

                    for i in range(episode_length):
                        step = episode[i]
                        
                        # 1. 保存当前帧图像
                        image_array = step['front_image']
                        image = Image.fromarray(image_array)
                        image.save(f'{save_dir}/image{i}.png')
                        
                        # 2. 保存当前帧点云 <--- Modified
                        if 'pointcloud' in step:
                            pc_array = step['pointcloud']
                            np.save(f'{save_dir}/pc{i}.npy', pc_array)
                            # print(save_dir)
                            # input()
                        else:
                            print('No pointcloud found for episode:', task, 'step:', i)
                            input()

                        # 3. 保存未来帧图像和点云 <--- Modified
                        for k in range(4): 
                            # Future Image
                            future_image_key = f'front_image_next_{k+1}'
                            if future_image_key in step:
                                future_image_array = step[future_image_key]
                                future_image = Image.fromarray(future_image_array)
                                future_image.save(f'{save_dir}/image{i}_future{k+1}.png')
                            
                            # Future Point Cloud <--- Modified
                            future_pc_key = f'pointcloud_next_{k+1}'
                            if future_pc_key in step:
                                future_pc_array = step[future_pc_key]
                                np.save(f'{save_dir}/pc{i}_future{k+1}.npy', future_pc_array)

                for i in range(episode_length):
                    step = episode[i]
                    image_old = f'{save_dir}/image{i}.png'
                    pc_old = f'{save_dir}/pc{i}.npy' # <--- Modified

                    image_new_list = []
                    pc_new_list = [] # <--- Modified
                    state_new_list = [] # <--- Modified

                    for k in range(4):
                        # Image path
                        future_image_path = f'{save_dir}/image{i}_future{k+1}.png'
                        image_new_list.append(future_image_path)
                        
                        # PC path <--- Modified
                        future_pc_path = f'{save_dir}/pc{i}_future{k+1}.npy'
                        pc_new_list.append(future_pc_path)
                        
                        # Future State data <--- Modified
                        future_state_key = f'state_next_{k+1}'
                        if future_state_key in step:
                            # 注意：如果 state 包含欧拉角，可能也需要像 action 那样处理一下角度
                            # 这里暂时直接存储原始值，或者你可以根据需要添加 unique_euler 处理
                            state_val = step[future_state_key]
                            state_val[3:6] = unique_euler_xyz_rad(state_val[3:6])
                            if isinstance(state_val, np.ndarray):
                                state_val = state_val.tolist()
                            state_new_list.append(state_val)

                    action_7d = step['action']
                    action_7d[3:6] = unique_euler_xyz_rad(action_7d[3:6])
                    
                    # Process current state euler angles
                    current_state = step['state']
                    # 假设 state 格式与 action 类似，包含位置+欧拉角+夹爪
                    # 如果 state 中 3:6 是欧拉角，进行处理
                    if len(current_state) >= 6:
                         current_state[3:6] = unique_euler_xyz_rad(current_state[3:6])

                    # Create dictionary for this step
                    episode_data = {
                        'image_old': image_old,
                        'input_pointcloud': pc_old, # <--- Modified
                        'image_new': image_new_list,
                        'output_pointcloud': pc_new_list, # <--- Modified
                        'output_state': state_new_list,   # <--- Modified
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
        
        new_item = {
            "input_prompt": item["language_instruction"],
            "input_image": [item["image_old"]],
            "input_pointcloud": [item["input_pointcloud"]], # <--- Modified
            "input_image_resolution": [384, 384],
            "output_image": item["image_new"], 
            "output_pointcloud": item["output_pointcloud"], # <--- Modified
            "output_image_resolution": [384, 384],
            "output_state": item["output_state"], # <--- Modified
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
            
            match = re.search(r'episode(\d+)', data['image_old'])
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
            "num_trajectories": max(episode_numbers) + 1  # episode编号从0开始
        }
    }

    output_path = jsonl_filename.replace("train.jsonl", "train_statistics.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Statistics have been saved to {output_path}")



######## ---------main---------- #########

data_root = "/media/liuzhuoyang/data/rlbench/data_rlbench_npy/keyframe_delta_position_abs_euler_1024_lcot_chunk4_helper_img+pc+state_1204/for_rlds"
img_save_root = "/media/liuzhuoyang/LCoT_VLA_MOT/training_data/rlbench_full"
json_save_root = "/media/liuzhuoyang/LCoT_VLA_MOT/training_data/json"
jsonl_filename = f'{json_save_root}/4tasks_chunk4_img+pc+state_train.jsonl'
json_file = f'{json_save_root}/4tasks_chunk4_img+pc+state_train.json'

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
#   'water_plants'
]

npy_2_jsonl(data_root, img_save_root, jsonl_filename, task_lists)
cal_stats(jsonl_filename)
jsonl_2_json(jsonl_filename, json_file)