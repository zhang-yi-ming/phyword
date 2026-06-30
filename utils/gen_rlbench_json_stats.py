import numpy as np
from PIL import Image
import json
import os
import re
from scipy.spatial.transform import Rotation as R

def unique_euler_xyz_rad(angles, range_style="2pi"):
    """
    输入: 欧拉角 (xyz 顺序)，弧度制 (任意范围, 可正可负, 可超过 2π)
    输出: 欧拉角 (xyz 顺序)，弧度制，严格唯一表示

    参数:
        precision: 保留小数位数
        range_style: "negpi" -> (-π, π], "2pi" -> [0, 2π)
    """
    # 输入是弧度
    rot = R.from_euler('xyz', angles, degrees=False)
    
    # 转回 xyz (弧度制)
    euler = rot.as_euler('xyz', degrees=False)
    
    # wrap 到 (-π, π]
    euler = (euler + np.pi) % (2 * np.pi) - np.pi
    
    # 约束: y ∈ [-π/2, π/2]
    if euler[1] > np.pi/2:
        euler[1] = np.pi - euler[1]
        euler[0] += np.pi
        euler[2] += np.pi
    elif euler[1] < -np.pi/2:
        euler[1] = -np.pi - euler[1]
        euler[0] += np.pi
        euler[2] += np.pi
    
    # 再 wrap 一次
    euler = (euler + np.pi) % (2 * np.pi) - np.pi
    
    # 如果要求 [0, 2π)，再转换
    if range_style == "2pi":
        euler = euler % (2 * np.pi)
    
    return euler

def npy_2_jsonl(data_root, img_save_root, jsonl_filename, task_lists):
    with open(jsonl_filename, 'w') as f:
        
        for task in task_lists:
            print(f'Processing task: {task}')

            if not os.path.exists(f'{img_save_root}/{task}'):
                os.mkdir(f'{img_save_root}/{task}')

            for file in os.listdir(f'{data_root}/{task}'):
                if not file.endswith('.npy'): 
                    continue

                print('generating:', file, end=' ')

                episode = np.load(f'{data_root}/{task}/{file}', allow_pickle=True)

                file = file.replace('.npy', '')
                episode_length = len(episode)
                print('episode_length:', episode_length)

                if not os.path.exists(f'{img_save_root}/{task}/{file}'):
                    os.mkdir(f'{img_save_root}/{task}/{file}')

                    for i in range(episode_length):
                        step = episode[i]
                        image_array = step['front_image']
                        image = Image.fromarray(image_array)
                        image.save(f'{img_save_root}/{task}/{file}/image{i}.png')

                for i in range(episode_length-1):
                    step = episode[i]
                    image_old = f'{img_save_root}/{task}/{file}/image{i}.png'
                    image_new = f'{img_save_root}/{task}/{file}/image{i+1}.png'

                    action = step['action']
                    action_56d = step['action'] # action chunk of 8
                    action_1 = action_56d[:7]  
                    action_2 = action_56d[7:14]
                    action_3 = action_56d[14:21]
                    action_4 = action_56d[21:28]
                    action_5 = action_56d[28:35]
                    action_6 = action_56d[35:42]
                    action_7 = action_56d[42:49]
                    action_8 = action_56d[49:56]

                    delta_position_1 = action_1[:3]
                    delta_position_2 = action_2[:3]
                    delta_position_3 = action_3[:3]
                    delta_position_4 = action_4[:3]
                    delta_position_5 = action_5[:3]
                    delta_position_6 = action_6[:3]
                    delta_position_7 = action_7[:3]
                    delta_position_8 = action_8[:3]
                    
                    abs_rot_1 = action_1[3:6]
                    abs_rot_2 = action_2[3:6]
                    abs_rot_3 = action_3[3:6]
                    abs_rot_4 = action_4[3:6]
                    abs_rot_5 = action_5[3:6]
                    abs_rot_6 = action_6[3:6]
                    abs_rot_7 = action_7[3:6]
                    abs_rot_8 = action_8[3:6]

                    gripper_1 = action_1[-1]
                    gripper_2 = action_2[-1]
                    gripper_3 = action_3[-1]
                    gripper_4 = action_4[-1]
                    gripper_5 = action_5[-1]
                    gripper_6 = action_6[-1]
                    gripper_7 = action_7[-1]
                    gripper_8 = action_8[-1]
                    
                    delta_position_total = delta_position_1 + delta_position_2 +\
                        delta_position_3 + delta_position_4 + delta_position_5 + \
                        delta_position_6 + delta_position_7 + delta_position_8       
                    
                    action_7d = np.concatenate([delta_position_total, abs_rot_8, [gripper_8]]).tolist()

                    action_7d[3:6] = unique_euler_xyz_rad(action_7d[3:6])
                    step['state'][3:6] = unique_euler_xyz_rad(step['state'][3:6])

                    # Create dictionary for this step
                    episode_data = {
                        'image_old': image_old,
                        'image_new': image_new,
                        'action': action_7d,
                        'state': step['state'].tolist(),
                        'language_instruction': step['language_instruction'],
                        'language_subgoals': step['language_subgoals']
                    }

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
            "input_image_resolution": [384, 384],
            "output_image": item["image_new"],
            "output_image_resolution": [384, 384],
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
            
            # 从image_old中提取episode编号
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

data_root = "/gpfs/0607-cluster/chenhao/data/rlbench/keyframe_fast_slow_chunk8_addlast_0806/for_rlds"
img_save_root = "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/rlbench"
json_save_root = "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/json"
jsonl_filename = f'{json_save_root}/close_box_train.jsonl'
json_file = f'{json_save_root}/close_box_train.json'

task_lists = [
  'close_box',
#   'close_fridge',
#   'close_laptop_lid',
#   'phone_on_base',
#   'place_wine_at_rack_location',
#   'sweep_to_dustpan',
#   'take_frame_off_hanger',
#   'take_umbrella_out_of_umbrella_stand',
#   'toilet_seat_down',
#   'water_plants'
]

npy_2_jsonl(data_root, img_save_root, jsonl_filename, task_lists)
cal_stats(jsonl_filename)
jsonl_2_json(jsonl_filename, json_file)