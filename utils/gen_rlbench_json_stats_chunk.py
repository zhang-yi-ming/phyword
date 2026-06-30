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

                        for k in range(4): 
                            future_image_key = f'front_image_next_{k+1}'
                            if future_image_key in step:
                                future_image_array = step[future_image_key]
                                future_image = Image.fromarray(future_image_array)
                                future_image.save(f'{img_save_root}/{task}/{file}/image{i}_future{k+1}.png')

                for i in range(episode_length):
                    step = episode[i]
                    image_old = f'{img_save_root}/{task}/{file}/image{i}.png'
                    image_new_list = []
                    for k in range(4):
                        future_image_path = f'{img_save_root}/{task}/{file}/image{i}_future{k+1}.png'
                        image_new_list.append(future_image_path)

                    action_7d = step['action']

                    action_7d[3:6] = unique_euler_xyz_rad(action_7d[3:6])
                    step['state'][3:6] = unique_euler_xyz_rad(step['state'][3:6])

                    # Create dictionary for this step
                    episode_data = {
                        'image_old': image_old,
                        'image_new': image_new_list,
                        'action': action_7d.tolist(),
                        'state': step['state'].tolist(),
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
            "input_image_resolution": [384, 384],
            "output_image": item["image_new"],  # 现在是一个列表
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

data_root = "/media/liuzhuoyang/data/rlbench/data_rlbench_npy/keyframe_delta_position_abs_euler_1024_lcot_chunk4_shortlang_1128/for_rlds"
img_save_root = "/media/liuzhuoyang/LCoT_VLA_MOT/training_data/rlbench"
json_save_root = "/media/liuzhuoyang/LCoT_VLA_MOT/training_data/json"
jsonl_filename = f'{json_save_root}/4tasks_chunk4_shortlang_train.jsonl'
json_file = f'{json_save_root}/4tasks_chunk4_shortlang_train.json'

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

