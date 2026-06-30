import json
import numpy as np
import os
import re

FILE_LIST = [
    "/media/liuzhuoyang/LCoT_VLA_MOT/training_data/rlbench_json/12tasks_1view_chunk1_fast4_sparse_fastslow_train.json",
    "/media/liuzhuoyang/LCoT_VLA_MOT/training_data/rlbench_json/12tasks_1view_chunk2_fast4_sparse_fastslow_train.json",
    "/media/liuzhuoyang/LCoT_VLA_MOT/training_data/rlbench_json/12tasks_1view_chunk4_fast4_sparse_fastslow_train.json",
]

OUTPUT_FILE = "/media/liuzhuoyang/LCoT_VLA_MOT/training_data/rlbench_json/12tasks_1view_chunk1+2+4_fast4_sparse_fastslow_train.json"

def calculate_stats(data, mask=None):
    if mask is None:
        mask = [True] * data.shape[1]

    data = np.array(data)
    
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

def merge_and_recalc(file_list, output_path):
    combined_data = []

    print("[1/5] Loading input files...")
    for idx, file_path in enumerate(file_list, 1):
        print(f"    - [{idx}/{len(file_list)}] {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
            combined_data.extend(data)

    print(f"[2/5] Merged {len(file_list)} files, total samples: {len(combined_data)}")

    dir_name = os.path.dirname(output_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    print(f"[3/5] Saving merged JSON to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=2)

    print("[4/5] Recalculating Statistics...")

    actions = []
    states = []
    episode_numbers = set()

    for item in combined_data:
        actions.append(item['action'])
        states.append(item['state'])

        if 'input_image_fast' in item and item['input_image_fast']:
            path = item['input_image_fast'][0]
            match = re.search(r'episode(\d+)', path)
            if match:
                episode_numbers.add(int(match.group(1)))

    actions = np.array(actions)
    states = np.array(states)

    print(f"    - Total Actions Shape: {actions.shape}")
    print(f"    - Total States Shape: {states.shape}")
    print(f"    - Total Unique Episodes: {len(episode_numbers)}")

    action_mask = [True] * actions.shape[1]
    if len(action_mask) > 6:
        action_mask[6] = False

    state_mask = [True] * states.shape[1]
    if len(state_mask) > 6:
        state_mask[6] = False

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

    output_stats_path = output_path.replace(".json", "_statistics.json")

    print(f"[5/5] Saving statistics to: {output_stats_path}")
    with open(output_stats_path, 'w') as f:
        json.dump(result, f, indent=2)

    print("Done! All Processed Successfully.")

if __name__ == "__main__":
    merge_and_recalc(FILE_LIST, OUTPUT_FILE)