import os
import json
import time
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


# def process_single_file(file, data_root):
#     file_path = os.path.join(data_root, file)
#     episode = np.load(file_path, allow_pickle=True)
#     if len(episode)<=5: 
#         return None
#     actions = np.stack([step["action"][0] for step in episode[:-1]])
#     L = len(episode) - 1
#     json_items = [{"idx": idx, "npy": file} for idx in range(L)]
#     return actions, json_items, L


def process_single_file(file, data_root):
    file_path = os.path.join(data_root, file)
    episode = np.load(file_path, allow_pickle=True)
    
    # 太短的 episode 直接跳过
    if len(episode) <= 5:
        return None

    actions_list = []
    for step in episode[:-1]:
        # 检查 action 长度必须是 1
        if len(step["action"]) != 1 or len(step["action"][0]) != 7:
            return None
        
        action = step["action"][0]
        # 如果 action 有任何元素 > 1，则丢弃整个 episode
        if np.any(action > 1):
            return None
        
        actions_list.append(action)

    actions = np.stack(actions_list)
    L = len(episode) - 1
    json_items = [{"idx": idx, "npy": file} for idx in range(L)]
    return actions, json_items, L



def process_npy_and_save(data_root, jsonl_path, json_path, stats_path, num_workers=16):
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    npy_files = [f for f in os.listdir(data_root) if f.endswith(".npy")]
    episode_num = len(npy_files)
    print(f"Found {episode_num} npy files.")

    all_actions = []
    total_transitions = 0
    t0 = time.time()
    valid_num = 0
    with open(jsonl_path, "w") as jsonl_file:
        json_items = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for result in tqdm(executor.map(lambda f: process_single_file(f, data_root), npy_files),
                            total=len(npy_files), desc="Processing npy", ncols=100):
                if result is None:
                    continue
                actions, items, L = result

                all_actions.append(actions)
                total_transitions += L
                valid_num += 1

                for item in items:
                    jsonl_file.write(json.dumps(item) + "\n")
                json_items.extend(items)
    print(f"Saved jsonl index to {jsonl_path}")


    # 直接用收集好的 json_items 写 json
    with open(json_path, "w") as f:
        json.dump(json_items, f)
    print(f"Saved json to {json_path}")


    # 计算统计信息
    all_actions = np.vstack(all_actions)
    print(f"Loaded {total_transitions} transitions in {(time.time()-t0):.2f}s, start computing statistics...")
    stats = {
        'mean': np.mean(all_actions, axis=0).tolist(),
        'std': np.std(all_actions, axis=0).tolist(),
        'max': np.max(all_actions, axis=0).tolist(),
        'min': np.min(all_actions, axis=0).tolist(),
        'q01': np.quantile(all_actions, 0.01, axis=0).tolist(),
        'q99': np.quantile(all_actions, 0.99, axis=0).tolist(),
        'mask': [True, True, True, True, True, True, False],
    }
    result = {
        "oxe_pretrain": {
            "action": stats,
            "num_transitions": total_transitions,
            "num_trajectories": episode_num,
            "valid_traj": valid_num,
        }
    }
    with open(stats_path, "w") as f:
        json.dump(result, f, indent=2)


    print(f"Statistics have been saved to {stats_path}")
    print(f"Processed {valid_num}/{episode_num} valid npy files.")



if __name__ == "__main__":
    data_root = "/media/realworld_data/rtx_npy_0912"
    json_save_root = "/media/chenhao/DoubleRL-VLA/training_data/json"
    jsonl_filename = os.path.join(json_save_root, "oxe_pretrain.jsonl")
    json_file = os.path.join(json_save_root, "oxe_pretrain.json")
    stats_file = os.path.join(json_save_root, "oxe_pretrain_statistics.json")
    process_npy_and_save(data_root, jsonl_filename, json_file, stats_file, num_workers=16)
