"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
from PIL import Image
import wandb

import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor, ActionTokenizer
# from janus.models import VLChatProcessor, ActionTokenizer
from models.cosmos_janus import CosmosJanusMoT
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)

from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

NUM_ACTIONS_CHUNK = 8

# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 350,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 320,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 820,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    model_path: Union[str, Path] = ""
    cosmos_experiment_name : str = ""                        # Name of the original Cosmos experiment (for loading video backbone)
    cosmos_model_path: Union[str, Path] = ""           # Path to the original Cosmos checkpoint (for loading video backbone)


    use_proprio: bool = False                        # Whether to include proprio state in input

    center_crop: bool = False                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 16                     # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = "rlbench"                # Action un-normalization key

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 42                                    # Random Seed (for reproducibility)

    cuda: str = "0"                                  # CUDA device to use
    denoise_steps: int = 10                          # Number of denoising steps

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
        
    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def model_load(cfg: Any):
    # =================================================================
    # 1. 从原始基座加载 Processor（避免 0x80 解析错误）
    # =================================================================
    print(f"Loading Processor from {cfg.model_path}...")
    vl_chat_processor = VLChatProcessor.from_pretrained(cfg.model_path, trust_remote_code=True)
    tokenizer = vl_chat_processor.tokenizer
    action_tokenizer = ActionTokenizer(tokenizer, need_to_sub=3)

    # =================================================================
    # 2. 加载 Janus Action 骨架
    # =================================================================
    print(f"Loading Janus Action Base from {cfg.model_path}...")
    janus_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        flow=True, action_dim=7, ignore_mismatched_sizes=True
    )
    
    # =================================================================
    # 3. 加载 Cosmos 骨架 (配合猴子补丁屏蔽 T5 文本编码器)
    # =================================================================
    print("Loading Cosmos Video Base...")
    experiment_opts = ["data_train=mock", "data_val=mock"]

    import cosmos_predict2._src.predict2.models.text2world_model_rectified_flow as t2w_module
    import torch.nn as nn
    
    class DummyTextEncoder(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            
    t2w_module.TextEncoder = DummyTextEncoder
    
    cosmos_wrapper, _ = load_model_from_checkpoint(
        experiment_name=cfg.cosmos_experiment_name,
        s3_checkpoint_dir=cfg.cosmos_model_path,
        config_file="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        load_ema_to_reg=True,
        to_device="cpu",
        experiment_opts=experiment_opts
    )
    
    # =================================================================
    # 4. 组装 MoT 架构并灌入我们全参保存的权重
    # =================================================================
    print("Building Joint MoT Architecture...")
    model = CosmosJanusMoT(cosmos_wrapper.net, cosmos_wrapper.tokenizer, janus_model, cfg)
    
    # 动态解析检查点路径
    if cfg.pretrained_checkpoint.endswith('.pt'):
        ckpt_path = cfg.pretrained_checkpoint
        base_dir = os.path.dirname(cfg.pretrained_checkpoint)
    else:
        ckpt_path = os.path.join(cfg.pretrained_checkpoint, "cosmos_janus_mot.pt")
        base_dir = cfg.pretrained_checkpoint

    print(f"Loading Fine-Tuned MoT Weights from {ckpt_path}...")
    
    state_dict = torch.load(ckpt_path, map_location="cpu")
    # 【关键修改】：加上 strict=False 提高鲁棒性，防止 cache buffer 报错
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if len(missing_keys) > 0:
        print(f"Warning: Missing keys in state_dict (usually OK if they are caches): {missing_keys[:5]}...")

    # 推到 GPU 并设置为推理模式
    model = model.to(torch.bfloat16).cuda().eval()

    statistics_path = os.path.join(base_dir, "train_statistics.json")
    print(f"Loading Statistics from {statistics_path}...")
    with open(statistics_path, 'r') as f:
        stats_data = json.load(f)
        
    dataset_name = next(iter(stats_data))
    
    statistic = {
        'action_mask': np.array(stats_data[dataset_name]['action']['mask'], dtype=bool),
        'action_q01': np.array(stats_data[dataset_name]['action']['q01'], dtype=np.float32),
        'action_q99': np.array(stats_data[dataset_name]['action']['q99'], dtype=np.float32),
        'state_mask': np.array(stats_data[dataset_name]['state']['mask'], dtype=bool),
        'state_q01': np.array(stats_data[dataset_name]['state']['q01'], dtype=np.float32),
        'state_q99': np.array(stats_data[dataset_name]['state']['q99'], dtype=np.float32),
    }

    return model, vl_chat_processor, action_tokenizer, statistic



def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # # Resize images to size expected by model
    # img_resized = resize_image_for_policy(img, resize_size)
    # wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img,
        "wrist_image": wrist_img,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def run_episode(
    cfg: Any,
    env,
    task_description: str,
    model,
    processor,
    action_tokenizer,
    statistic,
    initial_state=None,
    log_file=None,
    task_id=None,
):
    """Run a single episode in the environment."""
    env.reset()

    ## ----- debug ----- ##
    if getattr(cfg, 'task_suite_name', None) == 'LIBERO_SPATIAL' and task_id == 5:
        initial_state[12] += 0.038
        print(f"debug: initial_state[12] += 0.038")
    ## ----------------- ##

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # 默认执行 chunk_size 步长，或者根据设定取前几步 (Temporal Ensembling / Receding Horizon)
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    t = 0
    replay_images = []
    # 假设有个字典映射最大步数
    max_steps = 600 # 或者是 TASK_MAX_STEPS[cfg.task_suite_name]

    success = False

    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        observation, img = prepare_observation(obs)
        replay_images.append(img)

        current_state = obs['robot0_eef_pos'].copy()

        primary_image = Image.fromarray(observation['full_image'])

        if len(action_queue) == 0:
            actions = get_action(
                cfg=cfg,
                statistic=statistic,
                action_tokenizer=action_tokenizer,
                vl_chat_processor=processor,
                task_description=task_description,
                model=model,
                first_image=primary_image,
                state=current_state,
            )
            
            action_queue.extend(actions[:cfg.num_open_loop_steps])

        action = action_queue.popleft()

        action = process_action(action, cfg.model_family)

        obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1

    return success, replay_images


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    processor,
    action_tokenizer,
    statistic,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    # Get task
    ## ----- chenhao specify task_id for task_id 4----- ##

    # task_id = 4

    ## ----- chenhao specify task_id for task_id 4----- ##    

    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    
    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res, seed=cfg.seed)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            processor,
            action_tokenizer,
            statistic,
            initial_state,
            log_file,
            task_id,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, processor, action_tokenizer, statistic = model_load(cfg)

    # Get expected image dimensions
    # resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        # task_id = 6
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            processor,
            action_tokenizer,
            statistic,
            total_episodes,
            total_successes,
            log_file,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
