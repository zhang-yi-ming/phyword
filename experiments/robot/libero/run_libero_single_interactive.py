#!/usr/bin/env python3
"""Interactive single-sample LIBERO replay and branch runner.

This script intentionally lives next to the existing eval runner but does not
modify it. It keeps the model loaded once, then accepts commands such as:

  run task=5 episode=12
  branch task=5 episode=12 query=4 branches=10
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shlex
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

import imageio
import numpy as np
import torch
import torchvision.transforms as transforms
from libero.libero import benchmark
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    save_rollout_video,
)
from experiments.robot.libero.run_libero_eval_new import (  # noqa: E402
    DEFAULT_HISTORY_TRAJECTORY_CAMERA_CONFIG,
    JANUS_ACTION_PROMPT_SUFFIX,
    GenerateConfig,
    TASK_MAX_STEPS,
    build_eval_state_inputs,
    coerce_bool,
    load_initial_states,
    log_message,
    model_load,
    prepare_observation,
    process_action,
    resolve_eval_prompt,
    resolve_pad_token_id,
    save_predicted_video,
    set_seed_everywhere,
    validate_config,
)
from experiments.robot.libero.history_trajectory_utils import (  # noqa: E402
    draw_history_trajectory_on_image,
    load_history_trajectory_camera_config,
)


HELP_TEXT = """Commands:
  help
  quit | exit
  run task=<id> episode=<idx> [seed=<seed>] [max_steps=<n>] [note=<text>]
  branch task=<id> episode=<idx> query=<1-based> branches=<n> [seed=<seed>] [anchor=<path>] [branch_seed_base=<seed>] [reset_branch_rng=true|false] [max_steps=<n>] [note=<text>]
  branch task=<id> episode=<idx> query=<1-based> query_seeds=<s0,s1,...> branch_seeds=<a:b|s0,s1,...> [seed=<seed>] [post_seed=0] [max_steps=<n>] [note=<text>]

Examples:
  run task=5 episode=12
  branch task=5 episode=12 query=4 branches=10
  branch task=5 episode=12 query=4 branches=10 anchor=/path/to/task=05--episode=012--query=004.pt
  branch task=9 episode=7 query=3 query_seeds=0,1 branch_seeds=0:9 post_seed=0

Defaults:
  seed=0
  branch_seed_base=0, so branches=10 uses branch seeds 0..9
  branch_seeds ranges are inclusive, so 0:9 means 0,1,...,9
  reset_branch_rng=true gives each branch a fresh branch_seed. With anchor and reset_branch_rng=false, branches must be 1 and the saved eval RNG stream is continued exactly.
"""


def str_to_bool(value: Any) -> bool:
    return coerce_bool(value)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, deque):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def snapshot_rng_state() -> dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def parse_kv_command(line: str) -> tuple[str, dict[str, str]]:
    parts = shlex.split(line)
    if not parts:
        return "", {}
    command = parts[0].strip().lower()
    options: dict[str, str] = {}
    for token in parts[1:]:
        if "=" not in token:
            raise ValueError(f"Expected key=value token, got {token!r}.")
        key, value = token.split("=", 1)
        options[key.strip().lower()] = value.strip()
    return command, options


def option_int(options: dict[str, str], *names: str, default: Optional[int] = None) -> int:
    for name in names:
        if name in options:
            return int(options[name])
    if default is None:
        raise ValueError(f"Missing required option: {'/'.join(names)}")
    return int(default)


def option_str(options: dict[str, str], *names: str, default: str = "") -> str:
    for name in names:
        if name in options:
            return options[name]
    return default


def option_bool(options: dict[str, str], *names: str, default: bool = False) -> bool:
    for name in names:
        if name in options:
            return str_to_bool(options[name])
    return bool(default)


def parse_seed_list(value: str, *, name: str) -> list[int]:
    value = str(value or "").strip()
    if not value:
        return []

    seeds: list[int] = []
    for raw_piece in value.split(","):
        piece = raw_piece.strip()
        if not piece:
            continue
        if ":" in piece:
            parts = piece.split(":")
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                raise ValueError(f"{name} has invalid inclusive range {piece!r}.")
            start = int(parts[0])
            end = int(parts[1])
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(piece))
    return seeds


def has_any_option(options: dict[str, str], *names: str) -> bool:
    return any(name in options for name in names)


def action_summary(action_log: list[dict[str, Any]]) -> dict[str, Any]:
    if not action_log:
        return {"count": 0}
    actions = np.asarray([entry["action"] for entry in action_log], dtype=np.float32)
    return {
        "count": int(actions.shape[0]),
        "first": actions[0].tolist(),
        "last": actions[-1].tolist(),
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
    }


def save_trace(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_safe(payload), f, indent=2, sort_keys=True)


def write_video(path: str, frames: list[np.ndarray], fps: int = 30) -> Optional[str]:
    if not frames:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()
    return path


def close_env(env) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def get_sim_state(env):
    sim = getattr(env, "sim", None)
    if sim is None or not hasattr(sim, "get_state"):
        raise RuntimeError("Environment does not expose env.sim.get_state(); cannot branch exactly.")
    return copy.deepcopy(sim.get_state())


def set_sim_state(env, state) -> None:
    sim = getattr(env, "sim", None)
    if sim is None or not hasattr(sim, "set_state"):
        raise RuntimeError("Environment does not expose env.sim.set_state(); cannot restore branch state.")
    sim.set_state(copy.deepcopy(state))
    forward = getattr(sim, "forward", None)
    if callable(forward):
        forward()


class SingleLiberoInteractiveRunner:
    def __init__(self, cfg: GenerateConfig, single_output_dir: str):
        self.cfg = cfg
        self.single_output_dir = single_output_dir
        validate_config(self.cfg)
        set_seed_everywhere(int(self.cfg.seed))
        self.model, self.processor, self.action_tokenizer, self.statistic = model_load(self.cfg)
        self.post_model_seed = int(self.cfg.seed)
        self.post_model_rng_state = snapshot_rng_state()
        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[self.cfg.task_suite_name]()
        self.trajectory_camera_config = None
        if bool(getattr(self.cfg, "use_history_trajectory_janus_image", False)):
            self.trajectory_camera_config = load_history_trajectory_camera_config(
                self.cfg.task_suite_name,
                self.cfg.history_trajectory_camera_config_path,
            )
        self.video_transform = transforms.Compose(
            [
                transforms.Resize(min(self.cfg.video_h, self.cfg.video_w), antialias=True),
                transforms.CenterCrop((self.cfg.video_h, self.cfg.video_w)),
            ]
        )
        print(
            "Interactive runner ready. "
            f"task_suite={self.cfg.task_suite_name}, tasks={self.task_suite.n_tasks}. "
            "Type 'help' for commands."
        )

    def reset_rng_for_seed(self, seed: int) -> None:
        seed = int(seed)
        if seed == self.post_model_seed:
            restore_rng_state(self.post_model_rng_state)
        else:
            set_seed_everywhere(seed)

    def command_cfg(self, artifact_dir: str) -> GenerateConfig:
        cfg = copy.deepcopy(self.cfg)
        cfg.eval_artifact_name = os.path.basename(artifact_dir)
        cfg.predicted_video_save_dir = os.path.join(artifact_dir, "predicted")
        cfg.rollout_video_save_dir = os.path.join(artifact_dir, "rollout")
        cfg.value_visualization_dir = os.path.join(artifact_dir, "value")
        return cfg

    def artifact_dir(
        self,
        command: str,
        task_id: int,
        episode_idx: int,
        seed: int,
        query: Optional[int] = None,
        branch_id: Optional[int] = None,
        branch_seed: Optional[int] = None,
        note: str = "",
    ) -> str:
        timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
        pieces = [
            timestamp,
            f"cmd={command}",
            f"task={task_id:02d}",
            f"episode={episode_idx:03d}",
            f"seed={seed}",
        ]
        if query is not None:
            pieces.append(f"query={query:03d}")
        if branch_id is not None:
            pieces.append(f"branch={branch_id:03d}")
        if branch_seed is not None:
            pieces.append(f"branch_seed={branch_seed}")
        if note:
            safe_note = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in note)[:40]
            pieces.append(f"note={safe_note}")
        return os.path.join(self.single_output_dir, "--".join(pieces))

    def write_config_snapshot(self, artifact_dir: str, extra: dict[str, Any]) -> None:
        payload = {
            "config": json_safe(asdict(self.cfg)),
            "dynamic_config": json_safe(vars(self.cfg)),
            "extra": json_safe(extra),
        }
        save_trace(os.path.join(artifact_dir, "config.json"), payload)

    def get_task_env(self, task_id: int, seed: int):
        if task_id < 0 or task_id >= self.task_suite.n_tasks:
            raise ValueError(f"task must be in [0, {self.task_suite.n_tasks - 1}], got {task_id}.")
        task = self.task_suite.get_task(task_id)
        return get_libero_env(
            task,
            self.cfg.model_family,
            resolution=self.cfg.env_img_res,
            seed=seed,
            control_freq=self.cfg.control_freq,
        )

    def get_initial_state(self, task_id: int, episode_idx: int, task_description: str, log_file=None) -> np.ndarray:
        initial_states, all_initial_states = load_initial_states(self.cfg, self.task_suite, task_id, log_file)
        if self.cfg.initial_states_path == "DEFAULT":
            if episode_idx < 0 or episode_idx >= len(initial_states):
                raise ValueError(
                    f"episode must be in [0, {len(initial_states) - 1}] for task {task_id}, got {episode_idx}."
                )
            initial_state = np.array(initial_states[episode_idx], copy=True)
        else:
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"
            if initial_states_task_key not in all_initial_states:
                raise KeyError(f"Task key not found in initial states JSON: {initial_states_task_key}")
            if episode_key not in all_initial_states[initial_states_task_key]:
                raise KeyError(f"Episode key not found in initial states JSON: {episode_key}")
            entry = all_initial_states[initial_states_task_key][episode_key]
            if not entry["success"]:
                raise ValueError(f"Initial state entry {initial_states_task_key}/{episode_key} has success=false.")
            initial_state = np.array(entry["initial_state"], copy=True)

        if getattr(self.cfg, "task_suite_name", None) == "libero_spatial" and task_id == 5:
            initial_state[12] += 0.038
            log_message("debug: initial_state[12] += 0.038", log_file)
        return initial_state

    def reset_env_to_episode(self, env, initial_state: np.ndarray):
        env.reset()
        return env.set_init_state(initial_state)

    def replay_env_to_capture(self, env, initial_state: np.ndarray, capture: dict[str, Any], log_file=None):
        """Restore branch state by replaying actions, not by only setting MuJoCo state."""
        obs = self.reset_env_to_episode(env, initial_state)
        target_t = int(capture["t"])
        t = 0

        while t < int(getattr(self.cfg, "num_steps_wait", 0) or 0) and t < target_t:
            obs, reward, done, info = env.step(get_libero_dummy_action(self.cfg.model_family))
            if done:
                raise RuntimeError(f"Replay reached done during initial wait at step {t}.")
            t += 1

        for entry in capture.get("action_log", []):
            step = int(entry["step"])
            if step != t:
                raise RuntimeError(
                    "Cannot replay branch prefix exactly: "
                    f"next action_log step={step}, expected step={t}, target_t={target_t}."
                )
            if t >= target_t:
                break
            action = np.asarray(entry["action"], dtype=np.float64)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                raise RuntimeError(f"Replay reached done before branch start at step {t}.")
            t += 1

        if t != target_t:
            raise RuntimeError(
                "Cannot replay branch prefix exactly: "
                f"replayed to step {t}, but branch starts at step {target_t}. "
                f"action_log has {len(capture.get('action_log', []))} entries."
            )

        replay_observation, _ = prepare_observation(obs)
        capture_observation, _ = prepare_observation(capture["obs"])
        state_diff = float(
            np.max(np.abs(replay_observation["state"] - capture_observation["state"]))
        )
        if state_diff > 1e-6:
            raise RuntimeError(
                "Action replay did not reproduce captured robot state exactly: "
                f"max_abs_state_diff={state_diff:.8g} at step {target_t}."
            )
        for image_key in ("full_image", "wrist_image"):
            if not np.array_equal(replay_observation[image_key], capture_observation[image_key]):
                raise RuntimeError(
                    "Action replay did not reproduce captured observation image exactly: "
                    f"{image_key} differs at step {target_t}."
                )

        log_message(
            f"Replayed {len(capture.get('action_log', []))} logged actions to branch_start_step={target_t}.",
            log_file,
        )
        return obs

    def load_resume_anchor(
        self,
        path: str,
        task_id: int,
        episode_idx: int,
        query_1based: int,
    ) -> dict[str, Any]:
        if not path:
            raise ValueError("anchor path is empty.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Anchor file does not exist: {path}")
        anchor = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(anchor, dict):
            raise ValueError(f"Anchor file must contain a dict payload: {path}")

        expected = {
            "task_id": int(task_id),
            "episode_idx": int(episode_idx),
            "query_1based": int(query_1based),
        }
        for key, value in expected.items():
            if anchor.get(key) is not None and int(anchor[key]) != value:
                raise ValueError(
                    f"Anchor {key}={anchor[key]} does not match command {key}={value}: {path}"
                )
        if str(anchor.get("task_suite_name", "")) and str(anchor["task_suite_name"]) != str(self.cfg.task_suite_name):
            raise ValueError(
                f"Anchor task_suite_name={anchor['task_suite_name']!r} does not match "
                f"configured task_suite_name={self.cfg.task_suite_name!r}."
            )
        required = ["obs", "sim_state", "obs_history", "rng_state", "t", "inference_idx"]
        missing = [key for key in required if key not in anchor]
        if missing:
            raise ValueError(f"Anchor is missing required keys {missing}: {path}")
        return anchor

    def capture_from_anchor(
        self,
        anchor: dict[str, Any],
        env,
        task_description: str,
    ) -> dict[str, Any]:
        anchor_description = str(anchor.get("task_description", ""))
        if anchor_description and anchor_description != task_description:
            raise ValueError(
                "Anchor task description does not match current environment: "
                f"{anchor_description!r} != {task_description!r}."
            )
        set_sim_state(env, anchor["sim_state"])
        restore_rng_state(anchor["rng_state"])

        num_cond_input_frames = max(1, int(getattr(self.cfg, "num_cond_input_frames", 1) or 1))
        obs_history_raw = anchor["obs_history"]
        obs_history = deque(obs_history_raw, maxlen=num_cond_input_frames)
        ee_history = [
            np.asarray(point, dtype=np.float64).copy()
            for point in anchor.get("ee_history", [])
        ]
        ee_history_includes_current = bool(
            anchor.get("ee_history_includes_current", bool(ee_history))
        )
        return {
            "stopped": True,
            "success": False,
            "done": False,
            "obs": copy.deepcopy(anchor["obs"]),
            "env_state": copy.deepcopy(anchor["sim_state"]),
            "obs_history": obs_history,
            "obs_history_includes_current": bool(anchor.get("obs_history_includes_current", True)),
            "ee_history": ee_history,
            "ee_history_includes_current": ee_history_includes_current,
            "t": int(anchor["t"]),
            "inference_idx": int(anchor["inference_idx"]),
            "replay_images": [],
            "replay_value_scores": [],
            "value_trace": [],
            "action_log": [],
            "anchor_path": anchor.get("path", ""),
        }

    def append_history_frame(self, obs_history: deque, primary_image: Image.Image) -> None:
        primary_frame_np = np.array(primary_image)
        primary_frame_tensor = torch.from_numpy(primary_frame_np).permute(2, 0, 1).float() / 255.0
        primary_frame_tensor = self.video_transform(primary_frame_tensor)
        if len(obs_history) == 0:
            for _ in range(obs_history.maxlen or 1):
                obs_history.append(primary_frame_tensor.clone())
        else:
            obs_history.append(primary_frame_tensor)

    def cosmos_text_embeddings(self, cfg: GenerateConfig, eval_prompt: str, device, dtype):
        if not getattr(cfg, "cosmos_text_cache_path", ""):
            return None
        cosmos_text_cache = getattr(self.model, "cosmos_text_cache", None)
        cosmos_text_embedder = getattr(self.model, "cosmos_qwen_text_embedder", None)
        if cosmos_text_cache is None or cosmos_text_embedder is None:
            raise RuntimeError("cosmos_text_cache_path is set, but model lacks Cosmos text cache runtime.")

        def compute_cosmos_text(prompt: str) -> torch.Tensor:
            print(f"Cosmos text cache miss; computing native Qwen embedding for: {prompt!r}")
            return cosmos_text_embedder.compute_one(prompt)

        return cosmos_text_cache.get_or_compute(eval_prompt, compute_cosmos_text).unsqueeze(0).to(device).to(dtype)

    def infer_once(
        self,
        cfg: GenerateConfig,
        obs,
        obs_history: deque,
        task_description: str,
        task_id: int,
        episode_idx: int,
        inference_idx: int,
        step_idx: int,
        ee_history: Optional[list[np.ndarray]] = None,
        log_file=None,
    ) -> dict[str, Any]:
        device = next(self.model.parameters()).device
        dtype = torch.bfloat16
        eval_prompt = resolve_eval_prompt(cfg, task_description)
        observation, img = prepare_observation(obs)
        current_state = observation["state"].copy()
        primary_image = Image.fromarray(observation["full_image"])
        wrist_image = Image.fromarray(observation["wrist_image"])

        state_tokens_str, state_placeholder_ids, now_state = build_eval_state_inputs(
            cfg=cfg,
            processor=self.processor,
            action_tokenizer=self.action_tokenizer,
            statistic=self.statistic,
            current_state=current_state,
        )
        if now_state is not None:
            now_state = now_state.unsqueeze(0).to(device)

        user_content = (
            f"<image_placeholder>\n{eval_prompt}\n"
            f"{JANUS_ACTION_PROMPT_SUFFIX}{state_tokens_str}"
        )
        user_prompt = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=[{"role": "<|User|>", "content": user_content}],
            sft_format=self.processor.sft_format,
            system_prompt="",
        )
        prompt_text = user_prompt + "\n\n<|Assistant|>:"
        janus_primary_image = primary_image
        if self.trajectory_camera_config is not None:
            janus_primary_image = draw_history_trajectory_on_image(
                primary_image.copy(),
                [] if ee_history is None else ee_history,
                self.trajectory_camera_config,
            )
        janus_inputs = self.processor(prompt=prompt_text, images=[janus_primary_image], return_tensors="pt")

        janus_input_ids = janus_inputs.input_ids.to(device)
        janus_pixel_values = janus_inputs.pixel_values.to(device).to(dtype)
        janus_images_seq_mask = janus_inputs.images_seq_mask.to(device)
        janus_state_seq_mask = torch.zeros_like(janus_input_ids, dtype=torch.bool, device=device)
        if state_placeholder_ids is not None:
            from models.cosmos_janus_cot import build_token_sequence_mask

            janus_state_seq_mask = build_token_sequence_mask(
                janus_input_ids,
                state_placeholder_ids,
                require_match=bool(int(getattr(cfg, "robot_state", 0) or 0)),
                name="current state placeholder",
            ).to(device)
        janus_images_emb_mask = janus_inputs.images_emb_mask.to(device)
        if getattr(janus_inputs, "attention_mask", None) is None:
            janus_attention_mask = torch.ones_like(janus_input_ids, dtype=torch.bool, device=device)
        else:
            janus_attention_mask = janus_inputs.attention_mask.to(device).to(torch.bool)
        pad_token_id = resolve_pad_token_id(self.processor)
        janus_left_pad_lens = janus_input_ids.eq(pad_token_id).to(torch.long).cumprod(dim=1).sum(dim=1)
        janus_action_pixel_values = self.processor.image_processor(
            [wrist_image],
            return_tensors="pt",
        )["pixel_values"].unsqueeze(0).to(device).to(dtype)

        first_frame_tensor = torch.stack(list(obs_history), dim=1).unsqueeze(0).to(device).to(dtype)
        cosmos_text_embeddings = self.cosmos_text_embeddings(cfg, eval_prompt, device, dtype)
        with torch.inference_mode():
            fps_tensor = torch.tensor([cfg.fps], device=device, dtype=dtype)
            inference_outputs = self.model.forward_flow_joint_inference(
                janus_input_ids=janus_input_ids,
                janus_pixel_values=janus_pixel_values,
                janus_images_seq_mask=janus_images_seq_mask,
                janus_images_emb_mask=janus_images_emb_mask,
                first_frame=first_frame_tensor,
                action_denoise_steps=cfg.action_denoise_steps,
                cosmos_denoise_steps=cfg.cosmos_denoise_steps,
                fps=fps_tensor,
                action_self_causal_in_bridge=cfg.action_self_causal_in_bridge,
                num_latent_tokens=cfg.total_latent_tokens,
                janus_left_pad_lens=janus_left_pad_lens,
                janus_state_seq_mask=janus_state_seq_mask,
                janus_action_pixel_values=janus_action_pixel_values,
                janus_attention_mask=janus_attention_mask,
                now_state=now_state,
                cosmos_text_embeddings=cosmos_text_embeddings,
                return_value_prediction=cfg.use_value_prediction,
                return_action_value_prediction=cfg.use_action_value_prediction,
            )

        predicted_value = None
        predicted_action_value = None
        if cfg.use_action_value_prediction:
            pred_video, pred_action, predicted_value, predicted_action_value = inference_outputs
        elif cfg.use_value_prediction:
            pred_video, pred_action, predicted_value = inference_outputs
        else:
            pred_video, pred_action = inference_outputs

        cosmos_value_score = None
        action_value_score = None
        if predicted_value is not None:
            cosmos_value_score = float(np.mean(predicted_value.detach().cpu().float().view(-1).tolist()))
        if predicted_action_value is not None:
            action_value_score = float(np.mean(predicted_action_value.detach().cpu().float().view(-1).tolist()))

        pred_video_path = save_predicted_video(
            pred_video=pred_video,
            cfg=cfg,
            task_description=task_description,
            task_id=task_id,
            episode_idx=episode_idx,
            inference_idx=inference_idx,
            step_idx=step_idx,
            cosmos_value_score=cosmos_value_score,
            action_value_score=action_value_score,
            log_file=log_file,
        )

        normalized_actions = pred_action.squeeze(0).cpu().float().numpy()
        if normalized_actions.shape[1] in [7, 14]:
            normalized_actions[:, 6] = 1 - (normalized_actions[:, 6] >= 0.5).astype(int)
        action_pred = np.where(
            self.statistic["action_mask"],
            0.5
            * (normalized_actions + 1.0)
            * (self.statistic["action_q99"] - self.statistic["action_q01"])
            + self.statistic["action_q01"],
            normalized_actions,
        )

        value_scores = {}
        if cosmos_value_score is not None:
            value_scores["cosmos value"] = cosmos_value_score
        if action_value_score is not None:
            value_scores["action value"] = action_value_score

        return {
            "actions": action_pred,
            "image": img,
            "value_scores": value_scores or None,
            "predicted_video_path": pred_video_path,
            "cosmos_value_score": cosmos_value_score,
            "action_value_score": action_value_score,
        }

    def rollout_loop(
        self,
        cfg: GenerateConfig,
        env,
        task_description: str,
        obs,
        task_id: int,
        episode_idx: int,
        stop_before_query_idx: Optional[int] = None,
        start_t: int = 0,
        start_inference_idx: int = 0,
        start_obs_history: Optional[deque] = None,
        start_ee_history: Optional[list[np.ndarray]] = None,
        replay_images: Optional[list[np.ndarray]] = None,
        replay_value_scores: Optional[list[Optional[dict[str, float]]]] = None,
        value_trace: Optional[list[dict[str, Any]]] = None,
        action_log: Optional[list[dict[str, Any]]] = None,
        query_seed_trace: Optional[list[dict[str, Any]]] = None,
        query_seed_fn: Optional[Callable[[int], Optional[int]]] = None,
        max_steps_override: Optional[int] = None,
        start_obs_history_includes_current: bool = False,
        start_ee_history_includes_current: bool = False,
        log_file=None,
    ) -> dict[str, Any]:
        action_queue = deque(maxlen=cfg.num_open_loop_steps * cfg.action_repeat)
        t = int(start_t)
        inference_idx = int(start_inference_idx)
        replay_images = [] if replay_images is None else list(replay_images)
        replay_value_scores = [] if replay_value_scores is None else list(replay_value_scores)
        value_trace = [] if value_trace is None else list(value_trace)
        action_log = [] if action_log is None else list(action_log)
        query_seed_trace = [] if query_seed_trace is None else list(query_seed_trace)
        active_value_scores = None
        max_steps = TASK_MAX_STEPS.get(cfg.task_suite_name, 600)
        if max_steps_override is not None:
            max_steps = int(max_steps_override)
        num_cond_input_frames = max(1, int(getattr(cfg, "num_cond_input_frames", 1) or 1))
        obs_history = (
            deque(start_obs_history, maxlen=num_cond_input_frames)
            if start_obs_history is not None
            else deque(maxlen=num_cond_input_frames)
        )
        history_includes_current = bool(start_obs_history_includes_current)
        ee_history = (
            [np.asarray(point, dtype=np.float64).copy() for point in start_ee_history]
            if start_ee_history is not None
            else []
        )
        ee_history_includes_current = bool(start_ee_history_includes_current)

        success = False
        done = False
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, img = prepare_observation(obs)

            if len(action_queue) == 0:
                if stop_before_query_idx is not None and inference_idx == stop_before_query_idx:
                    return {
                        "stopped": True,
                        "success": False,
                        "done": False,
                        "obs": copy.deepcopy(obs),
                        "env_state": get_sim_state(env),
                        "obs_history": deque(obs_history, maxlen=obs_history.maxlen),
                        "ee_history": [point.copy() for point in ee_history],
                        "ee_history_includes_current": ee_history_includes_current,
                        "t": t,
                        "inference_idx": inference_idx,
                        "replay_images": replay_images,
                        "replay_value_scores": replay_value_scores,
                        "value_trace": value_trace,
                        "action_log": action_log,
                        "query_seed_trace": query_seed_trace,
                    }

            if history_includes_current:
                history_includes_current = False
            else:
                primary_image = Image.fromarray(observation["full_image"])
                self.append_history_frame(obs_history, primary_image)

            current_state = observation["state"].copy()
            if ee_history_includes_current:
                ee_history_includes_current = False
            else:
                ee_history.append(np.asarray(current_state[:3], dtype=np.float64).copy())

            if len(action_queue) == 0:
                query_seed = None
                if query_seed_fn is not None:
                    query_seed = query_seed_fn(inference_idx)
                    if query_seed is not None:
                        self.reset_rng_for_seed(int(query_seed))
                if query_seed is not None:
                    query_seed_trace.append(
                        {
                            "query_idx": int(inference_idx),
                            "query_1based": int(inference_idx + 1),
                            "step": int(t),
                            "seed": int(query_seed),
                        }
                    )
                result = self.infer_once(
                    cfg=cfg,
                    obs=obs,
                    obs_history=obs_history,
                    task_description=task_description,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    inference_idx=inference_idx,
                    step_idx=t,
                    ee_history=ee_history,
                    log_file=log_file,
                )
                active_value_scores = result["value_scores"]
                if active_value_scores is not None:
                    value_entry = {
                        "task_id": int(task_id),
                        "episode_idx": int(episode_idx),
                        "query_idx": int(inference_idx),
                        "query_1based": int(inference_idx + 1),
                        "step": int(t),
                    }
                    value_entry.update(active_value_scores)
                    value_trace.append(value_entry)
                inference_idx += 1
                for action in result["actions"][: cfg.num_open_loop_steps]:
                    action_queue.extend([action] * cfg.action_repeat)

            replay_images.append(img)
            if cfg.use_value_prediction or cfg.use_action_value_prediction:
                replay_value_scores.append(None if active_value_scores is None else dict(active_value_scores))

            action = action_queue.popleft()
            processed_action = process_action(action, cfg.model_family)
            action_log.append({"step": int(t), "action": np.asarray(processed_action).tolist()})
            obs, reward, done, info = env.step(processed_action.tolist())
            if done:
                success = True
                break
            t += 1

        return {
            "stopped": False,
            "success": bool(success),
            "done": bool(done),
            "obs": copy.deepcopy(obs),
            "env_state": get_sim_state(env),
            "obs_history": deque(obs_history, maxlen=obs_history.maxlen),
            "ee_history": [point.copy() for point in ee_history],
            "ee_history_includes_current": True,
            "t": t,
            "inference_idx": inference_idx,
            "replay_images": replay_images,
            "replay_value_scores": replay_value_scores,
            "value_trace": value_trace,
            "action_log": action_log,
            "query_seed_trace": query_seed_trace,
        }

    def save_rollout_outputs(
        self,
        cfg: GenerateConfig,
        artifact_dir: str,
        task_description: str,
        result: dict[str, Any],
        trace_extra: dict[str, Any],
    ) -> None:
        rollout_path = save_rollout_video(
            result["replay_images"],
            idx=1,
            success=result["success"],
            task_description=task_description,
            cosmos_denoise_steps=cfg.cosmos_denoise_steps,
            rollout_dir=cfg.rollout_video_save_dir,
            value_scores=(
                result["replay_value_scores"]
                if (cfg.use_value_prediction or cfg.use_action_value_prediction)
                else None
            ),
        )
        payload = {
            **trace_extra,
            "success": bool(result["success"]),
            "done": bool(result["done"]),
            "final_step": int(result["t"]),
            "num_queries": int(result["inference_idx"]),
            "rollout_path": rollout_path,
            "value_trace": result["value_trace"],
            "query_seed_trace": result.get("query_seed_trace", []),
            "action_summary": action_summary(result["action_log"]),
            "action_log": result["action_log"],
        }
        save_trace(os.path.join(artifact_dir, "trace.json"), payload)

    def run_single(self, options: dict[str, str]) -> None:
        task_id = option_int(options, "task", "task_id")
        episode_idx = option_int(options, "episode", "episode_idx")
        seed = option_int(options, "seed", default=0)
        max_steps = option_int(options, "max_steps", default=-1)
        note = option_str(options, "note", default="")
        self.reset_rng_for_seed(seed)
        artifact_dir = self.artifact_dir("run", task_id, episode_idx, seed, note=note)
        cfg = self.command_cfg(artifact_dir)
        os.makedirs(artifact_dir, exist_ok=True)
        self.write_config_snapshot(
            artifact_dir,
            {"command": "run", "task_id": task_id, "episode_idx": episode_idx, "seed": seed},
        )
        log_path = os.path.join(artifact_dir, "run.log")
        with open(log_path, "w") as log_file:
            env, task_description = self.get_task_env(task_id, seed)
            try:
                initial_state = self.get_initial_state(task_id, episode_idx, task_description, log_file)
                obs = self.reset_env_to_episode(env, initial_state)
                result = self.rollout_loop(
                    cfg,
                    env,
                    task_description,
                    obs,
                    task_id,
                    episode_idx,
                    max_steps_override=None if max_steps < 0 else max_steps,
                    log_file=log_file,
                )
                self.save_rollout_outputs(
                    cfg,
                    artifact_dir,
                    task_description,
                    result,
                    {
                        "command": "run",
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "seed": seed,
                        "task_description": task_description,
                    },
                )
                print(f"[run] success={result['success']} artifact={artifact_dir}")
            finally:
                close_env(env)

    def branch_single(self, options: dict[str, str]) -> None:
        task_id = option_int(options, "task", "task_id")
        episode_idx = option_int(options, "episode", "episode_idx")
        query_1based = option_int(options, "query")
        seed_schedule_mode = has_any_option(options, "query_seeds", "branch_seeds", "post_seed")
        branches = option_int(options, "branches", default=0 if seed_schedule_mode else None)
        seed = option_int(options, "seed", default=0)
        branch_seed_base = option_int(options, "branch_seed_base", default=0)
        reset_branch_rng = option_bool(options, "reset_branch_rng", default=True)
        max_steps = option_int(options, "max_steps", default=-1)
        anchor_path = option_str(options, "anchor", "anchor_path", default="")
        note = option_str(options, "note", default="")
        if query_1based < 1:
            raise ValueError("query must be >= 1 and is 1-based.")
        if seed_schedule_mode and anchor_path:
            raise ValueError("anchor mode is not supported together with query_seeds/branch_seeds/post_seed.")
        if not seed_schedule_mode and branches < 1:
            raise ValueError("branches must be positive.")
        if not seed_schedule_mode and not reset_branch_rng and branches != 1:
            raise ValueError("reset_branch_rng=false is only valid with branches=1.")

        query_seeds: list[int] = []
        branch_seeds: list[int] = []
        post_seed: Optional[int] = None
        if seed_schedule_mode:
            query_seeds = parse_seed_list(option_str(options, "query_seeds", default=""), name="query_seeds")
            if len(query_seeds) != query_1based - 1:
                raise ValueError(
                    f"query_seeds must contain exactly query-1 seeds; "
                    f"got {len(query_seeds)} for query={query_1based}."
                )
            branch_seeds_text = option_str(options, "branch_seeds", default="")
            if branch_seeds_text:
                branch_seeds = parse_seed_list(branch_seeds_text, name="branch_seeds")
            else:
                if branches < 1:
                    raise ValueError("branch_seeds is required in query seed schedule mode when branches is omitted.")
                branch_seeds = [branch_seed_base + branch_id for branch_id in range(branches)]
            if not branch_seeds:
                raise ValueError("branch_seeds must not be empty in query seed schedule mode.")
            if "branches" in options and int(branches) != len(branch_seeds):
                raise ValueError(
                    f"branches={branches} does not match number of branch_seeds={len(branch_seeds)}."
                )
            branches = len(branch_seeds)
            post_seed = option_int(options, "post_seed", default=0)

            def pre_branch_query_seed_fn(query_idx: int) -> Optional[int]:
                if query_idx < len(query_seeds):
                    return query_seeds[query_idx]
                return None
        else:
            pre_branch_query_seed_fn = None

        if not anchor_path:
            self.reset_rng_for_seed(seed)
        root_artifact = self.artifact_dir(
            "branch",
            task_id,
            episode_idx,
            seed,
            query=query_1based,
            note=note,
        )
        os.makedirs(root_artifact, exist_ok=True)
        cfg = self.command_cfg(root_artifact)
        self.write_config_snapshot(
            root_artifact,
            {
                "command": "branch",
                "task_id": task_id,
                "episode_idx": episode_idx,
                "seed": seed,
                "query_1based": query_1based,
                "branches": branches,
                "branch_seed_base": branch_seed_base,
                "reset_branch_rng": reset_branch_rng,
                "anchor_path": anchor_path,
                "seed_schedule_mode": seed_schedule_mode,
                "query_seeds": query_seeds,
                "branch_seeds": branch_seeds,
                "post_seed": post_seed,
            },
        )

        log_path = os.path.join(root_artifact, "branch.log")
        with open(log_path, "w") as log_file:
            env, task_description = self.get_task_env(task_id, seed)
            try:
                initial_state = self.get_initial_state(task_id, episode_idx, task_description, log_file)
                anchor = None
                if anchor_path:
                    anchor = self.load_resume_anchor(anchor_path, task_id, episode_idx, query_1based)
                    self.reset_env_to_episode(env, initial_state)
                    capture = self.capture_from_anchor(anchor, env, task_description)
                    capture["anchor_path"] = anchor_path
                    log_message(f"Loaded branch anchor: {anchor_path}", log_file)
                else:
                    obs = self.reset_env_to_episode(env, initial_state)
                    capture = self.rollout_loop(
                        cfg,
                        env,
                        task_description,
                        obs,
                        task_id,
                        episode_idx,
                        stop_before_query_idx=query_1based - 1,
                        max_steps_override=None if max_steps < 0 else max_steps,
                        query_seed_fn=pre_branch_query_seed_fn,
                        log_file=log_file,
                    )
                    if not capture["stopped"]:
                        raise RuntimeError(
                            f"Episode ended before reaching query={query_1based}; "
                            f"only reached next query index {capture['inference_idx'] + 1}."
                        )

                main_branch_rng_state = snapshot_rng_state()

                branch_summaries = []
                for branch_id in range(branches):
                    branch_seed = branch_seeds[branch_id] if seed_schedule_mode else branch_seed_base + branch_id
                    branch_artifact = self.artifact_dir(
                        "branch",
                        task_id,
                        episode_idx,
                        seed,
                        query=query_1based,
                        branch_id=branch_id,
                        branch_seed=branch_seed,
                        note=note,
                    )
                    branch_cfg = self.command_cfg(branch_artifact)
                    os.makedirs(branch_artifact, exist_ok=True)
                    branch_env = None
                    branch_owns_env = False
                    try:
                        # Replaying actions restores controller/wrapper state that is not
                        # represented in MuJoCo sim.get_state().
                        branch_env, branch_task_description = self.get_task_env(task_id, seed)
                        branch_owns_env = True
                        if branch_task_description != task_description:
                            raise RuntimeError(
                                "Branch env task description changed unexpectedly: "
                                f"{branch_task_description!r} != {task_description!r}"
                            )
                        branch_obs = self.replay_env_to_capture(
                            branch_env,
                            initial_state,
                            capture,
                            log_file=log_file,
                        )
                        if seed_schedule_mode:
                            self.reset_rng_for_seed(branch_seed)
                        elif reset_branch_rng:
                            self.reset_rng_for_seed(branch_seed)
                        else:
                            restore_rng_state(main_branch_rng_state)
                        if seed_schedule_mode:
                            def branch_query_seed_fn(
                                query_idx: int,
                                *,
                                first_idx: int = capture["inference_idx"],
                                first_seed: int = branch_seed,
                            ) -> Optional[int]:
                                if query_idx == first_idx:
                                    return first_seed
                                return post_seed
                        else:
                            branch_query_seed_fn = None
                        branch_result = self.rollout_loop(
                            branch_cfg,
                            branch_env,
                            task_description,
                            branch_obs,
                            task_id,
                            episode_idx,
                            start_t=capture["t"],
                            start_inference_idx=capture["inference_idx"],
                            start_obs_history=capture["obs_history"],
                            start_ee_history=capture.get("ee_history"),
                            replay_images=capture["replay_images"],
                            replay_value_scores=capture["replay_value_scores"],
                            value_trace=capture["value_trace"],
                            action_log=capture["action_log"],
                            query_seed_trace=capture.get("query_seed_trace"),
                            query_seed_fn=branch_query_seed_fn,
                            max_steps_override=None if max_steps < 0 else max_steps,
                            start_obs_history_includes_current=bool(
                                capture.get("obs_history_includes_current", False)
                            ),
                            start_ee_history_includes_current=bool(
                                capture.get("ee_history_includes_current", False)
                            ),
                            log_file=log_file,
                        )
                    finally:
                        if branch_owns_env and branch_env is not None:
                            close_env(branch_env)
                    trace_extra = {
                        "command": "branch",
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "seed": seed,
                        "branch_id": branch_id,
                        "branch_seed": branch_seed,
                        "reset_branch_rng": reset_branch_rng,
                        "query_1based": query_1based,
                        "query_idx": query_1based - 1,
                        "branch_start_step": int(capture["t"]),
                        "task_description": task_description,
                        "anchor_path": anchor_path,
                        "branch_restore_mode": "action_replay",
                        "seed_schedule_mode": seed_schedule_mode,
                        "query_seeds": query_seeds,
                        "branch_seeds": branch_seeds,
                        "post_seed": post_seed,
                    }
                    self.write_config_snapshot(branch_artifact, trace_extra)
                    self.save_rollout_outputs(
                        branch_cfg,
                        branch_artifact,
                        task_description,
                        branch_result,
                        trace_extra,
                    )
                    branch_summaries.append(
                        {
                            "branch_id": branch_id,
                            "branch_seed": branch_seed,
                            "query_seed_trace": branch_result.get("query_seed_trace", []),
                            "success": bool(branch_result["success"]),
                            "artifact": branch_artifact,
                            "final_step": int(branch_result["t"]),
                        }
                    )
                    print(
                        f"[branch {branch_id}] seed={branch_seed} "
                        f"success={branch_result['success']} artifact={branch_artifact}"
                    )

                save_trace(
                    os.path.join(root_artifact, "branch_summary.json"),
                    {
                        "command": "branch",
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "seed": seed,
                        "query_1based": query_1based,
                        "branch_start_step": int(capture["t"]),
                        "anchor_path": anchor_path,
                        "reset_branch_rng": reset_branch_rng,
                        "branches": branch_summaries,
                        "seed_schedule_mode": seed_schedule_mode,
                        "query_seeds": query_seeds,
                        "branch_seeds": branch_seeds,
                        "post_seed": post_seed,
                    },
                )
                print(f"[branch] wrote summary: {root_artifact}")
            finally:
                close_env(env)

    def execute_line(self, line: str) -> bool:
        command, options = parse_kv_command(line)
        if not command:
            return True
        if command in {"quit", "exit"}:
            return False
        if command == "help":
            print(HELP_TEXT)
            return True
        if command == "run":
            self.run_single(options)
            return True
        if command == "branch":
            self.branch_single(options)
            return True
        raise ValueError(f"Unknown command {command!r}. Type 'help' for usage.")

    def repl(self) -> None:
        print(HELP_TEXT)
        while True:
            try:
                line = input("libero-single> ").strip()
            except EOFError:
                print()
                break
            try:
                if not self.execute_line(line):
                    break
            except Exception as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--action_model_path", required=True)
    parser.add_argument("--cosmos_experiment_name", required=True)
    parser.add_argument("--cosmos_model_path", required=True)
    parser.add_argument("--cosmos_text_cache_path", default="")
    parser.add_argument("--rewrite_eval_prompt", type=str_to_bool, default=False)
    parser.add_argument("--use_history_trajectory_janus_image", type=str_to_bool, default=False)
    parser.add_argument(
        "--history_trajectory_camera_config_path",
        default=DEFAULT_HISTORY_TRAJECTORY_CAMERA_CONFIG,
    )
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--initial_states_path", default="DEFAULT")
    parser.add_argument("--env_img_res", type=int, default=256)
    parser.add_argument("--video_h", type=int, default=256)
    parser.add_argument("--video_w", type=int, default=256)
    parser.add_argument("--video_frames", type=int, default=17)
    parser.add_argument("--num_cond_input_frames", type=int, default=5)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--action_chunk", type=int, default=16)
    parser.add_argument("--robot_state", type=int, default=0)
    parser.add_argument("--state_placeholder_tokens", type=int, default=8)
    parser.add_argument("--state_encoding_mode", default="mlp")
    parser.add_argument("--action_intermediate_size", type=int, default=0)
    parser.add_argument("--total_latent_tokens", type=int, default=1)
    parser.add_argument("--img_latents_per_future", type=int, default=1)
    parser.add_argument("--state_latents_per_future", type=int, default=1)
    parser.add_argument("--num_future_frames", type=int, default=4)
    parser.add_argument("--future_frame_stride", type=int, default=8)
    parser.add_argument("--cosmos_self_only_bridge", type=str_to_bool, default=True)
    parser.add_argument("--decosmos", type=str_to_bool, default=False)
    parser.add_argument("--use_value_prediction", type=str_to_bool, default=False)
    parser.add_argument("--use_action_value_prediction", type=str_to_bool, default=False)
    parser.add_argument("--value_token_mask_video_to_value", type=str_to_bool, default=False)
    parser.add_argument("--value_token_mask_nonvalue_to_value", type=str_to_bool, default=False)
    parser.add_argument("--bridge_pos_scheme", default="mrope")
    parser.add_argument("--action_use_latent_prefix", type=str_to_bool, default=True)
    parser.add_argument("--control_freq", type=int, default=0)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--num_open_loop_steps", type=int, default=8)
    parser.add_argument("--action_repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--action_denoise_steps", type=int, default=10)
    parser.add_argument("--cosmos_denoise_steps", type=int, default=2)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--predicted_video_fps", type=int, default=10)
    parser.add_argument("--action_self_causal_in_bridge", type=str_to_bool, default=False)
    parser.add_argument("--single_output_dir", default="")
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--command", action="append", default=[])
    parser.add_argument("--repl_after_command", action="store_true")
    return parser


def cfg_from_args(args: argparse.Namespace) -> GenerateConfig:
    cfg = GenerateConfig()
    for key, value in vars(args).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.num_trials_per_task = 1
    return cfg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = cfg_from_args(args)
    validate_config(cfg)
    if args.print_config:
        print(json.dumps(json_safe(vars(cfg)), indent=2, sort_keys=True))
        return

    single_output_dir = args.single_output_dir
    if not single_output_dir:
        single_output_dir = os.path.join("../experiments/single_interactive", time.strftime("%Y_%m_%d-%H_%M_%S"))
    os.makedirs(single_output_dir, exist_ok=True)

    runner = SingleLiberoInteractiveRunner(cfg, single_output_dir=single_output_dir)
    for command in args.command:
        keep_running = runner.execute_line(command)
        if not keep_running:
            return
    if not args.command or args.repl_after_command:
        runner.repl()


if __name__ == "__main__":
    main()
