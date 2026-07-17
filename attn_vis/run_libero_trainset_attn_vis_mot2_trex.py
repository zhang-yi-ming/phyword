#!/usr/bin/env python3
"""Visualize T-Rex MoT2 Libero trainset attention over Cosmos video tokens."""

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
import types
from collections import OrderedDict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

from attn_vis.common import (  # noqa: E402
    PrintAccelerator,
    action_summary,
    coerce_bool,
    log_message,
    move_batch_to_device,
    resolve_spatial_token_args,
    set_seed,
    overlay_heatmap as _overlay_heatmap,
    sanitize_filename as _sanitize_filename,
    score_to_rgb as _score_to_rgb,
)
from attn_vis.run_rlbench_trainset_attn_vis_mot2_trex import model_load  # noqa: E402
from scripts.train_mot2_trex import VLACotDataset  # noqa: E402
from utils.model_config_manifest import load_model_manifest, validate_runtime_model_config  # noqa: E402


DEFAULT_SPECIAL_TOKEN_VOCAB = [
    "</MOVE>",
    "</BOWWL>",
    "</PICK>",
    "</PLACE>",
    "</APPROACH>",
    "</bowl>",
]


# label, latent start, latent end, representative-frame label, frame offset from t.
SLICE_SPECS = (
    ("slice0_t-4", 0, 1, "t-4", -4),
    ("slice1_t-3_to_t", 1, 2, "t-2", -2),
    ("slice2_t+1_to_t+4", 2, 3, "t+2", 2),
    ("slice3_t+5_to_t+8", 3, 4, "t+6", 6),
    ("slice4_t+9_to_t+12", 4, 5, "t+10", 10),
)


@dataclass
class LiberoTrainsetAttnVisConfig:
    pretrained_checkpoint: str = ""
    action_model_path: str = "/mnt/nas/zhangyiming/database/ckpt/pretrained/T-Rex_pretrain_mecka22k_epoch1"
    cosmos_model_path: str = (
        "/mnt/nas/zhangyiming/database/ckpt/pretrained/Cosmos-Predict2.5-2B/base/pre-trained/"
        "d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt"
    )
    cosmos_experiment_name: str = (
        "Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_"
        "high_sigma_loss_reweighted_1_1_rectified_flow_only"
    )
    cosmos_text_cache_path: str = ""
    data_path: str = (
        "/mnt/nas/zhangyiming/database/data/libero_training_data_last05_lastest/"
        "libero_spatial_20hz_224_dual/train_with_atomic_action_shared.json"
    )
    data_root: str = ""
    attention_visualization_dir: str = ""
    attention_visualization_tile_size: int = 160
    attention_visualization_alpha: float = 0.45
    attention_visualization_capture_mode: str = "last"
    attention_visualization_top_ratio: Optional[float] = None
    attention_visualization_top_softness: float = 0.05
    attention_visualization_gap: int = 8
    attention_visualization_header_height: int = 24
    bash_hparams_path: str = ""
    eval_artifact_name: str = ""
    task_names: str = ""
    num_trajectories_per_task: int = 1
    max_records_per_episode: int = 0
    max_total_records: int = 0
    cuda: str = "0"
    seed: int = 0
    video_h: int = 256
    video_w: int = 256
    video_frames: int = 17
    num_cond_input_frames: int = 5
    action_dim: int = 7
    action_chunk: int = 16
    robot_state: int = 0
    state_placeholder_tokens: int = 1
    state_dim: int = 8
    state_encoding_mode: str = "mlp"
    total_latent_tokens: str = ""
    latent_token_mode: str = ""
    special_token_vocab: str = ",".join(DEFAULT_SPECIAL_TOKEN_VOCAB)
    special_token_weight_tied: bool = True
    bridge_pos_scheme: str = "mrope"
    qwen3vl2b_model_path: str = "/mnt/amlfs-07/shared/physicalword/ckpt/pretraine/Qwen3-VL-2B-Instruct"
    right_single_attn_position: str = "first4"
    action_denoise_steps: int = 10
    cosmos_denoise_steps: int = 2
    fps: float = 20.0
    empty_cache_every: int = 10


def parse_args() -> LiberoTrainsetAttnVisConfig:
    parser = argparse.ArgumentParser()
    for field_def in fields(LiberoTrainsetAttnVisConfig):
        default = field_def.default
        arg_type = str if default is None or isinstance(default, bool) else type(default)
        parser.add_argument(f"--{field_def.name}", default=default, type=arg_type)
    ns = parser.parse_args()
    cfg = LiberoTrainsetAttnVisConfig(**vars(ns))
    top_ratio = cfg.attention_visualization_top_ratio
    if top_ratio is None or (isinstance(top_ratio, str) and top_ratio.strip() == ""):
        cfg.attention_visualization_top_ratio = None
    else:
        parsed = float(top_ratio)
        if not 0.0 <= parsed <= 1.0:
            raise ValueError(f"--attention_visualization_top_ratio must be in [0, 1], got {parsed}.")
        cfg.attention_visualization_top_ratio = parsed
    cfg.attention_visualization_top_softness = float(cfg.attention_visualization_top_softness)
    if not 0.0 <= cfg.attention_visualization_top_softness <= 1.0:
        raise ValueError("--attention_visualization_top_softness must be in [0, 1].")
    cfg.right_single_attn_position = str(getattr(cfg, "right_single_attn_position", "first4") or "first4").lower()
    if cfg.right_single_attn_position not in ("first4", "last4"):
        raise ValueError("right_single_attn_position must be 'first4' or 'last4'.")
    resolve_spatial_token_args(cfg)
    if isinstance(cfg.special_token_vocab, str):
        special_token_vocab = [
            token.strip()
            for token in str(cfg.special_token_vocab).split(",")
            if token.strip()
        ]
    else:
        special_token_vocab = list(cfg.special_token_vocab)
    if not special_token_vocab:
        raise ValueError("special_token_vocab must not be empty.")
    cfg.special_token_vocab = special_token_vocab
    cfg.special_token_to_id = {
        token: idx for idx, token in enumerate(special_token_vocab)
    }
    cfg.special_token_weight_tied = coerce_bool(cfg.special_token_weight_tied)
    if not cfg.special_token_weight_tied:
        raise ValueError("This model version requires special_token_weight_tied=true.")
    return cfg


class LiberoAttentionDataset(VLACotDataset):
    """Training-record dataset that decodes only real Cosmos history frames."""

    def _load_video(self, video_path, frame_idx):
        target_frames = int(self.config.num_cond_input_frames)
        start_frame = int(frame_idx) - (target_frames - 1)
        video_path = self._resolve_data_path(video_path)
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames <= 0:
            raise ValueError(f"Video has no frames: {video_path}")
        indices = np.arange(start_frame, start_frame + target_frames)
        indices = np.clip(indices, 0, total_frames - 1)
        frames = vr.get_batch(indices).asnumpy()
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        frames_tensor = self.video_transform(frames_tensor)
        return frames_tensor.permute(1, 0, 2, 3)


def build_dataset(cfg: LiberoTrainsetAttnVisConfig, processor, log_file=None) -> VLACotDataset:
    return LiberoAttentionDataset(cfg, processor, PrintAccelerator(log_file))


def parse_episode_index(video_path: str, fallback: int) -> int:
    match = re.search(r"episode[_=-]?(\d+)", str(video_path))
    if match:
        return int(match.group(1))
    return int(fallback)


def select_trainset_records(
    cfg: LiberoTrainsetAttnVisConfig,
    dataset: VLACotDataset,
    log_file=None,
) -> list[tuple[str, int, int]]:
    requested_tasks = [task.strip() for task in str(cfg.task_names or "").split(",") if task.strip()]
    requested_task_set = set(requested_tasks)
    grouped: OrderedDict[str, OrderedDict[str, list[int]]] = OrderedDict()
    for sample_index, sample in enumerate(dataset.data):
        task_name = str(sample.get("task_name") or sample.get("task") or sample.get("input_prompt") or "unknown_task")
        if requested_task_set and task_name not in requested_task_set:
            continue
        video_path = str(sample["video_path"])
        grouped.setdefault(task_name, OrderedDict()).setdefault(video_path, []).append(sample_index)
    if requested_task_set:
        missing = sorted(requested_task_set - set(grouped))
        if missing:
            raise ValueError(f"Requested task_names not found in Libero train JSON: {missing}")

    selected: list[tuple[str, int, int]] = []
    per_task = max(1, int(cfg.num_trajectories_per_task))
    for task_name in sorted(grouped):
        episodes = list(grouped[task_name].items())[:per_task]
        log_message(
            f"Selected task={task_name!r}: episodes={[parse_episode_index(path, idx) for idx, (path, _) in enumerate(episodes)]} "
            f"(num_trajectories_per_task={per_task})",
            log_file,
        )
        for episode_ordinal, (video_path, sample_indices) in enumerate(episodes):
            episode_index = parse_episode_index(video_path, episode_ordinal)
            ordered_indices = sorted(sample_indices, key=lambda idx: int(dataset.data[idx].get("frame_index", idx)))
            if int(cfg.max_records_per_episode) > 0:
                ordered_indices = ordered_indices[: int(cfg.max_records_per_episode)]
            for sample_index in ordered_indices:
                selected.append((task_name, int(episode_index), int(sample_index)))
                if int(cfg.max_total_records) > 0 and len(selected) >= int(cfg.max_total_records):
                    log_message(f"Reached max_total_records={cfg.max_total_records}; stopping selection.", log_file)
                    return selected
    return selected


def load_libero_frame(dataset: VLACotDataset, video_path: str, frame_idx: int) -> Image.Image:
    resolved = dataset._resolve_data_path(video_path)
    vr = VideoReader(resolved, ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise ValueError(f"Video has no frames: {resolved}")
    clipped = max(0, min(int(frame_idx), total - 1))
    frame_np = vr.get_batch([clipped]).asnumpy()[0]
    return Image.fromarray(frame_np.astype(np.uint8))


def load_cosmos_slice_base_images(
    dataset: VLACotDataset,
    sample: dict[str, Any],
    cfg: LiberoTrainsetAttnVisConfig,
) -> list[Image.Image]:
    frame_idx = int(sample.get("frame_index", 0))
    images = []
    for _, _, _, _, frame_offset in SLICE_SPECS:
        images.append(load_libero_frame(dataset, sample["video_path"], frame_idx + int(frame_offset)))
    return images


class LiberoAttentionMapRecorder:
    def __init__(
        self,
        cfg: LiberoTrainsetAttnVisConfig,
        *,
        right_layer_count: int,
        slow_layer_indices: list[int],
        action_layer_indices: list[int],
        log_file=None,
    ):
        self.cfg = cfg
        self.log_file = log_file
        self.output_dir = str(getattr(cfg, "attention_visualization_dir", "") or "").strip()
        self.action_chunk = int(getattr(cfg, "action_chunk", 16))
        self.spatial_token_count = int(getattr(cfg, "total_latent_tokens", 1) or 1)
        if self.spatial_token_count not in (1, 2):
            raise ValueError(f"Expected total_latent_tokens=1 or 2, got {self.spatial_token_count}.")
        if self.spatial_token_count == 1:
            self.slow_labels = ["context", str(getattr(cfg, "latent_token_mode", "v"))]
        else:
            self.slow_labels = ["context", "v", "n"]
        self.num_layers = int(right_layer_count)
        self.slow_layer_indices = tuple(int(idx) for idx in slow_layer_indices)
        self.action_layer_indices = tuple(int(idx) for idx in action_layer_indices)
        if not self.slow_layer_indices:
            raise ValueError("No paired prefix layers are available for Cosmos attention visualization.")
        if not self.action_layer_indices:
            raise ValueError(
                "No paired T-Rex action layers are available. "
                "Use right_single_attn_position='first4' for action-to-Cosmos attention visualization."
            )
        self._slow_layer_index_set = set(self.slow_layer_indices)
        self._action_layer_index_set = set(self.action_layer_indices)
        self.first_action_layer_idx = self.action_layer_indices[0]
        self.alpha = float(getattr(cfg, "attention_visualization_alpha", 0.45) or 0.45)
        self.tile_size = max(16, int(getattr(cfg, "attention_visualization_tile_size", 160) or 160))
        self.top_ratio = getattr(cfg, "attention_visualization_top_ratio", None)
        self.top_softness = float(getattr(cfg, "attention_visualization_top_softness", 0.05) or 0.0)
        self.gap = max(0, int(getattr(cfg, "attention_visualization_gap", 8) or 0))
        self.header_h = max(16, int(getattr(cfg, "attention_visualization_header_height", 24) or 24))
        self.capture_mode = str(getattr(cfg, "attention_visualization_capture_mode", "last") or "last").lower()
        if self.capture_mode not in {"first", "last"}:
            raise ValueError(
                "attention_visualization_capture_mode must be first/last; "
                f"got {self.capture_mode!r}."
            )
        self.active = False
        self.slice_base_images: list[Image.Image] = []
        self.metadata = {}
        self.action_denoise_step = -1
        self.capture_current_action_step = False
        self.slow_records: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
        self.action_records: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
        self._logged_incomplete_video_kv = False

    def start_query(
        self,
        *,
        slice_base_images: list[Image.Image],
        task_name: str,
        episode_index: int,
        record_index: int,
        sample_index: int,
        frame_index: int,
        prompt: str,
        action_denoise_steps: int,
    ) -> None:
        if not self.output_dir:
            self.active = False
            return
        if len(slice_base_images) != len(SLICE_SPECS):
            raise ValueError(f"Expected {len(SLICE_SPECS)} Cosmos slice base images, got {len(slice_base_images)}.")
        self.active = True
        self.slice_base_images = [img.copy().convert("RGB") for img in slice_base_images]
        self.metadata = {
            "task_name": str(task_name),
            "episode_index": int(episode_index),
            "record_index": int(record_index),
            "sample_index": int(sample_index),
            "frame_index": int(frame_index),
            "prompt": str(prompt),
            "action_denoise_steps": int(action_denoise_steps),
        }
        self.action_denoise_step = -1
        self.capture_current_action_step = False
        self.slow_records = {}
        self.action_records = {}
        self._logged_incomplete_video_kv = False

    def discard_query(self) -> None:
        self.active = False
        self.slice_base_images = []
        self.metadata = {}
        self.action_denoise_step = -1
        self.capture_current_action_step = False
        self.slow_records = {}
        self.action_records = {}

    def finish_query(self) -> list[str]:
        if not self.active:
            return []
        try:
            return self._save_query()
        finally:
            self.discard_query()

    def _target_action_step(self) -> int:
        if self.capture_mode == "first":
            return 0
        return max(0, int(self.metadata.get("action_denoise_steps", 1)) - 1)

    @staticmethod
    def _attention_scores(q_tokens: torch.Tensor, k_tokens: torch.Tensor, allowed_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if q_tokens.ndim != 3 or k_tokens.ndim != 3:
            raise ValueError(f"Expected q/k shapes [Q,H,D]/[K,H,D], got {tuple(q_tokens.shape)} and {tuple(k_tokens.shape)}.")
        scale = 1.0 / math.sqrt(float(q_tokens.shape[-1]))
        scores = torch.einsum("qhd,khd->qhk", q_tokens.to(torch.float32), k_tokens.to(torch.float32)) * scale
        if allowed_mask is not None:
            allowed = allowed_mask.to(device=scores.device, dtype=torch.bool)
            expected = (int(q_tokens.shape[0]), int(k_tokens.shape[0]))
            if tuple(allowed.shape) != expected:
                raise ValueError(f"Attention mask rows must have shape {expected}, got {tuple(allowed.shape)}.")
            scores = scores.masked_fill(~allowed[:, None, :], torch.finfo(scores.dtype).min)
        return scores

    @staticmethod
    def _resolve_video_grid(video_grid_thw: Optional[torch.Tensor], video_len: int) -> tuple[int, int, int]:
        if video_grid_thw is not None:
            grid = video_grid_thw.detach().to(device="cpu", dtype=torch.long)
            if grid.ndim == 2:
                grid = grid[0]
            if int(grid.numel()) == 3:
                t, h, w = (int(grid[0].item()), int(grid[1].item()), int(grid[2].item()))
                if t > 0 and h > 0 and w > 0 and t * h * w == int(video_len):
                    return t, h, w
        t = 5
        spatial = int(video_len) // t if int(video_len) % t == 0 else int(video_len)
        side = int(round(math.sqrt(spatial)))
        if t * side * side == int(video_len):
            return t, side, side
        raise ValueError(f"Could not resolve 5-slice Cosmos video grid for {video_len} tokens.")

    def _slice_records(
        self,
        *,
        scores: torch.Tensor,
        full_attn: torch.Tensor,
        video_grid_thw: Optional[torch.Tensor],
        video_len: int,
    ) -> dict[str, dict[str, Any]]:
        t, h, w = self._resolve_video_grid(video_grid_thw, video_len)
        if t != len(SLICE_SPECS) or h != 16 or w != 16:
            raise ValueError(f"Expected Cosmos video grid 5x16x16, got {t}x{h}x{w} for video_len={video_len}.")
        records = {}
        for slice_label, start_t, end_t, base_label, _ in SLICE_SPECS:
            start = int(start_t) * h * w
            end = int(end_t) * h * w
            region_attn = full_attn[:, start:end]
            # Keep the scalar slice mass globally comparable, but normalize
            # colors within each slice so its spatial pattern remains visible.
            score = float(region_attn.sum(dim=-1).mean().detach().cpu().item())
            region_scores = scores[:, :, start:end]
            local = torch.softmax(region_scores, dim=-1).mean(dim=(0, 1)).detach().to(torch.float32).cpu()
            heat = local.reshape(end_t - start_t, h, w).sum(dim=0).numpy().astype(np.float32)
            records[slice_label] = {"heat": heat, "score": score, "base_label": base_label}
        return records

    def _capture_tokens(
        self,
        *,
        target: dict[str, dict[int, dict[str, dict[str, Any]]]],
        labels: list[str],
        layer_idx: int,
        q_tokens: torch.Tensor,
        k_video: torch.Tensor,
        k_action: torch.Tensor,
        allowed_mask: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
    ) -> None:
        if not self.active:
            return
        if q_tokens.shape[0] != 1 or k_video.shape[0] != 1 or k_action.shape[0] != 1:
            raise ValueError(
                "Attention visualization expects batch size 1, "
                f"got q_tokens={tuple(q_tokens.shape)}, k_video={tuple(k_video.shape)}, k_action={tuple(k_action.shape)}."
            )
        if int(q_tokens.shape[1]) != len(labels):
            raise ValueError(f"Token label count {len(labels)} does not match q token count {int(q_tokens.shape[1])}.")
        video_len = int(k_video.shape[1])
        expected_video_len = len(SLICE_SPECS) * 16 * 16
        full_k = torch.cat([k_video, k_action], dim=1)
        scores = self._attention_scores(q_tokens[0], full_k[0], allowed_mask)
        full_attn = torch.softmax(scores, dim=-1).mean(dim=1)
        full_attn = torch.nan_to_num(full_attn, nan=0.0, posinf=0.0, neginf=0.0)
        if video_len != expected_video_len or int(scores.shape[-1]) < expected_video_len:
            if not self._logged_incomplete_video_kv:
                log_message(
                    "Skipping attention capture for incomplete Cosmos video KV: "
                    f"video_len={video_len}, scores_k={int(scores.shape[-1])}, expected={expected_video_len}, "
                    f"k_video_shape={tuple(k_video.shape)}, k_action_shape={tuple(k_action.shape)}.",
                    self.log_file,
                )
                self._logged_incomplete_video_kv = True
            return
        video_scores = scores[:, :, :video_len]
        video_attn = full_attn[:, :video_len]
        for q_idx, label in enumerate(labels):
            slice_records = self._slice_records(
                scores=video_scores[q_idx : q_idx + 1],
                full_attn=video_attn[q_idx : q_idx + 1],
                video_grid_thw=video_grid_thw,
                video_len=video_len,
            )
            target.setdefault(label, {})[int(layer_idx)] = slice_records

    def capture_slow_attention(
        self,
        *,
        layer_idx: int,
        q_tokens: torch.Tensor,
        k_video: torch.Tensor,
        k_action: torch.Tensor,
        allowed_mask: Optional[torch.Tensor],
        action_valid_mask: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
    ) -> None:
        if not self.active:
            return
        layer_idx = int(layer_idx)
        if layer_idx not in self._slow_layer_index_set:
            return
        if action_valid_mask is None:
            selected_count = len(self.slow_labels)
            selected = torch.arange(
                q_tokens.shape[1] - selected_count,
                q_tokens.shape[1],
                device=q_tokens.device,
                dtype=torch.long,
            )
        else:
            valid = action_valid_mask.to(device=q_tokens.device, dtype=torch.bool)[0]
            positions = torch.nonzero(valid, as_tuple=False).flatten()
            selected_count = len(self.slow_labels)
            if int(positions.numel()) < selected_count:
                raise ValueError(
                    f"Need at least {selected_count} valid slow-prefix tokens to visualize {self.slow_labels}."
                )
            selected = positions[-selected_count:]
        selected_q = q_tokens.index_select(1, selected)
        selected_mask = None
        if allowed_mask is not None:
            selected_mask = allowed_mask.index_select(0, selected)
        self._capture_tokens(
            target=self.slow_records,
            labels=self.slow_labels,
            layer_idx=layer_idx,
            q_tokens=selected_q,
            k_video=k_video,
            k_action=k_action,
            allowed_mask=selected_mask,
            video_grid_thw=video_grid_thw,
        )

    def capture_action_attention(
        self,
        *,
        layer_idx: int,
        q_tokens: torch.Tensor,
        k_video: torch.Tensor,
        k_action: torch.Tensor,
        allowed_mask: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
    ) -> None:
        if not self.active:
            return
        layer_idx = int(layer_idx)
        if layer_idx not in self._action_layer_index_set:
            return
        if layer_idx == self.first_action_layer_idx:
            self.action_denoise_step += 1
            self.capture_current_action_step = self.action_denoise_step == self._target_action_step()
        if not self.capture_current_action_step:
            return
        if int(q_tokens.shape[1]) != self.action_chunk:
            raise ValueError(f"Expected {self.action_chunk} action q tokens, got {int(q_tokens.shape[1])}.")
        labels = [f"action{i:02d}" for i in range(self.action_chunk)]
        self._capture_tokens(
            target=self.action_records,
            labels=labels,
            layer_idx=layer_idx,
            q_tokens=q_tokens,
            k_video=k_video,
            k_action=k_action,
            allowed_mask=allowed_mask,
            video_grid_thw=video_grid_thw,
        )

    def _query_dir(self) -> Path:
        task_name = _sanitize_filename(str(self.metadata["task_name"]))
        episode_label = f"{int(self.metadata['episode_index']):03d}"
        return Path(self.output_dir) / f"task={task_name}" / f"episode={episode_label}"

    def _output_path(self, kind: str) -> Path:
        task_name = _sanitize_filename(str(self.metadata["task_name"]))
        episode_label = f"{int(self.metadata['episode_index']):03d}"
        record_label = f"{int(self.metadata['record_index']):04d}"
        sample_label = f"{int(self.metadata['sample_index']):06d}"
        frame_label = f"{int(self.metadata['frame_index']):06d}"
        prompt = _sanitize_filename(str(self.metadata["prompt"]), max_len=60)
        filename = (
            f"task={task_name}--episode={episode_label}--record={record_label}"
            f"--sample={sample_label}--frame={frame_label}--{kind}--prompt={prompt}.png"
        )
        return self._query_dir() / filename

    def _blank_tile(self, label: str) -> Image.Image:
        cell = Image.new("RGB", (self.tile_size, self.header_h + self.tile_size), (255, 255, 255))
        draw = ImageDraw.Draw(cell)
        font = ImageFont.load_default()
        draw.text((5, 5), label, fill=(90, 90, 90), font=font)
        return cell

    def _render_summary(
        self,
        *,
        records: dict[str, dict[int, dict[str, dict[str, Any]]]],
        query_labels: list[str],
        layers: range,
        title: str,
    ) -> Image.Image:
        row_label_w = 92
        col_label_h = 38
        cell_w = self.tile_size
        cell_h = self.header_h + self.tile_size
        layer_list = list(layers)
        columns = [(label, spec[0], spec[3]) for label in query_labels for spec in SLICE_SPECS]
        width = row_label_w + len(columns) * cell_w + max(0, len(columns) - 1) * self.gap
        height = col_label_h + len(layer_list) * cell_h + max(0, len(layer_list) - 1) * self.gap
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        draw.text((8, 4), title, fill=(20, 20, 20), font=font)
        for col_idx, (query_label, slice_label, base_label) in enumerate(columns):
            x = row_label_w + col_idx * (cell_w + self.gap)
            short_query = query_label
            short_slice = slice_label.replace("slice", "s").replace("_to_", "-")
            draw.text((x + 3, 18), f"{short_query} {short_slice} base={base_label}", fill=(20, 20, 20), font=font)
        for row_idx, layer_idx in enumerate(layer_list):
            y = col_label_h + row_idx * (cell_h + self.gap)
            draw.text((8, y + self.header_h + max(4, self.tile_size // 2 - 6)), f"layer_{layer_idx:02d}", fill=(20, 20, 20), font=font)
            for col_idx, (query_label, slice_label, _) in enumerate(columns):
                x = row_label_w + col_idx * (cell_w + self.gap)
                record = records.get(query_label, {}).get(layer_idx, {}).get(slice_label)
                if record is None:
                    cell = self._blank_tile("n/a")
                else:
                    slice_idx = [spec[0] for spec in SLICE_SPECS].index(slice_label)
                    base = self.slice_base_images[slice_idx]
                    heat = np.asarray(record["heat"], dtype=np.float32)
                    score = float(record["score"])
                    overlay = _overlay_heatmap(base, heat, self.tile_size, self.alpha, self.top_ratio, self.top_softness)
                    cell = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
                    cell_draw = ImageDraw.Draw(cell)
                    cell_draw.text((5, 5), f"{score:.4f}", fill=_score_to_rgb(score), font=font)
                    cell.paste(overlay, (0, self.header_h))
                canvas.paste(cell, (x, y))
        return canvas

    def _validate_records(self, records: dict[str, dict[int, dict[str, dict[str, Any]]]], query_labels: list[str], layers: range, name: str) -> None:
        for label in query_labels:
            if label not in records:
                raise RuntimeError(f"Missing {name} attention records for query {label}.")
            missing_layers = [layer_idx for layer_idx in layers if layer_idx not in records[label]]
            if missing_layers:
                raise RuntimeError(f"Missing {name} attention layers for {label}: {missing_layers[:8]}.")
            for layer_idx in layers:
                missing_slices = [spec[0] for spec in SLICE_SPECS if spec[0] not in records[label][layer_idx]]
                if missing_slices:
                    raise RuntimeError(f"Missing {name} slices for {label} layer={layer_idx}: {missing_slices}.")

    def _save_query(self) -> list[str]:
        slow_labels = list(self.slow_labels)
        action_labels = [f"action{i:02d}" for i in range(self.action_chunk)]
        slow_layers = self.slow_layer_indices
        action_layers = self.action_layer_indices
        self._validate_records(self.slow_records, slow_labels, slow_layers, "slow")
        self._validate_records(self.action_records, action_labels, action_layers, "action")
        query_dir = self._query_dir()
        os.makedirs(query_dir, exist_ok=True)
        slow_img = self._render_summary(
            records=self.slow_records,
            query_labels=slow_labels,
            layers=slow_layers,
            title=(
                f"slow prefix: {'/'.join(slow_labels)} -> Cosmos slices "
                f"(right layers {self.slow_layer_indices[0]}-{self.slow_layer_indices[-1]})"
            ),
        )
        action_img = self._render_summary(
            records=self.action_records,
            query_labels=action_labels,
            layers=action_layers,
            title=(
                f"action denoise step {self._target_action_step()}: action00-action{self.action_chunk - 1:02d} "
                f"-> Cosmos slices (right layers {self.action_layer_indices[0]}-{self.action_layer_indices[-1]})"
            ),
        )
        slow_path = self._output_path("slow_summary")
        action_path = self._output_path("action_summary")
        slow_img.save(slow_path)
        action_img.save(action_path)
        log_message(f"Saved Libero slow attention summary PNG to {slow_path}", self.log_file)
        log_message(f"Saved Libero action attention summary PNG to {action_path}", self.log_file)
        return [str(slow_path), str(action_path)]


def install_attention_map_recorder(model, cfg: LiberoTrainsetAttnVisConfig, log_file=None) -> Optional[LiberoAttentionMapRecorder]:
    output_dir = str(getattr(cfg, "attention_visualization_dir", "") or "").strip()
    if not output_dir:
        return None
    if str(getattr(model, "right_single_attn_position", "")) != "first4":
        raise ValueError(
            "Libero action-to-Cosmos visualization requires right_single_attn_position='first4'. "
            "With last4, the four T-Rex action layers are standalone and have no Cosmos K/V."
        )

    right_layer_specs = getattr(model, "right_layer_specs", None)
    right_layers = getattr(model, "right_layers", None)
    prefix_layer_count = int(getattr(model, "prefix_layer_count", 0) or 0)
    right_layer_count = int(getattr(model, "right_layer_count", 0) or 0)
    if not right_layer_specs or right_layers is None or len(right_layer_specs) != right_layer_count:
        raise AttributeError("Model does not expose a complete right-layer topology for attention visualization.")

    paired_prefix_layers = [
        int(spec["right_idx"])
        for spec in right_layer_specs
        if spec["kind"] == "paired" and int(spec["right_idx"]) < prefix_layer_count
    ]
    paired_action_layers = [
        int(spec["right_idx"])
        for spec in right_layer_specs
        if spec["kind"] == "paired" and int(spec["right_idx"]) >= prefix_layer_count
    ]
    recorder = LiberoAttentionMapRecorder(
        cfg,
        right_layer_count=right_layer_count,
        slow_layer_indices=paired_prefix_layers,
        action_layer_indices=paired_action_layers,
        log_file=log_file,
    )

    def make_patched_forward_action_only(right_idx: int):
        def patched_forward_action_only(
            self,
            x_action: torch.Tensor,
            action_valid_mask: Optional[torch.Tensor] = None,
            rotary_payload=None,
            action_tail_token_count: int = 0,
        ) -> torch.Tensor:
            if self.cached_k_v is None or self.cached_v_v is None:
                raise RuntimeError("forward_action_only requires cached Cosmos KV from run_cosmos_once().")

            q_a, k_a, v_a = self.action_bridge.get_branch_qkv(x_action)
            q_a, k_a = self._apply_action_rotary(q_a, k_a, rotary_payload)
            video_len = int(self.cached_k_v.shape[1])
            action_len = int(q_a.shape[1])
            k = torch.cat([self.cached_k_v, k_a], dim=1)
            v = torch.cat([self.cached_v_v, v_a], dim=1)
            mask = self._build_action_only_mask(
                video_len,
                action_len,
                q_a.device,
                action_valid_mask=action_valid_mask,
                action_tail_token_count=action_tail_token_count,
            )

            tail_count = int(action_tail_token_count or 0)
            if tail_count > 0:
                if tail_count != recorder.action_chunk:
                    raise ValueError(
                        f"Expected {recorder.action_chunk} action-tail queries, got {tail_count}."
                    )
                recorder.capture_action_attention(
                    layer_idx=right_idx,
                    q_tokens=q_a[:, -tail_count:, :, :],
                    k_video=self.cached_k_v,
                    k_action=k_a,
                    allowed_mask=mask[0, 0, -tail_count:, :],
                    video_grid_thw=self.cached_video_grid_thw,
                )
            else:
                recorder.capture_slow_attention(
                    layer_idx=right_idx,
                    q_tokens=q_a,
                    k_video=self.cached_k_v,
                    k_action=k_a,
                    allowed_mask=mask[0, 0],
                    action_valid_mask=action_valid_mask,
                    video_grid_thw=self.cached_video_grid_thw,
                )

            query_valid_mask = action_valid_mask
            if query_valid_mask is None:
                query_valid_mask = torch.ones(
                    (q_a.shape[0], action_len),
                    device=q_a.device,
                    dtype=torch.bool,
                )
            result = self._bridge_sdpa(
                q_a,
                k,
                v,
                attn_mask=mask,
                query_valid_mask=query_valid_mask,
            )
            return self.action_bridge.post_attention(
                x_action,
                result,
                token_valid_mask=action_valid_mask,
            )

        return patched_forward_action_only

    for spec in right_layer_specs:
        if spec["kind"] != "paired":
            continue
        right_idx = int(spec["right_idx"])
        wrapper = right_layers[right_idx]
        if not hasattr(wrapper, "_libero_trex_attn_vis_original_forward_action_only"):
            wrapper._libero_trex_attn_vis_original_forward_action_only = wrapper.forward_action_only
            wrapper.forward_action_only = types.MethodType(
                make_patched_forward_action_only(right_idx),
                wrapper,
            )

    model._libero_trex_attention_map_recorder = recorder
    log_message(
        f"Installed Libero T-Rex MoT2 attention recorder: dir={output_dir}, "
        f"right_single_attn_position={cfg.right_single_attn_position}, "
        f"slow_right_layers={list(recorder.slow_layer_indices)}, "
        f"action_right_layers={list(recorder.action_layer_indices)}, "
        f"action_chunk={cfg.action_chunk}, capture_mode={cfg.attention_visualization_capture_mode}, tile={recorder.tile_size}, gap={recorder.gap}",
        log_file,
    )
    return recorder


def run_selected_records(
    cfg: LiberoTrainsetAttnVisConfig,
    dataset: VLACotDataset,
    selected: list[tuple[str, int, int]],
    model,
    recorder: LiberoAttentionMapRecorder,
    log_file=None,
) -> None:
    device = next(model.parameters()).device
    dtype = torch.bfloat16
    trace_path = Path(str(cfg.attention_visualization_dir)) / "inference_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with trace_path.open("a", encoding="utf-8") as trace_file:
        for ordinal, (task_name, episode_index, sample_index) in enumerate(selected, start=1):
            sample = dataset.data[sample_index]
            frame_index = int(sample.get("frame_index", 0))
            record_index = frame_index
            prompt = str(sample["input_prompt"])
            log_message(
                f"[{ordinal}/{len(selected)}] task={task_name!r} episode={episode_index} "
                f"frame={frame_index} sample={sample_index} prompt={prompt!r}",
                log_file,
            )
            item = dataset[sample_index]
            batch = move_batch_to_device(dataset.collate_fn([item]), device, dtype)
            slice_base_images = load_cosmos_slice_base_images(dataset, sample, cfg)
            now_state = batch.get("now_state")
            cosmos_text_embeddings = batch.get("cosmos_text_embeddings")
            if cosmos_text_embeddings is not None:
                cosmos_text_embeddings = cosmos_text_embeddings.to(device=device, dtype=dtype)
            condition_frame_count = int(cfg.num_cond_input_frames)
            condition_frames = batch["videos"][:, :, :condition_frame_count]
            if int(condition_frames.shape[2]) != condition_frame_count:
                raise ValueError(
                    f"Expected {condition_frame_count} Cosmos history frames, "
                    f"got shape={tuple(condition_frames.shape)}."
                )
            recorder.start_query(
                slice_base_images=slice_base_images,
                task_name=task_name,
                episode_index=episode_index,
                record_index=record_index,
                sample_index=sample_index,
                frame_index=frame_index,
                prompt=prompt,
                action_denoise_steps=int(cfg.action_denoise_steps),
            )
            try:
                with torch.inference_mode():
                    outputs = model.forward_flow_joint_inference(
                        janus_input_ids=batch["janus_input_ids"],
                        janus_pixel_values=batch["janus_pixel_values"],
                        janus_image_grid_thw=batch["janus_image_grid_thw"],
                        janus_images_seq_mask=batch["janus_images_seq_mask"],
                        janus_images_emb_mask=batch["janus_images_emb_mask"],
                        # Pass only t-4..t. The model initializes all future
                        # latent frames from random noise before Cosmos denoising.
                        first_frame=condition_frames,
                        action_denoise_steps=int(cfg.action_denoise_steps),
                        cosmos_denoise_steps=int(cfg.cosmos_denoise_steps),
                        fps=torch.tensor([float(cfg.fps)], device=device, dtype=dtype),
                        action_self_causal_in_bridge=bool(cfg.action_self_causal_in_bridge),
                        num_spatial_tokens=int(cfg.total_latent_tokens),
                        janus_left_pad_lens=batch["janus_left_pad_lens"],
                        janus_state_seq_mask=batch["janus_state_seq_mask"],
                        janus_attention_mask=batch["attention_mask"].to(torch.bool),
                        cosmos_janus_input_ids=batch["cosmos_janus_input_ids"],
                        cosmos_janus_image_grid_thw=batch["cosmos_janus_image_grid_thw"],
                        cosmos_janus_images_seq_mask=batch["cosmos_janus_images_seq_mask"],
                        cosmos_janus_state_seq_mask=batch["cosmos_janus_state_seq_mask"],
                        cosmos_janus_images_emb_mask=batch["cosmos_janus_images_emb_mask"],
                        now_state=now_state,
                        cosmos_text_embeddings=cosmos_text_embeddings,
                        decode_video=False,
                        return_spatial_debug=True,
                    )
                pred_video, pred_action, spatial_debug = outputs
                del pred_video
                saved_paths = recorder.finish_query()
                summary = action_summary(pred_action)
                spatial_token_ids = [
                    int(token_id)
                    for token_id in spatial_debug["spatial_token_ids"].detach().cpu().reshape(-1).tolist()
                ]
                spatial_tokens = [cfg.special_token_vocab[token_id] for token_id in spatial_token_ids]
                trace_file.write(
                    json.dumps(
                        {
                            "task": task_name,
                            "episode_index": int(episode_index),
                            "record_index": int(record_index),
                            "sample_index": int(sample_index),
                            "frame_index": int(frame_index),
                            "prompt": prompt,
                            "cosmos_condition_frames": condition_frame_count,
                            "spatial_token_ids": spatial_token_ids,
                            "spatial_tokens": spatial_tokens,
                            "saved_attention_paths": saved_paths,
                            "pred_action": summary,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                trace_file.flush()
                log_message(f"Spatial tokens: {spatial_tokens}", log_file)
                log_message(f"Action summary: {json.dumps(summary, sort_keys=True)}", log_file)
            except Exception:
                recorder.discard_query()
                raise
            processed += 1
            if int(cfg.empty_cache_every) > 0 and processed % int(cfg.empty_cache_every) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def main() -> None:
    cfg = parse_args()
    if not cfg.attention_visualization_dir:
        raise ValueError("--attention_visualization_dir is required.")
    if int(cfg.video_frames) != 17:
        raise ValueError(f"Libero 5-slice Cosmos visualization expects video_frames=17, got {cfg.video_frames}.")
    if int(cfg.num_cond_input_frames) != 5:
        raise ValueError(
            f"Libero 5-slice Cosmos visualization expects num_cond_input_frames=5, got {cfg.num_cond_input_frames}."
        )
    if int(cfg.action_chunk) != 16:
        raise ValueError(f"Libero action visualization expects action_chunk=16, got {cfg.action_chunk}.")
    if cfg.right_single_attn_position != "first4":
        raise ValueError(
            "Libero action-to-Cosmos visualization requires right_single_attn_position='first4'."
        )
    Path(str(cfg.attention_visualization_dir)).mkdir(parents=True, exist_ok=True)
    log_path = Path(str(cfg.attention_visualization_dir)) / "libero_trainset_attn_vis.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_message(f"=== Libero T-Rex trainset attention visualization start {time.strftime('%Y-%m-%d %H:%M:%S')} ===", log_file)
        log_message(f"Config: {json.dumps(vars(cfg), sort_keys=True, default=str)}", log_file)
        if cfg.bash_hparams_path:
            log_message(f"Bash hparams: {cfg.bash_hparams_path}", log_file)
        random.seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))
        set_seed(int(cfg.seed))
        model, processor, _statistic = model_load(cfg, log_file)
        set_seed(int(cfg.seed))
        cfg.special_token_to_id = {
            token: idx for idx, token in enumerate(cfg.special_token_vocab)
        }
        checkpoint_path = Path(str(cfg.pretrained_checkpoint))
        checkpoint_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
        model_manifest = load_model_manifest(str(checkpoint_dir))
        if model_manifest is None:
            log_message(
                "Checkpoint has no model_config.json; reconstructing topology from runtime arguments.",
                log_file,
            )
        else:
            validate_runtime_model_config(cfg, model_manifest)
        recorder = install_attention_map_recorder(model, cfg, log_file)
        if recorder is None:
            raise RuntimeError("Attention recorder was not installed.")
        dataset = build_dataset(cfg, processor, log_file)
        selected = select_trainset_records(cfg, dataset, log_file)
        if not selected:
            raise ValueError("No trainset records selected.")
        log_message(f"Total selected Libero records: {len(selected)}", log_file)
        run_selected_records(cfg, dataset, selected, model, recorder, log_file)
        log_message(f"Attention visualizations written under {cfg.attention_visualization_dir}", log_file)
        log_message("=== Libero T-Rex trainset attention visualization finished ===", log_file)


if __name__ == "__main__":
    main()
