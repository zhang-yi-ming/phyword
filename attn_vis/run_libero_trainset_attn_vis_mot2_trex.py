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

from attn_vis.run_rlbench_trainset_attn_vis_mot2_trex import (  # noqa: E402
    DEFAULT_SPECIAL_TOKEN_VOCAB,
    PrintAccelerator,
    action_summary,
    coerce_bool,
    log_message,
    model_load,
    move_batch_to_device,
    resolve_spatial_token_args,
    set_seed,
    _draw_text_with_outline,
    _overlay_heatmap,
    _sanitize_filename,
    _score_to_rgb,
)
from scripts.train_mot2_trex import VLACotDataset  # noqa: E402


SLICE_SPECS = (
    ("slice0_t-4", 0, 1, "t-4"),
    ("slice1_t-3_to_t", 1, 2, "t-3"),
    ("slice2_t+1_to_t+4", 2, 3, "t+1"),
    ("slice3_t+5_to_t+8", 3, 4, "t+5"),
    ("slice4_t+9_to_t+12", 4, 5, "t+9"),
)


@dataclass
class LiberoTrainsetAttnVisConfig:
    pretrained_checkpoint: str = ""
    model_path: str = "/mnt/nas/zhangyiming/database/ckpt/pretrained/T-Rex_pretrain_mecka22k_epoch1"
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
    state_placeholder_tokens: int = 8
    state_dim: int = 8
    state_encoding_mode: str = "mlp"
    total_latent_tokens: str = ""
    latent_token_mode: str = ""
    special_token_vocab: str = ",".join(DEFAULT_SPECIAL_TOKEN_VOCAB)
    img_latents_per_future: int = 0
    state_latents_per_future: int = 0
    num_future_frames: int = 0
    future_frame_stride: int = 8
    use_latent_hidden_sim_loss: int = 0
    latent_hidden_sim_loss_mode: str = "siglip"
    latent_hidden_sim_pool_mode: str = "pool"
    latent_hidden_sim_loss_weight: float = 1.0
    use_latent_hidden_wan_downsample_sim_loss: int = 0
    latent_hidden_wan_downsample_sim_loss_weight: float = 1.0
    wan21_vae_path: str = "/mnt/nas/zhangyiming/database/ckpt/pretrained/wan2.1_vae/original/Wan2.1_VAE.pth"
    cosmos_self_only_bridge: bool = False
    decosmos: bool = False
    bridge_pos_scheme: str = "mrope"
    action_use_latent_prefix: bool = True
    action_self_causal_in_bridge: bool = True
    action_insert_layer: int = 0
    detach_action_cosmos_kv: int = 0
    action_denoise_steps: int = 10
    cosmos_denoise_steps: int = 2
    fps: float = 10.0
    empty_cache_every: int = 10


def parse_args() -> LiberoTrainsetAttnVisConfig:
    parser = argparse.ArgumentParser()
    for field_def in fields(LiberoTrainsetAttnVisConfig):
        default = field_def.default
        arg_type = str if default is None or isinstance(default, bool) else type(default)
        parser.add_argument(f"--{field_def.name}", default=default, type=arg_type)
    ns = parser.parse_args()
    cfg = LiberoTrainsetAttnVisConfig(**vars(ns))
    for name in ("cosmos_self_only_bridge", "decosmos", "action_use_latent_prefix", "action_self_causal_in_bridge"):
        setattr(cfg, name, coerce_bool(getattr(cfg, name)))
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
    cfg.action_insert_layer = int(getattr(cfg, "action_insert_layer", 0) or 0)
    if cfg.action_insert_layer < 0 or cfg.action_insert_layer > 27:
        raise ValueError("action_insert_layer must be in [0, 27].")
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
    cfg.latent_hidden_sim_pool_mode = str(getattr(cfg, "latent_hidden_sim_pool_mode", "pool") or "pool").lower()
    if cfg.latent_hidden_sim_pool_mode not in ("pool", "one_mlp", "mlp"):
        raise ValueError("latent_hidden_sim_pool_mode must be 'pool', 'one_mlp', or 'mlp'.")
    cfg.spatial_hidden_sim_pool_mode = cfg.latent_hidden_sim_pool_mode
    return cfg


def build_dataset(cfg: LiberoTrainsetAttnVisConfig, processor, log_file=None) -> VLACotDataset:
    return VLACotDataset(cfg, processor, PrintAccelerator(log_file))


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
    start_frame = frame_idx - (int(cfg.num_cond_input_frames) - 1)
    images = []
    for _, frame_offset, _, _ in SLICE_SPECS:
        images.append(load_libero_frame(dataset, sample["video_path"], start_frame + int(frame_offset)))
    return images


class LiberoAttentionMapRecorder:
    def __init__(self, cfg: LiberoTrainsetAttnVisConfig, num_layers: int, log_file=None):
        self.cfg = cfg
        self.log_file = log_file
        self.output_dir = str(getattr(cfg, "attention_visualization_dir", "") or "").strip()
        self.action_chunk = int(getattr(cfg, "action_chunk", 16))
        self.spatial_token_count = int(getattr(cfg, "total_latent_tokens", 1) or 1)
        if self.spatial_token_count not in (1, 2):
            raise ValueError(f"Expected total_latent_tokens=1 or 2, got {self.spatial_token_count}.")
        self.slow_labels = ["v_prev", "v"] if self.spatial_token_count == 1 else ["v_prev", "v", "n"]
        self.num_layers = int(num_layers)
        self.slow_layer_count = self.num_layers
        self.first_action_layer_idx = int(getattr(cfg, "action_insert_layer", 0) or 0)
        if self.first_action_layer_idx < 0 or self.first_action_layer_idx >= self.num_layers:
            raise ValueError(
                f"action_insert_layer must be in [0, {self.num_layers - 1}], got {self.first_action_layer_idx}."
            )
        self.alpha = float(getattr(cfg, "attention_visualization_alpha", 0.45) or 0.45)
        self.tile_size = max(16, int(getattr(cfg, "attention_visualization_tile_size", 160) or 160))
        self.top_ratio = getattr(cfg, "attention_visualization_top_ratio", None)
        self.top_softness = float(getattr(cfg, "attention_visualization_top_softness", 0.05) or 0.0)
        self.gap = max(0, int(getattr(cfg, "attention_visualization_gap", 8) or 0))
        self.header_h = max(16, int(getattr(cfg, "attention_visualization_header_height", 24) or 24))
        self.capture_mode = str(getattr(cfg, "attention_visualization_capture_mode", "last") or "last").lower()
        if self.capture_mode not in {"all", "first", "last"}:
            raise ValueError(f"attention_visualization_capture_mode must be all/first/last, got {self.capture_mode!r}.")
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
        for slice_label, start_t, end_t, base_label in SLICE_SPECS:
            start = int(start_t) * h * w
            end = int(end_t) * h * w
            region_attn = full_attn[:, start:end]
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
        if layer_idx < 0 or layer_idx >= self.slow_layer_count:
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
        if layer_idx < self.first_action_layer_idx or layer_idx >= self.num_layers:
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
        slow_layers = range(0, self.slow_layer_count)
        action_layers = range(self.first_action_layer_idx, self.num_layers)
        self._validate_records(self.slow_records, slow_labels, slow_layers, "slow")
        self._validate_records(self.action_records, action_labels, action_layers, "action")
        query_dir = self._query_dir()
        os.makedirs(query_dir, exist_ok=True)
        slow_img = self._render_summary(
            records=self.slow_records,
            query_labels=slow_labels,
            layers=slow_layers,
            title=f"slow prefix: {'/'.join(slow_labels)} -> Cosmos slices (layers 0-{self.slow_layer_count - 1})",
        )
        action_img = self._render_summary(
            records=self.action_records,
            query_labels=action_labels,
            layers=action_layers,
            title=(
                f"action denoise step {self._target_action_step()}: action00-action{self.action_chunk - 1:02d} "
                f"-> Cosmos slices (layers {self.first_action_layer_idx}-{self.num_layers - 1})"
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
    wrappers = getattr(model, "mot_attention_wrappers", None)
    if wrappers is None or len(wrappers) == 0:
        raise AttributeError("Model has no mot_attention_wrappers to instrument for attention visualization.")
    recorder = LiberoAttentionMapRecorder(cfg, num_layers=len(wrappers), log_file=log_file)

    def make_patched_forward_action_prefix_and_cache(layer_idx: int):
        def patched_forward_action_prefix_and_cache(
            self,
            x_action: torch.Tensor,
            action_valid_mask: Optional[torch.Tensor] = None,
            rotary_payload=None,
            action_tail_token_count: int = 0,
            append_to_cache: bool = False,
        ) -> torch.Tensor:
            if self.cached_k_v is None or self.cached_v_v is None:
                raise RuntimeError("forward_action_prefix_and_cache requires cached Cosmos KV from run_cosmos_once().")
            q_a, k_a, v_a = self.action_bridge.get_branch_qkv(x_action)
            q_a, k_a = self._apply_action_rotary(q_a, k_a, rotary_payload)
            S_v = self.cached_k_v.shape[1]
            S_a = q_a.shape[1]
            if append_to_cache and self.cached_k_a_prefix is not None:
                prefix_len = self.cached_k_a_prefix.shape[1]
                k_action_full = torch.cat([self.cached_k_a_prefix, k_a], dim=1)
                k = torch.cat([self.cached_k_v, k_action_full], dim=1)
                v = torch.cat([self.cached_v_v, self.cached_v_a_prefix, v_a], dim=1)
                mask = self._build_action_cached_suffix_mask(
                    S_v,
                    prefix_len,
                    S_a,
                    q_a.device,
                    prefix_valid_mask=self.cached_action_prefix_valid_mask,
                    suffix_valid_mask=action_valid_mask,
                    action_tail_token_count=action_tail_token_count,
                )
            else:
                k_action_full = k_a
                k = torch.cat([self.cached_k_v, k_a], dim=1)
                v = torch.cat([self.cached_v_v, v_a], dim=1)
                mask = self._build_action_only_mask(
                    S_v,
                    S_a,
                    q_a.device,
                    action_valid_mask=action_valid_mask,
                    action_tail_token_count=action_tail_token_count,
                )
                recorder.capture_slow_attention(
                    layer_idx=layer_idx,
                    q_tokens=q_a,
                    k_video=self.cached_k_v,
                    k_action=k_action_full,
                    allowed_mask=mask[0, 0],
                    action_valid_mask=action_valid_mask,
                    video_grid_thw=self.cached_video_grid_thw,
                )
            query_valid_mask = action_valid_mask
            if query_valid_mask is None:
                query_valid_mask = torch.ones((q_a.shape[0], S_a), device=q_a.device, dtype=torch.bool)
            result = self._bridge_sdpa(q_a, k, v, attn_mask=mask, query_valid_mask=query_valid_mask)
            out = self.action_bridge.post_attention(x_action, result, token_valid_mask=action_valid_mask)
            if append_to_cache:
                k_store = k_a.detach()
                v_store = v_a.detach()
                valid_store = query_valid_mask.detach().clone()
                if self.cached_k_a_prefix is None:
                    self.cached_k_a_prefix = k_store
                    self.cached_v_a_prefix = v_store
                    self.cached_action_prefix_valid_mask = valid_store
                else:
                    self.cached_k_a_prefix = torch.cat([self.cached_k_a_prefix, k_store], dim=1)
                    self.cached_v_a_prefix = torch.cat([self.cached_v_a_prefix, v_store], dim=1)
                    self.cached_action_prefix_valid_mask = torch.cat([self.cached_action_prefix_valid_mask, valid_store], dim=1)
            return out

        return patched_forward_action_prefix_and_cache

    def make_patched_forward_action_suffix_only(layer_idx: int):
        def patched_forward_action_suffix_only(
            self,
            x_action_suffix: torch.Tensor,
            suffix_valid_mask: Optional[torch.Tensor] = None,
            rotary_payload=None,
            action_tail_token_count: int = 0,
        ) -> torch.Tensor:
            if self.cached_k_v is None or self.cached_v_v is None:
                raise RuntimeError("forward_action_suffix_only requires cached Cosmos KV from run_cosmos_once().")
            if self.cached_k_a_prefix is None or self.cached_v_a_prefix is None:
                raise RuntimeError("forward_action_suffix_only requires cached action prefix KV.")
            q_a, k_a, v_a = self.action_bridge.get_branch_qkv(x_action_suffix)
            q_a, k_a = self._apply_action_rotary(q_a, k_a, rotary_payload)
            S_v = self.cached_k_v.shape[1]
            S_prefix = self.cached_k_a_prefix.shape[1]
            S_suffix = q_a.shape[1]
            k_action_full = torch.cat([self.cached_k_a_prefix, k_a], dim=1)
            k = torch.cat([self.cached_k_v, k_action_full], dim=1)
            v = torch.cat([self.cached_v_v, self.cached_v_a_prefix, v_a], dim=1)
            mask = self._build_action_cached_suffix_mask(
                S_v,
                S_prefix,
                S_suffix,
                q_a.device,
                prefix_valid_mask=self.cached_action_prefix_valid_mask,
                suffix_valid_mask=suffix_valid_mask,
                action_tail_token_count=action_tail_token_count,
            )
            query_valid_mask = suffix_valid_mask
            if query_valid_mask is None:
                query_valid_mask = torch.ones((q_a.shape[0], S_suffix), device=q_a.device, dtype=torch.bool)
            expected = 1 + int(recorder.action_chunk)
            if int(q_a.shape[1]) != expected:
                raise ValueError(f"Expected suffix queries time+{recorder.action_chunk} actions={expected}, got {q_a.shape[1]}.")
            recorder.capture_action_attention(
                layer_idx=layer_idx,
                q_tokens=q_a[:, 1:, :, :],
                k_video=self.cached_k_v,
                k_action=k_action_full,
                allowed_mask=mask[0, 0, 1:, :],
                video_grid_thw=self.cached_video_grid_thw,
            )
            result = self._bridge_sdpa(q_a, k, v, attn_mask=mask, query_valid_mask=query_valid_mask)
            return self.action_bridge.post_attention(x_action_suffix, result, token_valid_mask=suffix_valid_mask)

        return patched_forward_action_suffix_only

    for layer_idx, wrapper in enumerate(wrappers):
        if not hasattr(wrapper, "_libero_trex_attn_vis_original_forward_action_prefix_and_cache"):
            wrapper._libero_trex_attn_vis_original_forward_action_prefix_and_cache = wrapper.forward_action_prefix_and_cache
            wrapper.forward_action_prefix_and_cache = types.MethodType(
                make_patched_forward_action_prefix_and_cache(layer_idx),
                wrapper,
            )
        if not hasattr(wrapper, "_libero_trex_attn_vis_original_forward_action_suffix_only"):
            wrapper._libero_trex_attn_vis_original_forward_action_suffix_only = wrapper.forward_action_suffix_only
            wrapper.forward_action_suffix_only = types.MethodType(make_patched_forward_action_suffix_only(layer_idx), wrapper)

    model._libero_trex_attention_map_recorder = recorder
    log_message(
        f"Installed Libero T-Rex MoT2 attention recorder: dir={output_dir}, layers={len(wrappers)}, "
        f"action_insert_layer={recorder.first_action_layer_idx}, action_layers={len(range(recorder.first_action_layer_idx, recorder.num_layers))}, "
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
                        first_frame=batch["videos"],
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
                    )
                pred_video, pred_action = outputs[:2]
                del pred_video
                saved_paths = recorder.finish_query()
                summary = action_summary(pred_action)
                trace_file.write(
                    json.dumps(
                        {
                            "task": task_name,
                            "episode_index": int(episode_index),
                            "record_index": int(record_index),
                            "sample_index": int(sample_index),
                            "frame_index": int(frame_index),
                            "prompt": prompt,
                            "saved_attention_paths": saved_paths,
                            "pred_action": summary,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                trace_file.flush()
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
