#!/usr/bin/env python3
"""Render global token-group bridge attention heatmaps for RLBench train records.

This script reads RLBench train JSON records, runs checkpoint inference, and
records action-denoise bridge attention for these query tokens:
  1. action token
  2. generated latent token(s): latent_0[, latent_1]
  3. final colon token before the generated latent token(s)

Each PNG is a layer x token-group heatmap. The columns are:
main_image, text_after_image, final_colon, latent_0[, latent_1],
latent_end, action_image, t, action_token.
"""

import argparse
import gc
import json
import logging
import math
import os
import random
import sys
import time
import types
from collections import OrderedDict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

import run_rlbench_trainset_attn_vis as base  # noqa: E402
from janus.models import ActionTokenizer, VLChatProcessor  # noqa: E402
from models.cosmos_janus_cot import CosmosJanusMoT3Expert, normalize_bridge_pos_scheme  # noqa: E402
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint  # noqa: E402
from utils.cosmos_text_cache import CosmosQwenTextEmbedder, CosmosTextEmbeddingCache  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RLBenchGlobalTokenAttnHeatmap")


LATENT_TOKEN_MODE_TO_FIELDS = {
    "v": ("gtlatent",),
    "n": ("gtlatent2",),
    "vn": ("gtlatent", "gtlatent2"),
}
LATENT_TOKEN_MODE_ALIASES = {
    "1": "v",
    "2": "vn",
}


def normalize_latent_token_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    mode = LATENT_TOKEN_MODE_ALIASES.get(mode, mode)
    if mode not in LATENT_TOKEN_MODE_TO_FIELDS:
        valid = ", ".join(sorted([*LATENT_TOKEN_MODE_TO_FIELDS.keys(), *LATENT_TOKEN_MODE_ALIASES.keys()]))
        raise ValueError(f"latent token mode must be one of {valid}, got {value!r}.")
    return mode


def resolve_latent_token_args(cfg: "TrainsetGlobalTokenAttnConfig") -> None:
    mode_arg = str(getattr(cfg, "latent_token_mode", "") or "").strip()
    count_or_mode_arg = str(getattr(cfg, "total_latent_tokens", "") or "").strip()

    if mode_arg:
        mode = normalize_latent_token_mode(mode_arg)
        fields_for_mode = list(LATENT_TOKEN_MODE_TO_FIELDS[mode])
        if count_or_mode_arg:
            try:
                token_count = int(count_or_mode_arg)
            except ValueError:
                token_mode = normalize_latent_token_mode(count_or_mode_arg)
                if token_mode != mode:
                    raise ValueError(
                        "Conflicting latent token settings: "
                        f"--latent_token_mode={mode_arg!r} but "
                        f"--total_latent_tokens={count_or_mode_arg!r}."
                    )
                token_count = len(LATENT_TOKEN_MODE_TO_FIELDS[token_mode])
            if token_count != len(fields_for_mode):
                raise ValueError(
                    "total_latent_tokens count does not match latent_token_mode: "
                    f"mode={mode!r} expects {len(fields_for_mode)}, got {token_count}."
                )
    else:
        mode = normalize_latent_token_mode(count_or_mode_arg or "1")
        fields_for_mode = list(LATENT_TOKEN_MODE_TO_FIELDS[mode])

    cfg.latent_token_mode = mode
    cfg.latent_token_fields = fields_for_mode
    cfg.total_latent_tokens = len(fields_for_mode)


def build_group_labels(total_latent_tokens: int) -> list[str]:
    return [
        "main_image",
        "text_after_image",
        "final_colon",
        *[f"latent_{idx}" for idx in range(int(total_latent_tokens))],
        "latent_end",
        "action_image",
        "t",
        "action_token",
    ]


def build_query_labels(total_latent_tokens: int) -> list[str]:
    return [
        "action_token",
        *[f"latent_{idx}" for idx in range(int(total_latent_tokens))],
        "final_colon",
    ]


@dataclass
class TrainsetGlobalTokenAttnConfig:
    pretrained_checkpoint: str = ""
    model_path: str = "/mnt/nas/zhangyiming/database/ckpt/pretrained/Janus-Pro-1B"
    action_model_path: str = "/mnt/nas/zhangyiming/database/ckpt/pretrained/LaST0_Pretrain_AE_chunk16/tfmr"
    cosmos_model_path: str = (
        "/mnt/nas/zhangyiming/database/ckpt/pretrained/Cosmos-Predict2.5-2B/base/pre-trained/"
        "d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt"
    )
    cosmos_experiment_name: str = (
        "Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_"
        "high_sigma_loss_reweighted_1_1_rectified_flow_only"
    )
    cosmos_text_cache_path: str = ""
    data_path: str = "/mnt/nas/zhangyiming/database/rlbench/train/json/train_action_chunk1_sumpos_lastrot.json"
    data_root: str = ""
    attention_visualization_dir: str = ""
    attention_visualization_capture_mode: str = "last"
    attention_visualization_cell_width: int = 112
    attention_visualization_cell_height: int = 28
    bash_hparams_path: str = ""
    eval_artifact_name: str = ""
    task_names: str = ""
    num_trajectories_per_task: int = 1
    max_records_per_episode: int = 0
    max_total_records: int = 0
    cuda: str = "0"
    seed: int = 0
    video_h: int = 32
    video_w: int = 32
    video_frames: int = 5
    num_cond_input_frames: int = 1
    action_dim: int = 7
    action_chunk: int = 1
    robot_state: int = 0
    state_placeholder_tokens: int = 1
    state_dim: int = 7
    state_encoding_mode: str = "mlp"
    action_intermediate_size: int = 5632
    total_latent_tokens: str = ""
    latent_token_mode: str = ""
    extra_special_tokens: str = ""
    img_latents_per_future: int = 0
    state_latents_per_future: int = 0
    num_future_frames: int = 0
    future_frame_stride: int = 1
    video_loss_weight: float = 0.0
    latent_loss_weight: float = 1.0
    use_latent_hidden_sim_loss: int = 1
    latent_hidden_sim_loss_mode: str = "wan_vae"
    latent_hidden_sim_loss_weight: float = 1.0
    use_latent_hidden_wan_downsample_sim_loss: int = 1
    latent_hidden_wan_downsample_sim_loss_weight: float = 1.0
    wan21_vae_path: str = "/mnt/nas/zhangyiming/database/ckpt/pretrained/wan2.1_vae/original/Wan2.1_VAE.pth"
    cosmos_self_only_bridge: bool = True
    train_embed_tokens: int = 1
    decosmos: bool = True
    use_value_prediction: bool = False
    use_action_value_prediction: bool = False
    value_token_mask_video_to_value: bool = False
    value_token_mask_nonvalue_to_value: bool = False
    bridge_pos_scheme: str = "llama1d"
    action_use_latent_prefix: bool = True
    action_self_causal_in_bridge: bool = True
    action_denoise_steps: int = 10
    cosmos_denoise_steps: int = 2
    fps: float = 20.0
    empty_cache_every: int = 10


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def parse_args() -> TrainsetGlobalTokenAttnConfig:
    parser = argparse.ArgumentParser()
    for field_def in fields(TrainsetGlobalTokenAttnConfig):
        default = field_def.default
        arg_type = str if isinstance(default, bool) else type(default)
        parser.add_argument(f"--{field_def.name}", default=default, type=arg_type)
    ns = parser.parse_args()
    cfg = TrainsetGlobalTokenAttnConfig(**vars(ns))
    for name in (
        "cosmos_self_only_bridge",
        "decosmos",
        "use_value_prediction",
        "use_action_value_prediction",
        "value_token_mask_video_to_value",
        "value_token_mask_nonvalue_to_value",
        "action_use_latent_prefix",
        "action_self_causal_in_bridge",
    ):
        setattr(cfg, name, coerce_bool(getattr(cfg, name)))
    cfg.latent_hidden_sim_loss_mode = str(cfg.latent_hidden_sim_loss_mode or "siglip").lower()
    cfg.attention_visualization_capture_mode = str(
        cfg.attention_visualization_capture_mode or "last"
    ).lower()
    if cfg.attention_visualization_capture_mode not in {"all", "first", "last"}:
        raise ValueError(
            "attention_visualization_capture_mode must be all/first/last, "
            f"got {cfg.attention_visualization_capture_mode!r}."
        )
    resolve_latent_token_args(cfg)
    return cfg


def log_message(message: str, log_file=None) -> None:
    logger.info(message)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_has_processor(checkpoint_dir: str) -> bool:
    required = ("tokenizer.json", "tokenizer_config.json", "processor_config.json", "preprocessor_config.json")
    return all(os.path.exists(os.path.join(checkpoint_dir, name)) for name in required)


def model_load(cfg: TrainsetGlobalTokenAttnConfig, log_file=None):
    cfg.bridge_pos_scheme = normalize_bridge_pos_scheme(cfg.bridge_pos_scheme)
    ckpt_path, base_dir = base.resolve_checkpoint_paths(cfg.pretrained_checkpoint)
    processor_path = base_dir if checkpoint_has_processor(base_dir) else cfg.model_path
    log_message(f"Loading VLChatProcessor/tokenizer from {processor_path}", log_file)
    if processor_path != base_dir:
        log_message(
            f"Checkpoint directory has no complete tokenizer/processor files; falling back to {cfg.model_path}",
            log_file,
        )
    processor = VLChatProcessor.from_pretrained(processor_path, trust_remote_code=True)
    if str(getattr(cfg, "extra_special_tokens", "") or "").strip():
        added_extra_special_tokens = processor.add_extra_special_tokens(cfg.extra_special_tokens)
        log_message(
            "Added extra special tokens for visualization tokenizer: "
            f"{added_extra_special_tokens}",
            log_file,
        )
    tokenizer = processor.tokenizer
    action_tokenizer = ActionTokenizer(tokenizer, need_to_sub=3)
    cfg.janus_image_start_id = getattr(processor, "image_start_id", None) or tokenizer.convert_tokens_to_ids("<begin_of_image>")
    cfg.janus_image_end_id = getattr(processor, "image_end_id", None) or tokenizer.convert_tokens_to_ids("<end_of_image>")
    cfg.latent_end_id = tokenizer.convert_tokens_to_ids("<|latent_end|>")

    log_message(f"Loading Janus action base from {cfg.action_model_path}", log_file)
    janus_model = AutoModelForCausalLM.from_pretrained(
        cfg.action_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        flow=True,
        action_dim=cfg.action_dim,
        ignore_mismatched_sizes=True,
    )

    import cosmos_predict2._src.predict2.models.text2world_model_rectified_flow as t2w_module

    class DummyTextEncoder(nn.Module):
        def __init__(self, *unused_args, **unused_kwargs):
            super().__init__()

    t2w_module.TextEncoder = DummyTextEncoder
    log_message(f"Loading Cosmos base from {cfg.cosmos_model_path}", log_file)
    cosmos_wrapper, cosmos_config = load_model_from_checkpoint(
        experiment_name=cfg.cosmos_experiment_name,
        s3_checkpoint_dir=cfg.cosmos_model_path,
        config_file="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        load_ema_to_reg=True,
        to_device="cpu",
        experiment_opts=["data_train=mock", "data_val=mock"],
    )
    base.resolve_video_condition_config(cfg, cosmos_wrapper.tokenizer)

    log_message(f"Loading fine-tuned state dict from {ckpt_path}", log_file)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    base.maybe_infer_action_intermediate_size(cfg, state_dict)
    base.ensure_janus_tokenizer_alignment(janus_model, tokenizer)
    base.validate_checkpoint_vocab_size(state_dict, tokenizer)

    model = CosmosJanusMoT3Expert(cosmos_wrapper.net, cosmos_wrapper.tokenizer, janus_model, cfg)
    model.set_cosmos_inference_runtime_from_wrapper(cosmos_wrapper)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log_message(f"Missing keys while loading checkpoint: {missing[:8]}", log_file)
    if unexpected:
        log_message(f"Unexpected keys while loading checkpoint: {unexpected[:8]}", log_file)

    device = torch.device(f"cuda:{cfg.cuda}" if torch.cuda.is_available() else "cpu")
    model = model.to(torch.bfloat16).to(device).eval()
    if cfg.cosmos_text_cache_path:
        text_encoder_config = getattr(cosmos_config.model.config, "text_encoder_config", None)
        model.cosmos_text_cache = CosmosTextEmbeddingCache(cfg.cosmos_text_cache_path, create=True)
        model.cosmos_qwen_text_embedder = CosmosQwenTextEmbedder(model.cosmos_dit, text_encoder_config, device=device)

    stats_path = os.path.join(base_dir, "train_statistics.json")
    with open(stats_path, "r", encoding="utf-8") as f:
        stats_data = json.load(f)
    dataset_name = next(iter(stats_data))
    stats = stats_data[dataset_name]
    statistic = {
        "action_mask": np.array(stats["action"]["mask"], dtype=bool),
        "action_q01": np.array(stats["action"]["q01"], dtype=np.float32),
        "action_q99": np.array(stats["action"]["q99"], dtype=np.float32),
        "state_mask": np.array(stats["state"]["mask"], dtype=bool),
        "state_q01": np.array(stats["state"]["q01"], dtype=np.float32),
        "state_q99": np.array(stats["state"]["q99"], dtype=np.float32),
    }
    log_message(
        f"Model ready on device={device}; tokenizer={processor_path}; train statistics={stats_path}",
        log_file,
    )
    return model, processor, action_tokenizer, statistic


def _sanitize_filename(value: str, max_len: int = 120) -> str:
    return base._sanitize_filename(value, max_len=max_len)


def _colorize(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 * values - 0.2, 0.0, 1.0)
    green = np.clip(1.5 - 3.0 * np.abs(values - 0.5), 0.0, 1.0)
    blue = np.clip(1.2 - 1.5 * values, 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def _valid_mean(values: torch.Tensor, indices: list[int]) -> float:
    if not indices:
        return 0.0
    selected = values[torch.as_tensor(indices, device=values.device, dtype=torch.long)]
    if selected.numel() == 0:
        return 0.0
    return float(selected.mean().detach().to(torch.float32).cpu().item())


class GlobalTokenAttentionRecorder:
    def __init__(self, cfg: TrainsetGlobalTokenAttnConfig, tokenizer, num_layers: int, log_file=None):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.log_file = log_file
        self.output_dir = str(getattr(cfg, "attention_visualization_dir", "") or "").strip()
        self.num_layers = int(num_layers)
        self.action_chunk = int(getattr(cfg, "action_chunk", 1))
        self.total_latent_tokens = int(getattr(cfg, "total_latent_tokens", 1))
        if self.total_latent_tokens not in (1, 2):
            raise ValueError(f"This recorder expects total_latent_tokens=1 or 2, got {self.total_latent_tokens}.")
        self.group_labels = build_group_labels(self.total_latent_tokens)
        self.query_labels = build_query_labels(self.total_latent_tokens)
        self.expected_image_tokens = 576
        self.capture_mode = str(getattr(cfg, "attention_visualization_capture_mode", "last") or "last").lower()
        self.cell_w = max(48, int(getattr(cfg, "attention_visualization_cell_width", 112) or 112))
        self.cell_h = max(20, int(getattr(cfg, "attention_visualization_cell_height", 28) or 28))
        self.special_ids = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) if x is not None)
        self.active = False
        self.current_step = -1
        self.records: dict[int, dict[int, torch.Tensor]] = {}
        self.metadata: dict[str, Any] = {}
        self.janus_input_ids: Optional[torch.Tensor] = None
        self.image_token_mask: Optional[torch.Tensor] = None
        self.attention_mask: Optional[torch.Tensor] = None

    def start_query(
        self,
        *,
        janus_input_ids: torch.Tensor,
        image_token_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        task_name: str,
        episode_index: int,
        record_index: int,
        sample_index: int,
        prompt: str,
        action_denoise_steps: int,
    ) -> None:
        if not self.output_dir:
            self.active = False
            return
        self.active = True
        self.janus_input_ids = janus_input_ids.detach().cpu()
        self.image_token_mask = image_token_mask.detach().cpu()
        self.attention_mask = attention_mask.detach().cpu().to(torch.bool)
        self.metadata = {
            "task_name": str(task_name),
            "episode_index": int(episode_index),
            "record_index": int(record_index),
            "sample_index": int(sample_index),
            "prompt": str(prompt),
            "action_denoise_steps": int(action_denoise_steps),
        }
        self.current_step = -1
        self.records = {}

    def discard_query(self) -> None:
        self.active = False
        self.current_step = -1
        self.records = {}
        self.metadata = {}
        self.janus_input_ids = None
        self.image_token_mask = None
        self.attention_mask = None

    def finish_query(self) -> list[str]:
        if not self.active:
            return []
        try:
            return self._save_query()
        finally:
            self.discard_query()

    def capture_action_only_attention(
        self,
        *,
        layer_idx: int,
        q_l: torch.Tensor,
        k_l: torch.Tensor,
        q_a: torch.Tensor,
        k_global: torch.Tensor,
        action_only_mask: torch.Tensor,
        action_value_token_count: int = 0,
    ) -> None:
        if not self.active:
            return
        if q_l.shape[0] != 1 or q_a.shape[0] != 1 or k_global.shape[0] != 1:
            raise ValueError(
                "Global token attention visualization expects batch size 1, "
                f"got q_l={tuple(q_l.shape)}, q_a={tuple(q_a.shape)}, k={tuple(k_global.shape)}."
            )
        if int(layer_idx) == 0:
            self.current_step += 1
            self.records[self.current_step] = {}
        if self.current_step < 0:
            raise RuntimeError("Attention recorder saw a nonzero layer before layer 0.")

        S_l = int(q_l.shape[1])
        S_a = int(q_a.shape[1])
        S_kv = int(k_global.shape[1])
        latent_start = S_l - int(self.total_latent_tokens)
        latent_indices = list(range(latent_start, S_l))
        final_colon_idx = latent_start - 1
        action_value_token_count = int(action_value_token_count or 0)
        action_start = S_a - action_value_token_count - self.action_chunk
        action_end = action_start + self.action_chunk
        if final_colon_idx < 0 or latent_start < 0 or len(latent_indices) != self.total_latent_tokens:
            raise ValueError(
                f"Could not locate final colon/latent tokens in S_l={S_l}, "
                f"total_latent_tokens={self.total_latent_tokens}."
            )
        if action_start < 0 or action_end > S_a:
            raise ValueError(
                f"Could not locate action tokens: S_a={S_a}, action_chunk={self.action_chunk}, "
                f"action_value_token_count={action_value_token_count}."
            )

        q_rows = {
            "final_colon": q_l[0, final_colon_idx:final_colon_idx + 1],
            "action_token": q_a[0, action_start:action_end],
        }
        query_mask_rows = {
            "final_colon": [final_colon_idx],
            "action_token": list(range(S_l + action_start, S_l + action_end)),
        }
        for latent_idx, token_idx in enumerate(latent_indices):
            query_name = f"latent_{latent_idx}"
            q_rows[query_name] = q_l[0, token_idx:token_idx + 1]
            query_mask_rows[query_name] = [token_idx]

        if action_only_mask.ndim == 4:
            mask_2d = action_only_mask[0, 0]
        elif action_only_mask.ndim == 3:
            mask_2d = action_only_mask[0]
        else:
            raise ValueError(f"Expected action_only_mask with 3/4 dims, got {tuple(action_only_mask.shape)}.")
        if tuple(mask_2d.shape) != (S_l + S_a, S_kv):
            raise ValueError(
                f"Unexpected action-only mask shape {tuple(mask_2d.shape)}, expected {(S_l + S_a, S_kv)}."
            )

        group_indices = self._group_global_indices(S_l=S_l, S_a=S_a, S_kv=S_kv, action_start=action_start, action_end=action_end)
        layer_values = []
        k = k_global[0]
        scale = 1.0 / math.sqrt(float(k.shape[-1]))
        for query_name in self.query_labels:
            query_tokens = q_rows[query_name]
            scores = torch.einsum("qhd,khd->qhk", query_tokens.to(torch.float32), k.to(torch.float32)) * scale
            rows = torch.as_tensor(query_mask_rows[query_name], device=mask_2d.device, dtype=torch.long)
            q_mask = mask_2d.index_select(0, rows).to(device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(~q_mask[:, None, :], torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1).mean(dim=1).mean(dim=0)
            layer_values.append([_valid_mean(attn, group_indices[label]) for label in self.group_labels])
        self.records[self.current_step][int(layer_idx)] = torch.tensor(layer_values, dtype=torch.float32)

    def _context_masks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.janus_input_ids is None or self.image_token_mask is None or self.attention_mask is None:
            raise RuntimeError("Recorder has no active token metadata.")
        ids = self.janus_input_ids
        image_mask = self.image_token_mask
        attention_mask = self.attention_mask
        if ids.ndim != 2 or image_mask.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError(
                "Expected janus_input_ids/image_token_mask/attention_mask shape [1, S], "
                f"got ids={tuple(ids.shape)}, image_mask={tuple(image_mask.shape)}, attention_mask={tuple(attention_mask.shape)}."
            )
        return ids[0], image_mask[0].to(torch.bool), attention_mask[0].to(torch.bool)

    def _group_global_indices(
        self,
        *,
        S_l: int,
        S_a: int,
        S_kv: int,
        action_start: int,
        action_end: int,
    ) -> dict[str, list[int]]:
        ids, image_mask, attention_mask = self._context_masks()
        S_v = int(S_kv) - int(S_l) - int(S_a)
        if S_v < 0:
            raise ValueError(f"Invalid KV layout: S_kv={S_kv}, S_l={S_l}, S_a={S_a}.")
        context_len = min(int(ids.shape[0]), S_l - self.total_latent_tokens)
        if context_len <= 0:
            raise ValueError(f"Invalid context_len={context_len} for S_l={S_l}.")
        image_positions = torch.nonzero(image_mask[:context_len], as_tuple=False).flatten().tolist()
        if len(image_positions) != self.expected_image_tokens:
            raise ValueError(
                f"Expected {self.expected_image_tokens} main image tokens, got {len(image_positions)}."
            )
        latent_start = S_l - int(self.total_latent_tokens)
        latent_indices = list(range(latent_start, S_l))
        final_colon_idx = latent_start - 1
        image_end_boundary_idx = min(max(image_positions) + 1, context_len - 1)
        text_indices = []
        for pos in range(image_end_boundary_idx + 1, min(final_colon_idx, context_len)):
            if not bool(attention_mask[pos].item()):
                continue
            token_id = int(ids[pos].item())
            if token_id in self.special_ids:
                continue
            text_indices.append(pos)

        latent_end_indices = [S_v + S_l] if S_a > 0 else []
        action_image_start = 2
        action_image_end = min(action_image_start + self.expected_image_tokens, S_a)
        action_image_indices = list(range(S_v + S_l + action_image_start, S_v + S_l + action_image_end))
        t_idx = action_start - 1
        t_indices = [S_v + S_l + t_idx] if 0 <= t_idx < S_a else []
        action_indices = [S_v + S_l + idx for idx in range(action_start, action_end)]

        groups = OrderedDict(
            [
                ("main_image", [S_v + idx for idx in image_positions]),
                ("text_after_image", [S_v + idx for idx in text_indices]),
                ("final_colon", [S_v + final_colon_idx]),
            ]
        )
        for latent_idx, token_idx in enumerate(latent_indices):
            groups[f"latent_{latent_idx}"] = [S_v + token_idx]
        groups.update(
            OrderedDict(
                [
                    ("latent_end", latent_end_indices),
                    ("action_image", action_image_indices),
                    ("t", t_indices),
                    ("action_token", action_indices),
                ]
            )
        )
        for label, indices in groups.items():
            bad = [idx for idx in indices if idx < 0 or idx >= S_kv]
            if bad:
                raise ValueError(f"Group {label} has indices outside KV length {S_kv}: {bad[:8]}.")
        return groups

    def _should_save_step(self, denoise_step: int) -> bool:
        if self.capture_mode == "all":
            return True
        if self.capture_mode == "first":
            return int(denoise_step) == 0
        return int(denoise_step) == int(self.metadata["action_denoise_steps"]) - 1

    def _query_dir(self) -> Path:
        task_name = _sanitize_filename(str(self.metadata["task_name"]))
        episode_label = f"{int(self.metadata['episode_index']):03d}"
        return Path(self.output_dir) / f"task={task_name}" / f"episode={episode_label}"

    def _output_stem(self, denoise_step: int, query_name: str) -> Path:
        task_name = _sanitize_filename(str(self.metadata["task_name"]))
        episode_label = f"{int(self.metadata['episode_index']):03d}"
        record_label = f"{int(self.metadata['record_index']):04d}"
        sample_label = f"{int(self.metadata['sample_index']):06d}"
        prompt = _sanitize_filename(str(self.metadata["prompt"]), max_len=60)
        filename = (
            f"task={task_name}--episode={episode_label}--record={record_label}"
            f"--sample={sample_label}--denoise={int(denoise_step):03d}"
            f"--query={query_name}--prompt={prompt}"
        )
        return self._query_dir() / filename

    def _save_query(self) -> list[str]:
        if not self.records:
            raise RuntimeError("No attention maps were captured for the active query.")
        query_dir = self._query_dir()
        os.makedirs(query_dir, exist_ok=True)
        saved_paths = []
        for denoise_step, layer_maps in sorted(self.records.items()):
            if not self._should_save_step(denoise_step):
                continue
            missing_layers = [idx for idx in range(self.num_layers) if idx not in layer_maps]
            if missing_layers:
                raise RuntimeError(
                    f"Missing attention maps for denoise_step={denoise_step}: layers={missing_layers[:8]}."
                )
            stacked = torch.stack([layer_maps[idx] for idx in range(self.num_layers)], dim=0).numpy()
            raw_payload = {
                "metadata": self.metadata,
                "denoise_step": int(denoise_step),
                "group_labels": self.group_labels,
                "query_labels": self.query_labels,
                "matrices": {
                    query_name: stacked[:, query_idx, :].tolist()
                    for query_idx, query_name in enumerate(self.query_labels)
                },
            }
            json_path = self._output_stem(denoise_step, "all_queries").with_suffix(".json")
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(raw_payload, f, indent=2, sort_keys=True)
            saved_paths.append(str(json_path))
            for query_idx, query_name in enumerate(self.query_labels):
                matrix = stacked[:, query_idx, :]
                image = self.render_heatmap(matrix, query_name=query_name, denoise_step=denoise_step)
                path = self._output_stem(denoise_step, query_name).with_suffix(".png")
                image.save(path)
                saved_paths.append(str(path))
        log_message(f"Saved {len(saved_paths)} global token attention artifacts to {query_dir}", self.log_file)
        return saved_paths

    def render_heatmap(self, matrix: np.ndarray, *, query_name: str, denoise_step: int) -> Image.Image:
        matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        n_layers, n_groups = matrix.shape
        if n_groups != len(self.group_labels):
            raise ValueError(f"Expected {len(self.group_labels)} groups, got {n_groups}.")
        label_w = 96
        header_h = 78
        footer_h = 34
        width = label_w + n_groups * self.cell_w
        height = header_h + n_layers * self.cell_h + footer_h
        canvas = Image.new("RGB", (width, height), (18, 22, 28))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        title = f"{query_name} attention, denoise={int(denoise_step):03d}"
        draw.text((8, 8), title, fill=(245, 248, 252), font=font)
        draw.text((8, 28), str(self.metadata.get("prompt", ""))[:110], fill=(190, 198, 210), font=font)

        for col_idx, label in enumerate(self.group_labels):
            x = label_w + col_idx * self.cell_w + 4
            draw.text((x, 54), label.replace("_", "\n"), fill=(235, 240, 248), font=font, spacing=1)

        min_v = float(matrix.min())
        max_v = float(matrix.max())
        if max_v > min_v:
            norm = (matrix - min_v) / (max_v - min_v)
        else:
            norm = np.zeros_like(matrix, dtype=np.float32)
        colors = _colorize(norm)
        for layer_idx in range(n_layers):
            y0 = header_h + layer_idx * self.cell_h
            draw.text((8, y0 + max(3, self.cell_h // 2 - 5)), f"layer_{layer_idx:02d}", fill=(235, 240, 248), font=font)
            for col_idx in range(n_groups):
                x0 = label_w + col_idx * self.cell_w
                color = tuple(int(x) for x in colors[layer_idx, col_idx])
                draw.rectangle([x0, y0, x0 + self.cell_w - 1, y0 + self.cell_h - 1], fill=color)
                value = float(matrix[layer_idx, col_idx])
                text = f"{value:.2e}" if value < 0.001 else f"{value:.4f}"
                draw.text((x0 + 5, y0 + max(3, self.cell_h // 2 - 5)), text, fill=(12, 16, 22), font=font)
        draw.text(
            (8, height - footer_h + 8),
            f"color normalized per image; raw group means in cells; min={min_v:.3e}, max={max_v:.3e}",
            fill=(190, 198, 210),
            font=font,
        )
        return canvas


def install_global_token_attention_recorder(
    model,
    cfg: TrainsetGlobalTokenAttnConfig,
    tokenizer,
    log_file=None,
) -> Optional[GlobalTokenAttentionRecorder]:
    output_dir = str(getattr(cfg, "attention_visualization_dir", "") or "").strip()
    if not output_dir:
        return None
    wrappers = getattr(model, "mot_attention_wrappers", None)
    if wrappers is None or len(wrappers) == 0:
        raise AttributeError("Model has no mot_attention_wrappers to instrument for attention visualization.")
    recorder = GlobalTokenAttentionRecorder(cfg, tokenizer=tokenizer, num_layers=len(wrappers), log_file=log_file)

    def make_patched_forward_action_only(layer_idx: int):
        def patched_forward_action_only(
            self,
            x_latent,
            x_action,
            latent_valid_mask=None,
            rotary_payload=None,
            action_value_token_count: int = 0,
        ):
            has_cached_video = self.cached_k_v is not None and self.cached_v_v is not None
            if self.decosmos:
                has_cached_video = False
            elif not has_cached_video:
                raise RuntimeError("forward_action_only requires cached video KV unless decosmos is enabled.")
            S_v = self.cached_k_v.shape[1] if has_cached_video else 0
            value_token_count = int(self.cached_value_token_count or 0) if has_cached_video else 0

            q_l, k_l, v_l = self.latent_bridge.get_branch_qkv(x_latent)
            S_l = q_l.shape[1]
            q_a, k_a, v_a = self.action_bridge.get_branch_qkv(x_action)
            S_a = q_a.shape[1]
            q_l, k_l, q_a, k_a = self._apply_branch_rotary(
                q_l=q_l,
                k_l=k_l,
                q_a=q_a,
                k_a=k_a,
                rotary_payload=rotary_payload,
            )

            q = torch.cat([q_l, q_a], dim=1)
            k_parts = []
            v_parts = []
            if has_cached_video:
                k_parts.append(self.cached_k_v)
                v_parts.append(self.cached_v_v)
            k_parts.extend([k_l, k_a])
            v_parts.extend([v_l, v_a])
            k = torch.cat(k_parts, dim=1)
            v = torch.cat(v_parts, dim=1)
            query_valid_mask = self._query_valid_mask(
                [latent_valid_mask if latent_valid_mask is not None else S_l, S_a],
                q.device,
                q.shape[0],
            )
            mask = self._build_action_only_mask(
                S_v,
                S_l,
                S_a,
                q.device,
                latent_valid_mask=latent_valid_mask,
                value_token_count=value_token_count,
                action_value_token_count=action_value_token_count,
            )

            recorder.capture_action_only_attention(
                layer_idx=layer_idx,
                q_l=q_l,
                k_l=k_l,
                q_a=q_a,
                k_global=k,
                action_only_mask=mask,
                action_value_token_count=action_value_token_count,
            )

            result = self._bridge_sdpa(q, k, v, attn_mask=mask, query_valid_mask=query_valid_mask)
            res_l = result[:, :S_l]
            res_a = result[:, S_l:]
            next_x_latent = self.latent_bridge.post_attention(
                x_latent,
                res_l,
                token_valid_mask=latent_valid_mask,
            )
            next_x_action = self.action_bridge.post_attention(x_action, res_a)
            return next_x_latent, next_x_action

        return patched_forward_action_only

    for layer_idx, wrapper in enumerate(wrappers):
        if hasattr(wrapper, "_global_token_attn_original_forward_action_only"):
            continue
        wrapper._global_token_attn_original_forward_action_only = wrapper.forward_action_only
        wrapper.forward_action_only = types.MethodType(make_patched_forward_action_only(layer_idx), wrapper)

    model._global_token_attention_recorder = recorder
    log_message(
        f"Installed global token attention recorder: dir={output_dir}, layers={len(wrappers)}, "
        f"groups={recorder.group_labels}, queries={recorder.query_labels}, "
        f"capture_mode={cfg.attention_visualization_capture_mode}",
        log_file,
    )
    return recorder


def run_selected_records(
    cfg: TrainsetGlobalTokenAttnConfig,
    dataset,
    selected: list[tuple[str, int, int]],
    model,
    recorder: GlobalTokenAttentionRecorder,
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
            record_index = int(sample.get("record_index", 0))
            prompt = str(sample["input_prompt"])
            log_message(
                f"[{ordinal}/{len(selected)}] task={task_name} episode={episode_index} "
                f"record={record_index} sample={sample_index} prompt={prompt!r}",
                log_file,
            )
            item = dataset[sample_index]
            batch = base.move_batch_to_device(dataset.collate_fn([item]), device, dtype)
            now_state = batch.get("now_state")
            cosmos_text_embeddings = batch.get("cosmos_text_embeddings")
            if cosmos_text_embeddings is not None:
                cosmos_text_embeddings = cosmos_text_embeddings.to(device=device, dtype=dtype)

            recorder.start_query(
                janus_input_ids=batch["janus_input_ids"].detach().cpu(),
                image_token_mask=batch["janus_images_seq_mask"].detach().cpu(),
                attention_mask=batch["attention_mask"].detach().cpu(),
                task_name=task_name,
                episode_index=episode_index,
                record_index=record_index,
                sample_index=sample_index,
                prompt=prompt,
                action_denoise_steps=int(cfg.action_denoise_steps),
            )
            try:
                with torch.inference_mode():
                    outputs = model.forward_flow_joint_inference(
                        janus_input_ids=batch["janus_input_ids"],
                        janus_pixel_values=batch["janus_pixel_values"],
                        janus_images_seq_mask=batch["janus_images_seq_mask"],
                        janus_images_emb_mask=batch["janus_images_emb_mask"],
                        first_frame=batch["videos"],
                        action_denoise_steps=int(cfg.action_denoise_steps),
                        cosmos_denoise_steps=int(cfg.cosmos_denoise_steps),
                        fps=torch.tensor([float(cfg.fps)], device=device, dtype=dtype),
                        action_self_causal_in_bridge=bool(cfg.action_self_causal_in_bridge),
                        num_latent_tokens=int(cfg.total_latent_tokens),
                        janus_left_pad_lens=batch["janus_left_pad_lens"],
                        janus_state_seq_mask=batch["janus_state_seq_mask"],
                        janus_action_pixel_values=batch["janus_action_pixel_values"],
                        janus_attention_mask=batch["attention_mask"].to(torch.bool),
                        now_state=now_state,
                        cosmos_text_embeddings=cosmos_text_embeddings,
                        return_value_prediction=False,
                        return_action_value_prediction=False,
                    )
                pred_video, pred_action = outputs[:2]
                del pred_video
                saved_paths = recorder.finish_query()
                summary = base.action_summary(pred_action)
                trace_file.write(
                    json.dumps(
                        {
                            "task": task_name,
                            "episode_index": int(episode_index),
                            "record_index": int(record_index),
                            "sample_index": int(sample_index),
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
    if int(cfg.total_latent_tokens) not in (1, 2):
        raise ValueError(
            f"Global token attention heatmap requires total_latent_tokens=1 or 2, got {cfg.total_latent_tokens}."
        )
    Path(str(cfg.attention_visualization_dir)).mkdir(parents=True, exist_ok=True)
    log_path = Path(str(cfg.attention_visualization_dir)) / "global_token_attn_heatmap.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_message(f"=== RLBench global token attention heatmap start {time.strftime('%Y-%m-%d %H:%M:%S')} ===", log_file)
        log_message(f"Config: {json.dumps(vars(cfg), sort_keys=True, default=str)}", log_file)
        if cfg.bash_hparams_path:
            log_message(f"Bash hparams: {cfg.bash_hparams_path}", log_file)
        set_seed(int(cfg.seed))
        model, processor, _action_tokenizer, _statistic = model_load(cfg, log_file)
        recorder = install_global_token_attention_recorder(model, cfg, processor.tokenizer, log_file)
        if recorder is None:
            raise RuntimeError("Global token attention recorder was not installed.")
        dataset = base.build_dataset(cfg, processor, log_file)
        selected = base.select_trainset_records(cfg, dataset, log_file)
        if not selected:
            raise ValueError("No trainset records selected.")
        log_message(f"Total selected records: {len(selected)}", log_file)
        run_selected_records(cfg, dataset, selected, model, recorder, log_file)
        log_message(f"Global token attention heatmaps written under {cfg.attention_visualization_dir}", log_file)
        log_message("=== RLBench global token attention heatmap finished ===", log_file)


if __name__ == "__main__":
    main()
