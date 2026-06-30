#!/usr/bin/env python3
"""Generate beta token-latent bridge attention visualizations for RLBench train records.

This script does not create an RLBench environment. It loads the train JSON,
reuses the beta training dataset input construction, runs checkpoint inference,
and records action-denoise bridge attention with runtime monkey patches.
"""

import argparse
import gc
import json
import logging
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
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

from janus.models import ActionTokenizer, VLChatProcessor  # noqa: E402
from models.cosmos_janus_cot import CosmosJanusMoT3Expert, normalize_bridge_pos_scheme  # noqa: E402
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint  # noqa: E402
from utils.cosmos_text_cache import CosmosQwenTextEmbedder, CosmosTextEmbeddingCache  # noqa: E402
from scripts.train_cot_rlbench_keyframe import VLACotDataset, resolve_rlbench_episode_key  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BetaRLBenchTrainsetAttnVis")


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


def resolve_latent_token_args(cfg: "TrainsetAttnVisConfig") -> None:
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


def build_latent_labels(total_latent_tokens: int) -> list[str]:
    return [f"latent_{idx}" for idx in range(int(total_latent_tokens))]


@dataclass
class TrainsetAttnVisConfig:
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
    attention_visualization_tile_size: int = 256
    attention_visualization_alpha: float = 0.45
    attention_visualization_capture_mode: str = "all"
    attention_visualization_top_ratio: Optional[float] = None
    attention_visualization_top_softness: float = 0.05
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
    latent_hidden_sim_loss_weight: float = 1.0
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


def parse_args() -> TrainsetAttnVisConfig:
    parser = argparse.ArgumentParser()
    for field_def in fields(TrainsetAttnVisConfig):
        default = field_def.default
        if default is None:
            arg_type = str
        else:
            arg_type = str if isinstance(default, bool) else type(default)
        parser.add_argument(f"--{field_def.name}", default=default, type=arg_type)
    ns = parser.parse_args()
    cfg = TrainsetAttnVisConfig(**vars(ns))
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
    top_ratio = cfg.attention_visualization_top_ratio
    if top_ratio is None or (isinstance(top_ratio, str) and top_ratio.strip() == ""):
        cfg.attention_visualization_top_ratio = None
    else:
        try:
            parsed_top_ratio = float(top_ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "--attention_visualization_top_ratio must be empty or a float in [0, 1], "
                f"got {top_ratio!r}."
            ) from exc
        if not 0.0 <= parsed_top_ratio <= 1.0:
            raise ValueError(
                "--attention_visualization_top_ratio must be empty or a float in [0, 1], "
                f"got {parsed_top_ratio}."
            )
        cfg.attention_visualization_top_ratio = parsed_top_ratio
    cfg.attention_visualization_top_softness = float(cfg.attention_visualization_top_softness)
    if not 0.0 <= cfg.attention_visualization_top_softness <= 1.0:
        raise ValueError(
            "--attention_visualization_top_softness must be a float in [0, 1], "
            f"got {cfg.attention_visualization_top_softness}."
        )
    resolve_latent_token_args(cfg)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_message(message: str, log_file=None) -> None:
    logger.info(message)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


class PrintAccelerator:
    def __init__(self, log_file=None):
        self.log_file = log_file

    def print(self, *args, **kwargs) -> None:
        sep = kwargs.get("sep", " ")
        log_message(sep.join(str(arg) for arg in args), self.log_file)


def get_video_latent_num_frames(video_tokenizer, pixel_frames: int) -> int:
    if video_tokenizer is not None and callable(getattr(video_tokenizer, "get_latent_num_frames", None)):
        return int(video_tokenizer.get_latent_num_frames(int(pixel_frames)))
    return 1 + (int(pixel_frames) - 1) // 4


def resolve_video_condition_config(cfg: TrainsetAttnVisConfig, video_tokenizer=None) -> None:
    cfg.video_frames = int(cfg.video_frames)
    cfg.num_cond_input_frames = int(cfg.num_cond_input_frames)
    cfg.num_cond_latent_frames = get_video_latent_num_frames(video_tokenizer, cfg.num_cond_input_frames)
    cfg.total_video_latent_frames = get_video_latent_num_frames(video_tokenizer, cfg.video_frames)


def resolve_checkpoint_paths(pretrained_checkpoint: str) -> tuple[str, str]:
    if not pretrained_checkpoint:
        raise ValueError("--pretrained_checkpoint is required.")
    if pretrained_checkpoint.endswith(".pt"):
        return pretrained_checkpoint, os.path.dirname(pretrained_checkpoint)
    for name in ("cosmos_janus_mot.pt", "cosmos_janus_mot3.pt", "mot_action_weights.pt"):
        candidate = os.path.join(pretrained_checkpoint, name)
        if os.path.exists(candidate):
            return candidate, pretrained_checkpoint
    return os.path.join(pretrained_checkpoint, "cosmos_janus_mot.pt"), pretrained_checkpoint


def infer_action_intermediate_size_from_state_dict(state_dict: dict[str, Any]) -> Optional[int]:
    for key, value in state_dict.items():
        if key.endswith("action_bridge.mlp.gate_proj.weight") and hasattr(value, "shape") and len(value.shape) == 2:
            return int(value.shape[0])
    return None


def maybe_infer_action_intermediate_size(cfg: TrainsetAttnVisConfig, state_dict: dict[str, Any]) -> None:
    inferred = infer_action_intermediate_size_from_state_dict(state_dict)
    if inferred is None:
        return
    if int(cfg.action_intermediate_size or 0) > 0 and int(cfg.action_intermediate_size) != inferred:
        raise ValueError(f"action_intermediate_size={cfg.action_intermediate_size} but checkpoint requires {inferred}.")
    cfg.action_intermediate_size = inferred


def ensure_janus_tokenizer_alignment(janus_model, tokenizer) -> None:
    target_vocab = int(len(tokenizer))
    language_model = janus_model.language_model
    embed = language_model.get_input_embeddings()
    current_vocab = int(embed.weight.shape[0])
    lm_head = getattr(language_model, "lm_head", None)
    lm_head_vocab = None
    if lm_head is not None and getattr(lm_head, "weight", None) is not None:
        lm_head_vocab = int(lm_head.weight.shape[0])
    covered_vocab = current_vocab if lm_head_vocab is None else min(current_vocab, lm_head_vocab)
    if covered_vocab >= target_vocab:
        logger.info(
            "Janus tokenizer covered by embedding/lm_head vocab: tokenizer=%s, embedding=%s, lm_head=%s",
            target_vocab,
            current_vocab,
            lm_head_vocab if lm_head_vocab is not None else "N/A",
        )
        return
    logger.info("Resizing Janus token embeddings for tokenizer alignment: %s -> %s", current_vocab, target_vocab)
    language_model.resize_token_embeddings(target_vocab)
    if hasattr(janus_model.config, "vocab_size"):
        janus_model.config.vocab_size = target_vocab
    if hasattr(janus_model.config, "language_config"):
        janus_model.config.language_config.vocab_size = target_vocab
    if hasattr(language_model, "config"):
        language_model.config.vocab_size = target_vocab


def validate_checkpoint_vocab_size(state_dict: dict[str, Any], tokenizer) -> None:
    target_vocab = int(len(tokenizer))
    vocab_keys = [
        "janus.language_model.model.embed_tokens.weight",
        "janus.language_model.lm_head.weight",
    ]
    mismatches = []
    for key in vocab_keys:
        value = state_dict.get(key)
        if value is not None and hasattr(value, "shape") and int(value.shape[0]) < target_vocab:
            mismatches.append(f"{key}: checkpoint={int(value.shape[0])}, tokenizer={target_vocab}")
    if mismatches:
        raise ValueError(
            "Checkpoint vocab size is incompatible with this RLBench beta tokenizer. "
            "Use a checkpoint trained after adding </NONE> and </MOVE><PICK> as special tokens. "
            f"Mismatches: {', '.join(mismatches)}"
        )


def model_load(cfg: TrainsetAttnVisConfig, log_file=None):
    cfg.bridge_pos_scheme = normalize_bridge_pos_scheme(cfg.bridge_pos_scheme)
    processor = VLChatProcessor.from_pretrained(cfg.model_path, trust_remote_code=True)
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
    resolve_video_condition_config(cfg, cosmos_wrapper.tokenizer)

    ckpt_path, base_dir = resolve_checkpoint_paths(cfg.pretrained_checkpoint)
    log_message(f"Loading fine-tuned state dict from {ckpt_path}", log_file)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    maybe_infer_action_intermediate_size(cfg, state_dict)
    ensure_janus_tokenizer_alignment(janus_model, tokenizer)
    validate_checkpoint_vocab_size(state_dict, tokenizer)

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
    log_message(f"Model ready on device={device}; train statistics={stats_path}", log_file)
    return model, processor, action_tokenizer, statistic


def _sanitize_filename(value: str, max_len: int = 120) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_")
    return (cleaned or "unknown")[:max_len]


def _colorize_heatmap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 * values - 0.2, 0.0, 1.0)
    green = np.clip(1.5 - 3.0 * np.abs(values - 0.5), 0.0, 1.0)
    blue = np.clip(1.2 - 1.5 * values, 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def _smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def _overlay_heatmap(
    base: Image.Image,
    heat: np.ndarray,
    tile_size: int,
    alpha: float,
    top_ratio: Optional[float] = None,
    top_softness: float = 0.05,
) -> Image.Image:
    heat = np.asarray(heat, dtype=np.float32)
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)
    heat_min = float(heat.min())
    heat_max = float(heat.max())
    if heat_max > heat_min:
        heat = (heat - heat_min) / (heat_max - heat_min)
    else:
        heat = np.zeros_like(heat, dtype=np.float32)

    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    base = base.convert("RGB").resize((tile_size, tile_size), resample=resample)
    if top_ratio is None:
        heat_img = Image.fromarray(_colorize_heatmap(heat), mode="RGB").resize((tile_size, tile_size), resample=resample)
    else:
        heat = np.asarray(
            Image.fromarray(heat.astype(np.float32)).resize((tile_size, tile_size), resample=resample),
            dtype=np.float32,
        )
        ratio = float(top_ratio)
        if ratio <= 0.0:
            heat = np.zeros_like(heat, dtype=np.float32)
        elif ratio < 1.0:
            flat = heat.reshape(-1)
            keep_count = int(math.ceil(flat.size * ratio))
            keep_count = max(0, min(int(flat.size), keep_count))
            softness = float(np.clip(top_softness, 0.0, 1.0))
            if keep_count <= 0:
                heat = np.zeros_like(heat, dtype=np.float32)
            elif softness <= 0.0:
                mask = np.zeros(flat.shape, dtype=bool)
                keep_indices = np.argpartition(flat, -keep_count)[-keep_count:]
                mask[keep_indices] = True
                heat = np.where(mask.reshape(heat.shape), heat, 0.0).astype(np.float32)
            else:
                cutoff = float(np.partition(flat, flat.size - keep_count)[flat.size - keep_count])
                weights = _smoothstep((heat - (cutoff - softness)) / (2.0 * softness))
                heat = (heat * weights).astype(np.float32)
        heat_img = Image.fromarray(_colorize_heatmap(heat), mode="RGB")
    return Image.blend(base, heat_img, float(np.clip(alpha, 0.0, 1.0)))


class AttentionMapRecorder:
    """Collect and render action-denoise bridge attention maps for one trainset record."""

    def __init__(self, cfg: TrainsetAttnVisConfig, num_layers: int, log_file=None):
        self.cfg = cfg
        self.log_file = log_file
        self.output_dir = str(getattr(cfg, "attention_visualization_dir", "") or "").strip()
        self.action_chunk = int(getattr(cfg, "action_chunk", 1))
        self.total_latent_tokens = int(getattr(cfg, "total_latent_tokens", 1))
        if self.total_latent_tokens not in (1, 2):
            raise ValueError(f"total_latent_tokens must be 1 or 2, got {self.total_latent_tokens}.")
        self.latent_labels = build_latent_labels(self.total_latent_tokens)
        self.num_layers = int(num_layers)
        self.expected_image_tokens = 576
        self.expected_grid_side = 24
        self.alpha = float(getattr(cfg, "attention_visualization_alpha", 0.45) or 0.45)
        self.tile_size = max(16, int(getattr(cfg, "attention_visualization_tile_size", 256) or 256))
        self.top_ratio = getattr(cfg, "attention_visualization_top_ratio", None)
        self.top_softness = float(getattr(cfg, "attention_visualization_top_softness", 0.05) or 0.0)
        self.capture_mode = str(getattr(cfg, "attention_visualization_capture_mode", "all") or "all").lower()
        if self.capture_mode not in {"all", "first", "last"}:
            raise ValueError(f"attention_visualization_capture_mode must be all/first/last, got {self.capture_mode!r}.")
        if self.action_chunk <= 0:
            raise ValueError(f"action_chunk must be positive, got {self.action_chunk}.")
        self.active = False
        self.image_token_mask = None
        self.base_image = None
        self.metadata = {}
        self.current_step = -1
        self.records: dict[int, dict[int, torch.Tensor]] = {}

    def start_query(
        self,
        *,
        image_token_mask: torch.Tensor,
        base_image: Image.Image,
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
        self.image_token_mask = image_token_mask.detach()
        self.base_image = base_image.copy().convert("RGB")
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

    def finish_query(self) -> list[str]:
        if not self.active:
            return []
        try:
            return self._save_query()
        finally:
            self.active = False
            self.image_token_mask = None
            self.base_image = None
            self.metadata = {}
            self.current_step = -1
            self.records = {}

    def discard_query(self) -> None:
        self.active = False
        self.image_token_mask = None
        self.base_image = None
        self.metadata = {}
        self.current_step = -1
        self.records = {}

    def capture_layer_attention(
        self,
        *,
        layer_idx: int,
        q_l: torch.Tensor,
        k_l: torch.Tensor,
        q_a: torch.Tensor,
        k_a: torch.Tensor,
        action_value_token_count: int = 0,
    ) -> None:
        if not self.active:
            return
        if q_l.shape[0] != 1 or k_l.shape[0] != 1 or q_a.shape[0] != 1 or k_a.shape[0] != 1:
            raise ValueError(
                "Attention visualization currently expects batch size 1, "
                f"got q_l={tuple(q_l.shape)}, k_l={tuple(k_l.shape)}, "
                f"q_a={tuple(q_a.shape)}, k_a={tuple(k_a.shape)}."
            )
        if int(layer_idx) == 0:
            self.current_step += 1
            self.records[self.current_step] = {}
        if self.current_step < 0:
            raise RuntimeError("Attention recorder saw a nonzero layer before layer 0.")

        latent_start = int(q_l.shape[1]) - self.total_latent_tokens
        latent_end = int(q_l.shape[1])
        final_colon_idx = latent_start - 1
        if final_colon_idx < 0 or latent_start < 0:
            raise ValueError(
                "Could not locate final_colon and latent tokens: "
                f"S_latent={q_l.shape[1]}, total_latent_tokens={self.total_latent_tokens}."
            )
        image_positions = self._image_positions(device=k_l.device, latent_seq_len=k_l.shape[1])
        action_value_token_count = int(action_value_token_count or 0)
        action_start = int(q_a.shape[1]) - action_value_token_count - self.action_chunk
        action_end = action_start + self.action_chunk
        if action_start < 0 or action_end > int(q_a.shape[1]):
            raise ValueError(
                "Could not locate action chunk q tokens: "
                f"S_action={q_a.shape[1]}, action_chunk={self.action_chunk}, "
                f"action_value_token_count={action_value_token_count}."
            )

        image_keys = k_l[0, image_positions, :, :]
        main_image_query_tokens = torch.cat(
            [
                q_l[0, final_colon_idx:final_colon_idx + 1, :, :],
                q_l[0, latent_start:latent_end, :, :],
                q_a[0, action_start:action_end, :, :],
            ],
            dim=0,
        )
        scale = 1.0 / math.sqrt(float(main_image_query_tokens.shape[-1]))
        main_scores = (
            torch.einsum("qhd,khd->qhk", main_image_query_tokens.to(torch.float32), image_keys.to(torch.float32))
            * scale
        )
        main_attn = torch.softmax(main_scores, dim=-1).mean(dim=1)

        action_image_start = 2
        action_image_end = action_image_start + self.expected_image_tokens
        if action_image_end > int(k_a.shape[1]):
            raise ValueError(
                "Could not locate action-branch image key tokens: "
                f"S_action={k_a.shape[1]}, expected range=[{action_image_start}, {action_image_end})."
            )
        action_image_keys = k_a[0, action_image_start:action_image_end, :, :]
        action_query_tokens = q_a[0, action_start:action_end, :, :]
        action_scores = (
            torch.einsum("qhd,khd->qhk", action_query_tokens.to(torch.float32), action_image_keys.to(torch.float32))
            * scale
        )
        action_image_attn = torch.softmax(action_scores, dim=-1).mean(dim=1)

        attn = torch.cat([main_attn, action_image_attn], dim=0)
        expected_queries = 1 + self.total_latent_tokens + self.action_chunk * 2
        if int(attn.shape[0]) != expected_queries:
            raise ValueError(f"Expected {expected_queries} query maps, got {attn.shape[0]}.")
        self.records[self.current_step][int(layer_idx)] = attn.detach().cpu()

    def _image_positions(self, *, device: torch.device, latent_seq_len: int) -> torch.Tensor:
        if self.image_token_mask is None:
            raise RuntimeError("Attention recorder has no image token mask for the active query.")
        mask = self.image_token_mask.to(device=device, dtype=torch.bool)
        if mask.ndim != 2 or int(mask.shape[0]) != 1:
            raise ValueError(f"Expected image token mask shape [1, S], got {tuple(mask.shape)}.")
        positions = torch.nonzero(mask[0], as_tuple=False).flatten()
        if int(positions.numel()) != self.expected_image_tokens:
            meta = self.metadata
            raise ValueError(
                f"Expected exactly {self.expected_image_tokens} Janus primary-image key tokens, "
                f"got {int(positions.numel())} for task={meta.get('task_name')}, "
                f"episode={meta.get('episode_index')}, record={meta.get('record_index')}."
            )
        if int(positions[-1].item()) >= int(latent_seq_len):
            raise ValueError(
                "Image token positions do not fit latent KV sequence: "
                f"max_image_pos={int(positions[-1].item())}, latent_seq_len={latent_seq_len}."
            )
        return positions

    def _should_save_step(self, denoise_step: int) -> bool:
        if self.capture_mode == "all":
            return True
        if self.capture_mode == "first":
            return int(denoise_step) == 0
        return int(denoise_step) == int(self.metadata["action_denoise_steps"]) - 1

    def _save_query(self) -> list[str]:
        if self.base_image is None:
            raise RuntimeError("Attention recorder has no base image for the active query.")
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
                    f"Missing attention maps for denoise_step={denoise_step}: "
                    f"layers={missing_layers[:8]}."
                )
            overview = self.render_overview(layer_maps)
            path = self._output_path(denoise_step)
            overview.save(path)
            saved_paths.append(str(path))
        log_message(f"Saved {len(saved_paths)} attention overview PNGs to {query_dir}", self.log_file)
        return saved_paths

    def _query_dir(self) -> Path:
        task_name = _sanitize_filename(str(self.metadata["task_name"]))
        episode_label = f"{int(self.metadata['episode_index']):03d}"
        return Path(self.output_dir) / f"task={task_name}" / f"episode={episode_label}"

    def _output_path(self, denoise_step: int) -> Path:
        task_name = _sanitize_filename(str(self.metadata["task_name"]))
        episode_label = f"{int(self.metadata['episode_index']):03d}"
        record_label = f"{int(self.metadata['record_index']):04d}"
        sample_label = f"{int(self.metadata['sample_index']):06d}"
        prompt = _sanitize_filename(str(self.metadata["prompt"]), max_len=60)
        filename = (
            f"task={task_name}--episode={episode_label}--record={record_label}"
            f"--sample={sample_label}--denoise={int(denoise_step):03d}--prompt={prompt}.png"
        )
        return self._query_dir() / filename

    def render_overview(self, layer_maps: dict[int, torch.Tensor]) -> Image.Image:
        row_label_w = 86
        col_label_h = 42
        num_cols = 1 + self.total_latent_tokens + self.action_chunk * 2
        canvas = Image.new(
            "RGB",
            (row_label_w + num_cols * self.tile_size, col_label_h + self.num_layers * self.tile_size),
            (18, 22, 28),
        )
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        col_labels = (
            ["final_colon"]
            + self.latent_labels
            + [f"action_{idx:02d}" for idx in range(self.action_chunk)]
            + [f"action_{idx:02d}_fast_image" for idx in range(self.action_chunk)]
        )
        for col_idx, label in enumerate(col_labels):
            x = row_label_w + col_idx * self.tile_size + 5
            draw.text((x, 13), label, fill=(245, 248, 252), font=font)

        for layer_idx in range(self.num_layers):
            y = col_label_h + layer_idx * self.tile_size
            draw.text(
                (8, y + max(4, self.tile_size // 2 - 6)),
                f"layer_{layer_idx:02d}",
                fill=(245, 248, 252),
                font=font,
            )
            maps = layer_maps[layer_idx]
            expected_shape = (num_cols, self.expected_image_tokens)
            if tuple(maps.shape) != expected_shape:
                raise ValueError(f"Layer {layer_idx} attention map shape must be {expected_shape}, got {tuple(maps.shape)}.")
            for col_idx in range(num_cols):
                heat = maps[col_idx].reshape(self.expected_grid_side, self.expected_grid_side).numpy()
                overlay = _overlay_heatmap(
                    self.base_image,
                    heat,
                    self.tile_size,
                    self.alpha,
                    self.top_ratio,
                    self.top_softness,
                )
                canvas.paste(overlay, (row_label_w + col_idx * self.tile_size, y))
        return canvas


def install_attention_map_recorder(model, cfg: TrainsetAttnVisConfig, log_file=None) -> Optional[AttentionMapRecorder]:
    output_dir = str(getattr(cfg, "attention_visualization_dir", "") or "").strip()
    if not output_dir:
        return None
    wrappers = getattr(model, "mot_attention_wrappers", None)
    if wrappers is None or len(wrappers) == 0:
        raise AttributeError("Model has no mot_attention_wrappers to instrument for attention visualization.")
    recorder = AttentionMapRecorder(cfg, num_layers=len(wrappers), log_file=log_file)

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

            recorder.capture_layer_attention(
                layer_idx=layer_idx,
                q_l=q_l,
                k_l=k_l,
                q_a=q_a,
                k_a=k_a,
                action_value_token_count=action_value_token_count,
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
            result = self._bridge_sdpa(q, k, v, attn_mask=mask, query_valid_mask=query_valid_mask)
            res_l = result[:, :S_l]
            res_a = result[:, S_l:]
            next_x_latent = self.latent_bridge.post_attention(x_latent, res_l, token_valid_mask=latent_valid_mask)
            next_x_action = self.action_bridge.post_attention(x_action, res_a)
            return next_x_latent, next_x_action

        return patched_forward_action_only

    for layer_idx, wrapper in enumerate(wrappers):
        if hasattr(wrapper, "_attn_vis_original_forward_action_only"):
            continue
        wrapper._attn_vis_original_forward_action_only = wrapper.forward_action_only
        wrapper.forward_action_only = types.MethodType(make_patched_forward_action_only(layer_idx), wrapper)

    model._attention_map_recorder = recorder
    log_message(
        f"Installed attention recorder: dir={output_dir}, layers={len(wrappers)}, "
        f"action_chunk={cfg.action_chunk}, capture_mode={cfg.attention_visualization_capture_mode}, "
        f"top_ratio={cfg.attention_visualization_top_ratio}, "
        f"top_softness={cfg.attention_visualization_top_softness}",
        log_file,
    )
    return recorder


def build_dataset(cfg: TrainsetAttnVisConfig, processor, log_file=None) -> VLACotDataset:
    return VLACotDataset(cfg, processor, PrintAccelerator(log_file))


def select_trainset_records(cfg: TrainsetAttnVisConfig, dataset: VLACotDataset, log_file=None) -> list[tuple[str, int, int]]:
    requested_tasks = [task.strip() for task in str(cfg.task_names or "").split(",") if task.strip()]
    requested_task_set = set(requested_tasks)
    grouped: OrderedDict[str, OrderedDict[int, list[int]]] = OrderedDict()
    for sample_index, sample in enumerate(dataset.data):
        task_name, episode_index = resolve_rlbench_episode_key(sample)
        if requested_task_set and task_name not in requested_task_set:
            continue
        grouped.setdefault(task_name, OrderedDict()).setdefault(int(episode_index), []).append(sample_index)

    if requested_task_set:
        missing = sorted(requested_task_set - set(grouped))
        if missing:
            raise ValueError(f"Requested task_names not found in train JSON: {missing}")

    selected: list[tuple[str, int, int]] = []
    per_task = max(1, int(cfg.num_trajectories_per_task))
    for task_name in sorted(grouped):
        episodes = sorted(grouped[task_name].items(), key=lambda item: item[0])[:per_task]
        log_message(
            f"Selected task={task_name}: episodes={[episode for episode, _ in episodes]} "
            f"(num_trajectories_per_task={per_task})",
            log_file,
        )
        for episode_index, sample_indices in episodes:
            ordered_indices = sorted(sample_indices, key=lambda idx: int(dataset.data[idx].get("record_index", idx)))
            if int(cfg.max_records_per_episode) > 0:
                ordered_indices = ordered_indices[: int(cfg.max_records_per_episode)]
            for sample_index in ordered_indices:
                selected.append((task_name, int(episode_index), int(sample_index)))
                if int(cfg.max_total_records) > 0 and len(selected) >= int(cfg.max_total_records):
                    log_message(f"Reached max_total_records={cfg.max_total_records}; stopping selection.", log_file)
                    return selected
    return selected


def move_batch_to_device(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue
        if key in {"janus_pixel_values", "janus_action_pixel_values", "videos", "cosmos_text_embeddings"}:
            moved[key] = value.to(device=device, dtype=dtype, non_blocking=True)
        elif key == "now_state" and value.is_floating_point():
            moved[key] = value.to(device=device, dtype=torch.float32, non_blocking=True)
        else:
            moved[key] = value.to(device=device, non_blocking=True)
    return moved


def action_summary(pred_action: torch.Tensor) -> dict[str, Any]:
    action = pred_action.detach().to(torch.float32).cpu()
    return {
        "shape": list(action.shape),
        "mean": round(float(action.mean().item()), 6),
        "min": round(float(action.min().item()), 6),
        "max": round(float(action.max().item()), 6),
    }


def run_selected_records(
    cfg: TrainsetAttnVisConfig,
    dataset: VLACotDataset,
    selected: list[tuple[str, int, int]],
    model,
    recorder: AttentionMapRecorder,
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
            batch = move_batch_to_device(dataset.collate_fn([item]), device, dtype)
            base_image = dataset._load_image_pil(sample["front_pic"])
            now_state = batch.get("now_state")
            cosmos_text_embeddings = batch.get("cosmos_text_embeddings")
            if cosmos_text_embeddings is not None:
                cosmos_text_embeddings = cosmos_text_embeddings.to(device=device, dtype=dtype)

            recorder.start_query(
                image_token_mask=batch["janus_images_seq_mask"].detach().cpu(),
                base_image=base_image,
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
                summary = action_summary(pred_action)
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
            f"Beta token-latent visualization requires total_latent_tokens=1 or 2, got {cfg.total_latent_tokens}."
        )
    Path(str(cfg.attention_visualization_dir)).mkdir(parents=True, exist_ok=True)
    log_path = Path(str(cfg.attention_visualization_dir)) / "trainset_attn_vis.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_message(f"=== Beta RLBench trainset attention visualization start {time.strftime('%Y-%m-%d %H:%M:%S')} ===", log_file)
        log_message(f"Config: {json.dumps(vars(cfg), sort_keys=True, default=str)}", log_file)
        if cfg.bash_hparams_path:
            log_message(f"Bash hparams: {cfg.bash_hparams_path}", log_file)
        set_seed(int(cfg.seed))
        model, processor, _action_tokenizer, _statistic = model_load(cfg, log_file)
        recorder = install_attention_map_recorder(model, cfg, log_file)
        if recorder is None:
            raise RuntimeError("Attention recorder was not installed.")
        dataset = build_dataset(cfg, processor, log_file)
        selected = select_trainset_records(cfg, dataset, log_file)
        if not selected:
            raise ValueError("No trainset records selected.")
        log_message(f"Total selected records: {len(selected)}", log_file)
        run_selected_records(cfg, dataset, selected, model, recorder, log_file)
        log_message(f"Attention visualizations written under {cfg.attention_visualization_dir}", log_file)
        log_message("=== Beta RLBench trainset attention visualization finished ===", log_file)


if __name__ == "__main__":
    main()
