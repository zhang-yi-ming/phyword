#!/usr/bin/env python3
"""Visualize T-Rex 2-MoT action-spatial attention on RLBench train records."""

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
from transformers import AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint  # noqa: E402
from models.cosmos_janus_action_spatial import (  # noqa: E402
    CosmosJanusActionSpatialMoT2Expert,
    normalize_bridge_pos_scheme,
)
from models.trex_action_backend import TrexActionModel, resolve_trex_checkpoint_path  # noqa: E402
from scripts.train_mot2_trex_rlbench_keyframe import (  # noqa: E402
    VLACotDataset,
    build_front_pic_path,
    clipped_keyframe_indices,
    parse_front_pic_index,
    resolve_rlbench_episode_key,
)
from experiments.robot.rlbench.run_rlbench_eval_keyframe_mot2 import (  # noqa: E402
    DEFAULT_SPECIAL_TOKEN_VOCAB,
    load_special_token_vocab,
    resolve_special_token_init_ids,
    validate_special_token_checkpoint_rows,
)
from utils.cosmos_text_cache import CosmosQwenTextEmbedder, CosmosTextEmbeddingCache  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MoT2TrexRLBenchTrainsetAttnVis")


SPATIAL_TOKEN_MODE_TO_FIELDS = {
    "v": ("gtlatent",),
    "n": ("gtlatent2",),
    "vn": ("gtlatent", "gtlatent2"),
}
SPATIAL_TOKEN_MODE_ALIASES = {
    "1": "v",
    "2": "vn",
}


def normalize_spatial_token_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    mode = SPATIAL_TOKEN_MODE_ALIASES.get(mode, mode)
    if mode not in SPATIAL_TOKEN_MODE_TO_FIELDS:
        valid = ", ".join(sorted([*SPATIAL_TOKEN_MODE_TO_FIELDS.keys(), *SPATIAL_TOKEN_MODE_ALIASES.keys()]))
        raise ValueError(f"spatial token mode must be one of {valid}, got {value!r}.")
    return mode


def resolve_spatial_token_args(cfg: "TrainsetAttnVisConfig") -> None:
    mode_arg = str(getattr(cfg, "latent_token_mode", "") or "").strip()
    count_or_mode_arg = str(getattr(cfg, "total_latent_tokens", "") or "").strip()

    if mode_arg:
        mode = normalize_spatial_token_mode(mode_arg)
        fields_for_mode = list(SPATIAL_TOKEN_MODE_TO_FIELDS[mode])
        if count_or_mode_arg:
            try:
                token_count = int(count_or_mode_arg)
            except ValueError:
                token_mode = normalize_spatial_token_mode(count_or_mode_arg)
                if token_mode != mode:
                    raise ValueError(
                        "Conflicting spatial token settings: "
                        f"--latent_token_mode={mode_arg!r} but "
                        f"--total_latent_tokens={count_or_mode_arg!r}."
                    )
                token_count = len(SPATIAL_TOKEN_MODE_TO_FIELDS[token_mode])
            if token_count != len(fields_for_mode):
                raise ValueError(
                    "total_latent_tokens count does not match latent_token_mode: "
                    f"mode={mode!r} expects {len(fields_for_mode)}, got {token_count}."
                )
    else:
        mode = normalize_spatial_token_mode(count_or_mode_arg or "1")
        fields_for_mode = list(SPATIAL_TOKEN_MODE_TO_FIELDS[mode])

    cfg.latent_token_mode = mode
    cfg.latent_token_fields = fields_for_mode
    cfg.total_latent_tokens = len(fields_for_mode)
    cfg.total_spatial_tokens = cfg.total_latent_tokens


@dataclass
class TrainsetAttnVisConfig:
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
    data_path: str = "/mnt/nas/zhangyiming/database/rlbench/train/json/train_action_chunk1_sumpos_lastrot.json"
    data_root: str = ""
    attention_visualization_dir: str = ""
    attention_visualization_tile_size: int = 256
    attention_visualization_alpha: float = 0.45
    attention_visualization_capture_mode: str = "last"
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
    video_h: int = 256
    video_w: int = 256
    video_frames: int = 9
    num_cond_input_frames: int = 5
    action_dim: int = 7
    action_chunk: int = 1
    robot_state: int = 0
    state_placeholder_tokens: int = 1
    state_dim: int = 7
    state_encoding_mode: str = "mlp"
    total_latent_tokens: str = ""
    latent_token_mode: str = ""
    special_token_vocab: str = ",".join(DEFAULT_SPECIAL_TOKEN_VOCAB)
    img_latents_per_future: int = 0
    state_latents_per_future: int = 0
    num_future_frames: int = 0
    future_frame_stride: int = 1
    use_latent_hidden_sim_loss: int = 1
    latent_hidden_sim_loss_mode: str = "siglip"
    cosmos_self_only_bridge: bool = False
    decosmos: bool = False
    bridge_pos_scheme: str = "mrope"
    action_use_latent_prefix: bool = True
    action_self_causal_in_bridge: bool = True
    qwen3vl2b_model_path: str = "/mnt/amlfs-07/shared/physicalword/ckpt/pretraine/Qwen3-VL-2B-Instruct"
    right_single_attn_position: str = "last4"
    action_denoise_steps: int = 10
    cosmos_denoise_steps: int = 2
    fps: float = 10.0
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
        arg_type = str if default is None or isinstance(default, bool) else type(default)
        parser.add_argument(f"--{field_def.name}", default=default, type=arg_type)
    ns = parser.parse_args()
    cfg = TrainsetAttnVisConfig(**vars(ns))
    for name in (
        "cosmos_self_only_bridge",
        "decosmos",
        "action_use_latent_prefix",
        "action_self_causal_in_bridge",
    ):
        setattr(cfg, name, coerce_bool(getattr(cfg, name)))
    top_ratio = cfg.attention_visualization_top_ratio
    if top_ratio is None or (isinstance(top_ratio, str) and top_ratio.strip() == ""):
        cfg.attention_visualization_top_ratio = None
    else:
        parsed_top_ratio = float(top_ratio)
        if not 0.0 <= parsed_top_ratio <= 1.0:
            raise ValueError(f"--attention_visualization_top_ratio must be in [0, 1], got {parsed_top_ratio}.")
        cfg.attention_visualization_top_ratio = parsed_top_ratio
    cfg.attention_visualization_top_softness = float(cfg.attention_visualization_top_softness)
    if not 0.0 <= cfg.attention_visualization_top_softness <= 1.0:
        raise ValueError(
            "--attention_visualization_top_softness must be a float in [0, 1], "
            f"got {cfg.attention_visualization_top_softness}."
        )
    resolve_spatial_token_args(cfg)
    cfg.right_single_attn_position = str(getattr(cfg, "right_single_attn_position", "last4") or "last4").lower()
    if cfg.right_single_attn_position not in ("first4", "last4"):
        raise ValueError("right_single_attn_position must be 'first4' or 'last4'.")
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


def checkpoint_has_processor_files(checkpoint_dir: str) -> bool:
    if not checkpoint_dir:
        return False
    return any(
        os.path.exists(os.path.join(checkpoint_dir, name))
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "processor_config.json",
            "preprocessor_config.json",
        )
    )


def load_processor_for_checkpoint(model_path: str, checkpoint_dir: str):
    candidate_paths = []
    if checkpoint_dir:
        candidate_paths.append(checkpoint_dir)
    if model_path:
        try:
            trex_ckpt = resolve_trex_checkpoint_path(model_path)
            processor_dir = os.path.join(trex_ckpt, "processor")
            if processor_dir not in candidate_paths:
                candidate_paths.append(processor_dir)
        except Exception:
            if model_path not in candidate_paths:
                candidate_paths.append(model_path)

    last_error = None
    for candidate in candidate_paths:
        try:
            processor = AutoProcessor.from_pretrained(candidate, trust_remote_code=True)
            log_message(f"Loaded Qwen/T-Rex processor from {candidate}")
            return processor
        except Exception as exc:
            last_error = exc
            logger.info("Failed to load Qwen/T-Rex processor from %s: %s", candidate, exc)
            if candidate == checkpoint_dir and checkpoint_has_processor_files(checkpoint_dir):
                raise RuntimeError(
                    "Checkpoint directory contains tokenizer/processor files but they could not be loaded: "
                    f"{checkpoint_dir}"
                ) from exc
    raise last_error


def infer_checkpoint_vocab_size(state_dict: dict[str, Any]) -> Optional[int]:
    vocab_keys = [
        "janus.language_model.model.embed_tokens.weight",
        "janus.language_model.lm_head.weight",
        "janus.vla.model.embed_tokens.weight",
    ]
    sizes = []
    for key in vocab_keys:
        value = state_dict.get(key)
        if value is not None and hasattr(value, "shape") and len(value.shape) >= 1:
            sizes.append(int(value.shape[0]))
    if not sizes:
        return None
    if len(set(sizes)) != 1:
        raise ValueError(f"Checkpoint embedding/lm_head vocab sizes disagree: {dict(zip(vocab_keys, sizes))}")
    return sizes[0]


def ensure_janus_tokenizer_alignment(janus_model, tokenizer, target_vocab_size: Optional[int] = None) -> None:
    tokenizer_vocab = int(len(tokenizer))
    target_vocab = int(target_vocab_size or tokenizer_vocab)
    if target_vocab < tokenizer_vocab:
        raise ValueError(f"Target vocab size {target_vocab} is smaller than tokenizer length {tokenizer_vocab}.")
    language_model = janus_model.language_model
    embed = language_model.get_input_embeddings()
    current_vocab = int(embed.weight.shape[0])
    lm_head = getattr(language_model, "lm_head", None)
    lm_head_vocab = None
    if lm_head is not None and getattr(lm_head, "weight", None) is not None:
        lm_head_vocab = int(lm_head.weight.shape[0])
    if current_vocab == target_vocab and (lm_head_vocab is None or lm_head_vocab == target_vocab):
        logger.info("Action backend vocab matches target=%s.", target_vocab)
        return
    logger.info(
        "Resizing action backend token embeddings/lm_head: tokenizer=%s target=%s embedding=%s lm_head=%s",
        tokenizer_vocab,
        target_vocab,
        current_vocab,
        lm_head_vocab if lm_head_vocab is not None else "N/A",
    )
    language_model.resize_token_embeddings(target_vocab)
    if hasattr(janus_model.config, "vocab_size"):
        janus_model.config.vocab_size = target_vocab
    if hasattr(language_model, "config"):
        language_model.config.vocab_size = target_vocab


def validate_checkpoint_vocab_size(state_dict: dict[str, Any], tokenizer) -> None:
    tokenizer_vocab = int(len(tokenizer))
    vocab_keys = [
        "janus.language_model.model.embed_tokens.weight",
        "janus.language_model.lm_head.weight",
        "janus.vla.model.embed_tokens.weight",
    ]
    mismatches = []
    checkpoint_vocab_sizes = []
    for key in vocab_keys:
        value = state_dict.get(key)
        if value is not None and hasattr(value, "shape"):
            checkpoint_vocab = int(value.shape[0])
            checkpoint_vocab_sizes.append(checkpoint_vocab)
            if checkpoint_vocab < tokenizer_vocab:
                mismatches.append(f"{key}: checkpoint={checkpoint_vocab}, tokenizer={tokenizer_vocab}")
    if len(set(checkpoint_vocab_sizes)) > 1:
        mismatches.append(f"checkpoint embedding/lm_head sizes disagree: {checkpoint_vocab_sizes}")
    if mismatches:
        raise ValueError(
            "Checkpoint vocab size is incompatible with this tokenizer. "
            "Use the processor/tokenizer saved with the checkpoint, or evaluate with the same extra special tokens. "
            f"Mismatches: {', '.join(mismatches)}"
        )


def model_load(cfg: TrainsetAttnVisConfig, log_file=None):
    cfg.bridge_pos_scheme = normalize_bridge_pos_scheme(cfg.bridge_pos_scheme)
    cfg.cosmos_self_only_bridge = False
    cfg.decosmos = False
    cfg.action_use_latent_prefix = True
    cfg.action_self_causal_in_bridge = True
    cfg.use_value_prediction = False
    cfg.use_action_value_prediction = False
    cfg.total_spatial_tokens = int(cfg.total_latent_tokens)

    ckpt_path, base_dir = resolve_checkpoint_paths(cfg.pretrained_checkpoint)
    processor = load_processor_for_checkpoint(cfg.qwen3vl2b_model_path or cfg.action_model_path or cfg.model_path, base_dir)
    tokenizer = processor.tokenizer
    cfg.janus_image_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    cfg.janus_image_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    cfg.latent_end_id = tokenizer.eos_token_id
    cfg.trex_image_token_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))

    if not str(getattr(cfg, "qwen3vl2b_model_path", "") or "").strip():
        raise ValueError("--qwen3vl2b_model_path is required for the 32-layer right-branch architecture.")
    log_message(f"Loading Qwen3VL2B base from {cfg.qwen3vl2b_model_path}", log_file)
    janus_model, _ = TrexActionModel.from_qwen3vl_checkpoint(
        cfg.qwen3vl2b_model_path,
        action_dim=cfg.action_dim,
        action_chunk=cfg.action_chunk,
        torch_dtype=torch.bfloat16,
        use_robot_state=bool(cfg.robot_state),
        verbose=True,
    )
    log_message(f"Loading T-Rex action backend from {cfg.action_model_path}", log_file)
    trex_action_model, _ = TrexActionModel.from_checkpoint(
        cfg.action_model_path,
        action_dim=cfg.action_dim,
        action_chunk=cfg.action_chunk,
        torch_dtype=torch.bfloat16,
        use_robot_state=bool(cfg.robot_state),
        verbose=True,
    )
    janus_model.transplant_action_components_from(trex_action_model, fast_layer_count=4)
    del trex_action_model
    cfg.trex_spatial_merge_size = int(getattr(janus_model.visual, "spatial_merge_size", 2) or 2)

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
        experiment_opts=["data_train=mock", "data_val=mock", "model.config.net.sac_config.mode=none"],
    )
    resolve_video_condition_config(cfg, cosmos_wrapper.tokenizer)

    log_message(f"Loading fine-tuned state dict from {ckpt_path}", log_file)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    validate_checkpoint_vocab_size(state_dict, tokenizer)
    cfg.special_token_vocab = load_special_token_vocab(base_dir, getattr(cfg, "special_token_vocab", ""))
    cfg.special_token_init_ids = resolve_special_token_init_ids(tokenizer, cfg.special_token_vocab)
    validate_special_token_checkpoint_rows(state_dict, cfg.special_token_vocab)
    checkpoint_vocab_size = infer_checkpoint_vocab_size(state_dict)
    ensure_janus_tokenizer_alignment(janus_model, tokenizer, target_vocab_size=checkpoint_vocab_size)
    cfg.valid_token_vocab_size = int(len(tokenizer))
    if checkpoint_vocab_size is not None:
        cfg.checkpoint_vocab_size = int(checkpoint_vocab_size)

    model = CosmosJanusActionSpatialMoT2Expert(cosmos_wrapper.net, cosmos_wrapper.tokenizer, janus_model, cfg)
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
    return model, processor, statistic


def _sanitize_filename(value: str, max_len: int = 120) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_")
    return (cleaned or "unknown")[:max_len]


def _colorize_heatmap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 * values - 0.2, 0.0, 1.0)
    green = np.clip(1.5 - 3.0 * np.abs(values - 0.5), 0.0, 1.0)
    blue = np.clip(1.2 - 1.5 * values, 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def _score_to_rgb(score: float) -> tuple[int, int, int]:
    score = float(np.clip(score, 0.0, 1.0))
    low = np.array([80.0, 145.0, 255.0], dtype=np.float32)
    high = np.array([255.0, 48.0, 42.0], dtype=np.float32)
    rgb = low * (1.0 - score) + high * score
    return tuple(int(round(v)) for v in rgb)


def _draw_text_with_outline(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill, font=None) -> None:
    x, y = int(xy[0]), int(xy[1])
    outline = (4, 6, 10)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + dx, y + dy), text, fill=outline, font=font)
    draw.text((x, y), text, fill=fill, font=font)


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
    """Collect action-spatial bridge attention maps for one trainset record."""

    def __init__(self, cfg: TrainsetAttnVisConfig, num_layers: int, log_file=None):
        self.cfg = cfg
        self.log_file = log_file
        self.output_dir = str(getattr(cfg, "attention_visualization_dir", "") or "").strip()
        self.action_chunk = int(getattr(cfg, "action_chunk", 1))
        self.total_spatial_tokens = int(getattr(cfg, "total_latent_tokens", 1))
        if self.total_spatial_tokens not in (1, 2):
            raise ValueError(f"total_latent_tokens must be 1 or 2, got {self.total_spatial_tokens}.")
        self.num_layers = int(num_layers)
        self.first_action_layer_idx = 28
        if self.first_action_layer_idx < 0 or self.first_action_layer_idx >= self.num_layers:
            raise ValueError(
                f"first action layer must be in [0, {self.num_layers - 1}], got {self.first_action_layer_idx}."
            )
        self.alpha = float(getattr(cfg, "attention_visualization_alpha", 0.45) or 0.45)
        self.tile_size = max(16, int(getattr(cfg, "attention_visualization_tile_size", 256) or 256))
        self.top_ratio = getattr(cfg, "attention_visualization_top_ratio", None)
        self.top_softness = float(getattr(cfg, "attention_visualization_top_softness", 0.05) or 0.0)
        self.capture_mode = str(getattr(cfg, "attention_visualization_capture_mode", "last") or "last").lower()
        if self.capture_mode not in {"all", "first", "last"}:
            raise ValueError(f"attention_visualization_capture_mode must be all/first/last, got {self.capture_mode!r}.")
        self.active = False
        self.image_token_mask = None
        self.image_grid_thw = None
        self.action_base_image = None
        self.cosmos_cond_base_image = None
        self.cosmos_future_base_image = None
        self.metadata = {}
        self.action_denoise_step = -1
        self.spatial_call_index = -1
        self.capture_current_action_step = False
        self.summary_records: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}

    def start_query(
        self,
        *,
        image_token_mask: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
        base_image: Image.Image,
        cosmos_future_image: Optional[Image.Image],
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
        self.image_grid_thw = None if image_grid_thw is None else image_grid_thw.detach().cpu()
        self.action_base_image = base_image.copy().convert("RGB")
        self.cosmos_cond_base_image = base_image.copy().convert("RGB")
        if cosmos_future_image is None:
            self.cosmos_future_base_image = base_image.copy().convert("RGB")
        else:
            self.cosmos_future_base_image = cosmos_future_image.copy().convert("RGB")
        self.metadata = {
            "task_name": str(task_name),
            "episode_index": int(episode_index),
            "record_index": int(record_index),
            "sample_index": int(sample_index),
            "prompt": str(prompt),
            "action_denoise_steps": int(action_denoise_steps),
        }
        self.action_denoise_step = -1
        self.spatial_call_index = -1
        self.capture_current_action_step = False
        self.summary_records = {}

    def finish_query(self) -> list[str]:
        if not self.active:
            return []
        try:
            return self._save_query()
        finally:
            self.discard_query()

    def discard_query(self) -> None:
        self.active = False
        self.image_token_mask = None
        self.image_grid_thw = None
        self.action_base_image = None
        self.cosmos_cond_base_image = None
        self.cosmos_future_base_image = None
        self.metadata = {}
        self.action_denoise_step = -1
        self.spatial_call_index = -1
        self.capture_current_action_step = False
        self.summary_records = {}

    def _image_positions(self, *, device: torch.device, action_seq_len: int, prefix_len: int = 0) -> torch.Tensor:
        if self.image_token_mask is None:
            raise RuntimeError("Attention recorder has no image token mask for the active query.")
        mask = self.image_token_mask.to(device=device, dtype=torch.bool)
        if mask.ndim != 2 or int(mask.shape[0]) != 1:
            raise ValueError(f"Expected image token mask shape [1, S], got {tuple(mask.shape)}.")
        positions = torch.nonzero(mask[0], as_tuple=False).flatten()
        if int(positions.numel()) == 0:
            raise ValueError("No T-Rex image token positions found for the active query.")
        max_action_position = int(prefix_len) + int(action_seq_len)
        if int(positions[-1].item()) >= max_action_position:
            raise ValueError(
                "Image token positions do not fit action KV sequence: "
                f"max_image_pos={int(positions[-1].item())}, action_seq_len={action_seq_len}, "
                f"prefix_len={prefix_len}."
            )
        return positions

    def _grid_shape(self, image_token_count: int) -> tuple[int, int]:
        if self.image_grid_thw is not None:
            grid = self.image_grid_thw
            if grid.ndim == 2 and int(grid.shape[-1]) == 3 and int(grid.shape[0]) >= 1:
                first_grid = grid[0]
                merge = int(getattr(self.cfg, "trex_spatial_merge_size", 2) or 2)
                grid_t = max(1, int(first_grid[0].item()))
                grid_h = max(1, int(first_grid[1].item()) // merge)
                grid_w = max(1, int(first_grid[2].item()) // merge)
                if grid_t == 1 and grid_h * grid_w == int(image_token_count):
                    return grid_h, grid_w

        side = int(round(math.sqrt(int(image_token_count))))
        if side * side == int(image_token_count):
            return side, side
        raise ValueError(
            "Could not infer a 2D T-Rex image-token grid from image_grid_thw or square fallback; "
            f"got {image_token_count} tokens."
        )

    @staticmethod
    def _full_attention(q_tokens: torch.Tensor, k_tokens: torch.Tensor, allowed_mask: Optional[torch.Tensor]) -> torch.Tensor:
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
        attn = torch.softmax(scores, dim=-1).mean(dim=1)
        return torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

    def _resolve_video_grid(self, video_grid_thw: Optional[torch.Tensor], video_len: int) -> tuple[int, int, int]:
        if video_grid_thw is not None:
            grid = video_grid_thw.detach().to(device="cpu", dtype=torch.long)
            if grid.ndim == 2:
                grid = grid[0]
            if int(grid.numel()) == 3:
                t, h, w = (int(grid[0].item()), int(grid[1].item()), int(grid[2].item()))
                if t > 0 and h > 0 and w > 0 and t * h * w == int(video_len):
                    return t, h, w
        t = max(1, int(getattr(self.cfg, "total_video_latent_frames", 1) or 1))
        spatial = int(video_len) // t if int(video_len) % t == 0 else int(video_len)
        side = int(round(math.sqrt(spatial)))
        if t * side * side == int(video_len):
            return t, side, side
        side = int(round(math.sqrt(int(video_len))))
        if side * side == int(video_len):
            return 1, side, side
        raise ValueError(f"Could not resolve Cosmos video grid for {video_len} tokens.")

    def _video_region_record(
        self,
        *,
        attn: torch.Tensor,
        video_grid_thw: Optional[torch.Tensor],
        video_len: int,
        region: str,
    ) -> dict[str, Any]:
        t, h, w = self._resolve_video_grid(video_grid_thw, video_len)
        cond_frames = int(getattr(self.cfg, "num_cond_latent_frames", 1) or 1)
        cond_frames = max(1, min(cond_frames, t))
        if region == "cosmos_cond":
            start_t, end_t = 0, cond_frames
        elif region == "cosmos_future":
            start_t, end_t = cond_frames, t
        else:
            raise ValueError(f"Unknown Cosmos attention region: {region}")
        start = start_t * h * w
        end = end_t * h * w
        if end <= start:
            return {"heat": np.zeros((h, w), dtype=np.float32), "score": 0.0}
        region_attn = attn[:, start:end]
        score = float(region_attn.sum(dim=-1).mean().detach().cpu().item())
        flat = region_attn.mean(dim=0).detach().to(torch.float32).cpu()
        heat = flat.reshape(end_t - start_t, h, w).sum(dim=0).numpy().astype(np.float32)
        return {"heat": heat, "score": score}

    def _action_image_region_record(self, *, attn: torch.Tensor, video_len: int, action_seq_len: int) -> dict[str, Any]:
        image_positions = self._image_positions(device=attn.device, action_seq_len=action_seq_len, prefix_len=0)
        full_positions = image_positions + int(video_len)
        if int(full_positions[-1].item()) >= int(attn.shape[1]):
            raise ValueError(
                "Image token positions do not fit full KV sequence: "
                f"max_full_pos={int(full_positions[-1].item())}, full_k_len={int(attn.shape[1])}."
            )
        region_attn = attn[:, full_positions]
        score = float(region_attn.sum(dim=-1).mean().detach().cpu().item())
        flat = region_attn.mean(dim=0).detach().to(torch.float32).cpu()
        grid_h, grid_w = self._grid_shape(int(flat.numel()))
        heat = flat.reshape(grid_h, grid_w).numpy().astype(np.float32)
        return {"heat": heat, "score": score}

    def _target_action_step(self) -> int:
        if self.capture_mode == "first":
            return 0
        return max(0, int(self.metadata.get("action_denoise_steps", 1)) - 1)

    def capture_summary_attention(
        self,
        *,
        query_kind: str,
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
                "Attention visualization currently expects batch size 1, "
                f"got q_tokens={tuple(q_tokens.shape)}, k_video={tuple(k_video.shape)}, k_action={tuple(k_action.shape)}."
            )
        layer_idx = int(layer_idx)
        query_kind = str(query_kind)
        if query_kind == "spatial":
            if layer_idx == 0:
                self.spatial_call_index += 1
            if self.spatial_call_index < 0:
                raise RuntimeError("Spatial recorder saw a nonzero layer before layer 0.")
            label = f"spatial{self.spatial_call_index}"
            if label not in {"spatial0", "spatial1"}:
                return
        elif query_kind == "action":
            if layer_idx == self.first_action_layer_idx:
                self.action_denoise_step += 1
                self.capture_current_action_step = self.action_denoise_step == self._target_action_step()
            if layer_idx < self.first_action_layer_idx:
                return
            if self.action_denoise_step < 0:
                raise RuntimeError("Action recorder saw a nonzero layer before layer 0.")
            if not self.capture_current_action_step:
                return
            label = "action"
        else:
            raise ValueError(f"Unknown attention query kind: {query_kind}")

        video_len = int(k_video.shape[1])
        full_k = torch.cat([k_video, k_action], dim=1)
        attn = self._full_attention(q_tokens[0], full_k[0], allowed_mask)
        record = {
            "cosmos_cond": self._video_region_record(
                attn=attn,
                video_grid_thw=video_grid_thw,
                video_len=video_len,
                region="cosmos_cond",
            ),
            "cosmos_future": self._video_region_record(
                attn=attn,
                video_grid_thw=video_grid_thw,
                video_len=video_len,
                region="cosmos_future",
            ),
            "action_image": self._action_image_region_record(
                attn=attn,
                video_len=video_len,
                action_seq_len=int(k_action.shape[1]),
            ),
        }
        self.summary_records.setdefault(label, {})[layer_idx] = record

    def _query_dir(self) -> Path:
        task_name = _sanitize_filename(str(self.metadata["task_name"]))
        episode_label = f"{int(self.metadata['episode_index']):03d}"
        return Path(self.output_dir) / f"task={task_name}" / f"episode={episode_label}"

    def _output_path(self, kind: str, index: int) -> Path:
        task_name = _sanitize_filename(str(self.metadata["task_name"]))
        episode_label = f"{int(self.metadata['episode_index']):03d}"
        record_label = f"{int(self.metadata['record_index']):04d}"
        sample_label = f"{int(self.metadata['sample_index']):06d}"
        prompt = _sanitize_filename(str(self.metadata["prompt"]), max_len=60)
        filename = (
            f"task={task_name}--episode={episode_label}--record={record_label}"
            f"--sample={sample_label}--{kind}={int(index):03d}--prompt={prompt}.png"
        )
        return self._query_dir() / filename

    def _summary_columns(self) -> list[tuple[str, str, Image.Image, str]]:
        if self.action_base_image is None or self.cosmos_cond_base_image is None or self.cosmos_future_base_image is None:
            raise RuntimeError("Attention recorder has no base images for the active query.")
        return [
            ("spatial0", "cosmos_cond", self.cosmos_cond_base_image, "s0 -> cosmos cond"),
            ("spatial0", "cosmos_future", self.cosmos_future_base_image, "s0 -> cosmos future"),
            ("spatial0", "action_image", self.action_base_image, "s0 -> action img"),
            ("spatial1", "cosmos_cond", self.cosmos_cond_base_image, "s1 -> cosmos cond"),
            ("spatial1", "cosmos_future", self.cosmos_future_base_image, "s1 -> cosmos future"),
            ("spatial1", "action_image", self.action_base_image, "s1 -> action img"),
            ("action", "cosmos_cond", self.cosmos_cond_base_image, "act -> cosmos cond"),
            ("action", "cosmos_future", self.cosmos_future_base_image, "act -> cosmos future"),
            ("action", "action_image", self.action_base_image, "act -> action img"),
        ]

    def _render_summary_overview(self) -> Image.Image:
        row_label_w = 86
        col_label_h = 44
        columns = self._summary_columns()
        canvas = Image.new(
            "RGB",
            (row_label_w + len(columns) * self.tile_size, col_label_h + self.num_layers * self.tile_size),
            (18, 22, 28),
        )
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for col_idx, (_, _, _, label) in enumerate(columns):
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
            for col_idx, (query_label, region_name, base_image, _) in enumerate(columns):
                record = self.summary_records[query_label][layer_idx][region_name]
                heat = np.asarray(record["heat"], dtype=np.float32)
                score = float(record["score"])
                overlay = _overlay_heatmap(
                    base_image,
                    heat,
                    self.tile_size,
                    self.alpha,
                    self.top_ratio,
                    self.top_softness,
                )
                tile_draw = ImageDraw.Draw(overlay)
                _draw_text_with_outline(tile_draw, (7, 5), f"{score:.3f}", fill=_score_to_rgb(score), font=font)
                canvas.paste(overlay, (row_label_w + col_idx * self.tile_size, y))
        return canvas

    def _save_query(self) -> list[str]:
        if not self.summary_records:
            raise RuntimeError("No attention maps were captured for the active query.")
        required = ["spatial0", "spatial1", "action"]
        missing_queries = [label for label in required if label not in self.summary_records]
        if missing_queries:
            raise RuntimeError(f"Missing attention summary query records: {missing_queries}.")
        for label in required:
            missing_layers = [idx for idx in range(self.num_layers) if idx not in self.summary_records[label]]
            if missing_layers:
                raise RuntimeError(f"Missing summary maps for {label}: layers={missing_layers[:8]}.")
            for layer_idx in range(self.num_layers):
                missing_regions = [
                    region for region in ("cosmos_cond", "cosmos_future", "action_image")
                    if region not in self.summary_records[label][layer_idx]
                ]
                if missing_regions:
                    raise RuntimeError(f"Missing summary regions for {label} layer={layer_idx}: {missing_regions}.")
        query_dir = self._query_dir()
        os.makedirs(query_dir, exist_ok=True)
        overview = self._render_summary_overview()
        path = self._output_path("summary", self._target_action_step())
        overview.save(path)
        log_message(f"Saved attention summary PNG to {path}", self.log_file)
        return [str(path)]


def install_attention_map_recorder(model, cfg: TrainsetAttnVisConfig, log_file=None) -> Optional[AttentionMapRecorder]:
    output_dir = str(getattr(cfg, "attention_visualization_dir", "") or "").strip()
    if not output_dir:
        return None
    wrappers = getattr(model, "mot_attention_wrappers", None)
    if wrappers is None or len(wrappers) == 0:
        raise AttributeError("Model has no mot_attention_wrappers to instrument for attention visualization.")
    recorder = AttentionMapRecorder(cfg, num_layers=len(wrappers), log_file=log_file)

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
                k = torch.cat([self.cached_k_v, self.cached_k_a_prefix, k_a], dim=1)
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
                recorder.capture_summary_attention(
                    query_kind="spatial",
                    layer_idx=layer_idx,
                    q_tokens=q_a,
                    k_video=self.cached_k_v,
                    k_action=torch.cat([self.cached_k_a_prefix, k_a], dim=1),
                    allowed_mask=mask[0, 0],
                    video_grid_thw=self.cached_video_grid_thw,
                )
            else:
                prefix_len = 0
                k = torch.cat([self.cached_k_v, k_a], dim=1)
                v = torch.cat([self.cached_v_v, v_a], dim=1)
                mask = self._build_action_only_mask(
                    S_v,
                    S_a,
                    q_a.device,
                    action_valid_mask=action_valid_mask,
                    action_tail_token_count=action_tail_token_count,
                )
                if append_to_cache:
                    if action_valid_mask is None:
                        q_for_spatial = q_a[:, -1:, :, :]
                        q_allowed_mask = mask[0, 0, -1:, :]
                    else:
                        valid = action_valid_mask.to(device=q_a.device, dtype=torch.bool)
                        positions = torch.arange(S_a, device=q_a.device, dtype=torch.long).unsqueeze(0)
                        last_valid = torch.where(valid, positions, torch.zeros_like(positions)).max(dim=1).values
                        gather_idx = last_valid.view(-1, 1, 1, 1).expand(-1, 1, q_a.shape[2], q_a.shape[3])
                        q_for_spatial = q_a.gather(1, gather_idx)
                        if int(last_valid.numel()) != 1:
                            raise ValueError(f"Attention visualization expects batch size 1, got last_valid={tuple(last_valid.shape)}.")
                        row_idx = int(last_valid[0].item())
                        q_allowed_mask = mask[0, 0, row_idx : row_idx + 1, :]
                    recorder.capture_summary_attention(
                        query_kind="spatial",
                        layer_idx=layer_idx,
                        q_tokens=q_for_spatial,
                        k_video=self.cached_k_v,
                        k_action=k_a,
                        allowed_mask=q_allowed_mask,
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
                    self.cached_action_prefix_valid_mask = torch.cat(
                        [self.cached_action_prefix_valid_mask, valid_store],
                        dim=1,
                    )
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
            k = torch.cat([self.cached_k_v, self.cached_k_a_prefix, k_a], dim=1)
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
            recorder.capture_summary_attention(
                query_kind="action",
                layer_idx=layer_idx,
                q_tokens=q_a[:, 1:, :, :],
                k_video=self.cached_k_v,
                k_action=torch.cat([self.cached_k_a_prefix, k_a], dim=1),
                allowed_mask=mask[0, 0, 1:, :],
                video_grid_thw=self.cached_video_grid_thw,
            )
            result = self._bridge_sdpa(q_a, k, v, attn_mask=mask, query_valid_mask=query_valid_mask)
            return self.action_bridge.post_attention(x_action_suffix, result, token_valid_mask=suffix_valid_mask)

        return patched_forward_action_suffix_only

    for layer_idx, wrapper in enumerate(wrappers):
        if not hasattr(wrapper, "_attn_vis_original_forward_action_prefix_and_cache"):
            wrapper._attn_vis_original_forward_action_prefix_and_cache = wrapper.forward_action_prefix_and_cache
            wrapper.forward_action_prefix_and_cache = types.MethodType(
                make_patched_forward_action_prefix_and_cache(layer_idx),
                wrapper,
            )
        if not hasattr(wrapper, "_attn_vis_original_forward_action_suffix_only"):
            wrapper._attn_vis_original_forward_action_suffix_only = wrapper.forward_action_suffix_only
            wrapper.forward_action_suffix_only = types.MethodType(
                make_patched_forward_action_suffix_only(layer_idx),
                wrapper,
            )

    model._attention_map_recorder = recorder
    log_message(
        f"Installed MoT2 attention recorder: dir={output_dir}, layers={len(wrappers)}, "
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
    float_dtype_keys = {
        "janus_pixel_values",
        "cosmos_janus_pixel_values",
        "videos",
        "cosmos_text_embeddings",
    }
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue
        if key in float_dtype_keys:
            moved[key] = value.to(device=device, dtype=dtype, non_blocking=True)
        elif key == "now_state" and value is not None and value.is_floating_point():
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


def load_visualization_future_frame(dataset: VLACotDataset, sample: dict[str, Any], cfg: TrainsetAttnVisConfig) -> Image.Image:
    cur = parse_front_pic_index(sample["front_pic"])
    pic_num = int(sample["pic_num"])
    indices = clipped_keyframe_indices(
        cur,
        pic_num,
        int(cfg.video_frames),
        int(cfg.num_cond_input_frames),
    )
    target_idx = int(indices[-1])
    frame_path = build_front_pic_path(dataset._resolve_data_path(sample["front_pic"]), target_idx)
    return dataset._load_image_pil(frame_path)


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
            future_image = load_visualization_future_frame(dataset, sample, cfg)
            now_state = batch.get("now_state")
            cosmos_text_embeddings = batch.get("cosmos_text_embeddings")
            if cosmos_text_embeddings is not None:
                cosmos_text_embeddings = cosmos_text_embeddings.to(device=device, dtype=dtype)

            recorder.start_query(
                image_token_mask=batch["janus_images_seq_mask"].detach().cpu(),
                image_grid_thw=batch.get("janus_image_grid_thw"),
                base_image=base_image,
                cosmos_future_image=future_image,
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
        raise ValueError(f"MoT2 spatial visualization requires total_latent_tokens=1 or 2, got {cfg.total_latent_tokens}.")
    Path(str(cfg.attention_visualization_dir)).mkdir(parents=True, exist_ok=True)
    log_path = Path(str(cfg.attention_visualization_dir)) / "trainset_attn_vis.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_message(f"=== MoT2 T-Rex RLBench trainset attention visualization start {time.strftime('%Y-%m-%d %H:%M:%S')} ===", log_file)
        log_message(f"Config: {json.dumps(vars(cfg), sort_keys=True, default=str)}", log_file)
        if cfg.bash_hparams_path:
            log_message(f"Bash hparams: {cfg.bash_hparams_path}", log_file)
        set_seed(int(cfg.seed))
        model, processor, _statistic = model_load(cfg, log_file)
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
        log_message("=== MoT2 T-Rex RLBench trainset attention visualization finished ===", log_file)


if __name__ == "__main__":
    main()
