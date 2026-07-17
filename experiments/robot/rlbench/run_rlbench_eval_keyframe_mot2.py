#!/usr/bin/env python3
"""Evaluate 2-MoT Cosmos + action-spatial keyframe policy on RLBench via lift3d."""

import argparse
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from collections import deque
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, Optional

import imageio
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.transform import Rotation as R  # noqa: F401 - imported for lift3d compatibility in some envs

PROJECT_ROOT = Path(__file__).resolve().parents[3]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from models.cosmos_janus_action_spatial import (
    CosmosJanusActionSpatialMoT2Expert,
    normalize_bridge_pos_scheme,
)
from models.cosmos_janus_cot import build_token_sequence_mask
from models.trex_action_backend import TrexActionModel, resolve_trex_checkpoint_path
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint
from utils.cosmos_text_cache import CosmosQwenTextEmbedder, CosmosTextEmbeddingCache
from utils.trex_processor import load_trex_processor

from lift3d.envs.rlbench_env import RLBenchActionMode, RLBenchEnv, RLBenchObservationConfig
from lift3d.helpers.gymnasium import VideoWrapper
from lift3d.helpers.graphics import EEpose


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RLBenchKeyframeEval")

TASK_PROMPTS = {
    "close_box": ("close box", "grasp the lid and turn it to shut the box"),
    "close_laptop_lid": ("close laptop lid", "hold the laptop lid and rotate it closed"),
    "sweep_to_dustpan": ("sweep the dirt up", "grasping the broom by its handle, clear way the dirt from the table"),
    "phone_on_base": ("put the phone on the base", "grasp the phone and put it on the base"),
    "toilet_seat_down": ("toilet seat down", "grasp the toilet seat and turn it down to close"),
    "close_fridge": ("close fridge", "grasp the fridge door and turn it shut"),
    "place_wine_at_rack_location": ("put the wine on the middle", "grasp the bottle and put it away on the middle of the rack"),
    "water_plants": ("water plant", "pick up the watering can by its handle and water the plant"),
    "take_umbrella_out_of_umbrella_stand": (
        "get the umbrella",
        "grasping the umbrella by its handle, lift it up and out of the stand",
    ),
    "take_frame_off_hanger": (
        "take frame off hanger",
        "grasping the picture frame, take it off the wall and place it on the table top",
    ),
}
DEFAULT_SPECIAL_TOKEN_VOCAB = [
    "</PAD>",
    "</MOVE>",
    "</PICK>",
    "</PLACE>",
    "</ROTATE>",
    "</PULL>",
    "</PUSH>",
    "</NONE>",
    "</box>",
    "</broom>",
    "</charger>",
    "</frame>",
    "</fridge>",
    "</lamp>",
    "</laptop>",
    "</phone>",
    "</toilet>",
    "</umbrella>",
    "</watering_can>",
    "</wine>",
]
SPECIAL_TOKEN_VOCAB_FILENAME = "special_token_vocab.json"


DEFAULT_PROMPT_SCHEDULE_PATH = str(Path(__file__).with_name("rlbench_keyframe_prompt_schedule.json"))
DEFAULT_TRAIN_PROMPT_JSON_PATH = "/mnt/nas/zhangyiming/database/rlbench/train/json/train_action_chunk1_sumpos_lastrot.json"
JANUS_ACTION_PROMPT_SUFFIX = (
    "Please refer to the current image and task instruction, predict the spatial token "
    "and output the action to execute now."
)


def build_qwen_chat_prompt(processor, user_text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@dataclass
class EvalConfig:
    pretrained_checkpoint: str
    model_path: str
    action_model_path: str
    cosmos_model_path: str
    cosmos_experiment_name: str
    cosmos_text_cache_path: str = ""
    train_prompt_json_path: str = DEFAULT_TRAIN_PROMPT_JSON_PATH
    prompt_schedule_path: str = ""
    task_names: str = ",".join(TASK_PROMPTS)
    num_episodes: int = 20
    max_steps: int = 10
    result_dir: str = "./rlbench_eval"
    bash_hparams_path: str = ""
    eval_artifact_name: str = ""
    cuda: str = "0"
    seed: int = 0
    video_h: int = 256
    video_w: int = 256
    video_frames: int = 5
    num_cond_input_frames: int = 1
    action_dim: int = 7
    action_chunk: int = 1
    robot_state: int = 0
    state_placeholder_tokens: int = 1
    state_dim: int = 7
    state_encoding_mode: str = "mlp"
    total_latent_tokens: int = 1
    img_latents_per_future: int = 0
    state_latents_per_future: int = 0
    num_future_frames: int = 0
    future_frame_stride: int = 1
    special_token_vocab: str = ",".join(DEFAULT_SPECIAL_TOKEN_VOCAB)
    cosmos_self_only_bridge: bool = False
    decosmos: bool = False
    use_value_prediction: bool = False
    use_action_value_prediction: bool = False
    value_token_mask_video_to_value: bool = False
    value_token_mask_nonvalue_to_value: bool = False
    bridge_pos_scheme: str = "mrope"
    action_use_latent_prefix: bool = True
    action_self_causal_in_bridge: bool = True
    qwen3vl2b_model_path: str = "/mnt/amlfs-07/shared/physicalword/ckpt/pretraine/Qwen3-VL-2B-Instruct"
    right_single_attn_position: str = "last4"
    action_denoise_steps: int = 10
    cosmos_denoise_steps: int = 2
    fps: float = 20.0
    num_open_loop_steps: int = 1
    action_repeat: int = 1
    env_img_res: int = 224
    reset_retries: int = 10
    reset_retry_delay_seconds: float = 1.0
    save_rollout_images: bool = False
    save_keyframe_video: bool = False
    save_cosmos_videos: bool = True


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def recreate_directory(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def parse_special_token_vocab(value) -> list[str]:
    if value is None:
        tokens = list(DEFAULT_SPECIAL_TOKEN_VOCAB)
    elif isinstance(value, str):
        raw = value.strip()
        tokens = list(DEFAULT_SPECIAL_TOKEN_VOCAB) if not raw else [
            part.strip() for part in (raw.split(",") if "," in raw else raw.split()) if part.strip()
        ]
    else:
        tokens = [str(part).strip() for part in value if str(part).strip()]
    if not tokens:
        raise ValueError("special_token_vocab must not be empty.")
    seen = set()
    deduped = []
    for token in tokens:
        if token in seen:
            raise ValueError(f"Duplicate special token in vocab: {token!r}")
        seen.add(token)
        deduped.append(token)
    return deduped


def load_special_token_vocab(checkpoint_dir: str, fallback) -> list[str]:
    vocab_path = os.path.join(str(checkpoint_dir), SPECIAL_TOKEN_VOCAB_FILENAME) if checkpoint_dir else ""
    if vocab_path and os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            return parse_special_token_vocab(json.load(f))
    return parse_special_token_vocab(fallback)


def derive_special_token_source_words(token_text: str) -> list[str]:
    chunks = re.findall(r"</([^>]+)>", str(token_text))
    if not chunks or "".join(f"</{chunk}>" for chunk in chunks) != str(token_text):
        raise ValueError(f"Cannot derive source words from special token {token_text!r}.")
    source_words = []
    for chunk in chunks:
        source_words.extend(part for part in re.split(r"[^A-Za-z0-9]+", chunk.lower()) if part)
    if not source_words:
        raise ValueError(f"Cannot derive non-empty source words from special token {token_text!r}.")
    return source_words


def resolve_special_token_init_ids(tokenizer, special_token_vocab: list[str]) -> list[list[int]]:
    unk_id = getattr(tokenizer, "unk_token_id", None)
    all_source_ids = []
    for token_text in special_token_vocab:
        source_ids = []
        for source_word in derive_special_token_source_words(token_text):
            encoded = tokenizer.encode(source_word, add_special_tokens=False)
            if not encoded:
                raise ValueError(f"Source word {source_word!r} for {token_text!r} encoded to no tokens.")
            for token_id in encoded:
                token_id = int(token_id)
                if unk_id is not None and token_id == int(unk_id) and source_word != getattr(tokenizer, "unk_token", None):
                    raise ValueError(f"Source word {source_word!r} for {token_text!r} encoded to unk id {unk_id}.")
                source_ids.append(token_id)
        all_source_ids.append(source_ids)
    return all_source_ids


def validate_special_token_checkpoint_rows(state_dict: dict[str, Any], special_token_vocab: list[str]) -> None:
    expected = len(special_token_vocab)
    row_keys = ["special_token_embedding.weight"]
    missing = [key for key in row_keys if key not in state_dict]
    if missing:
        raise ValueError(f"Checkpoint is missing tied special-token weights: {missing}")
    mismatches = []
    for key in row_keys:
        rows = int(state_dict[key].shape[0])
        if rows != expected:
            mismatches.append(f"{key}: checkpoint_rows={rows}, special_token_vocab={expected}")
    if mismatches:
        raise ValueError("Special-token checkpoint rows are incompatible: " + ", ".join(mismatches))


def log_message(message: str, log_file=None) -> None:
    logger.info(message)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


def rounded_array(value, decimals: int = 6):
    array = np.asarray(value, dtype=np.float64)
    return np.round(array, decimals=decimals).tolist()


def save_rgb_frames_as_images(frames: np.ndarray, save_dir: str) -> None:
    recreate_directory(save_dir)
    for idx, frame in enumerate(frames):
        Image.fromarray(np.asarray(frame).astype(np.uint8)).save(os.path.join(save_dir, f"{idx:03d}.png"))


def save_rgb_frames_as_video(frames: np.ndarray, save_path: str, fps: float) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer = imageio.get_writer(save_path, fps=fps, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame).astype(np.uint8))
    finally:
        writer.close()


def select_keyframe_boundary_frames(frames: np.ndarray, chunk_stride: int) -> np.ndarray:
    """Select reset frame and post-chunk frames; rollout frames between them are not training keyframes."""
    if len(frames) == 0:
        return frames
    stride = max(1, int(chunk_stride))
    indices = [0]
    indices.extend(range(stride, len(frames), stride))
    return frames[indices]


def predicted_video_to_numpy_frames(pred_video: torch.Tensor) -> np.ndarray:
    video = pred_video.detach().to(torch.float32).cpu()
    if video.dim() == 5:
        video = video[0]
    if video.dim() != 4:
        raise ValueError(f"Expected pred_video with 4 or 5 dims, got shape={tuple(video.shape)}")
    if video.shape[0] in (1, 3):
        video = video.permute(1, 2, 3, 0)
    elif video.shape[1] in (1, 3):
        video = video.permute(0, 2, 3, 1)
    elif video.shape[-1] not in (1, 3):
        raise ValueError(f"Unrecognized pred_video layout with shape={tuple(video.shape)}")
    if video.shape[-1] == 1:
        video = video.repeat(1, 1, 1, 3)
    if float(video.min()) < 0.0 or float(video.max()) > 1.0:
        video = (video + 1.0) / 2.0
    return (video.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).contiguous().numpy()


def get_video_latent_num_frames(video_tokenizer, pixel_frames: int) -> int:
    if video_tokenizer is not None and callable(getattr(video_tokenizer, "get_latent_num_frames", None)):
        return int(video_tokenizer.get_latent_num_frames(int(pixel_frames)))
    return 1 + (int(pixel_frames) - 1) // 4


def resolve_video_condition_config(cfg: EvalConfig, video_tokenizer=None) -> None:
    cfg.video_frames = int(cfg.video_frames)
    cfg.num_cond_input_frames = int(cfg.num_cond_input_frames)
    cfg.num_cond_latent_frames = get_video_latent_num_frames(video_tokenizer, cfg.num_cond_input_frames)
    cfg.total_video_latent_frames = get_video_latent_num_frames(video_tokenizer, cfg.video_frames)


def resolve_checkpoint_paths(pretrained_checkpoint: str):
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
    if checkpoint_dir and checkpoint_dir not in candidate_paths:
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
            processor = load_trex_processor(candidate)
            if candidate != model_path:
                logger.info("Loaded Qwen/T-Rex processor from %s", candidate)
            return processor
        except Exception as exc:
            last_error = exc
            logger.info("Failed to load VLChatProcessor from %s: %s", candidate, exc)
            if candidate == checkpoint_dir and checkpoint_has_processor_files(checkpoint_dir):
                raise RuntimeError(
                    f"Checkpoint directory contains tokenizer/processor files but they could not be loaded: "
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
        raise ValueError(
            f"Target Janus vocab size {target_vocab} is smaller than tokenizer length {tokenizer_vocab}."
        )
    language_model = janus_model.language_model
    embed = language_model.get_input_embeddings()
    current_vocab = int(embed.weight.shape[0])
    lm_head = getattr(language_model, "lm_head", None)
    lm_head_vocab = None
    if lm_head is not None and getattr(lm_head, "weight", None) is not None:
        lm_head_vocab = int(lm_head.weight.shape[0])
    if current_vocab == target_vocab and (lm_head_vocab is None or lm_head_vocab == target_vocab):
        logger.info(
            "Janus vocab exactly matches target: tokenizer=%s, target=%s, embedding=%s, lm_head=%s",
            tokenizer_vocab,
            target_vocab,
            current_vocab,
            lm_head_vocab if lm_head_vocab is not None else "N/A",
        )
        return
    logger.info(
        "Resizing Janus token embeddings/lm_head for tokenizer alignment: "
        "tokenizer=%s, target=%s, embedding=%s, lm_head=%s",
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
            "Checkpoint vocab size is incompatible with this RLBench beta tokenizer. "
            "Use the processor/tokenizer saved with the checkpoint, or evaluate with the same extra special tokens. "
            f"Mismatches: {', '.join(mismatches)}"
        )
    if checkpoint_vocab_sizes and checkpoint_vocab_sizes[0] > tokenizer_vocab:
        logger.info(
            "Checkpoint vocab rows (%s) exceed tokenizer length (%s); treating extra rows as padded lm_head/embed rows.",
            checkpoint_vocab_sizes[0],
            tokenizer_vocab,
        )


def model_load(cfg: EvalConfig):
    cfg.bridge_pos_scheme = normalize_bridge_pos_scheme(cfg.bridge_pos_scheme)
    cfg.cosmos_self_only_bridge = False
    cfg.decosmos = False
    cfg.action_use_latent_prefix = True
    cfg.action_self_causal_in_bridge = True
    cfg.use_value_prediction = False
    cfg.use_action_value_prediction = False
    cfg.total_spatial_tokens = int(cfg.total_latent_tokens)
    logger.info("Resolved right_single_attn_position=%s", getattr(cfg, "right_single_attn_position", "last4"))
    ckpt_path, base_dir = resolve_checkpoint_paths(cfg.pretrained_checkpoint)
    processor = load_processor_for_checkpoint(cfg.qwen3vl2b_model_path or cfg.action_model_path or cfg.model_path, base_dir)
    tokenizer = processor.tokenizer
    action_tokenizer = None
    cfg.janus_image_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    cfg.janus_image_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    cfg.latent_end_id = tokenizer.eos_token_id
    cfg.trex_image_token_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))

    if not str(cfg.qwen3vl2b_model_path or "").strip():
        raise ValueError("--qwen3vl2b_model_path is required for the 32-layer right-branch architecture.")
    janus_model, _ = TrexActionModel.from_qwen3vl_checkpoint(
        cfg.qwen3vl2b_model_path,
        action_dim=cfg.action_dim,
        action_chunk=cfg.action_chunk,
        torch_dtype=torch.bfloat16,
        use_robot_state=bool(cfg.robot_state),
        verbose=True,
    )
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

    import cosmos_predict2._src.predict2.models.text2world_model_rectified_flow as t2w_module

    class DummyTextEncoder(nn.Module):
        def __init__(self, *unused_args, **unused_kwargs):
            super().__init__()

    t2w_module.TextEncoder = DummyTextEncoder
    cosmos_wrapper, cosmos_config = load_model_from_checkpoint(
        experiment_name=cfg.cosmos_experiment_name,
        s3_checkpoint_dir=cfg.cosmos_model_path,
        config_file="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        load_ema_to_reg=True,
        to_device="cpu",
        experiment_opts=["data_train=mock", "data_val=mock", "model.config.net.sac_config.mode=none"],
    )
    resolve_video_condition_config(cfg, cosmos_wrapper.tokenizer)

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
        logger.warning("Missing keys while loading checkpoint: %s", missing[:8])
    if unexpected:
        logger.warning("Unexpected keys while loading checkpoint: %s", unexpected[:8])

    device = torch.device(f"cuda:{cfg.cuda}" if torch.cuda.is_available() else "cpu")
    model = model.to(torch.bfloat16).to(device).eval()
    if cfg.cosmos_text_cache_path:
        text_encoder_config = getattr(cosmos_config.model.config, "text_encoder_config", None)
        model.cosmos_text_cache = CosmosTextEmbeddingCache(cfg.cosmos_text_cache_path, create=True)
        model.cosmos_qwen_text_embedder = CosmosQwenTextEmbedder(model.cosmos_dit, text_encoder_config, device=device)

    stats_path = os.path.join(base_dir, "train_statistics.json")
    with open(stats_path, "r") as f:
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
    return model, processor, action_tokenizer, statistic


def resolve_pad_token_id(processor) -> int:
    return int(processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0)


def resolve_pad_token_text(processor) -> str:
    token = processor.tokenizer.pad_token
    if token is not None:
        return token
    return processor.tokenizer.convert_ids_to_tokens(resolve_pad_token_id(processor))


def normalize_state_for_eval(state, statistic) -> np.ndarray:
    state = np.array(state, dtype=np.float32)
    return np.where(
        statistic["state_mask"],
        np.clip(2 * (state - statistic["state_q01"]) / (statistic["state_q99"] - statistic["state_q01"] + 1e-8) - 1.0, -1.0, 1.0),
        state,
    )


def build_eval_state_inputs(cfg, processor, action_tokenizer, statistic, current_state):
    if not int(cfg.robot_state):
        return "", None, None
    norm_state = normalize_state_for_eval(current_state, statistic)
    if cfg.state_encoding_mode == "mlp":
        if norm_state.shape[-1] != int(cfg.state_dim):
            raise ValueError(f"Expected state_dim={cfg.state_dim}, got {norm_state.shape}.")
        now_state = torch.tensor(norm_state, dtype=torch.float32)
    else:
        raise ValueError("T-Rex action eval supports state_encoding_mode='mlp' only.")
    placeholder_count = int(cfg.state_placeholder_tokens)
    placeholder_text = " ".join([resolve_pad_token_text(processor)] * placeholder_count)
    placeholder_ids = torch.full((placeholder_count,), resolve_pad_token_id(processor), dtype=torch.long)
    return placeholder_text, placeholder_ids, now_state


def load_prompt_schedule(path: str) -> dict[str, list[dict[str, Any]]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw_schedule = json.load(f)
    task_items = raw_schedule.get("tasks", raw_schedule)
    schedule = {}
    for task_name, task_schedule in task_items.items():
        keyframes = task_schedule.get("keyframes", task_schedule)
        if not isinstance(keyframes, list):
            raise ValueError(f"Invalid prompt schedule for task={task_name}: expected keyframes list.")
        schedule[task_name] = keyframes
    return schedule


def task_name_from_train_record(record: dict[str, Any]) -> str:
    front_pic = record.get("front_pic", "")
    parts = Path(front_pic).parts
    if "images" in parts:
        image_index = parts.index("images")
        if image_index + 1 < len(parts):
            return parts[image_index + 1]
    raise ValueError(f"Cannot infer RLBench task name from front_pic={front_pic!r}.")


def load_train_input_prompts(path: str) -> dict[str, str]:
    if not path:
        raise ValueError("--train_prompt_json_path is required for RLBench keyframe eval.")
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected train prompt JSON to contain a list, got {type(records)!r}: {path}")

    prompts_by_task: dict[str, set[str]] = {}
    for index, record in enumerate(records):
        if "input_prompt" not in record:
            raise KeyError(f"Sample {index} in {path} has no input_prompt.")
        task_name = task_name_from_train_record(record)
        prompts_by_task.setdefault(task_name, set()).add(str(record["input_prompt"]))

    task_prompts = {}
    for task_name, prompts in sorted(prompts_by_task.items()):
        if len(prompts) != 1:
            raise ValueError(
                f"Expected exactly one training input_prompt for task={task_name}, "
                f"got {len(prompts)}: {sorted(prompts)!r}"
            )
        task_prompts[task_name] = next(iter(prompts))
    return task_prompts


def make_prompt(task_name: str, train_input_prompts: dict[str, str]) -> str:
    if task_name not in train_input_prompts:
        raise KeyError(
            f"Task {task_name!r} is missing from train_prompt_json_path. "
            "Eval prompt must come from the training JSON for train/test consistency."
        )
    return train_input_prompts[task_name]


def rlbench_state_from_obs(obs_dict, gripper_open) -> np.ndarray:
    robot_state = np.array(obs_dict["robot_state"], dtype=np.float32)
    pose_6d = EEpose.pose_7DoF_to_6DoF(robot_state[7:14])
    gripper = np.array([1.0 if gripper_open is None else float(gripper_open)], dtype=np.float32)
    return np.concatenate([pose_6d, gripper], axis=0)


def build_cosmos_frames(obs_history, video_transform, device, dtype):
    frames = torch.stack(list(obs_history), dim=1).unsqueeze(0)
    return frames.to(device=device, dtype=dtype)


def predict_actions(cfg, model, processor, action_tokenizer, statistic, prompt, image, state, obs_history):
    device = next(model.parameters()).device
    dtype = torch.bfloat16
    state_tokens, state_placeholder_ids, now_state = build_eval_state_inputs(cfg, processor, action_tokenizer, statistic, state)
    if now_state is not None:
        now_state = now_state.unsqueeze(0).to(device)

    user_content = f"{prompt}\n{JANUS_ACTION_PROMPT_SUFFIX}"
    if state_tokens:
        user_content += "\n" + state_tokens
    prompt_text = build_qwen_chat_prompt(processor, user_content)
    janus_inputs = processor(text=prompt_text, images=[image], return_tensors="pt", padding=False)
    janus_input_ids = janus_inputs.input_ids.to(device)

    cosmos_user_content = f"{prompt}"
    if state_tokens:
        cosmos_user_content += "\n" + state_tokens
    cosmos_prompt_text = build_qwen_chat_prompt(processor, cosmos_user_content)
    cosmos_janus_inputs = processor(text=cosmos_prompt_text, images=[image], return_tensors="pt", padding=False)
    cosmos_janus_input_ids = cosmos_janus_inputs.input_ids.to(device)

    janus_state_seq_mask = build_token_sequence_mask(
        janus_input_ids,
        state_placeholder_ids,
        require_match=bool(int(cfg.robot_state)),
        name="current state placeholder",
    ).to(device)
    cosmos_janus_state_seq_mask = build_token_sequence_mask(
        cosmos_janus_input_ids,
        state_placeholder_ids,
        require_match=bool(int(cfg.robot_state)),
        name="current state placeholder in Cosmos prompt",
    ).to(device)
    pad_token_id = resolve_pad_token_id(processor)
    janus_left_pad_lens = janus_input_ids.eq(pad_token_id).to(torch.long).cumprod(dim=1).sum(dim=1)
    attention_mask = janus_inputs.attention_mask.to(device).to(torch.bool)
    image_token_id = int(getattr(cfg, "trex_image_token_id", processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")))
    janus_image_mask = janus_input_ids.eq(image_token_id)
    cosmos_janus_image_mask = cosmos_janus_input_ids.eq(image_token_id)
    cosmos_text_embeddings = None
    if cfg.cosmos_text_cache_path:
        cache = getattr(model, "cosmos_text_cache")
        embedder = getattr(model, "cosmos_qwen_text_embedder")
        cosmos_text_embeddings = cache.get_or_compute(prompt, embedder.compute_one).unsqueeze(0).to(device).to(dtype)

    with torch.inference_mode():
        inference_outputs = model.forward_flow_joint_inference(
            janus_input_ids=janus_input_ids,
            janus_pixel_values=janus_inputs.pixel_values.to(device).to(dtype),
            janus_image_grid_thw=janus_inputs.image_grid_thw.to(device),
            janus_images_seq_mask=janus_image_mask,
            janus_images_emb_mask=janus_image_mask,
            first_frame=build_cosmos_frames(obs_history, None, device, dtype),
            action_denoise_steps=cfg.action_denoise_steps,
            cosmos_denoise_steps=cfg.cosmos_denoise_steps,
            fps=torch.tensor([cfg.fps], device=device, dtype=dtype),
            num_spatial_tokens=cfg.total_latent_tokens,
            janus_left_pad_lens=janus_left_pad_lens,
            janus_state_seq_mask=janus_state_seq_mask,
            janus_attention_mask=attention_mask,
            cosmos_janus_input_ids=cosmos_janus_input_ids,
            cosmos_janus_image_grid_thw=cosmos_janus_inputs.image_grid_thw.to(device),
            cosmos_janus_images_seq_mask=cosmos_janus_image_mask,
            cosmos_janus_state_seq_mask=cosmos_janus_state_seq_mask,
            cosmos_janus_images_emb_mask=cosmos_janus_image_mask,
            now_state=now_state,
            cosmos_text_embeddings=cosmos_text_embeddings,
        )
    pred_video, pred_action = inference_outputs[:2]
    pred_video_to_save = pred_video.detach().cpu() if cfg.save_cosmos_videos and pred_video is not None else None
    raw_normalized = pred_action.squeeze(0).cpu().float().numpy()
    normalized = raw_normalized.copy()
    action_mask = statistic["action_mask"]
    continuous_mask = action_mask.copy()
    if normalized.shape[1] == 7:
        gripper_raw = raw_normalized[:, 6].copy()
        normalized[:, 6] = (normalized[:, 6] >= 0.5).astype(np.float32)
        continuous_mask[6] = False
    else:
        gripper_raw = None
    continuous_values = raw_normalized[:, continuous_mask] if np.any(continuous_mask) else np.empty((raw_normalized.shape[0], 0))
    continuous_clipped = raw_normalized.copy()
    if np.any(continuous_mask):
        continuous_clipped[:, continuous_mask] = np.clip(continuous_clipped[:, continuous_mask], -1.0, 1.0)
        continuous_oob = np.maximum(np.abs(continuous_values) - 1.0, 0.0)
        continuous_oob_count = int(np.count_nonzero(continuous_oob > 0.0))
        continuous_oob_max = float(continuous_oob.max()) if continuous_oob.size else 0.0
        continuous_min = float(continuous_values.min()) if continuous_values.size else 0.0
        continuous_max = float(continuous_values.max()) if continuous_values.size else 0.0
    else:
        continuous_oob_count = 0
        continuous_oob_max = 0.0
        continuous_min = 0.0
        continuous_max = 0.0
    denormalized = np.where(
        action_mask,
        0.5 * (normalized + 1.0) * (statistic["action_q99"] - statistic["action_q01"]) + statistic["action_q01"],
        normalized,
    )
    action_debug = {
        "raw_normalized": rounded_array(raw_normalized),
        "normalized_after_gripper_threshold": rounded_array(normalized),
        "continuous_clipped_for_debug_only": rounded_array(continuous_clipped),
        "continuous_min": round(continuous_min, 6),
        "continuous_max": round(continuous_max, 6),
        "continuous_oob_count": continuous_oob_count,
        "continuous_oob_max": round(continuous_oob_max, 6),
        "gripper_raw": None if gripper_raw is None else rounded_array(gripper_raw),
        "gripper_thresholded": None if gripper_raw is None else rounded_array(normalized[:, 6]),
        "denormalized_model_action": rounded_array(denormalized),
    }
    return denormalized, pred_video_to_save, action_debug


def env_action_from_model_action(model_action, obs_dict) -> np.ndarray:
    action = np.array(model_action, dtype=np.float32).copy()
    action[:3] += np.array(obs_dict["robot_state"], dtype=np.float32)[7:10]
    gripper_open = float(action[-1])
    pose_7d = EEpose.pose_6DoF_to_7DoF(action[:-1])
    return np.append(pose_7d, gripper_open), gripper_open


def close_rlbench_env(env, log_file):
    try:
        env.close()
    except Exception as exc:  # noqa: BLE001 - teardown should not hide eval results
        log_message(f"VideoWrapper close warning={type(exc).__name__}: {exc}", log_file)

    base_env = getattr(env, "unwrapped", None) or getattr(env, "env", None)
    inner_env = getattr(base_env, "env", None)
    inner_close = getattr(inner_env, "close", None)
    if inner_close is None:
        log_message("Inner RLBench env close skipped: close method not found", log_file)
        return
    try:
        inner_close()
        log_message("Inner RLBench env closed", log_file)
    except Exception as exc:  # noqa: BLE001 - teardown should not hide eval results
        log_message(f"Inner RLBench env close warning={type(exc).__name__}: {exc}", log_file)


def build_rlbench_env(cfg, task_name):
    action_mode = RLBenchActionMode.eepose_then_gripper_action_mode(absolute=True)
    obs_config = RLBenchObservationConfig.single_view_config(camera_name="front", image_size=(cfg.env_img_res, cfg.env_img_res))
    env = RLBenchEnv(
        task_name=task_name,
        action_mode=action_mode,
        obs_config=obs_config,
        point_cloud_camera_names=["front"],
        cinematic_record_enabled=True,
        num_points=1024,
        use_point_crop=True,
    )
    return VideoWrapper(env)


def build_rlbench_env_with_retries(cfg, task_name, log_file, context):
    # Retrying RLBench construction in-process can wedge CoppeliaSim/Qt after a
    # failed placement. Let the caller record failures and move on instead.
    attempts = 1
    for attempt in range(attempts):
        try:
            return build_rlbench_env(cfg, task_name)
        except Exception as exc:  # noqa: BLE001 - RLBench task placement can fail before reset
            log_message(
                f"{task_name} env_build_error context={context} attempt={attempt + 1}/{attempts} "
                f"{type(exc).__name__}: {exc}",
                log_file,
            )
            if attempt + 1 < attempts:
                time.sleep(float(cfg.reset_retry_delay_seconds))
    return None


def log_failed_episodes(task_name, start_episode, num_episodes, log_file, reason):
    for failed_episode in range(int(start_episode), int(num_episodes)):
        log_message(f"{task_name} episode={failed_episode} skipped_failed reason={reason}", log_file)
        log_message(f"Task {task_name} episode {failed_episode}: success=False", log_file)


def run_task(cfg, task_name, model, processor, action_tokenizer, statistic, train_input_prompts, log_file):
    task_dir = os.path.join(cfg.result_dir, task_name)
    video_dir = os.path.join(task_dir, "videos")
    image_root = os.path.join(task_dir, "images")
    cosmos_video_dir = os.path.join(task_dir, "cosmos_videos")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(image_root, exist_ok=True)
    keyframe_video_dir = os.path.join(task_dir, "keyframe_videos")
    rollout_image_root = os.path.join(task_dir, "rollout_images")
    if cfg.save_keyframe_video:
        os.makedirs(keyframe_video_dir, exist_ok=True)
    if cfg.save_rollout_images:
        os.makedirs(rollout_image_root, exist_ok=True)
    if cfg.save_cosmos_videos:
        os.makedirs(cosmos_video_dir, exist_ok=True)

    video_transform = transforms.Compose([
        transforms.Resize(min(cfg.video_h, cfg.video_w), antialias=True),
        transforms.CenterCrop((cfg.video_h, cfg.video_w)),
    ])
    successes = 0
    env = build_rlbench_env_with_retries(cfg, task_name, log_file, "initial")

    for episode in range(cfg.num_episodes):
        if env is None:
            env = build_rlbench_env_with_retries(cfg, task_name, log_file, f"episode_{episode}")
            if env is None:
                log_message(f"{task_name} episode={episode} env_build_failed; marking episode failed", log_file)
                log_message(f"Task {task_name} episode {episode}: success=False", log_file)
                log_message("Exiting after env_build_failed to avoid in-process Qt/CoppeliaSim rebuild hang", log_file)
                log_file.flush()
                os._exit(75)

        obs = None
        episode_error = None
        for reset_attempt in range(int(cfg.reset_retries)):
            try:
                obs = env.reset()
                break
            except Exception as exc:  # noqa: BLE001 - reset can fail transiently in RLBench
                log_message(
                    f"{task_name} episode={episode} reset_error attempt={reset_attempt + 1}/{int(cfg.reset_retries)} "
                    f"{type(exc).__name__}: {exc}",
                    log_file,
                )
                if reset_attempt + 1 < int(cfg.reset_retries):
                    time.sleep(float(cfg.reset_retry_delay_seconds))
        if obs is None:
            log_message(
                f"{task_name} episode={episode} reset_failed after {int(cfg.reset_retries)} attempts; marking failed",
                log_file,
            )
            log_message(f"Task {task_name} episode {episode}: success=False", log_file)
            close_rlbench_env(env, log_file)
            env = None
            continue
        action_queue = deque(maxlen=cfg.num_open_loop_steps * cfg.action_repeat)
        obs_history = deque(maxlen=max(1, int(cfg.num_cond_input_frames)))
        success = False
        gripper_open = None
        inference_idx = 0
        for step in range(cfg.max_steps):
            image = Image.fromarray(obs["image"]).convert("RGB")
            frame_tensor = video_transform(torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0)
            if len(obs_history) == 0:
                for _ in range(obs_history.maxlen):
                    obs_history.append(frame_tensor.clone())
            else:
                obs_history.append(frame_tensor)
            if len(action_queue) == 0:
                state = rlbench_state_from_obs(obs, gripper_open)
                prompt = make_prompt(task_name, train_input_prompts)
                log_message(f"{task_name} episode={episode} query={inference_idx} prompt={prompt}", log_file)
                actions, pred_video, action_debug = predict_actions(cfg, model, processor, action_tokenizer, statistic, prompt, image, state, obs_history)
                log_message(
                    f"{task_name} episode={episode} query={inference_idx} action_debug="
                    f"{json.dumps(action_debug, sort_keys=True)}",
                    log_file,
                )
                if pred_video is not None:
                    cosmos_path = os.path.join(cosmos_video_dir, f"episode{episode}_query{inference_idx}.mp4")
                    save_rgb_frames_as_video(predicted_video_to_numpy_frames(pred_video), cosmos_path, fps=cfg.fps)
                for act in actions[: cfg.num_open_loop_steps]:
                    for _ in range(cfg.action_repeat):
                        action_queue.append(act)
                inference_idx += 1
            model_action = action_queue.popleft()
            current_ee_xyz = np.array(obs["robot_state"], dtype=np.float32)[7:10]
            env_action, gripper_open = env_action_from_model_action(model_action, obs)
            log_message(
                f"{task_name} episode={episode} step={step} "
                f"model_action={rounded_array(model_action)} "
                f"current_ee_xyz={rounded_array(current_ee_xyz)} "
                f"env_action={rounded_array(env_action)}",
                log_file,
            )
            try:
                obs, reward, terminated, truncated, info = env.step(env_action)
            except Exception as exc:  # noqa: BLE001 - RLBench/CoppeliaSim can leave env state invalid
                episode_error = exc
                log_message(
                    f"{task_name} episode={episode} step={step} env_step_error={type(exc).__name__}: {exc}",
                    log_file,
                )
                break
            success = success or bool(reward)
            if success or terminated or truncated:
                break
        if success:
            successes += 1
        env.save_video(os.path.join(video_dir, f"episode{episode}_success{int(success)}.mp4"))
        frames = env.get_frames()
        chunk_stride = int(cfg.num_open_loop_steps) * int(cfg.action_repeat)
        keyframe_frames = select_keyframe_boundary_frames(frames, chunk_stride)
        if cfg.save_keyframe_video:
            save_rgb_frames_as_video(
                keyframe_frames,
                os.path.join(keyframe_video_dir, f"episode{episode}_success{int(success)}.mp4"),
                fps=1.0,
            )
        ep_image_dir = os.path.join(image_root, f"episode{episode}")
        save_rgb_frames_as_images(keyframe_frames, ep_image_dir)
        if cfg.save_rollout_images:
            save_rgb_frames_as_images(frames, os.path.join(rollout_image_root, f"episode{episode}"))
        log_message(f"Task {task_name} episode {episode}: success={success}", log_file)
        if episode_error is not None:
            log_message(
                f"{task_name} episode={episode} keeping env after step error; next episode will reset",
                log_file,
            )
    if env is not None:
        close_rlbench_env(env, log_file)
    return successes


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    for field_def in fields(EvalConfig):
        field_name = field_def.name
        default = field_def.default
        if default is MISSING:
            parser.add_argument(f"--{field_name}", required=True, type=str)
            continue
        arg_type = str if default is None else type(default)
        if arg_type is bool:
            arg_type = str
        parser.add_argument(f"--{field_name}", default=default, type=arg_type)
    ns = parser.parse_args()
    cfg = EvalConfig(**vars(ns))
    for name in ("cosmos_self_only_bridge", "decosmos", "use_value_prediction", "use_action_value_prediction",
                 "value_token_mask_video_to_value", "value_token_mask_nonvalue_to_value",
                 "action_use_latent_prefix", "action_self_causal_in_bridge",
                 "save_rollout_images", "save_keyframe_video", "save_cosmos_videos"):
        setattr(cfg, name, coerce_bool(getattr(cfg, name)))
    cfg.cosmos_self_only_bridge = False
    cfg.decosmos = False
    cfg.action_use_latent_prefix = True
    cfg.action_self_causal_in_bridge = True
    cfg.use_value_prediction = False
    cfg.use_action_value_prediction = False
    cfg.right_single_attn_position = str(getattr(cfg, "right_single_attn_position", "last4") or "last4").lower()
    if cfg.right_single_attn_position not in ("first4", "last4"):
        raise ValueError("right_single_attn_position must be 'first4' or 'last4'.")
    if int(cfg.total_latent_tokens) not in (1, 2):
        raise ValueError(
            f"Beta token-latent RLBench eval requires total_latent_tokens=1 or 2, got {cfg.total_latent_tokens}."
        )
    if int(cfg.img_latents_per_future) != 0 or int(cfg.state_latents_per_future) != 0 or int(cfg.num_future_frames) != 0:
        raise ValueError(
            "Beta token-latent RLBench eval does not use continuous future latents: "
            f"img_latents_per_future={cfg.img_latents_per_future}, "
            f"state_latents_per_future={cfg.state_latents_per_future}, "
            f"num_future_frames={cfg.num_future_frames}."
        )
    if cfg.state_encoding_mode == "mlp" and int(cfg.robot_state) and int(cfg.state_placeholder_tokens) != 1:
        raise ValueError("state_encoding_mode='mlp' requires state_placeholder_tokens=1 when robot_state is enabled.")
    if cfg.use_value_prediction and cfg.decosmos:
        raise ValueError("use_value_prediction requires decosmos=false.")
    return cfg


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    os.makedirs(cfg.result_dir, exist_ok=True)
    log_path = os.path.join(cfg.result_dir, "eval.log")
    log_file = open(log_path, "a")
    log_message(f"=== Eval run start {time.strftime('%Y-%m-%d %H:%M:%S')} artifact={cfg.eval_artifact_name} ===", log_file)
    if cfg.bash_hparams_path and os.path.exists(cfg.bash_hparams_path):
        log_message(f"Bash hparams: {cfg.bash_hparams_path}", log_file)
    model, processor, action_tokenizer, statistic = model_load(cfg)
    train_input_prompts = load_train_input_prompts(cfg.train_prompt_json_path)
    log_message(f"Loaded train input prompts: {cfg.train_prompt_json_path}", log_file)
    for task_name, prompt in sorted(train_input_prompts.items()):
        log_message(f"Train prompt task={task_name} prompt={prompt}", log_file)
    task_names = [task.strip() for task in cfg.task_names.split(",") if task.strip()]
    total_successes = 0
    total_episodes = 0
    summary = {}
    for task_name in task_names:
        successes = run_task(cfg, task_name, model, processor, action_tokenizer, statistic, train_input_prompts, log_file)
        summary[task_name] = {"successes": successes, "episodes": cfg.num_episodes, "success_rate": successes / max(cfg.num_episodes, 1)}
        total_successes += successes
        total_episodes += cfg.num_episodes
        log_message(f"Task {task_name}: {summary[task_name]}", log_file)
        with open(os.path.join(cfg.result_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    summary["overall"] = {
        "successes": total_successes,
        "episodes": total_episodes,
        "success_rate": total_successes / max(total_episodes, 1),
    }
    with open(os.path.join(cfg.result_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log_message(f"Overall: {summary['overall']}", log_file)
    log_file.close()


if __name__ == "__main__":
    main()
