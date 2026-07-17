"""
RLBench keyframe training entry for the 2-MoT Cosmos + Janus-action architecture.

The old middle latent expert is removed. v/n/vn spatial labels from the keyframe
JSON are inserted into the Janus action sequence and supervised with next-token
CE, while the final action tokens are supervised with flow matching. Janus is
loaded only from --action_expert_path.
"""

import os
import sys
import json
import torch
import logging
import argparse
import random
import shutil
import math
import re
import wandb
import numpy as np
import gc
from contextlib import contextmanager
from typing import List, Dict

import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator
from transformers import set_seed
from transformers.utils import logging as transformers_logging
from PIL import Image

# Make imports resolve to this checkout even when launched from scripts/ with
# another last05 checkout already present in PYTHONPATH.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import torchvision.transforms as transforms

import models.cosmos_janus_action_spatial as cosmos_janus_mot2_module
from models.cosmos_janus_action_spatial import CosmosJanusActionSpatialMoT2Expert
from models.cosmos_janus_cot import build_token_sequence_mask
from models.trex_action_backend import TrexActionModel, resolve_trex_checkpoint_path
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint
from utils.cosmos_text_cache import CosmosTextEmbeddingCache
from utils.trex_processor import load_trex_processor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

LATENT_TOKEN_MODE_TO_FIELDS = {
    "v": ("gtlatent",),
    "n": ("gtlatent2",),
    "vn": ("gtlatent", "gtlatent2"),
}
LATENT_TOKEN_MODE_ALIASES = {
    "1": "v",
    "2": "vn",
}

JANUS_ACTION_PROMPT_SUFFIX = (
    "Please refer to the current image and task instruction, predict the spatial token "
    "and output the action to execute now."
)

DEFAULT_TREX_PROCESSOR_PATH = (
    "/mnt/nas/zhangyiming/database/ckpt/pretrained/"
    "T-Rex_pretrain_mecka22k_epoch1/checkpoint-0-610000/processor"
)
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


def parse_special_token_vocab(value) -> List[str]:
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


def derive_special_token_source_words(token_text: str) -> List[str]:
    chunks = re.findall(r"</([^>]+)>", str(token_text))
    if not chunks or "".join(f"</{chunk}>" for chunk in chunks) != str(token_text):
        raise ValueError(f"Cannot derive source words from special token {token_text!r}.")
    source_words = []
    for chunk in chunks:
        source_words.extend(part for part in re.split(r"[^A-Za-z0-9]+", chunk.lower()) if part)
    if not source_words:
        raise ValueError(f"Cannot derive non-empty source words from special token {token_text!r}.")
    return source_words


def resolve_special_token_init_ids(tokenizer, special_token_vocab: List[str]) -> List[List[int]]:
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


def resolve_special_token_vocab_args(args):
    vocab = parse_special_token_vocab(getattr(args, "special_token_vocab", None))
    args.special_token_vocab = vocab
    args.special_token_to_id = {token: idx for idx, token in enumerate(vocab)}
    return args


def normalize_latent_token_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    mode = LATENT_TOKEN_MODE_ALIASES.get(mode, mode)
    if mode not in LATENT_TOKEN_MODE_TO_FIELDS:
        valid = ", ".join(sorted([*LATENT_TOKEN_MODE_TO_FIELDS.keys(), *LATENT_TOKEN_MODE_ALIASES.keys()]))
        raise ValueError(f"latent token mode must be one of {valid}, got {value!r}.")
    return mode


def resolve_latent_token_args(args):
    mode_arg = str(getattr(args, "latent_token_mode", "") or "").strip()
    count_or_mode_arg = str(getattr(args, "total_latent_tokens", "") or "").strip()

    if mode_arg:
        mode = normalize_latent_token_mode(mode_arg)
        fields = list(LATENT_TOKEN_MODE_TO_FIELDS[mode])
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
            if token_count != len(fields):
                raise ValueError(
                    "total_latent_tokens count does not match latent_token_mode: "
                    f"mode={mode!r} expects {len(fields)}, got {token_count}."
                )
    else:
        mode = normalize_latent_token_mode(count_or_mode_arg or "1")
        fields = list(LATENT_TOKEN_MODE_TO_FIELDS[mode])

    args.latent_token_mode = mode
    args.latent_token_fields = fields
    args.total_latent_tokens = len(fields)
    return args


def parse_front_pic_index(path):
    match = re.search(r"front_(\d+)\.[^.]+$", os.path.basename(str(path)))
    if match is None:
        raise ValueError(f"Could not parse RLBench front image index from path: {path}")
    return int(match.group(1))


def build_front_pic_path(current_path, idx):
    dirname = os.path.dirname(str(current_path))
    basename = os.path.basename(str(current_path))
    next_basename = re.sub(r"front_\d+(\.[^.]+)$", f"front_{int(idx)}\\1", basename)
    if next_basename == basename and parse_front_pic_index(current_path) != int(idx):
        raise ValueError(f"Could not replace RLBench front image index in path: {current_path}")
    return os.path.join(dirname, next_basename)


def clipped_keyframe_indices(cur, pic_num, video_frames, num_cond_input_frames):
    cur = int(cur)
    pic_num = int(pic_num)
    video_frames = int(video_frames)
    num_cond_input_frames = int(num_cond_input_frames)
    if video_frames <= 0:
        raise ValueError(f"video_frames must be positive, got {video_frames}.")
    if num_cond_input_frames <= 0:
        raise ValueError(f"num_cond_input_frames must be positive, got {num_cond_input_frames}.")
    start = cur - (num_cond_input_frames - 1)
    indices = np.arange(start, start + video_frames)
    return np.clip(indices, 0, pic_num).astype(int)


def resolve_rlbench_episode_key(sample):
    if "task_name" in sample:
        task = str(sample["task_name"])
    else:
        front_pic = str(sample["front_pic"])
        episode_dir = os.path.dirname(front_pic)
        task_dir = os.path.dirname(episode_dir)
        task = os.path.basename(task_dir) or os.path.basename(episode_dir)
    return task, int(sample["episode_index"])


@contextmanager
def suppress_transformers_loading_warnings():
    """Temporarily silence Transformers weight-loading warnings."""
    previous_verbosity = transformers_logging.get_verbosity()
    try:
        transformers_logging.set_verbosity_error()
        yield
    finally:
        transformers_logging.set_verbosity(previous_verbosity)


def extract_unloaded_keys(loading_info: Dict) -> List[str]:
    unloaded_keys: List[str] = []

    unloaded_keys.extend(loading_info.get("missing_keys", []) or [])
    unloaded_keys.extend(loading_info.get("unexpected_keys", []) or [])

    for item in loading_info.get("mismatched_keys", []) or []:
        if isinstance(item, (list, tuple)) and item:
            unloaded_keys.append(item[0])
        else:
            unloaded_keys.append(str(item))

    return unloaded_keys


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=True)
    return batch


def get_video_latent_num_frames(video_tokenizer, pixel_frames: int) -> int:
    pixel_frames = int(pixel_frames)
    if pixel_frames < 1:
        raise ValueError(f"pixel_frames must be positive, got {pixel_frames}.")

    if video_tokenizer is not None:
        latent_num_frames = getattr(video_tokenizer, "get_latent_num_frames", None)
        if callable(latent_num_frames):
            return int(latent_num_frames(pixel_frames))

        temporal_compression_factor = getattr(video_tokenizer, "temporal_compression_factor", None)
        if callable(temporal_compression_factor):
            temporal_compression_factor = temporal_compression_factor()
        if temporal_compression_factor is not None:
            temporal_compression_factor = int(temporal_compression_factor)
        else:
            temporal_compression_factor = 4
    else:
        temporal_compression_factor = 4

    if temporal_compression_factor < 1:
        raise ValueError(
            "temporal_compression_factor must be positive, "
            f"got {temporal_compression_factor}."
        )
    return 1 + (pixel_frames - 1) // temporal_compression_factor


def resolve_video_condition_args(args, video_tokenizer=None):
    args.video_frames = int(args.video_frames)
    args.num_cond_input_frames = int(getattr(args, "num_cond_input_frames", 1))
    if args.video_frames < 1:
        raise ValueError("video_frames must be positive.")
    if args.num_cond_input_frames < 1:
        raise ValueError("num_cond_input_frames must be positive.")
    if args.num_cond_input_frames > args.video_frames:
        raise ValueError(
            "num_cond_input_frames must be <= video_frames, "
            f"got {args.num_cond_input_frames} > {args.video_frames}."
        )

    args.num_cond_latent_frames = get_video_latent_num_frames(
        video_tokenizer,
        args.num_cond_input_frames,
    )
    args.total_video_latent_frames = get_video_latent_num_frames(
        video_tokenizer,
        args.video_frames,
    )
    if args.num_cond_latent_frames > args.total_video_latent_frames:
        raise ValueError(
            "num_cond_latent_frames must be <= total_video_latent_frames, "
            f"got {args.num_cond_latent_frames} > {args.total_video_latent_frames}."
        )
    return args


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


def get_custom_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
        scaled_factor = (1 - min_lr_ratio) * cosine_factor + min_lr_ratio
        return scaled_factor
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


class VLACotDataset(Dataset):
    """Dataset with future frame/state loading for latent CoT ground truth."""

    def __init__(self, config, processor, accelerator):
        self.config = config
        self.processor = processor
        self.accelerator = accelerator
        self.tokenizer = processor.tokenizer
        self.pad_token_id = self._resolve_pad_token_id()
        self.pad_token_text = None
        self.use_cosmos_text_cache = bool(getattr(config, "cosmos_text_cache_path", ""))
        self.cosmos_text_cache = None
        if self.use_cosmos_text_cache:
            self.cosmos_text_cache = CosmosTextEmbeddingCache(
                config.cosmos_text_cache_path,
                create=False,
            )
            self.accelerator.print(
                f"Using cached Cosmos text embeddings from {self.cosmos_text_cache.root}"
            )

        self.accelerator.print(f"Loading RLBench keyframe dataset from {config.data_path} ...")
        with open(config.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        statistics_path = config.data_path.replace(".json", "_statistics.json")
        with open(statistics_path, 'r', encoding='utf-8') as f:
            self.stats_data = json.load(f)

        self.dataset_name = next(iter(self.stats_data))
        self.action_q01 = np.array(self.stats_data[self.dataset_name]['action']['q01'])
        self.action_q99 = np.array(self.stats_data[self.dataset_name]['action']['q99'])
        self.action_mask = np.array(self.stats_data[self.dataset_name]['action']['mask'])
        self.state_q01 = np.array(self.stats_data[self.dataset_name]['state']['q01'])
        self.state_q99 = np.array(self.stats_data[self.dataset_name]['state']['q99'])
        self.state_mask = np.array(self.stats_data[self.dataset_name]['state']['mask'])

        self.video_transform = transforms.Compose([
            transforms.Resize(min(config.video_h, config.video_w), antialias=True),
            transforms.CenterCrop((config.video_h, config.video_w)),
        ])
        self.latent_hidden_sim_loss_mode = str(
            getattr(config, "latent_hidden_sim_loss_mode", "siglip")
        ).lower()
        if self.latent_hidden_sim_loss_mode == "wan_vae":
            self.latent_hidden_sim_transform = transforms.Compose([
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop((256, 256)),
                transforms.ToTensor(),
            ])
        else:
            self.latent_hidden_sim_transform = None

        # Build episode index for diagnostics and optional future lookups.
        self._build_episode_index()

    def _build_episode_index(self):
        """Group samples by RLBench task/episode."""
        self.episode_records = {}
        for i, sample in enumerate(self.data):
            key = resolve_rlbench_episode_key(sample)
            record_index = int(sample["record_index"])
            self.episode_records.setdefault(key, {})[record_index] = i
        self.accelerator.print(f"Indexed {len(self.episode_records)} RLBench task/episode groups.")

    def __len__(self):
        return len(self.data)

    def _normalize(self, data_array, q01, q99, mask):
        return np.where(
            mask,
            np.clip(2 * (data_array - q01) / (q99 - q01 + 1e-8) - 1.0, -1.0, 1.0),
            data_array
        )

    def _resolve_data_path(self, path):
        if hasattr(self.config, 'data_root') and self.config.data_root and not os.path.isabs(path):
            return os.path.join(self.config.data_root, path)
        return path

    def _resolve_pad_token_id(self):
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None and hasattr(self.processor, 'pad_id'):
            pad_token_id = self.processor.pad_id
        if pad_token_id is None:
            pad_token_id = 0
        return int(pad_token_id)

    def _resolve_pad_token_text(self):
        candidates = []
        if self.tokenizer.pad_token is not None:
            candidates.append(self.tokenizer.pad_token)

        token_text = self.tokenizer.convert_ids_to_tokens(self.pad_token_id)
        if token_text is not None:
            candidates.append(token_text)

        for candidate in candidates:
            encoded = self.tokenizer.encode(candidate, add_special_tokens=False)
            if len(encoded) == 1 and int(encoded[0]) == self.pad_token_id:
                return candidate

        raise ValueError(
            f"Could not resolve a text form for pad token id {self.pad_token_id}."
        )

    def _state_placeholder_count(self):
        return int(getattr(self.config, 'state_placeholder_tokens', 8))

    def _state_encoding_mode(self):
        return getattr(self.config, 'state_encoding_mode', 'token')

    def _build_state_placeholder_ids(self):
        placeholder_count = self._state_placeholder_count()
        return torch.full((placeholder_count,), self.pad_token_id, dtype=torch.long)

    def _build_state_placeholder_text(self):
        if self.pad_token_text is None:
            self.pad_token_text = self._resolve_pad_token_text()
        return " ".join([self.pad_token_text] * self._state_placeholder_count())

    def _validate_current_state_ids(self, current_state_ids):
        placeholder_count = self._state_placeholder_count()
        if current_state_ids.numel() != placeholder_count:
            raise ValueError(
                f"Encoded current state has {current_state_ids.numel()} tokens, "
                f"but state_placeholder_tokens is {placeholder_count}. "
                "Set STATE_PLACEHOLDER_TOKENS to match the encoded state length."
            )
        return current_state_ids

    def _normalize_state(self, state):
        state_arr = np.array(state, dtype=np.float32)
        norm_state = self._normalize(state_arr, self.state_q01, self.state_q99, self.state_mask)
        expected_dim = int(getattr(self.config, "state_dim", 7))
        if self._state_encoding_mode() == 'mlp' and norm_state.shape[-1] != expected_dim:
            raise ValueError(f"MLP state encoding expects state dim {expected_dim}, got shape {norm_state.shape}.")
        return norm_state

    def _load_image_pil(self, image_path):
        image_path_abs = self._resolve_data_path(image_path)
        return Image.open(image_path_abs).convert("RGB")

    def _load_future_frame_pil(self, sample, stride):
        stride = int(stride)
        if stride <= 0:
            raise ValueError("future_frame_stride must be positive when use_latent_hidden_sim_loss=1.")
        cur = parse_front_pic_index(sample["front_pic"])
        pic_num = int(sample["pic_num"])
        target_idx = int(np.clip(cur + stride, 0, pic_num))
        frame_path = build_front_pic_path(self._resolve_data_path(sample["front_pic"]), target_idx)
        return self._load_image_pil(frame_path)

    def _load_video(self, sample):
        target_frames = int(self.config.video_frames)
        num_cond_input_frames = int(getattr(self.config, "num_cond_input_frames", 1))
        cur = parse_front_pic_index(sample["front_pic"])
        pic_num = int(sample["pic_num"])
        indices = clipped_keyframe_indices(cur, pic_num, target_frames, num_cond_input_frames)
        frames = []
        for idx in indices:
            frame_path = build_front_pic_path(self._resolve_data_path(sample["front_pic"]), idx)
            frames.append(np.array(self._load_image_pil(frame_path), dtype=np.uint8))
        frames = np.stack(frames, axis=0)
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        frames_tensor = self.video_transform(frames_tensor)
        frames_tensor = frames_tensor.permute(1, 0, 2, 3)
        return frames_tensor

    def _compute_value_target(self, sample):
        record_count = int(sample.get("record_count", 1))
        record_index = int(sample.get("record_index", 0))
        alpha = max(record_count - record_index, 1)
        value = 0.99 ** alpha
        return torch.tensor(value, dtype=torch.float32)

    def _latent_token_count(self) -> int:
        return int(getattr(self.config, "total_latent_tokens", 1) or 1)

    def _latent_token_fields(self):
        fields = getattr(self.config, "latent_token_fields", None)
        if fields is None:
            token_count = self._latent_token_count()
            if token_count == 1:
                fields = ["gtlatent"]
            elif token_count == 2:
                fields = ["gtlatent", "gtlatent2"]
            else:
                raise ValueError(f"total_latent_tokens must be 1 or 2, got {token_count}.")
        elif isinstance(fields, str):
            fields = [field.strip() for field in fields.split(",") if field.strip()]
        else:
            fields = [str(field).strip() for field in fields if str(field).strip()]
        if not fields:
            raise ValueError("latent_token_fields must not be empty.")
        return fields

    def _encode_gt_latent_token(self, sample, index, field_name: str):
        if field_name not in sample:
            raise KeyError(f"Dataset sample index={index} is missing required '{field_name}' field.")
        gt_text = str(sample[field_name])
        token_to_id = getattr(self.config, "special_token_to_id", None)
        if token_to_id is None:
            raise ValueError("config.special_token_to_id is required for spatial token labels.")
        if gt_text not in token_to_id:
            raise ValueError(
                f"Dataset sample index={index} {field_name}={gt_text!r} is not in special_token_vocab. "
                f"Known tokens: {getattr(self.config, 'special_token_vocab', [])}"
            )
        return torch.tensor(int(token_to_id[gt_text]), dtype=torch.long)

    def _encode_gt_latent_tokens(self, sample, index):
        token_count = self._latent_token_count()
        fields = self._latent_token_fields()
        if len(fields) != token_count:
            raise ValueError(
                "latent_token_fields length must match total_latent_tokens: "
                f"fields={fields}, total_latent_tokens={token_count}."
            )
        tokens = [
            self._encode_gt_latent_token(sample, index, field_name)
            for field_name in fields
        ]
        return torch.stack(tokens)

    def __getitem__(self, index):
        sample = self.data[index]
        cosmos_text_embedding = None
        if self.use_cosmos_text_cache:
            try:
                cosmos_text_embedding = self.cosmos_text_cache.load(sample["input_prompt"])
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"{exc}\nDataset sample index={index}, data_path={self.config.data_path}"
                ) from exc

        state_tokens_str = ""
        state_placeholder_ids = None
        now_state = None
        if self.config.robot_state:
            if 'state' not in sample:
                raise KeyError("robot_state is enabled, but sample has no 'state' field.")
            if self.state_mask is None:
                raise ValueError("robot_state is enabled, but state_mask is missing.")
            norm_state = self._normalize_state(sample['state'])
            if self._state_encoding_mode() == 'mlp':
                now_state = torch.tensor(norm_state, dtype=torch.float32)
            else:
                raise ValueError("T-Rex action training supports state_encoding_mode='mlp' only.")
            state_tokens_str = self._build_state_placeholder_text()
            state_placeholder_ids = self._build_state_placeholder_ids()

        user_content = f"{sample['input_prompt']}\n{JANUS_ACTION_PROMPT_SUFFIX}"
        if state_tokens_str:
            user_content += "\n" + state_tokens_str
        prompt = build_qwen_chat_prompt(self.processor, user_content)

        video_tensor = self._load_video(sample)

        first_frame_pil = self._load_image_pil(sample['front_pic'])

        janus_inputs = self.processor(
            text=prompt,
            images=[first_frame_pil],
            return_tensors="pt",
            padding=False,
        )
        cosmos_user_content = f"{sample['input_prompt']}"
        if state_tokens_str:
            cosmos_user_content += "\n" + state_tokens_str
        cosmos_prompt = build_qwen_chat_prompt(self.processor, cosmos_user_content)
        cosmos_janus_inputs = self.processor(
            text=cosmos_prompt,
            images=[first_frame_pil],
            return_tensors="pt",
            padding=False,
        )
        janus_input_ids = janus_inputs.input_ids.squeeze(0)
        cosmos_janus_input_ids = cosmos_janus_inputs.input_ids.squeeze(0)
        janus_state_seq_mask = build_token_sequence_mask(
            janus_input_ids,
            state_placeholder_ids,
            require_match=bool(self.config.robot_state),
            name="current state placeholder",
        )
        cosmos_janus_state_seq_mask = build_token_sequence_mask(
            cosmos_janus_input_ids,
            state_placeholder_ids,
            require_match=bool(self.config.robot_state),
            name="current state placeholder in Cosmos prompt",
        )
        attention_mask = janus_inputs.attention_mask.squeeze(0).to(torch.bool)

        # Action
        action_arr = np.array(sample['action'], dtype=np.float32).reshape(-1, self.config.action_dim)
        if action_arr.shape[0] < self.config.action_chunk:
            pad_len = self.config.action_chunk - action_arr.shape[0]
            action_arr = np.concatenate([action_arr, np.repeat(action_arr[-1:], pad_len, axis=0)], axis=0)
        elif action_arr.shape[0] > self.config.action_chunk:
            action_arr = action_arr[:self.config.action_chunk]

        norm_action = self._normalize(action_arr, self.action_q01, self.action_q99, self.action_mask)
        actions_tensor = torch.tensor(norm_action, dtype=torch.float32)

        gt_latent_token_ids = self._encode_gt_latent_tokens(sample, index)
        latent_hidden_sim_pixel_values = None
        latent_hidden_sim_image_grid_thw = None
        if int(getattr(self.config, "use_latent_hidden_sim_loss", 0) or 0):
            future_frame_pil = self._load_future_frame_pil(
                sample,
                self.config.future_frame_stride,
            )
            if self.latent_hidden_sim_loss_mode == "wan_vae":
                latent_hidden_sim_pixel_values = self.latent_hidden_sim_transform(
                    future_frame_pil
                ).unsqueeze(0)
            else:
                future_inputs = self.processor.image_processor(
                    [future_frame_pil],
                    return_tensors="pt",
                )
                latent_hidden_sim_pixel_values = future_inputs["pixel_values"]
                latent_hidden_sim_image_grid_thw = future_inputs["image_grid_thw"]

        item = {
            "janus_input_ids": janus_input_ids,
            "janus_pixel_values": janus_inputs.pixel_values,
            "janus_image_grid_thw": janus_inputs.image_grid_thw,
            "janus_images_seq_mask": janus_input_ids.eq(
                int(getattr(self.config, "trex_image_token_id", 151655))
            ),
            "janus_state_seq_mask": janus_state_seq_mask,
            "janus_images_emb_mask": janus_input_ids.eq(
                int(getattr(self.config, "trex_image_token_id", 151655))
            ),
            "cosmos_janus_input_ids": cosmos_janus_input_ids,
            "cosmos_janus_pixel_values": cosmos_janus_inputs.pixel_values,
            "cosmos_janus_image_grid_thw": cosmos_janus_inputs.image_grid_thw,
            "cosmos_janus_images_seq_mask": cosmos_janus_input_ids.eq(
                int(getattr(self.config, "trex_image_token_id", 151655))
            ),
            "cosmos_janus_state_seq_mask": cosmos_janus_state_seq_mask,
            "cosmos_janus_images_emb_mask": cosmos_janus_input_ids.eq(
                int(getattr(self.config, "trex_image_token_id", 151655))
            ),
            "attention_mask": attention_mask,
            "now_state": now_state,
            "actions": actions_tensor,
            "videos": video_tensor,
            "gt_latent_token_ids": gt_latent_token_ids,
            "cosmos_text_embeddings": cosmos_text_embedding,
        }
        if latent_hidden_sim_pixel_values is not None:
            item["latent_hidden_sim_pixel_values"] = latent_hidden_sim_pixel_values
        if latent_hidden_sim_image_grid_thw is not None:
            item["latent_hidden_sim_image_grid_thw"] = latent_hidden_sim_image_grid_thw
        return item

    def collate_fn(self, batch):
        input_ids_list = [x['janus_input_ids'] for x in batch]
        seq_mask_list = [x['janus_images_seq_mask'] for x in batch]
        state_mask_list = [x['janus_state_seq_mask'] for x in batch]
        attention_mask_list = [x['attention_mask'] for x in batch]
        max_len = max(len(ids) for ids in input_ids_list)
        cosmos_input_ids_list = [x['cosmos_janus_input_ids'] for x in batch]
        cosmos_seq_mask_list = [x['cosmos_janus_images_seq_mask'] for x in batch]
        cosmos_state_mask_list = [x['cosmos_janus_state_seq_mask'] for x in batch]
        cosmos_emb_mask_list = [x['cosmos_janus_images_emb_mask'] for x in batch]
        cosmos_max_len = max(len(ids) for ids in cosmos_input_ids_list)

        padded_input_ids = []
        padded_seq_masks = []
        padded_state_masks = []
        padded_attention_masks = []
        left_pad_lens = []
        padded_cosmos_input_ids = []
        padded_cosmos_seq_masks = []
        padded_cosmos_state_masks = []
        padded_cosmos_emb_masks = []
        pad_token_id = self.pad_token_id
        #print("pad_token_id: ", pad_token_id)

        for ids, seq_mask, state_mask, attention_mask in zip(
            input_ids_list,
            seq_mask_list,
            state_mask_list,
            attention_mask_list,
        ):
            pad_len = max_len - len(ids)
            # Keep Janus inputs left-padded end-to-end so the latent bridge sees the
            # same layout as prepare_inputs_embeds.
            padded_ids = F.pad(ids, (pad_len, 0), value=pad_token_id)
            padded_seq_mask = F.pad(seq_mask, (pad_len, 0), value=False)
            padded_state_mask = F.pad(state_mask, (pad_len, 0), value=False)
            padded_attention_mask = F.pad(attention_mask, (pad_len, 0), value=False)
            padded_input_ids.append(padded_ids)
            padded_seq_masks.append(padded_seq_mask)
            padded_state_masks.append(padded_state_mask)
            padded_attention_masks.append(padded_attention_mask)
            left_pad_lens.append(pad_len)

        for ids, seq_mask, state_mask in zip(
            cosmos_input_ids_list,
            cosmos_seq_mask_list,
            cosmos_state_mask_list,
        ):
            pad_len = cosmos_max_len - len(ids)
            padded_cosmos_input_ids.append(F.pad(ids, (pad_len, 0), value=pad_token_id))
            padded_cosmos_seq_masks.append(F.pad(seq_mask, (pad_len, 0), value=False))
            padded_cosmos_state_masks.append(F.pad(state_mask, (pad_len, 0), value=False))
        for ids, emb_mask in zip(cosmos_input_ids_list, cosmos_emb_mask_list):
            pad_len = cosmos_max_len - len(ids)
            padded_cosmos_emb_masks.append(F.pad(emb_mask, (pad_len, 0), value=False))

        now_state = None
        if any(x['now_state'] is not None for x in batch):
            if not all(x['now_state'] is not None for x in batch):
                raise ValueError("Mixed batch contains both present and missing now_state values.")
            now_state = torch.stack([x['now_state'] for x in batch])

        cosmos_text_embeddings = None
        if self.use_cosmos_text_cache:
            cosmos_text_embeddings = torch.stack([x['cosmos_text_embeddings'] for x in batch])

        out = {
            "janus_input_ids": torch.stack(padded_input_ids),
            "janus_left_pad_lens": torch.tensor(left_pad_lens, dtype=torch.long),
            "janus_pixel_values": torch.cat([x['janus_pixel_values'] for x in batch], dim=0),
            "janus_image_grid_thw": torch.cat([x['janus_image_grid_thw'] for x in batch], dim=0),
            "janus_images_seq_mask": torch.stack(padded_seq_masks),
            "janus_state_seq_mask": torch.stack(padded_state_masks),
            "janus_images_emb_mask": torch.stack(padded_seq_masks),
            "cosmos_janus_input_ids": torch.stack(padded_cosmos_input_ids),
            "cosmos_janus_pixel_values": torch.cat([x['cosmos_janus_pixel_values'] for x in batch], dim=0),
            "cosmos_janus_image_grid_thw": torch.cat([x['cosmos_janus_image_grid_thw'] for x in batch], dim=0),
            "cosmos_janus_images_seq_mask": torch.stack(padded_cosmos_seq_masks),
            "cosmos_janus_state_seq_mask": torch.stack(padded_cosmos_state_masks),
            "cosmos_janus_images_emb_mask": torch.stack(padded_cosmos_emb_masks),
            "attention_mask": torch.stack(padded_attention_masks),
            "now_state": now_state,
            "actions": torch.stack([x['actions'] for x in batch]),
            "videos": torch.stack([x['videos'] for x in batch]),
            "gt_latent_token_ids": torch.stack([x['gt_latent_token_ids'] for x in batch]),
            "cosmos_text_embeddings": cosmos_text_embeddings,
        }
        if int(getattr(self.config, "use_latent_hidden_sim_loss", 0) or 0):
            if self.latent_hidden_sim_loss_mode == "wan_vae":
                out["latent_hidden_sim_pixel_values"] = torch.stack([
                    x["latent_hidden_sim_pixel_values"] for x in batch
                ])
            else:
                out["latent_hidden_sim_pixel_values"] = torch.cat([
                    x["latent_hidden_sim_pixel_values"] for x in batch
                ], dim=0)
                out["latent_hidden_sim_image_grid_thw"] = torch.cat([
                    x["latent_hidden_sim_image_grid_thw"] for x in batch
                ], dim=0)
        return out


def ensure_janus_tokenizer_alignment(janus_model, tokenizer, accelerator=None):
    """Resize Janus token embeddings/lm_head when this checkout adds new special tokens."""
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
        if accelerator is not None:
            accelerator.print(
                "Janus tokenizer covered by embedding/lm_head vocab: "
                f"tokenizer={target_vocab}, embedding={current_vocab}, "
                f"lm_head={lm_head_vocab if lm_head_vocab is not None else 'N/A'}."
            )
        return
    if accelerator is not None:
        accelerator.print(
            f"Resizing Janus token embeddings/lm_head from {current_vocab} to {target_vocab} "
            "for RLBench token-latent special tokens."
        )
    language_model.resize_token_embeddings(target_vocab)
    if hasattr(janus_model.config, "vocab_size"):
        janus_model.config.vocab_size = target_vocab
    if hasattr(language_model, "config"):
        language_model.config.vocab_size = target_vocab


def initialize_action_special_token_rows(janus_model, tokenizer, accelerator=None):
    """Initialize action special-token rows from their natural-language token rows."""
    token_sources = {
        "</MOVE>": ("move",),
        "</PICK>": ("pick",),
        "</PLACE>": ("place",),
        "</ROTATE>": ("rotate",),
        "</PULL>": ("pull",),
        "</PUSH>": ("push",),
        "</NONE>": ("none",),
    }

    def derive_source_words(token_text: str):
        chunks = re.findall(r"</([^>]+)>", str(token_text))
        if not chunks:
            return None
        if "".join(f"</{chunk}>" for chunk in chunks) != str(token_text):
            return None
        source_words = []
        for chunk in chunks:
            source_words.extend(
                part for part in re.split(r"[^A-Za-z0-9]+", chunk.lower()) if part
            )
        return tuple(source_words) if source_words else None

    for token_text in getattr(tokenizer, "additional_special_tokens", []):
        if token_text in token_sources:
            continue
        derived_sources = derive_source_words(token_text)
        if derived_sources is not None:
            token_sources[token_text] = derived_sources

    language_model = janus_model.language_model
    embed = language_model.get_input_embeddings()
    embed_weight = embed.weight
    lm_head = getattr(language_model, "lm_head", None)
    lm_head_weight = None if lm_head is None else getattr(lm_head, "weight", None)
    weights_tied = (
        lm_head_weight is not None
        and embed_weight.shape == lm_head_weight.shape
        and embed_weight.data_ptr() == lm_head_weight.data_ptr()
    )
    unk_id = getattr(tokenizer, "unk_token_id", None)

    def log(message: str):
        if accelerator is not None:
            accelerator.print(message)

    def resolve_target_id(token_text: str):
        token_id = tokenizer.convert_tokens_to_ids(token_text)
        if token_id is None or int(token_id) < 0:
            log(f"  skip {token_text}: token is not in tokenizer.")
            return None
        if unk_id is not None and int(token_id) == int(unk_id) and token_text != getattr(tokenizer, "unk_token", None):
            log(f"  skip {token_text}: token resolves to unk id {unk_id}.")
            return None
        if int(token_id) >= int(embed_weight.shape[0]):
            log(f"  skip {token_text}: token id {int(token_id)} exceeds embedding rows {embed_weight.shape[0]}.")
            return None
        if lm_head_weight is not None and int(token_id) >= int(lm_head_weight.shape[0]):
            log(f"  skip {token_text}: token id {int(token_id)} exceeds lm_head rows {lm_head_weight.shape[0]}.")
            return None
        return int(token_id)

    def encode_source_ids(source_words):
        ids = []
        for word in source_words:
            word_ids = tokenizer.encode(str(word), add_special_tokens=False)
            ids.extend(int(idx) for idx in word_ids if 0 <= int(idx) < int(embed_weight.shape[0]))
        return ids

    log("Initializing action special token embedding/lm_head rows from word tokens...")
    with torch.no_grad():
        for target_token, source_words in token_sources.items():
            target_id = resolve_target_id(target_token)
            if target_id is None:
                continue
            source_ids = encode_source_ids(source_words)
            if not source_ids:
                log(f"  skip {target_token}: could not encode source words {source_words}.")
                continue
            source_tensor = torch.tensor(source_ids, device=embed_weight.device, dtype=torch.long)
            embed_weight[target_id].copy_(embed_weight.index_select(0, source_tensor).mean(dim=0))
            if lm_head_weight is not None and not weights_tied:
                head_source_tensor = source_tensor.to(device=lm_head_weight.device)
                lm_head_weight[target_id].copy_(lm_head_weight.index_select(0, head_source_tensor).mean(dim=0))
            log(
                f"  {target_token} <- mean({', '.join(source_words)}) "
                f"target_id={target_id}, source_ids={source_ids}, tied_lm_head={int(weights_tied)}"
            )


def save_checkpoint(model, processor, accelerator, args, epoch, global_step, stats_data=None):
    save_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}-step-{global_step}")

    if accelerator.is_main_process:
        if hasattr(args, 'max_ckpts') and args.max_ckpts > 0:
            checkpoint_dirs = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint-")]
            if len(checkpoint_dirs) >= args.max_ckpts:
                oldest_ckpt = min(checkpoint_dirs, key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
                shutil.rmtree(os.path.join(args.output_dir, oldest_ckpt))

        os.makedirs(save_dir, exist_ok=True)
        full_state_dict = accelerator.get_state_dict(model)
        torch.save(full_state_dict, os.path.join(save_dir, "cosmos_janus_mot.pt"))
        processor.save_pretrained(save_dir)

        if stats_data is not None:
            with open(os.path.join(save_dir, 'train_statistics.json'), 'w') as f:
                json.dump(stats_data, f, indent=2)
        with open(os.path.join(save_dir, SPECIAL_TOKEN_VOCAB_FILENAME), 'w') as f:
            json.dump(list(args.special_token_vocab), f, indent=2)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info(f'Checkpoint {epoch}-{global_step} saved to {save_dir}')


def should_use_gt_latent_action_single_forward(epoch_idx, action_gt_latent_after_epoch):
    """Return whether the current epoch should use GT-latent single-forward action loss.

    Args:
        epoch_idx: zero-based epoch index from the training loop.
        action_gt_latent_after_epoch: scheduling threshold. -1 keeps legacy
            two-pass training forever; 0 switches immediately from epoch 1;
            N >= 1 switches starting from epoch N + 1.
    """
    if action_gt_latent_after_epoch == -1:
        return False
    return (epoch_idx + 1) > action_gt_latent_after_epoch


def train(args):
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        wandb.init(project=args.experiment_name, name=args.run_name, config=args, dir=args.log_dir)

    if not args.model_path:
        args.model_path = args.qwen3vl2b_model_path
    processor_path = args.qwen3vl2b_model_path
    processor = load_trex_processor(processor_path)
    args.special_token_init_ids = resolve_special_token_init_ids(processor.tokenizer, args.special_token_vocab)
    args.trex_image_token_id = int(processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    accelerator.print(f"trex_processor={processor_path}")
    accelerator.print(f"special_token_vocab={args.special_token_vocab}")
    accelerator.print(f"special_token_vocab_size={len(args.special_token_vocab)}")

    # ----------------------------------------------------------------
    # Load Qwen3VL base for the first 28 right-branch layers, then transplant
    # T-Rex flow/action modules and last 4 action layers.
    # ----------------------------------------------------------------
    accelerator.print("Loading Qwen3VL2B base checkpoint...")
    janus_model, _ = TrexActionModel.from_qwen3vl_checkpoint(
        args.qwen3vl2b_model_path,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
        torch_dtype=torch.bfloat16,
        use_robot_state=bool(args.robot_state),
        verbose=accelerator.is_main_process,
    )
    accelerator.print("Loading T-Rex action expert checkpoint...")
    trex_action_model, trex_loading_info = TrexActionModel.from_checkpoint(
        args.action_expert_path,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
        torch_dtype=torch.bfloat16,
        use_robot_state=bool(args.robot_state),
        verbose=accelerator.is_main_process,
    )
    janus_model.transplant_action_components_from(trex_action_model, fast_layer_count=4)
    del trex_action_model
    gc.collect()
    accelerator.print(f"T-Rex skipped mismatched tensors={len(trex_loading_info['skipped_mismatch'])}")

    # ----------------------------------------------------------------
    # Load Cosmos
    # ----------------------------------------------------------------
    accelerator.print("Loading Cosmos Video Base...")
    experiment_opts = ["data_train=mock", "data_val=mock","model.config.net.sac_config.mode=none"]
    import cosmos_predict2._src.predict2.models.text2world_model_rectified_flow as t2w_module
    class DummyTextEncoder(nn.Module):
        def __init__(self, *a, **kw): super().__init__()
    t2w_module.TextEncoder = DummyTextEncoder

    cosmos_wrapper, _ = load_model_from_checkpoint(
        experiment_name=args.cosmos_experiment_name,
        s3_checkpoint_dir=args.cosmos_model_path,
        config_file="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        load_ema_to_reg=True,
        to_device=accelerator.device.type,
        experiment_opts=experiment_opts,
    )
    resolve_video_condition_args(args, cosmos_wrapper.tokenizer)

    # ----------------------------------------------------------------
    # Build Cosmos + action MoT
    # ----------------------------------------------------------------
    accelerator.print("Building 2-MoT Cosmos-Janus action-spatial VLA...")
    args.janus_image_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    args.janus_image_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    args.latent_end_id = processor.tokenizer.eos_token_id

    args.total_spatial_tokens = args.total_latent_tokens
    args.use_value_prediction = 0
    args.use_action_value_prediction = 0
    model = CosmosJanusActionSpatialMoT2Expert(
        cosmos_wrapper.net, cosmos_wrapper.tokenizer, janus_model, args
    ).to(accelerator.device, torch.bfloat16)
    accelerator.print(f"action_chunk={args.action_chunk}")
    accelerator.print(f"future_frame_stride={args.future_frame_stride}")
    accelerator.print(
        "cosmos_frame_window="
        f"pixel_total={args.video_frames}, "
        f"pixel_condition={args.num_cond_input_frames}, "
        f"latent_total={args.total_video_latent_frames}, "
        f"latent_condition={args.num_cond_latent_frames}"
    )
    accelerator.print(
        "attention=cosmos_full_action_causal_action_to_cosmos"
    )
    accelerator.print("train_embed_tokens=1")
    accelerator.print("no_detach_spatial_input=1")
    accelerator.print(f"state_encoding_mode={getattr(args, 'state_encoding_mode', 'token')}")
    accelerator.print("decosmos=0")
    accelerator.print(
        f"cosmos_text_cache_enabled={int(bool(getattr(args, 'cosmos_text_cache_path', '')))}"
    )
    accelerator.print(
        "action_context_prefix=image_prompt_spatial_time_action"
    )
    accelerator.print(f"spatial_token_mode={getattr(args, 'latent_token_mode', 'v')}")
    accelerator.print(f"spatial_token_fields={','.join(getattr(args, 'latent_token_fields', ['gtlatent']))}")
    accelerator.print(f"total_spatial_tokens={getattr(args, 'total_latent_tokens', 1)}")
    accelerator.print(
        f"use_history_trajectory_janus_image={int(bool(getattr(args, 'use_history_trajectory_janus_image', 0)))}"
    )
    accelerator.print(
        "use_action_value_prediction=0"
    )
    accelerator.print(f"use_latent_hidden_sim_loss={int(bool(getattr(args, 'use_latent_hidden_sim_loss', 0)))}")
    accelerator.print(f"latent_hidden_sim_loss_mode={getattr(args, 'latent_hidden_sim_loss_mode', 'siglip')}")
    accelerator.print(f"latent_hidden_sim_pool_mode={getattr(args, 'latent_hidden_sim_pool_mode', 'pool')}")
    accelerator.print(
        "use_latent_hidden_wan_downsample_sim_loss="
        f"{int(bool(getattr(args, 'use_latent_hidden_wan_downsample_sim_loss', 0)))}"
    )
    accelerator.print(
        "latent_hidden_wan_downsample_sim_loss_weight="
        f"{getattr(args, 'latent_hidden_wan_downsample_sim_loss_weight', 0.0)}"
    )
    accelerator.print(f"wan21_vae_path={getattr(args, 'wan21_vae_path', '')}")
    accelerator.print("value_prediction=0")
    accelerator.print(f"state_latents_per_future={getattr(args, 'state_latents_per_future', 0)}")
    accelerator.print(f"bridge_pos_scheme={getattr(args, 'bridge_pos_scheme', 'mrope')}")
    accelerator.print(f"qwen3vl2b_model_path={getattr(args, 'qwen3vl2b_model_path', '')}")
    accelerator.print(f"right_single_attn_position={getattr(args, 'right_single_attn_position', 'last4')}")
    accelerator.print("right_branch_layers=32 prefix_layers=28 action_layers=4")
    accelerator.print(f"cosmos_janus_mot2_module={getattr(cosmos_janus_mot2_module, '__file__', 'N/A')}")

    accelerator.print("检查词表")
    accelerator.print(f"len(processor.tokenizer) = {len(processor.tokenizer)}")
    accelerator.print(f"processor.original_tokenizer_len = {getattr(processor, 'original_tokenizer_len', 'N/A')}")
    accelerator.print(f"processor.num_add_tokens = {getattr(processor, 'num_add_tokens', 'N/A')}")

    accelerator.print("检查词嵌入矩阵")
    accelerator.print(
        f"embed_tokens.weight.shape = "
        f"{tuple(model.janus.language_model.model.embed_tokens.weight.shape)}"
    )


    accelerator.print("==== 检查 tokenizer / embedding 对齐 ====")

    tokenizer = processor.tokenizer
    embed = model.janus.language_model.model.embed_tokens

    accelerator.print(f"len(tokenizer) = {len(tokenizer)}")
    accelerator.print(f"embed_tokens.weight.shape = {tuple(embed.weight.shape)}")
    accelerator.print(f"aligned = {len(tokenizer) == embed.weight.shape[0]}")

    lm_head = getattr(model.janus.language_model, "lm_head", None)
    if lm_head is not None and getattr(lm_head, "weight", None) is not None:
        accelerator.print(f"lm_head.weight.shape = {tuple(lm_head.weight.shape)}")
        accelerator.print(f"lm_head aligned = {len(tokenizer) == lm_head.weight.shape[0]}")

    for tok in [
        "<image_placeholder>",
        "<begin_of_image>",
        "<end_of_image>",
        "<|latent_pad|>",
        "<|latent_end|>",
        "</MOVE>",
        "</PICK>",
        "</PLACE>",
        "</ROTATE>",
        "</PULL>",
        "</PUSH>",
        "</NONE>",
    ]:
        accelerator.print(f"{tok:<22} -> {tokenizer.convert_tokens_to_ids(tok)}")

    # Freeze non-trainable modules
    for param in model.parameters():
        param.requires_grad = True

    frozen_modules = [
        "janus.vla.visual",
        "cosmos_vae",
    ]
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in frozen_modules):
            param.requires_grad = False

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    frozen_count = total_count - trainable_count

    accelerator.print("\n==== Parameter Status ====")
    accelerator.print(f"Total:     {total_count / 1e9:.2f}B")
    accelerator.print(f"Trainable: {trainable_count / 1e9:.2f}B ({trainable_count / total_count * 100:.1f}%)")
    accelerator.print(f"Frozen:    {frozen_count / 1e9:.2f}B")
    accelerator.print(f"Frozen:    {', '.join(frozen_modules)}\n")

    # Optimizer with separate LR for cosmos core vs bridge/expert vs janus
    # Bridge & expert params live under cosmos_dit.blocks.*.self_attn.{latent,action}_bridge
    # but should train at full LR, not cosmos LR.
    cosmos_core_params = []
    bridge_expert_params = []
    janus_params = []

    cosmos_debug_name=[]
    bridge_debug_name=[]
    janus_debug_name=[]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'cosmos_dit' in name:
            if 'action_bridge' in name:
                bridge_debug_name.append(name)
                bridge_expert_params.append(param)
            else:
                cosmos_debug_name.append(name)
                cosmos_core_params.append(param)
        else:
            janus_debug_name.append(name)
            janus_params.append(param)
    #accelerator.print("==== Model Architecture ====")
    #accelerator.print(model)
    #accelerator.print("Parameter groups for janus:")
    #accelerator.print(janus_debug_name)
    #accelerator.print("Parameter groups for bridge_expert:")
    #accelerator.print(bridge_debug_name)
    #accelerator.print("Parameter groups for cosmos_core:")
    #accelerator.print(cosmos_debug_name)

    video_lr = args.learning_rate * args.cosmos_core_lr_ratio
    action_lr = args.learning_rate

    accelerator.print(f"Param groups: cosmos_core={len(cosmos_core_params)}, "
                      f"bridge_expert={len(bridge_expert_params)}, janus={len(janus_params)}")
    accelerator.print(
        f"Learning rates: cosmos_core={video_lr:.6g} "
        f"(base_lr * {args.cosmos_core_lr_ratio}), "
        f"bridge_expert={action_lr:.6g}, janus={action_lr:.6g}"
    )

    optimizer = torch.optim.AdamW([
        {"params": cosmos_core_params, "lr": video_lr, "weight_decay": args.weight_decay},
        {"params": bridge_expert_params, "lr": action_lr, "weight_decay": args.weight_decay},
        {"params": janus_params, "lr": action_lr, "weight_decay": args.weight_decay},
    ])

    # Dataset & DataLoader
    train_dataset = VLACotDataset(args, processor, accelerator)
    use_pin_memory = bool(args.pin_memory)
    num_workers = max(int(args.num_workers), 0)
    use_persistent_workers = bool(args.persistent_workers) and num_workers > 0
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_bsz_per_gpu,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
        prefetch_factor=4,
    )

    world_size = int(getattr(accelerator, "num_processes", 1) or 1)
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    num_training_steps = int(len(train_dataloader) * args.n_epochs) // accelerator.gradient_accumulation_steps // world_size
    accelerator.print(
        "num_training_steps="
        f"{num_training_steps} "
        f"(len_dataloader={len(train_dataloader)}, epochs={args.n_epochs}, "
        f"grad_accum={accelerator.gradient_accumulation_steps}, world_size={world_size})"
    )
    lr_scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    model.train()
    global_step = 0
    video_frozen = False

    for epoch in range(args.n_epochs):
        accelerator.print(
            f">>> Epoch {epoch + 1}/{args.n_epochs}: action training mode = token_ce_gt_embedding"
        )

        # Freeze video backbone at or after the specified epoch.
        if not video_frozen and args.freeze_video_after >= 0 and epoch >= args.freeze_video_after:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.freeze_video_backbone()
            video_frozen = True
            accelerator.print(f">>> Epoch {epoch}: Froze Cosmos video backbone.")

        train_iter = train_dataloader
        if accelerator.is_main_process:
            from tqdm import tqdm
            train_iter = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

        for batch in train_iter:
            with accelerator.accumulate(model):
                janus_input_ids = batch['janus_input_ids'].to(accelerator.device)
                janus_left_pad_lens = batch['janus_left_pad_lens'].to(accelerator.device)
                janus_pixel_values = batch['janus_pixel_values'].to(accelerator.device).to(torch.bfloat16)
                janus_image_grid_thw = batch['janus_image_grid_thw'].to(accelerator.device)
                janus_images_seq_mask = batch['janus_images_seq_mask'].to(accelerator.device)
                janus_state_seq_mask = batch['janus_state_seq_mask'].to(accelerator.device)
                janus_images_emb_mask = batch['janus_images_emb_mask'].to(accelerator.device)
                cosmos_janus_input_ids = batch['cosmos_janus_input_ids'].to(accelerator.device)
                cosmos_janus_image_grid_thw = batch['cosmos_janus_image_grid_thw'].to(accelerator.device)
                cosmos_janus_images_seq_mask = batch['cosmos_janus_images_seq_mask'].to(accelerator.device)
                cosmos_janus_state_seq_mask = batch['cosmos_janus_state_seq_mask'].to(accelerator.device)
                cosmos_janus_images_emb_mask = batch['cosmos_janus_images_emb_mask'].to(accelerator.device)
                attention_mask = batch['attention_mask'].to(accelerator.device)
                now_state = batch['now_state']
                if now_state is not None:
                    now_state = now_state.to(accelerator.device)
                actions = batch['actions'].to(accelerator.device)
                videos = batch['videos'].to(accelerator.device).to(torch.bfloat16)
                latent_gt_token_ids = batch['gt_latent_token_ids'].to(accelerator.device)
                spatial_hidden_sim_pixel_values = batch.get('latent_hidden_sim_pixel_values')
                if spatial_hidden_sim_pixel_values is not None:
                    spatial_hidden_sim_pixel_values = spatial_hidden_sim_pixel_values.to(
                        accelerator.device,
                        non_blocking=True,
                    ).to(torch.bfloat16)
                spatial_hidden_sim_image_grid_thw = batch.get('latent_hidden_sim_image_grid_thw')
                if spatial_hidden_sim_image_grid_thw is not None:
                    spatial_hidden_sim_image_grid_thw = spatial_hidden_sim_image_grid_thw.to(
                        accelerator.device,
                        non_blocking=True,
                    )
                cosmos_text_embeddings = batch.get('cosmos_text_embeddings')
                if cosmos_text_embeddings is not None:
                    cosmos_text_embeddings = cosmos_text_embeddings.to(
                        accelerator.device,
                        non_blocking=True,
                    ).to(torch.bfloat16)
                first_frame = videos[:, :, 0]
                video_frames = videos[:, :, 1:]

                # ----- Forward -----
                video_loss_weight = 0.0 if video_frozen else args.video_loss_weight
                loss_weights = (
                    video_loss_weight,
                    1.0,
                    args.latent_loss_weight,
                )
                if args.use_latent_hidden_sim_loss:
                    loss_weights = loss_weights + (args.latent_hidden_sim_loss_weight,)
                if args.use_latent_hidden_wan_downsample_sim_loss:
                    loss_weights = loss_weights + (args.latent_hidden_wan_downsample_sim_loss_weight,)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss, v_loss, a_loss, spatial_ce_loss, aux_metrics = model(
                        first_frame=first_frame,
                        video_frames=video_frames,
                        actions=actions,
                        janus_input_ids=janus_input_ids,
                        janus_left_pad_lens=janus_left_pad_lens,
                        janus_pixel_values=janus_pixel_values,
                        janus_image_grid_thw=janus_image_grid_thw,
                        janus_images_seq_mask=janus_images_seq_mask,
                        janus_state_seq_mask=janus_state_seq_mask,
                        janus_images_emb_mask=janus_images_emb_mask,
                        janus_attention_mask=attention_mask,
                        cosmos_janus_input_ids=cosmos_janus_input_ids,
                        janus_action_pixel_values=None,
                        cosmos_janus_images_seq_mask=cosmos_janus_images_seq_mask,
                        cosmos_janus_state_seq_mask=cosmos_janus_state_seq_mask,
                        cosmos_janus_images_emb_mask=cosmos_janus_images_emb_mask,
                        cosmos_janus_image_grid_thw=cosmos_janus_image_grid_thw,
                        now_state=now_state,
                        fps=args.fps,
                        spatial_gt_token_ids=latent_gt_token_ids,
                        spatial_hidden_sim_pixel_values=spatial_hidden_sim_pixel_values,
                        spatial_hidden_sim_image_grid_thw=spatial_hidden_sim_image_grid_thw,
                        cosmos_text_embeddings=cosmos_text_embeddings,
                        loss_weights=loss_weights,
                    )

                accelerator.backward(loss)

                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                lr_scheduler.step()
                global_step += 1
                if accelerator.is_main_process:
                    log_payload = {
                        'total_loss': loss.item(),
                        'video_loss': v_loss,
                        'action_loss': a_loss,
                        'spatial_ce_loss': spatial_ce_loss,
                        'lr': lr_scheduler.get_last_lr()[1],
                    }
                    for key in (
                        "spatial_hidden_wan_vae_mse_loss",
                        "spatial_hidden_wan_downsample_sim_loss",
                    ):
                        if key in aux_metrics:
                            log_payload[key] = aux_metrics[key]
                    if "spatial_hidden_sim_loss" in aux_metrics:
                        log_payload["spatial_hidden_sim_loss"] = aux_metrics["spatial_hidden_sim_loss"]
                    wandb.log(log_payload, step=global_step)

                    postfix = {
                        "v": f"{v_loss:.4f}",
                        "a": f"{a_loss:.4f}",
                        "sp": f"{spatial_ce_loss:.4f}",
                    }
                    if "spatial_hidden_sim_loss" in aux_metrics:
                        postfix["hsim"] = f"{aux_metrics['spatial_hidden_sim_loss']:.4f}"
                    if "spatial_hidden_wan_vae_mse_loss" in aux_metrics:
                        postfix["wvae"] = f"{aux_metrics['spatial_hidden_wan_vae_mse_loss']:.4f}"
                    if "spatial_hidden_wan_downsample_sim_loss" in aux_metrics:
                        postfix["wdsim"] = f"{aux_metrics['spatial_hidden_wan_downsample_sim_loss']:.4f}"
                    train_iter.set_postfix(postfix,refresh=False)

        if ((epoch + 1) % args.save_freq == 0) or (epoch == args.n_epochs - 1):
            accelerator.wait_for_everyone()
            save_checkpoint(
                model=model, processor=processor, accelerator=accelerator,
                args=args, epoch=epoch, global_step=global_step,
                stats_data=train_dataset.stats_data,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Base config
    parser.add_argument('--experiment_name', type=str, default='cosmos_janus_mot2_action_spatial')
    parser.add_argument('--run_name', type=str, default='run_1')
    parser.add_argument('--model_path', type=str, default='',
                        help='Deprecated compatibility alias; action_expert_path is used for Janus loading.')
    parser.add_argument('--action_expert_path', type=str, required=True,
                        help='Path to action-only pretrained checkpoint (provides action expert + flow matching weights)')

    # Data
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--use_history_trajectory_janus_image', type=int, default=0,
                        help='Compatibility option; RLBench image JSON does not use history-trajectory videos.')

    # Video
    parser.add_argument('--video_h', type=int, default=256)
    parser.add_argument('--video_w', type=int, default=256)
    parser.add_argument('--video_frames', type=int, default=5)
    parser.add_argument('--num_cond_input_frames', type=int, default=1,
                        help='Number of raw pixel frames used as Cosmos history conditioning.')
    parser.add_argument('--fps', type=int, default=10)

    # Cosmos
    parser.add_argument('--cosmos_model_path', type=str, required=True)
    parser.add_argument('--cosmos_experiment_name', type=str,
                        default='Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only')
    parser.add_argument('--cosmos_text_cache_path', type=str, default='',
                        help='If non-empty, load precomputed native Cosmos Qwen text embeddings from this sidecar cache.')

    # Training
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--train_bsz_per_gpu', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--cosmos_core_lr_ratio', type=float, default=0.1,
                        help='LR multiplier for cosmos_core params relative to --learning_rate')
    parser.add_argument('--min_lr_ratio', type=float, default=0.05)
    parser.add_argument('--warmup_rates', type=float, default=0.05)
    parser.add_argument('--robot_state', type=int, default=0)
    parser.add_argument('--state_placeholder_tokens', type=int, default=1,
                        help='Number of pad placeholder tokens reserved for current robot state.')
    parser.add_argument('--state_dim', type=int, default=7,
                        help='RLBench robot state/action state dimension used by MLP state encoding.')
    parser.add_argument('--action_dim', type=int, default=7)
    parser.add_argument('--action_chunk', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader worker processes per training process.')
    parser.add_argument('--pin_memory', type=int, default=1,
                        help='If 1, pin CPU batch memory before host-to-device transfer.')
    parser.add_argument('--persistent_workers', type=int, default=1,
                        help='If 1 and num_workers > 0, keep DataLoader workers alive across epochs.')

    # Action expert
    parser.add_argument('--qwen3vl2b_model_path', type=str,
                        default='/mnt/amlfs-07/shared/physicalword/ckpt/pretraine/Qwen3-VL-2B-Instruct',
                        help='Path to the plain Qwen3VL2B checkpoint for the first 28 right-branch layers.')
    parser.add_argument('--action_intermediate_size', type=int, default=0,
                        help='If >0, slim action MLP intermediate size')
    parser.add_argument('--action_self_causal_in_bridge', type=int, default=1,
                        help='If 1, training uses causal action->action bridge attention; if 0, action->action is bidirectional')
    parser.add_argument('--right_single_attn_position', type=str, default='last4',
                        choices=['first4', 'last4'],
                        help='Where to place the 4 right-branch standalone layers needed to align 32 right layers with 28 Cosmos layers.')


    # Latent CoT
    parser.add_argument('--total_latent_tokens', type=str, default=None,
                        help='Latent token mode/count. Accepts v/1, n, or vn/2; defaults to v.')
    parser.add_argument('--latent_token_mode', type=str, default='',
                        help='Explicit latent token supervision mode: v=gtlatent, n=gtlatent2, vn=gtlatent+gtlatent2.')
    parser.add_argument('--special_token_vocab', type=str, default=','.join(DEFAULT_SPECIAL_TOKEN_VOCAB),
                        help='Comma- or whitespace-separated independent spatial-token vocabulary.')
    parser.add_argument('--img_latents_per_future', type=int, default=0,
                        help='Legacy compatibility option; ignored by token latent CE training')
    parser.add_argument('--state_latents_per_future', type=int, default=0,
                        help='Legacy compatibility option; ignored by token latent CE training')
    parser.add_argument('--num_future_frames', type=int, default=0,
                        help='Legacy compatibility option; ignored by token latent CE training')
    parser.add_argument('--future_frame_stride', type=int, default=1,
                        help='Legacy compatibility option; ignored by token latent CE training.')
    parser.add_argument('--video_loss_weight', type=float, default=0.0,
                        help='Weight for the video loss before video freeze. After freeze it is forced to 0.')
    parser.add_argument('--latent_loss_weight', type=float, default=1.0,
                        help='Weight for latent token CE loss')
    parser.add_argument('--use_latent_hidden_sim_loss', type=int, default=0,
                        help='If 1, add hidden-state cosine loss against Janus-encoded frame_index + future_frame_stride image.')
    parser.add_argument('--latent_hidden_sim_loss_mode', type=str, default='siglip',
                        choices=['siglip', 'wan_vae'],
                        help='siglip keeps the current Janus future-image cosine loss; wan_vae uses frozen Wan2.1 VAE latent MSE.')
    parser.add_argument('--latent_hidden_sim_pool_mode', type=str, default='pool',
                        choices=['pool', 'one_mlp', 'mlp'],
                        help='SigLIP hidden sim target pooling: pool keeps mean pooling; one_mlp shares attention/proj MLPs; mlp uses per-spatial-token MLPs.')
    parser.add_argument('--latent_hidden_sim_loss_weight', type=float, default=1.0,
                        help='Independent weight for latent hidden-state cosine loss when --use_latent_hidden_sim_loss=1.')
    parser.add_argument('--use_latent_hidden_wan_downsample_sim_loss', type=int, default=0,
                        help='If 1 with wan_vae sim loss, add cosine loss from frozen Wan target latent downsampled to the latent hidden state.')
    parser.add_argument('--latent_hidden_wan_downsample_sim_loss_weight', type=float, default=1.0,
                        help='Independent weight for the Wan-latent downsample hidden cosine loss.')
    parser.add_argument('--wan21_vae_path', type=str,
                        default='/mnt/nas/zhangyiming/database/ckpt/pretrained/wan2.1_vae/original/Wan2.1_VAE.pth',
                        help='Wan2.1 VAE checkpoint used only by latent_hidden_sim_loss_mode=wan_vae.')
    parser.add_argument('--use_value_prediction', type=int, default=0,
                        help='If 1, append an extra Cosmos latent time slice for scalar value prediction.')
    parser.add_argument('--use_action_value_prediction', type=int, default=0,
                        help='If 1, append one noisy scalar value token to the action branch and train its velocity.')
    parser.add_argument('--value_token_mask_video_to_value', type=int, default=0,
                        help='If 1 with value prediction, non-value video tokens cannot attend to value tokens in MoT attention.')
    parser.add_argument('--value_token_mask_nonvalue_to_value', type=int, default=0,
                        help='If 1 with value prediction, all non-value tokens cannot attend to value tokens in MoT attention.')
    parser.add_argument('--value_loss_weight', type=float, default=1.0,
                        help='Weight for the Cosmos scalar value prediction loss when --use_value_prediction=1.')
    parser.add_argument('--action_value_loss_weight', type=float, default=1.0,
                        help='Weight for the action-branch scalar value prediction loss when --use_action_value_prediction=1.')
    parser.add_argument('--action_gt_latent_after_epoch', type=int, default=-1,
                        help='-1 keeps legacy two-pass action training forever; 0 starts single-forward GT-latent action training from epoch 1; N>=1 keeps the first N epochs in legacy mode, then switches starting from epoch N+1.')
    parser.add_argument('--cosmos_self_only_bridge', type=int, default=0,
                        help='If 1, restrict cosmos bridge attention to video-only KV during training/eval')
    parser.add_argument('--train_embed_tokens', type=int, default=0,
                        help='If 1, compute Janus prepare_inputs_embeds with grad so embed_tokens can train. If 0, keep it under no_grad.')
    parser.add_argument('--no_detach_latent_input', type=int, default=0,
                        help='If 1, do not detach latent branch inputs during training, so context token embeddings can receive gradients. Default 0 keeps legacy context-detach behavior.')
    parser.add_argument('--state_encoding_mode', type=str, default='mlp',
                        choices=['token', 'mlp'],
                        help='Encode robot state as tokenizer ids ("token") or normalized float values through a trainable MLP ("mlp").')
    parser.add_argument('--decosmos', type=int, default=0,
                        help='If 1, train latent/action branches without attending to Cosmos KV and project Cosmos cross-attention embeddings under no_grad.')
    parser.add_argument('--action_use_latent_prefix', type=int, default=0,
                        help='If 1, prepend encoded wrist image and current state tokens to the action branch.')
    parser.add_argument('--bridge_pos_scheme', type=str, default='mrope',
                        choices=['mrope', 'mrope_interleave', 'llama1d', 'qwen', 'local', 'last0'],
                        help='Bridge rotary mode for right-branch QK. `qwen` uses native Qwen3-VL M-RoPE; `mrope` uses Cosmos-aligned 3D RoPE; `llama1d` uses 1-D RoPE.')
    # Freeze
    parser.add_argument('--freeze_video_after', type=int, default=-1,
                        help='Freeze Cosmos when epoch >= this value (0-indexed). 0 = freeze from epoch 0. -1 = never.')

    args = parser.parse_args()
    args.cosmos_text_cache_path = str(args.cosmos_text_cache_path or "").strip()
    args.latent_hidden_sim_loss_mode = str(args.latent_hidden_sim_loss_mode or "siglip").lower()
    args.latent_hidden_sim_pool_mode = str(args.latent_hidden_sim_pool_mode or "pool").lower()
    args.spatial_hidden_sim_pool_mode = args.latent_hidden_sim_pool_mode
    args.wan21_vae_path = str(args.wan21_vae_path or "").strip()
    args.bridge_pos_scheme = cosmos_janus_mot2_module.normalize_bridge_pos_scheme(args.bridge_pos_scheme)
    resolve_special_token_vocab_args(args)
    resolve_latent_token_args(args)
    resolve_video_condition_args(args)
    args.total_spatial_tokens = args.total_latent_tokens
    args.train_embed_tokens = 1
    args.no_detach_latent_input = 1
    args.decosmos = 0
    args.cosmos_self_only_bridge = 0
    args.action_use_latent_prefix = 1
    args.action_self_causal_in_bridge = 1
    args.use_value_prediction = 0
    args.use_action_value_prediction = 0
    args.value_token_mask_video_to_value = 0
    args.value_token_mask_nonvalue_to_value = 0

    if args.cosmos_core_lr_ratio <= 0:
        raise ValueError("cosmos_core_lr_ratio must be positive.")
    if args.video_loss_weight < 0:
        raise ValueError("video_loss_weight must be non-negative.")
    if args.state_placeholder_tokens < 0:
        raise ValueError("state_placeholder_tokens must be non-negative.")
    if args.robot_state and args.state_placeholder_tokens <= 0:
        raise ValueError("state_placeholder_tokens must be positive when robot_state is enabled.")
    if args.state_encoding_mode == 'mlp' and args.robot_state and args.state_placeholder_tokens != 1:
        raise ValueError("state_encoding_mode='mlp' requires state_placeholder_tokens=1 when robot_state is enabled.")
    if args.use_history_trajectory_janus_image not in (0, 1):
        raise ValueError("use_history_trajectory_janus_image must be 0 or 1.")
    if args.right_single_attn_position not in ("first4", "last4"):
        raise ValueError("right_single_attn_position must be 'first4' or 'last4'.")
    if args.use_latent_hidden_sim_loss not in (0, 1):
        raise ValueError("use_latent_hidden_sim_loss must be 0 or 1.")
    if args.use_latent_hidden_wan_downsample_sim_loss not in (0, 1):
        raise ValueError("use_latent_hidden_wan_downsample_sim_loss must be 0 or 1.")
    if args.latent_hidden_sim_loss_mode not in ("siglip", "wan_vae"):
        raise ValueError("latent_hidden_sim_loss_mode must be 'siglip' or 'wan_vae'.")
    if args.latent_hidden_sim_pool_mode not in ("pool", "one_mlp", "mlp"):
        raise ValueError("latent_hidden_sim_pool_mode must be 'pool', 'one_mlp', or 'mlp'.")
    if args.use_latent_hidden_sim_loss and args.future_frame_stride <= 0:
        raise ValueError("use_latent_hidden_sim_loss requires future_frame_stride > 0.")
    if args.use_latent_hidden_sim_loss and args.latent_hidden_sim_loss_mode == "wan_vae":
        if not args.wan21_vae_path:
            raise ValueError("wan_vae sim loss requires --wan21_vae_path.")
        if not os.path.exists(args.wan21_vae_path):
            raise FileNotFoundError(f"Wan2.1 VAE checkpoint not found: {args.wan21_vae_path}")
    if args.latent_hidden_sim_loss_weight < 0:
        raise ValueError("latent_hidden_sim_loss_weight must be non-negative.")
    if args.use_latent_hidden_wan_downsample_sim_loss and not (
        args.use_latent_hidden_sim_loss and args.latent_hidden_sim_loss_mode == "wan_vae"
    ):
        raise ValueError(
            "use_latent_hidden_wan_downsample_sim_loss requires "
            "use_latent_hidden_sim_loss=1 and latent_hidden_sim_loss_mode='wan_vae'."
        )
    if args.latent_hidden_wan_downsample_sim_loss_weight < 0:
        raise ValueError("latent_hidden_wan_downsample_sim_loss_weight must be non-negative.")
    args.log_dir = os.path.join(args.log_dir, args.run_name)
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.log_dir, exist_ok=True)

    train(args)
