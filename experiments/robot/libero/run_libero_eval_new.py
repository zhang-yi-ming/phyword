"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""
import faulthandler
faulthandler.enable()

import json
import logging
import os
import random
import re
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import draccus
import imageio
import numpy as np
import tqdm
from libero.libero import benchmark
from PIL import Image, ImageDraw, ImageFont
import wandb

import torch
import torchvision.transforms as transforms
from transformers import AutoModelForCausalLM

# Resolve imports relative to the repository root, not the launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from janus.models import VLChatProcessor, ActionTokenizer
from models.cosmos_janus_action_spatial import (
    CosmosJanusActionSpatialMoT2Expert,
    normalize_bridge_pos_scheme,
)
from models.cosmos_janus_cot import build_token_sequence_mask
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint
from scripts.rewrite_input_prompts import PROMPT_REPLACEMENTS
from utils.cosmos_text_cache import CosmosQwenTextEmbedder, CosmosTextEmbeddingCache

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.libero.history_trajectory_utils import (
    DEFAULT_HISTORY_TRAJECTORY_CAMERA_CONFIG,
    draw_history_trajectory_on_image,
    load_history_trajectory_camera_config,
)

JANUS_ACTION_PROMPT_SUFFIX = (
    "Please refer to the current image and task instruction, predict the spatial token "
    "and output the action to execute now."
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


def build_janus_action_prompt_suffix(use_history_trajectory: bool) -> str:
    return JANUS_ACTION_PROMPT_SUFFIX

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
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    normalized_action = action.copy()
    normalized_action[..., -1] = 2 * normalized_action[..., -1] - 1
    if binarize:
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])
    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    inverted_action = action.copy()
    inverted_action[..., -1] *= -1.0
    return inverted_action

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path (our saved MoT weights)
    model_path: Union[str, Path] = ""                # Path to original Janus Processor & Tokenizer
    action_model_path: Union[str, Path] = ""         # Path to Janus Action Base
    cosmos_experiment_name : str = ""                # Name of the original Cosmos experiment
    cosmos_model_path: Union[str, Path] = ""         # Path to the original Cosmos .pt checkpoint
    cosmos_text_cache_path: str = ""                 # Optional sidecar cache of projected native Cosmos Qwen text embeddings
    rewrite_eval_prompt: bool = False                # If true, map LIBERO task descriptions through scripts/rewrite_input_prompts.py

    use_proprio: bool = False                        # Whether to include proprio state in input

    center_crop: bool = False                        # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy
    action_repeat: int = 1                           # Number of env steps to repeat each queued action

    unnorm_key: Union[str, Path] = "rlbench"         # Action un-normalization key
    
    # Model Architecture Overrides (Matching Training)
    video_h: int = 256
    video_w: int = 256
    video_frames: int = 1
    num_cond_input_frames: int = 1
    action_dim: int = 7
    action_chunk: int = 16
    robot_state: int = 0
    state_placeholder_tokens: int = 8
    state_encoding_mode: str = "mlp"
    action_intermediate_size: int = 0                # If 0, infer slim MLP size from checkpoint; if >0, require an exact match
    model_variant: str = "mot2_action_spatial"      # This eval script builds the 2-MoT Cosmos + action-spatial model.
    total_latent_tokens: int = 1                     # Number of latent token CE tokens to generate at eval time
    special_token_vocab: str = ",".join(DEFAULT_SPECIAL_TOKEN_VOCAB)  # Fallback independent spatial-token vocab
    img_latents_per_future: int = 0
    state_latents_per_future: int = 0
    num_future_frames: int = 0
    future_frame_stride: int = 8                     # Training-time interval between latent CoT future observations
    joint_action_prefill: bool = False               # Legacy compatibility; ignored by the 2-MoT action-spatial model
    cosmos_self_only_bridge: bool = False            # Fixed 2-MoT behavior: action can see Cosmos, Cosmos cannot see action
    decosmos: bool = False                           # Match training option: latent/action do not attend to Cosmos KV; skip Cosmos inference
    use_value_prediction: bool = False               # Fixed 2-MoT behavior: no value prediction branch
    use_action_value_prediction: bool = False        # Fixed 2-MoT behavior: no action value token
    value_token_mask_video_to_value: bool = False     # If true, video tokens cannot attend to value tokens
    value_token_mask_nonvalue_to_value: bool = False  # If true, all non-value tokens cannot attend to value tokens
    bridge_pos_scheme: str = "mrope"                 # Accepts mrope/mrope_interleave/llama1d plus legacy aliases local/last0
    action_use_latent_prefix: bool = True            # Match training option that prepends wrist image + current state to the action branch
    action_use_image_prefix: bool = False            # Legacy alias for action_use_latent_prefix
    use_history_trajectory_janus_image: bool = False  # If true, draw EE history on the Janus primary image at eval time
    history_trajectory_camera_config_path: str = DEFAULT_HISTORY_TRAJECTORY_CAMERA_CONFIG



    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    control_freq: int = 0                           # LIBERO env control frequency (Hz)
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    task_ids: str = ""                               # Optional comma-separated task ids to evaluate
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "../experiments/logs"        # Local directory for eval logs
    predicted_video_save_dir: str = ""               # If set, save each predicted Cosmos video to this directory
    rollout_video_save_dir: str = ""                 # If set, save rollout videos under this directory
    disable_rollout_video: bool = False              # If true, skip rollout mp4 saving entirely
    value_visualization_dir: str = ""                # If set, save value traces and summary plots under this directory
    bash_hparams_path: str = ""                      # Optional shell-generated hparams file to copy into eval logs
    eval_artifact_name: str = ""                     # Shared date+runname stem for eval artifacts
    predicted_video_fps: int = 10                    # FPS for saved predicted Cosmos videos

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 42                                   # Random Seed (for reproducibility)

    cuda: str = "0"                                  # CUDA device to use
    action_denoise_steps: int = 10                     # Number of action denoising steps
    cosmos_denoise_steps: int = 2                      # Number of Cosmos scheduler steps before KV reuse
    fps: float = 20.0                                  # FPS for predicted Cosmos videos
    action_self_causal_in_bridge: bool = True          # Fixed 2-MoT behavior: causal action prefix, final action tokens mutually visible

    # fmt: on


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


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
    row_keys = [
        "special_token_embedding.weight",
        "special_token_lm_head.weight",
    ]
    missing = [key for key in row_keys if key not in state_dict]
    if missing:
        raise ValueError(f"Checkpoint is missing independent special-token weights: {missing}")
    mismatches = []
    for key in row_keys:
        rows = int(state_dict[key].shape[0])
        if rows != expected:
            mismatches.append(f"{key}: checkpoint_rows={rows}, vocab_size={expected}")
    if mismatches:
        raise ValueError("Special-token vocab/checkpoint mismatch: " + ", ".join(mismatches))


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


def resolve_video_condition_config(cfg: Any, video_tokenizer=None) -> None:
    cfg.video_frames = int(getattr(cfg, "video_frames", 16))
    cfg.num_cond_input_frames = int(getattr(cfg, "num_cond_input_frames", 1))
    if cfg.video_frames < 1:
        raise ValueError("video_frames must be positive.")
    if cfg.num_cond_input_frames < 1:
        raise ValueError("num_cond_input_frames must be positive.")
    if cfg.num_cond_input_frames > cfg.video_frames:
        raise ValueError(
            "num_cond_input_frames must be <= video_frames, "
            f"got {cfg.num_cond_input_frames} > {cfg.video_frames}."
        )

    cfg.num_cond_latent_frames = get_video_latent_num_frames(
        video_tokenizer,
        cfg.num_cond_input_frames,
    )
    cfg.total_video_latent_frames = get_video_latent_num_frames(
        video_tokenizer,
        cfg.video_frames,
    )
    if cfg.num_cond_latent_frames > cfg.total_video_latent_frames:
        raise ValueError(
            "num_cond_latent_frames must be <= total_video_latent_frames, "
            f"got {cfg.num_cond_latent_frames} > {cfg.total_video_latent_frames}."
        )


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"
    assert cfg.num_open_loop_steps > 0, "num_open_loop_steps must be positive!"
    assert cfg.action_repeat > 0, "action_repeat must be positive!"
    assert cfg.cosmos_denoise_steps > 0, "cosmos_denoise_steps must be positive!"
    cfg.cosmos_text_cache_path = str(getattr(cfg, "cosmos_text_cache_path", "") or "").strip()
    cfg.predicted_video_save_dir = str(getattr(cfg, "predicted_video_save_dir", "") or "").strip()
    cfg.rollout_video_save_dir = str(getattr(cfg, "rollout_video_save_dir", "") or "").strip()
    cfg.value_visualization_dir = str(getattr(cfg, "value_visualization_dir", "") or "").strip()
    cfg.bash_hparams_path = str(getattr(cfg, "bash_hparams_path", "") or "").strip()
    cfg.eval_artifact_name = str(getattr(cfg, "eval_artifact_name", "") or "").strip()
    resolve_video_condition_config(cfg)
    cfg.robot_state = int(getattr(cfg, "robot_state", 0) or 0)
    cfg.action_use_latent_prefix = coerce_bool(getattr(cfg, "action_use_latent_prefix", False))
    cfg.action_use_image_prefix = coerce_bool(getattr(cfg, "action_use_image_prefix", False))
    if cfg.action_use_image_prefix:
        cfg.action_use_latent_prefix = True
    cfg.use_history_trajectory_janus_image = coerce_bool(
        getattr(cfg, "use_history_trajectory_janus_image", False)
    )
    cfg.history_trajectory_camera_config_path = str(
        getattr(cfg, "history_trajectory_camera_config_path", "") or DEFAULT_HISTORY_TRAJECTORY_CAMERA_CONFIG
    ).strip()
    cfg.cosmos_self_only_bridge = False
    cfg.decosmos = False
    cfg.use_value_prediction = False
    cfg.use_action_value_prediction = False
    cfg.value_token_mask_nonvalue_to_value = False
    cfg.value_token_mask_video_to_value = False
    cfg.action_self_causal_in_bridge = True
    cfg.rewrite_eval_prompt = coerce_bool(getattr(cfg, "rewrite_eval_prompt", False))
    cfg.use_history_trajectory_janus_image = False
    cfg.action_use_latent_prefix = True
    cfg.action_use_image_prefix = False
    cfg.cosmos_self_only_bridge = False
    cfg.decosmos = False
    cfg.use_value_prediction = False
    cfg.use_action_value_prediction = False
    cfg.value_token_mask_nonvalue_to_value = False
    cfg.value_token_mask_video_to_value = False
    cfg.action_self_causal_in_bridge = True
    if int(getattr(cfg, "future_frame_stride", 0) or 0) <= 0:
        cfg.future_frame_stride = cfg.action_chunk
    cfg.bridge_pos_scheme = normalize_bridge_pos_scheme(cfg.bridge_pos_scheme)
    cfg.state_encoding_mode = str(getattr(cfg, "state_encoding_mode", "token")).lower()
    if cfg.state_encoding_mode not in {"token", "mlp"}:
        raise ValueError(f"state_encoding_mode must be 'token' or 'mlp', got {cfg.state_encoding_mode!r}.")
    if int(getattr(cfg, "state_placeholder_tokens", 0) or 0) < 0:
        raise ValueError("state_placeholder_tokens must be non-negative.")
    if cfg.robot_state and int(cfg.state_placeholder_tokens) <= 0:
        raise ValueError("state_placeholder_tokens must be positive when robot_state is enabled.")
    if cfg.state_encoding_mode == "mlp" and cfg.robot_state and int(cfg.state_placeholder_tokens) != 1:
        raise ValueError("state_encoding_mode='mlp' requires state_placeholder_tokens=1 when robot_state is enabled.")
    if int(cfg.total_latent_tokens) not in (1, 2):
        raise ValueError(
            f"Token latent CE inference requires total_latent_tokens=1 or 2, got {cfg.total_latent_tokens}."
        )

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    if cfg.use_history_trajectory_janus_image:
        if cfg.task_suite_name != TaskSuite.LIBERO_SPATIAL.value:
            raise ValueError(
                "use_history_trajectory_janus_image currently supports only libero_spatial."
            )
        if not os.path.exists(cfg.history_trajectory_camera_config_path):
            raise FileNotFoundError(
                "History trajectory camera config not found: "
                f"{cfg.history_trajectory_camera_config_path}"
            )
        
    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def resolve_checkpoint_paths(pretrained_checkpoint: Union[str, Path]):
    pretrained_checkpoint = str(pretrained_checkpoint)
    if pretrained_checkpoint.endswith(".pt"):
        return pretrained_checkpoint, os.path.dirname(pretrained_checkpoint)

    base_dir = pretrained_checkpoint
    candidate_names = ["cosmos_janus_mot.pt", "cosmos_janus_mot3.pt", "mot_action_weights.pt"]

    for name in candidate_names:
        ckpt_path = os.path.join(base_dir, name)
        if os.path.exists(ckpt_path):
            return ckpt_path, base_dir

    return os.path.join(base_dir, candidate_names[0]), base_dir


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
    if model_path and model_path not in candidate_paths:
        candidate_paths.append(model_path)

    last_error = None
    for candidate in candidate_paths:
        try:
            processor = VLChatProcessor.from_pretrained(
                candidate,
                trust_remote_code=True,
                skip_output_special_tokens=True,
            )
            if candidate != model_path:
                logger.info("Loaded VLChatProcessor from %s", candidate)
            return processor
        except Exception as exc:
            last_error = exc
            logger.info("Failed to load VLChatProcessor from %s: %s", candidate, exc)
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
    if hasattr(janus_model.config, "language_config"):
        janus_model.config.language_config.vocab_size = target_vocab
    if hasattr(language_model, "config"):
        language_model.config.vocab_size = target_vocab


def validate_checkpoint_vocab_size(state_dict: dict[str, Any], tokenizer) -> None:
    tokenizer_vocab = int(len(tokenizer))
    vocab_keys = [
        "janus.language_model.model.embed_tokens.weight",
        "janus.language_model.lm_head.weight",
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
            "Checkpoint vocab size is incompatible with this LIBERO tokenizer. "
            "Use the processor/tokenizer saved with the checkpoint, or evaluate with the same extra special tokens. "
            f"Mismatches: {', '.join(mismatches)}"
        )
    if checkpoint_vocab_sizes and checkpoint_vocab_sizes[0] > tokenizer_vocab:
        logger.info(
            "Checkpoint vocab rows (%s) exceed tokenizer length (%s); treating extra rows as padded lm_head/embed rows.",
            checkpoint_vocab_sizes[0],
            tokenizer_vocab,
        )


def infer_action_intermediate_size_from_state_dict(state_dict: dict[str, Any]) -> Optional[int]:
    for key, value in state_dict.items():
        if key.endswith("action_bridge.mlp.gate_proj.weight") and hasattr(value, "shape") and len(value.shape) == 2:
            inferred_size = int(value.shape[0])
            if inferred_size > 0:
                return inferred_size
            return None
    return None


def maybe_infer_action_intermediate_size(cfg: Any, state_dict: dict[str, Any]) -> Optional[int]:
    current_value = int(getattr(cfg, "action_intermediate_size", 0) or 0)
    inferred_size = infer_action_intermediate_size_from_state_dict(state_dict)
    if inferred_size is None:
        return None

    if current_value > 0:
        if current_value != inferred_size:
            raise ValueError(
                "Configured action_intermediate_size does not match checkpoint: "
                f"got {current_value}, but checkpoint requires {inferred_size}. "
                "Set --action_intermediate_size 0 (or omit it) to auto-detect, "
                "or pass the exact checkpoint value."
            )
        print(f"Validated action_intermediate_size={current_value} against checkpoint.")
        return inferred_size

    cfg.action_intermediate_size = inferred_size
    print(f"Auto-detected action_intermediate_size={inferred_size} from checkpoint.")
    return inferred_size


def log_resolved_inference_config(cfg: Any) -> None:
    print("Resolved inference config:")
    print(f"  action_intermediate_size={int(getattr(cfg, 'action_intermediate_size', 0) or 0)}")
    print(f"  bridge_pos_scheme={getattr(cfg, 'bridge_pos_scheme', '')}")
    print(f"  action_use_latent_prefix={int(bool(getattr(cfg, 'action_use_latent_prefix', False)))}")
    print(f"  action_use_image_prefix_alias={int(bool(getattr(cfg, 'action_use_image_prefix', False)))}")
    print(f"  use_history_trajectory_janus_image={int(bool(getattr(cfg, 'use_history_trajectory_janus_image', False)))}")
    print(f"  history_trajectory_camera_config_path={getattr(cfg, 'history_trajectory_camera_config_path', '')}")
    print(f"  robot_state={int(getattr(cfg, 'robot_state', 0) or 0)}")
    print(f"  state_placeholder_tokens={int(getattr(cfg, 'state_placeholder_tokens', 0) or 0)}")
    print(f"  state_encoding_mode={getattr(cfg, 'state_encoding_mode', 'token')}")
    print(f"  total_latent_tokens={int(getattr(cfg, 'total_latent_tokens', 0) or 0)}")
    print(f"  img_latents_per_future={int(getattr(cfg, 'img_latents_per_future', 0) or 0)}")
    print(f"  state_latents_per_future={int(getattr(cfg, 'state_latents_per_future', 0) or 0)}")
    print(f"  num_future_frames={int(getattr(cfg, 'num_future_frames', 0) or 0)}")
    print(f"  future_frame_stride={int(getattr(cfg, 'future_frame_stride', 0) or 0)}")
    print(f"  video_frames={int(getattr(cfg, 'video_frames', 0) or 0)}")
    print(f"  num_cond_input_frames={int(getattr(cfg, 'num_cond_input_frames', 1) or 1)}")
    print(f"  total_video_latent_frames={int(getattr(cfg, 'total_video_latent_frames', 0) or 0)}")
    print(f"  num_cond_latent_frames={int(getattr(cfg, 'num_cond_latent_frames', 1) or 1)}")
    print(f"  cosmos_self_only_bridge={int(bool(getattr(cfg, 'cosmos_self_only_bridge', False)))}")
    print(f"  decosmos={int(bool(getattr(cfg, 'decosmos', False)))}")
    print(f"  use_value_prediction={int(bool(getattr(cfg, 'use_value_prediction', False)))}")
    print(f"  use_action_value_prediction={int(bool(getattr(cfg, 'use_action_value_prediction', False)))}")
    print(f"  value_token_mask_video_to_value={int(bool(getattr(cfg, 'value_token_mask_video_to_value', False)))}")
    print(f"  value_token_mask_nonvalue_to_value={int(bool(getattr(cfg, 'value_token_mask_nonvalue_to_value', False)))}")
    print(f"  cosmos_text_cache_enabled={int(bool(getattr(cfg, 'cosmos_text_cache_path', '')))}")
    print(f"  rewrite_eval_prompt={int(bool(getattr(cfg, 'rewrite_eval_prompt', False)))}")
    print(f"  action_self_causal_in_bridge={int(bool(getattr(cfg, 'action_self_causal_in_bridge', False)))}")


def attach_cosmos_inference_runtime(model, cosmos_wrapper):
    """Attach inference-only Cosmos sampling state to a loaded MoT model."""
    model.set_cosmos_inference_runtime_from_wrapper(cosmos_wrapper)
    return model


def model_load(cfg: Any):
    cfg.bridge_pos_scheme = normalize_bridge_pos_scheme(getattr(cfg, "bridge_pos_scheme", "mrope"))
    cfg.cosmos_self_only_bridge = False
    cfg.decosmos = False
    cfg.action_use_latent_prefix = True
    cfg.action_self_causal_in_bridge = True
    cfg.use_value_prediction = False
    cfg.use_action_value_prediction = False
    cfg.value_token_mask_video_to_value = False
    cfg.value_token_mask_nonvalue_to_value = False
    cfg.total_spatial_tokens = int(getattr(cfg, "total_latent_tokens", 1) or 1)
    ckpt_path, base_dir = resolve_checkpoint_paths(cfg.pretrained_checkpoint)

    # =================================================================
    # 1. 从 checkpoint 优先加载 Processor / Tokenizer
    # =================================================================
    print(f"Loading Processor from checkpoint/base: {base_dir} / {cfg.action_model_path or cfg.model_path}...")
    vl_chat_processor = load_processor_for_checkpoint(cfg.action_model_path or cfg.model_path, base_dir)
    tokenizer = vl_chat_processor.tokenizer
    action_tokenizer = ActionTokenizer(tokenizer, need_to_sub=3)
    cfg.janus_image_start_id = getattr(vl_chat_processor, "image_start_id", None)
    cfg.janus_image_end_id = getattr(vl_chat_processor, "image_end_id", None)
    if cfg.janus_image_start_id is None:
        cfg.janus_image_start_id = tokenizer.convert_tokens_to_ids("<begin_of_image>")
    if cfg.janus_image_end_id is None:
        cfg.janus_image_end_id = tokenizer.convert_tokens_to_ids("<end_of_image>")
    if cfg.janus_image_start_id is None or cfg.janus_image_end_id is None:
        raise ValueError("Could not resolve Janus image start/end token ids.")

    # =================================================================
    # 2. 加载 Janus Action 骨架 (注意这里改为了 action_model_path)
    # =================================================================
    print(f"Loading Janus Action Base from {cfg.action_model_path}...")
    janus_model = AutoModelForCausalLM.from_pretrained(
        cfg.action_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        flow=True, action_dim=cfg.action_dim, ignore_mismatched_sizes=True
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
    
    cosmos_wrapper, cosmos_config = load_model_from_checkpoint(
        experiment_name=cfg.cosmos_experiment_name,
        s3_checkpoint_dir=cfg.cosmos_model_path,
        config_file="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        load_ema_to_reg=True,
        to_device="cpu",
        experiment_opts=experiment_opts
    )
    resolve_video_condition_config(cfg, cosmos_wrapper.tokenizer)
    
    # =================================================================
    # 4. 组装 2-MoT 架构并灌入我们全参保存的权重
    # =================================================================
    cfg.model_variant = "mot2_action_spatial"

    print(f"Loading Fine-Tuned MoT Weights from {ckpt_path}...")
    
    state_dict = torch.load(ckpt_path, map_location="cpu")
    cfg.special_token_vocab = load_special_token_vocab(base_dir, getattr(cfg, "special_token_vocab", ""))
    cfg.special_token_init_ids = resolve_special_token_init_ids(tokenizer, cfg.special_token_vocab)
    validate_special_token_checkpoint_rows(state_dict, cfg.special_token_vocab)
    cfg.valid_token_vocab_size = int(len(tokenizer))
    log_resolved_inference_config(cfg)

    print("Building 2-MoT Cosmos + action-spatial Architecture...")
    model = CosmosJanusActionSpatialMoT2Expert(cosmos_wrapper.net, cosmos_wrapper.tokenizer, janus_model, cfg)
    attach_cosmos_inference_runtime(model, cosmos_wrapper)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if len(missing_keys) > 0:
        print(f"Warning: Missing keys in state_dict (usually OK if they are caches): {missing_keys[:5]}...")

    # 推到 GPU 并设置为推理模式
    device = torch.device(f"cuda:{cfg.cuda}" if torch.cuda.is_available() else "cuda")
    model = model.to(torch.bfloat16).to(device).eval()
    if getattr(cfg, "cosmos_text_cache_path", ""):
        text_encoder_config = getattr(cosmos_config.model.config, "text_encoder_config", None)
        if text_encoder_config is None:
            raise ValueError(
                "cosmos_text_cache_path was set, but the Cosmos experiment has no text_encoder_config."
            )
        model.cosmos_text_cache = CosmosTextEmbeddingCache(cfg.cosmos_text_cache_path, create=True)
        model.cosmos_qwen_text_embedder = CosmosQwenTextEmbedder(
            cosmos_dit=model.cosmos_dit,
            text_encoder_config=text_encoder_config,
            device=device,
        )
        print(f"Using Cosmos text cache: {model.cosmos_text_cache.root}")
        if bool(getattr(cfg, "decosmos", False)):
            print("Note: decosmos=true skips Cosmos video denoising, so cached Cosmos text has little/no effect.")

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


def parse_task_ids(task_ids: str, num_tasks: int) -> list[int]:
    """Parse a comma-separated task id list, defaulting to all tasks."""
    task_ids = str(task_ids or "").strip()
    if not task_ids:
        return list(range(num_tasks))

    parsed: list[int] = []
    for item in task_ids.split(","):
        item = item.strip()
        if not item:
            continue
        task_id = int(item)
        if task_id < 0 or task_id >= num_tasks:
            raise ValueError(f"task id {task_id} is outside [0, {num_tasks}).")
        parsed.append(task_id)

    if not parsed:
        raise ValueError("task_ids was provided but no valid ids were parsed.")
    return parsed


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value


def log_eval_start_config(cfg: GenerateConfig, log_file=None) -> None:
    """Record shell hparams and resolved draccus config before expensive eval work starts."""
    log_message("Evaluation start config:", log_file)
    if getattr(cfg, "eval_artifact_name", ""):
        log_message(f"eval_artifact_name={cfg.eval_artifact_name}", log_file)

    bash_hparams_path = str(getattr(cfg, "bash_hparams_path", "") or "").strip()
    if bash_hparams_path:
        log_message(f"Bash hparams path: {bash_hparams_path}", log_file)
        if os.path.exists(bash_hparams_path):
            log_message("Bash hparams begin", log_file)
            with open(bash_hparams_path, "r") as f:
                for line in f:
                    log_message(line.rstrip("\n"), log_file)
            log_message("Bash hparams end", log_file)
        else:
            log_message(f"Bash hparams file not found: {bash_hparams_path}", log_file)

    cfg_dict = {key: _json_safe(value) for key, value in asdict(cfg).items()}
    log_message("GenerateConfig begin", log_file)
    for line in json.dumps(cfg_dict, indent=2, sort_keys=True).splitlines():
        log_message(line, log_file)
    log_message("GenerateConfig end", log_file)


def _sanitize_filename(text: str) -> str:
    sanitized = text.lower().replace(" ", "_").replace("\n", "_").replace(".", "_").replace("/", "_")
    sanitized = sanitized.replace("\\", "_").replace(":", "_")
    return sanitized[:80]


def predicted_video_to_numpy_frames(pred_video: torch.Tensor) -> np.ndarray:
    """Convert model video output to uint8 frames in [T, H, W, C]."""
    video = pred_video.detach().to(torch.float32).cpu()

    if video.dim() == 5:
        video = video[0]
    if video.dim() != 4:
        raise ValueError(f"Expected pred_video with 4 or 5 dims, got shape={tuple(video.shape)}")

    if video.shape[0] in (1, 3):
        video = video.permute(1, 2, 3, 0)  # [C, T, H, W] -> [T, H, W, C]
    elif video.shape[1] in (1, 3):
        video = video.permute(0, 2, 3, 1)  # [T, C, H, W] -> [T, H, W, C]
    elif video.shape[-1] not in (1, 3):
        raise ValueError(f"Unrecognized pred_video layout with shape={tuple(video.shape)}")

    if video.shape[-1] == 1:
        video = video.repeat(1, 1, 1, 3)

    if float(video.min()) < 0.0 or float(video.max()) > 1.0:
        video = (video + 1.0) / 2.0

    video = video.clamp(0.0, 1.0)
    video = (video * 255.0).round().to(torch.uint8).contiguous()
    return video.numpy()


def _value_score_items(
    cosmos_value_score: Optional[float] = None,
    action_value_score: Optional[float] = None,
) -> list[tuple[str, float]]:
    items = []
    if cosmos_value_score is not None:
        items.append(("cosmos value", float(cosmos_value_score)))
    if action_value_score is not None:
        items.append(("action value", float(action_value_score)))
    return items


def _draw_value_header(frame: np.ndarray, value_items: list[tuple[str, float]]) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] > 3:
        frame = frame[..., :3]

    header_height = 18 + 18 * max(1, len(value_items))
    pil_frame = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    canvas = Image.new("RGB", (pil_frame.width, pil_frame.height + header_height), (18, 22, 28))
    canvas.paste(pil_frame, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for idx, (label, score) in enumerate(value_items):
        draw.text((8, 8 + 18 * idx), f"{label}: {score:.4f}", fill=(245, 248, 252), font=font)
    return np.asarray(canvas, dtype=np.uint8)


def add_value_header_to_frames(frames: np.ndarray, value_items: list[tuple[str, float]]) -> np.ndarray:
    return np.stack([_draw_value_header(frame, value_items) for frame in frames], axis=0)


def save_predicted_video(
    pred_video: Optional[torch.Tensor],
    cfg: GenerateConfig,
    task_description: str,
    task_id: Optional[int],
    episode_idx: Optional[int],
    inference_idx: int,
    step_idx: int,
    cosmos_value_score: Optional[float] = None,
    action_value_score: Optional[float] = None,
    log_file=None,
):
    """Save a predicted Cosmos video if a save directory is configured."""
    if not cfg.predicted_video_save_dir:
        return None
    if pred_video is None:
        return None

    save_dir = cfg.predicted_video_save_dir
    os.makedirs(save_dir, exist_ok=True)

    processed_task_description = _sanitize_filename(task_description)
    task_label = f"{task_id:02d}" if task_id is not None else "na"
    episode_label = f"{episode_idx:03d}" if episode_idx is not None else "na"
    value_items = _value_score_items(cosmos_value_score, action_value_score)
    value_label = ""
    if value_items:
        value_label = "".join(
            f"--{label.replace(' ', '_')}={score:.4f}" for label, score in value_items
        )
    filename = (
        f"{DATE_TIME}--task={task_label}--episode={episode_label}"
        f"--query={inference_idx:03d}--step={step_idx:04d}"
        f"{value_label}--desc={processed_task_description}.mp4"
    )
    mp4_path = os.path.join(save_dir, filename)

    frames = predicted_video_to_numpy_frames(pred_video)
    if value_items:
        frames = add_value_header_to_frames(frames, value_items)
    writer = imageio.get_writer(mp4_path, fps=cfg.predicted_video_fps, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()

    log_message(f"Saved predicted Cosmos video to {mp4_path}", log_file)
    return mp4_path


def save_value_visualizations(value_traces: list[dict[str, Any]], cfg: GenerateConfig, log_file=None):
    if not (cfg.use_value_prediction or cfg.use_action_value_prediction) or not cfg.value_visualization_dir:
        return None

    save_dir = cfg.value_visualization_dir
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "value_traces.json")
    payload = {
        "eval_artifact_name": getattr(cfg, "eval_artifact_name", ""),
        "num_points": len(value_traces),
        "traces": value_traces,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    log_message(f"Saved value trace JSON to {json_path}", log_file)

    if not value_traces:
        log_message("No value traces collected; skipping value summary plot.", log_file)
        return {"json": json_path, "plot": None}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log_message(f"Could not render value summary plot: {exc}", log_file)
        return {"json": json_path, "plot": None}

    grouped: dict[tuple[Any, Any, bool], list[dict[str, Any]]] = {}
    for entry in value_traces:
        key = (entry.get("task_id"), entry.get("episode_idx"), bool(entry.get("success", False)))
        grouped.setdefault(key, []).append(entry)

    trace_specs = [
        ("cosmos_value_score", "cosmos value", "#1f9d55"),
        ("action_value_score", "action value", "#f59e0b"),
    ]
    series_by_field: dict[tuple[str, bool], list[list[float]]] = {
        (field, success): [] for field, _, _ in trace_specs for success in (False, True)
    }
    for (task_id, episode_idx, success), entries in sorted(grouped.items(), key=lambda item: str(item[0])):
        del task_id, episode_idx
        entries = sorted(entries, key=lambda item: (int(item.get("query_idx", 0)), int(item.get("step", 0))))
        for field, _, _ in trace_specs:
            series = [float(item[field]) for item in entries if item.get(field) is not None]
            if series:
                series_by_field[(field, success)].append(series)

    def plot_series(ax, series_list: list[list[float]], color: str, label: str, linestyle: str) -> None:
        for idx, series in enumerate(series_list):
            ax.plot(
                np.arange(len(series)),
                series,
                color=color,
                alpha=0.22,
                linewidth=1.0,
                linestyle=linestyle,
                label=label if idx == 0 else None,
            )

    def mean_curve(series_list: list[list[float]]) -> Optional[np.ndarray]:
        if not series_list:
            return None
        max_len = max(len(series) for series in series_list)
        values = np.full((len(series_list), max_len), np.nan, dtype=np.float32)
        for row_idx, series in enumerate(series_list):
            values[row_idx, : len(series)] = np.asarray(series, dtype=np.float32)
        return np.nanmean(values, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    for field, label, color in trace_specs:
        good_series = series_by_field[(field, True)]
        bad_series = series_by_field[(field, False)]
        plot_series(ax, good_series, color, f"{label} good samples", "-")
        plot_series(ax, bad_series, color, f"{label} bad samples", "--")

        good_mean = mean_curve(good_series)
        if good_mean is not None:
            ax.plot(
                np.arange(len(good_mean)),
                good_mean,
                color=color,
                linewidth=2.5,
                linestyle="-",
                label=f"{label} good mean",
            )
        bad_mean = mean_curve(bad_series)
        if bad_mean is not None:
            ax.plot(
                np.arange(len(bad_mean)),
                bad_mean,
                color=color,
                linewidth=2.5,
                linestyle="--",
                label=f"{label} bad mean",
            )

    ax.set_title("Value traces by rollout outcome")
    ax.set_xlabel("Query index")
    ax.set_ylabel("Value score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best")
    fig.tight_layout()

    plot_path = os.path.join(save_dir, "value_summary.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    log_message(f"Saved value summary plot to {plot_path}", log_file)
    return {"json": json_path, "plot": plot_path}


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


def resolve_pad_token_id(processor) -> int:
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None and hasattr(processor, "pad_id"):
        pad_token_id = processor.pad_id
    if pad_token_id is None:
        pad_token_id = 0
    return int(pad_token_id)


def resolve_pad_token_text(processor) -> str:
    tokenizer = processor.tokenizer
    pad_token_id = resolve_pad_token_id(processor)
    candidates = []
    if tokenizer.pad_token is not None:
        candidates.append(tokenizer.pad_token)
    if hasattr(processor, "pad_tag"):
        candidates.append(processor.pad_tag)

    token_text = tokenizer.convert_ids_to_tokens(pad_token_id)
    if token_text is not None:
        candidates.append(token_text)

    for candidate in candidates:
        encoded = tokenizer.encode(candidate, add_special_tokens=False)
        if len(encoded) == 1 and int(encoded[0]) == pad_token_id:
            return candidate

    raise ValueError(f"Could not resolve a text form for pad token id {pad_token_id}.")


def normalize_state_for_eval(state, statistic) -> np.ndarray:
    state_arr = np.array(state, dtype=np.float32)
    return np.where(
        statistic["state_mask"],
        np.clip(
            2 * (state_arr - statistic["state_q01"]) / (statistic["state_q99"] - statistic["state_q01"] + 1e-8) - 1.0,
            -1.0,
            1.0,
        ),
        state_arr,
    )


def build_eval_state_inputs(cfg, processor, action_tokenizer, statistic, current_state):
    """Mirror train_cot.py current-state prompt placeholders and now_state encoding."""
    if not int(getattr(cfg, "robot_state", 0) or 0):
        return "", None, None
    if current_state is None:
        raise ValueError("robot_state is enabled, but current_state is None.")

    placeholder_count = int(getattr(cfg, "state_placeholder_tokens", 8))
    pad_token_id = resolve_pad_token_id(processor)
    placeholder_text = resolve_pad_token_text(processor) * placeholder_count
    placeholder_ids = torch.full((placeholder_count,), pad_token_id, dtype=torch.long)

    norm_state = normalize_state_for_eval(current_state, statistic)
    state_encoding_mode = str(getattr(cfg, "state_encoding_mode", "token")).lower()
    if state_encoding_mode == "mlp":
        if norm_state.shape[-1] != 8:
            raise ValueError(f"MLP state encoding expects state dim 8, got shape {norm_state.shape}.")
        now_state = torch.tensor(norm_state, dtype=torch.float32)
    elif state_encoding_mode == "token":
        state_token_str = action_tokenizer(norm_state)
        current_state_ids = torch.LongTensor(
            processor.tokenizer.encode(state_token_str, add_special_tokens=False)
        )
        if current_state_ids.numel() != placeholder_count:
            raise ValueError(
                f"Encoded current state has {current_state_ids.numel()} tokens, "
                f"but state_placeholder_tokens is {placeholder_count}."
            )
        now_state = current_state_ids
    else:
        raise ValueError(f"Unsupported state_encoding_mode={state_encoding_mode!r}.")

    return placeholder_text, placeholder_ids, now_state


def resolve_eval_prompt(cfg: Any, task_description: str) -> str:
    """Optionally map LIBERO's task description to the rewritten training prompt."""
    prompt = str(task_description).strip()
    if not bool(getattr(cfg, "rewrite_eval_prompt", False)):
        return prompt
    if prompt not in PROMPT_REPLACEMENTS:
        raise KeyError(
            "rewrite_eval_prompt is enabled, but this task description has no replacement: "
            f"{prompt!r}"
        )
    return PROMPT_REPLACEMENTS[prompt]


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
    episode_idx=None,
):
    """Run a single episode in the environment."""
    env.reset()
    eval_prompt = resolve_eval_prompt(cfg, task_description)
    if eval_prompt != task_description and (episode_idx is None or int(episode_idx) == 0):
        log_message(f"Rewritten eval prompt: {eval_prompt}", log_file)
    
    ## ----- debug ----- ##
    if getattr(cfg, 'task_suite_name', None) == 'libero_spatial' and task_id == 5:
        initial_state[12] += 0.038
        print(f"debug: initial_state[12] += 0.038")
    ## ----------------- ##

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # 默认执行 chunk_size 步长，或者根据设定取前几步 (Temporal Ensembling / Receding Horizon)
    action_queue = deque(maxlen=cfg.num_open_loop_steps * cfg.action_repeat)

    t = 0
    inference_idx = 0
    replay_images = []
    replay_value_scores = []
    episode_value_trace = []
    active_value_scores = None
    trajectory_camera_config = None
    if bool(getattr(cfg, "use_history_trajectory_janus_image", False)):
        trajectory_camera_config = load_history_trajectory_camera_config(
            cfg.task_suite_name,
            cfg.history_trajectory_camera_config_path,
        )
    ee_history = []
    # 假设有个字典映射最大步数
    max_steps = TASK_MAX_STEPS.get(cfg.task_suite_name, 600)

    success = False

    # Data transformation for Cosmos
    device = next(model.parameters()).device
    dtype = torch.bfloat16
    video_transform = transforms.Compose([
        transforms.Resize(min(cfg.video_h, cfg.video_w), antialias=True),
        transforms.CenterCrop((cfg.video_h, cfg.video_w)),
    ])
    num_cond_input_frames = max(1, int(getattr(cfg, "num_cond_input_frames", 1) or 1))
    obs_history = deque(maxlen=num_cond_input_frames)

    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        observation, img = prepare_observation(obs)

        current_state = observation['state'].copy()
        primary_image = Image.fromarray(observation['full_image'])
        wrist_image = Image.fromarray(observation['wrist_image'])
        primary_frame_np = np.array(primary_image)
        primary_frame_tensor = torch.from_numpy(primary_frame_np).permute(2, 0, 1).float() / 255.0
        primary_frame_tensor = video_transform(primary_frame_tensor)
        if len(obs_history) == 0:
            for _ in range(num_cond_input_frames):
                obs_history.append(primary_frame_tensor.clone())
        else:
            obs_history.append(primary_frame_tensor)

        if trajectory_camera_config is not None:
            ee_history.append(np.asarray(current_state[:3], dtype=np.float64).copy())

        if len(action_queue) == 0:
            # =================================================================
            # 1. 构造多模态对话格式 & 状态拼接
            # =================================================================
            state_tokens_str, state_placeholder_ids, now_state = build_eval_state_inputs(
                cfg=cfg,
                processor=processor,
                action_tokenizer=action_tokenizer,
                statistic=statistic,
                current_state=current_state,
            )
            if now_state is not None:
                now_state = now_state.unsqueeze(0).to(device)

            action_prompt_suffix = build_janus_action_prompt_suffix(
                bool(getattr(cfg, "use_history_trajectory_janus_image", False))
            )
            user_content = f"<image_placeholder>\n{eval_prompt}"
            if action_prompt_suffix or state_tokens_str:
                user_content += "\n"
                if action_prompt_suffix:
                    user_content += action_prompt_suffix
                if state_tokens_str:
                    user_content += state_tokens_str
            user_prompt = processor.apply_sft_template_for_multi_turn_prompts(
                conversations=[{"role": "<|User|>", "content": user_content}],
                sft_format=processor.sft_format,
                system_prompt="",
            )
            # Keep Assistant as an open prefix; adding Assistant content through the
            # template would append the assistant end marker.
            prompt_text = user_prompt + "\n\n<|Assistant|>:"
            
            # =================================================================
            # 2. 交给 Processor 生成 Janus 专属特征
            # =================================================================
            janus_primary_image = primary_image
            if trajectory_camera_config is not None:
                janus_primary_image = draw_history_trajectory_on_image(
                    primary_image.copy(),
                    ee_history,
                    trajectory_camera_config,
                )

            janus_inputs = processor(
                prompt=prompt_text,
                images=[janus_primary_image],
                return_tensors="pt"
            )
            cosmos_user_content = f"<image_placeholder>\n{eval_prompt}"
            if state_tokens_str:
                cosmos_user_content += f"\n{state_tokens_str}"
            cosmos_user_prompt = processor.apply_sft_template_for_multi_turn_prompts(
                conversations=[{"role": "<|User|>", "content": cosmos_user_content}],
                sft_format=processor.sft_format,
                system_prompt="",
            )
            cosmos_prompt_text = cosmos_user_prompt + "\n\n<|Assistant|>:"
            cosmos_janus_inputs = processor(
                prompt=cosmos_prompt_text,
                images=[janus_primary_image],
                return_tensors="pt",
            )

            janus_input_ids = janus_inputs.input_ids.to(device)
            janus_pixel_values = janus_inputs.pixel_values.to(device).to(dtype)
            janus_images_seq_mask = janus_inputs.images_seq_mask.to(device)
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
            pad_token_id = resolve_pad_token_id(processor)
            janus_left_pad_lens = janus_input_ids.eq(pad_token_id).to(torch.long).cumprod(dim=1).sum(dim=1)
            cosmos_janus_input_ids = cosmos_janus_inputs.input_ids.to(device)
            cosmos_janus_state_seq_mask = build_token_sequence_mask(
                cosmos_janus_input_ids,
                state_placeholder_ids,
                require_match=bool(int(getattr(cfg, "robot_state", 0) or 0)),
                name="current state placeholder in Cosmos prompt",
            ).to(device)

            if inference_idx == 0:
                log_message(
                    "Eval input smoke: "
                    f"janus_state_seq_mask.sum={int(janus_state_seq_mask.sum().item())}, "
                    f"attention_mask.sum={int(janus_attention_mask.sum().item())}, "
                    f"cosmos_history_frames={len(obs_history)}",
                    log_file,
                )

            # =================================================================
            # 3. 处理 Cosmos 视角的历史条件帧
            # =================================================================
            first_frame_tensor = torch.stack(list(obs_history), dim=1).unsqueeze(0).to(device).to(dtype)

            # =================================================================
            # 4. 执行多模态联合 ODE 去噪推理
            # =================================================================
            cosmos_text_embeddings = None
            if getattr(cfg, "cosmos_text_cache_path", ""):
                cosmos_text_cache = getattr(model, "cosmos_text_cache", None)
                cosmos_text_embedder = getattr(model, "cosmos_qwen_text_embedder", None)
                if cosmos_text_cache is None or cosmos_text_embedder is None:
                    raise RuntimeError(
                        "cosmos_text_cache_path is set, but model was not initialized with "
                        "Cosmos text cache/runtime embedder."
                    )

                def compute_cosmos_text(prompt: str) -> torch.Tensor:
                    log_message(
                        f"Cosmos text cache miss; computing native Qwen embedding for: {prompt!r}",
                        log_file,
                    )
                    return cosmos_text_embedder.compute_one(prompt)

                cosmos_text_embeddings = cosmos_text_cache.get_or_compute(
                    eval_prompt,
                    compute_cosmos_text,
                ).unsqueeze(0).to(device).to(dtype)

            with torch.inference_mode():
                fps_tensor = torch.tensor([cfg.fps], device=device, dtype=dtype)
                inference_kwargs = dict(
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
                    janus_attention_mask=janus_attention_mask,
                    now_state=now_state,
                    cosmos_text_embeddings=cosmos_text_embeddings,
                    cosmos_janus_input_ids=cosmos_janus_input_ids,
                    cosmos_janus_images_seq_mask=cosmos_janus_inputs.images_seq_mask.to(device),
                    cosmos_janus_state_seq_mask=cosmos_janus_state_seq_mask,
                    cosmos_janus_images_emb_mask=cosmos_janus_inputs.images_emb_mask.to(device),
                )

                inference_outputs = model.forward_flow_joint_inference(**inference_kwargs)
                predicted_value = None
                predicted_action_value = None
                pred_video, pred_action = inference_outputs[:2]

            cosmos_value_score = None
            action_value_score = None
            if predicted_value is not None:
                value_list = predicted_value.detach().cpu().float().view(-1).tolist()
                if value_list:
                    cosmos_value_score = float(np.mean(value_list))
                log_message(
                    f"Predicted Cosmos value at step {t}: "
                    + ", ".join(f"{value:.4f}" for value in value_list),
                    log_file,
                )
            if predicted_action_value is not None:
                action_value_list = predicted_action_value.detach().cpu().float().view(-1).tolist()
                if action_value_list:
                    action_value_score = float(np.mean(action_value_list))
                log_message(
                    f"Predicted action-branch value at step {t}: "
                    + ", ".join(f"{value:.4f}" for value in action_value_list),
                    log_file,
                )
            if cosmos_value_score is not None or action_value_score is not None:
                trace_entry = {
                    "task_id": None if task_id is None else int(task_id),
                    "episode_idx": None if episode_idx is None else int(episode_idx),
                    "query_idx": int(inference_idx),
                    "step": int(t),
                }
                if cosmos_value_score is not None:
                    trace_entry["cosmos_value_score"] = cosmos_value_score
                if action_value_score is not None:
                    trace_entry["action_value_score"] = action_value_score
                episode_value_trace.append(trace_entry)
            if cfg.use_value_prediction or cfg.use_action_value_prediction:
                active_value_scores = {}
                if cfg.use_value_prediction:
                    active_value_scores["cosmos value"] = cosmos_value_score
                if cfg.use_action_value_prediction:
                    active_value_scores["action value"] = action_value_score

            save_predicted_video(
                pred_video=pred_video,
                cfg=cfg,
                task_description=task_description,
                task_id=task_id,
                episode_idx=episode_idx,
                inference_idx=inference_idx,
                step_idx=t,
                cosmos_value_score=cosmos_value_score,
                action_value_score=action_value_score,
                log_file=log_file,
            )
            inference_idx += 1

            # =================================================================
            # 5. 反归一化并压入队列
            # =================================================================
            normalized_actions = pred_action.squeeze(0).cpu().float().numpy()
            if normalized_actions.shape[1] in [7, 14]:
                normalized_actions[:, 6] = 1-(normalized_actions[:, 6] >= 0.5).astype(int)
                
                
            action_pred = np.where(
                statistic['action_mask'],
                0.5 * (normalized_actions + 1.0) * (statistic['action_q99'] - statistic['action_q01']) + statistic['action_q01'],
                normalized_actions
            )
            
            for action in action_pred[:cfg.num_open_loop_steps]:
                action_queue.extend([action] * cfg.action_repeat)

        replay_images.append(img)
        if cfg.use_value_prediction or cfg.use_action_value_prediction:
            replay_value_scores.append(None if active_value_scores is None else dict(active_value_scores))

        # 执行动作
        action = action_queue.popleft()
        action = process_action(action, cfg.model_family)

        obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1

    return success, replay_images, replay_value_scores, episode_value_trace


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
    value_traces=None,
    log_file=None,
):
    """Run evaluation for a single task."""
    if value_traces is None:
        value_traces = []

    # Get task
    ## ----- chenhao specify task_id for task_id 4----- ##
    # task_id = 4
    ## ----- chenhao specify task_id for task_id 4----- ##    

    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    
    # Initialize environment and get task description
    env, task_description = get_libero_env(
        task,
        cfg.model_family,
        resolution=cfg.env_img_res,
        seed=cfg.seed,
        control_freq=cfg.control_freq,
    )

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
        success, replay_images, replay_value_scores, episode_value_trace = run_episode(
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
            episode_idx,
        )
        for entry in episode_value_trace:
            entry["success"] = bool(success)
            entry["task_description"] = task_description
        value_traces.extend(episode_value_trace)

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video only when explicitly enabled. Long ep50 sweeps otherwise
        # produce a large amount of I/O and increase render-process fragility.
        if not getattr(cfg, "disable_rollout_video", False):
            save_rollout_video(
                replay_images,
                total_episodes,
                success=success,
                task_description=task_description,
                log_file=log_file,
                cosmos_denoise_steps=cfg.cosmos_denoise_steps,
                rollout_dir=cfg.rollout_video_save_dir,
                value_scores=replay_value_scores if (cfg.use_value_prediction or cfg.use_action_value_prediction) else None,
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

    return total_episodes, total_successes, value_traces


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Setup logging before loading the model so hparams are captured at eval start.
    log_file, local_log_filepath, run_id = setup_logging(cfg)
    log_eval_start_config(cfg, log_file)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, processor, action_tokenizer, statistic = model_load(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Cosmos denoise steps: {cfg.cosmos_denoise_steps}", log_file)
    selected_task_ids = parse_task_ids(cfg.task_ids, num_tasks)
    log_message(f"Selected task ids: {selected_task_ids}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    all_value_traces = []
    for task_id in tqdm.tqdm(selected_task_ids):
        total_episodes, total_successes, all_value_traces = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            processor,
            action_tokenizer,
            statistic,
            total_episodes,
            total_successes,
            all_value_traces,
            log_file,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    save_value_visualizations(all_value_traces, cfg, log_file)

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
