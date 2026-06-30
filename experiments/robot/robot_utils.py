"""Utils for evaluating robot policies in various environments."""

import os
import random
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

# from experiments.robot.openvla_utils import (
#     get_vla,
#     get_vla_action,
# )

import torchvision.transforms as transforms
from models.cosmos_janus_cot import CosmosJanusMoT3Expert, build_token_sequence_mask

# Initialize important constants
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# Model image size configuration
MODEL_IMAGE_SIZES = {
    "openvla": 224,
    # Add other models as needed
}

from dataclasses import dataclass

@dataclass
class VLChatProcessorOutput():
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor

    def __len__(self):
        return len(self.input_ids)

def set_seed_everywhere(seed: int) -> None:
    """
    Set random seed for all random number generators for reproducibility.

    Args:
        seed: The random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_action(
    cfg: Any,
    statistic: Dict,
    action_tokenizer,
    vl_chat_processor,
    task_description: str,
    model: Any,
    first_image: Image.Image,
    state=None,
) -> List[np.ndarray]:
    
    device = next(model.parameters()).device
    dtype = model.dtype

    state_tokens_str = ""
    if cfg.use_proprio and state is not None:
        state_arr = np.array(state, dtype=np.float32)
        norm_state = np.where(
            statistic['state_mask'],
            np.clip(2 * (state_arr - statistic['state_q01']) / (statistic['state_q99'] - statistic['state_q01'] + 1e-8) - 1.0, -1.0, 1.0),
            state_arr
        )
        state_tokens_str = action_tokenizer(norm_state)

    full_prompt = f"<|User|>{task_description}{state_tokens_str}<|Assistant|>"
    input_ids = vl_chat_processor.tokenizer.encode(full_prompt, add_special_tokens=False)
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device) # [1, Seq_len]

    video_transform = transforms.Compose([
        transforms.Resize(min(cfg.video_h, cfg.video_w), antialias=True),
        transforms.CenterCrop((cfg.video_h, cfg.video_w)),
    ])
    img_tensor = torch.from_numpy(np.array(first_image)).permute(2, 0, 1).float() / 255.0
    first_frame_tensor = video_transform(img_tensor).unsqueeze(0).to(device).to(dtype)

    with torch.inference_mode():
        text_embeds = model.janus.language_model.get_input_embeddings()(input_ids)
        fps_tensor = torch.tensor([10.0], device=device, dtype=dtype)

        _, action_latent = model.forward_flow_joint_inference(
            text_embeds=text_embeds,
            first_frame=first_frame_tensor,
            action_denoise_steps=getattr(cfg, 'action_denoise_steps', 10),
            fps=fps_tensor
        )

    normalized_actions = action_latent.squeeze(0).cpu().numpy() # [chunk_size, 7]
    
    dim = normalized_actions.shape[1]
    if dim in [7, 14]:
        normalized_actions[:, 6] = (normalized_actions[:, 6] >= 0.5).astype(int)
    
    actions = np.where(
        statistic['action_mask'],
        0.5 * (normalized_actions + 1.0) * (statistic['action_q99'] - statistic['action_q01']) + statistic['action_q01'],
        normalized_actions
    )

    return list(actions)


def get_action_cot(
    cfg: Any,
    statistic: Dict,
    action_tokenizer,
    vl_chat_processor,
    task_description: str,
    model: CosmosJanusMoT3Expert,
    first_image: Image.Image,
    state=None,
) -> List[np.ndarray]:
    """Get action using the 3-expert MoT model (cosmos → CoT → action)."""
    device = next(model.parameters()).device
    dtype = model.dtype

    state_tokens_str = ""
    current_state_ids = None
    if cfg.use_proprio and state is not None:
        state_arr = np.array(state, dtype=np.float32)
        norm_state = np.where(
            statistic['state_mask'],
            np.clip(2 * (state_arr - statistic['state_q01']) / (statistic['state_q99'] - statistic['state_q01'] + 1e-8) - 1.0, -1.0, 1.0),
            state_arr
        )
        state_tokens_str = action_tokenizer(norm_state)
        current_state_ids = torch.LongTensor(
            vl_chat_processor.tokenizer.encode(state_tokens_str, add_special_tokens=False)
        )

    conversation = [
        {"role": "<|User|>", "content": f"<image_placeholder>\n{task_description}{state_tokens_str}"},
        {"role": "<|Assistant|>", "content": ""},
    ]

    first_image_pil = first_image if isinstance(first_image, Image.Image) else Image.fromarray(first_image)
    janus_inputs = vl_chat_processor(conversations=conversation, images=[first_image_pil], return_tensors="pt")

    janus_input_ids = janus_inputs.input_ids.to(device)
    janus_pixel_values = janus_inputs.pixel_values.to(device).to(dtype)
    janus_images_seq_mask = janus_inputs.images_seq_mask.to(device)
    janus_state_seq_mask = build_token_sequence_mask(
        janus_input_ids,
        current_state_ids,
        require_match=int(getattr(model, "state_latents_per_future", 0) or 0) > 0,
        name="current state",
    ).to(device)
    janus_images_emb_mask = janus_inputs.images_emb_mask.to(device)

    video_transform = transforms.Compose([
        transforms.Resize(min(cfg.video_h, cfg.video_w), antialias=True),
        transforms.CenterCrop((cfg.video_h, cfg.video_w)),
    ])
    img_tensor = torch.from_numpy(np.array(first_image)).permute(2, 0, 1).float() / 255.0
    first_frame_tensor = video_transform(img_tensor).unsqueeze(0).to(device).to(dtype)

    with torch.inference_mode():
        fps_tensor = torch.tensor([10.0], device=device, dtype=dtype)
        _, action_latent = model.forward_flow_joint_inference(
            janus_input_ids=janus_input_ids,
            janus_pixel_values=janus_pixel_values,
            janus_images_seq_mask=janus_images_seq_mask,
            janus_state_seq_mask=janus_state_seq_mask,
            janus_images_emb_mask=janus_images_emb_mask,
            first_frame=first_frame_tensor,
            action_denoise_steps=getattr(cfg, 'action_denoise_steps', 10),
            num_latent_tokens=getattr(cfg, 'num_latent_tokens', 8),
            fps=fps_tensor,
        )

    normalized_actions = action_latent.squeeze(0).cpu().numpy()

    dim = normalized_actions.shape[1]
    if dim in [7, 14]:
        normalized_actions[:, 6] = (normalized_actions[:, 6] >= 0.5).astype(int)

    actions = np.where(
        statistic['action_mask'],
        0.5 * (normalized_actions + 1.0) * (statistic['action_q99'] - statistic['action_q01']) + statistic['action_q01'],
        normalized_actions
    )

    return list(actions)


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] *= -1.0

    return inverted_action
