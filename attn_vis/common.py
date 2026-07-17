"""Lightweight helpers shared by train-set attention visualizers."""

from __future__ import annotations

import logging
import math
import random
import re
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image


logger = logging.getLogger(__name__)

SPATIAL_TOKEN_MODE_TO_FIELDS = {
    "v": ("gtlatent",),
    "n": ("gtlatent2",),
    "vn": ("gtlatent", "gtlatent2"),
}
SPATIAL_TOKEN_MODE_ALIASES = {
    "1": "v",
    "2": "vn",
}


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def normalize_spatial_token_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    mode = SPATIAL_TOKEN_MODE_ALIASES.get(mode, mode)
    if mode not in SPATIAL_TOKEN_MODE_TO_FIELDS:
        valid = ", ".join(
            sorted(
                [
                    *SPATIAL_TOKEN_MODE_TO_FIELDS.keys(),
                    *SPATIAL_TOKEN_MODE_ALIASES.keys(),
                ]
            )
        )
        raise ValueError(f"spatial token mode must be one of {valid}, got {value!r}.")
    return mode


def resolve_spatial_token_args(config: Any) -> None:
    mode_arg = str(getattr(config, "latent_token_mode", "") or "").strip()
    count_or_mode_arg = str(getattr(config, "total_latent_tokens", "") or "").strip()
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
                        f"latent_token_mode={mode_arg!r}, "
                        f"total_latent_tokens={count_or_mode_arg!r}."
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

    config.latent_token_mode = mode
    config.latent_token_fields = fields_for_mode
    config.total_latent_tokens = len(fields_for_mode)
    config.total_spatial_tokens = config.total_latent_tokens


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


def sanitize_filename(value: str, max_len: int = 120) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_")
    return (cleaned or "unknown")[:max_len]


def score_to_rgb(score: float) -> tuple[int, int, int]:
    score = float(np.clip(score, 0.0, 1.0))
    low = np.array([80.0, 145.0, 255.0], dtype=np.float32)
    high = np.array([255.0, 48.0, 42.0], dtype=np.float32)
    rgb = low * (1.0 - score) + high * score
    return tuple(int(round(value)) for value in rgb)


def _colorize_heatmap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 * values - 0.2, 0.0, 1.0)
    green = np.clip(1.5 - 3.0 * np.abs(values - 0.5), 0.0, 1.0)
    blue = np.clip(1.2 - 1.5 * values, 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def _smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def overlay_heatmap(
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
        heat_img = Image.fromarray(_colorize_heatmap(heat), mode="RGB").resize(
            (tile_size, tile_size),
            resample=resample,
        )
    else:
        heat = np.asarray(
            Image.fromarray(heat.astype(np.float32)).resize(
                (tile_size, tile_size),
                resample=resample,
            ),
            dtype=np.float32,
        )
        ratio = float(top_ratio)
        if ratio <= 0.0:
            heat = np.zeros_like(heat, dtype=np.float32)
        elif ratio < 1.0:
            flat = heat.reshape(-1)
            keep_count = max(0, min(flat.size, int(math.ceil(flat.size * ratio))))
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


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    moved = {}
    float_dtype_keys = {
        "janus_pixel_values",
        "videos",
        "cosmos_text_embeddings",
    }
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif key in float_dtype_keys:
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
