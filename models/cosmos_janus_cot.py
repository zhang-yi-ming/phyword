"""
3-Expert MoT: Cosmos (video) + Janus Latent CoT + Janus Action

Architecture:
  - Cosmos DIT processes video tokens (flow matching, bidirectional)
  - Janus latent expert provides latent branch LayerNorm/MLP updates
  - Janus action expert provides action branch LayerNorm/MLP updates

Bridge attention features:
  - Per-head QK RMSNorm on bridge Q/K (matching cosmos DIT's QK normalization)
  - Selectable bridge rotary encoding on latent/action Q/K
    (`mrope` = current A1/Qwen3-VL-style multimodal 3D RoPE,
     `mrope_interleave` = mrope with THW-interleaved bridge basis,
     `llama1d` = native Llama-style 1D RoPE)
  - Action self-attention is full (bidirectional) since actions are generated in parallel
    via flow matching

Bridge attention mask (cosmos-level, default):
              | Cosmos | Latent_i | Action_j |
  Cosmos      |  full  |   full   |   full   |
  Latent_i    |  full  |  causal  |    0     |
  Action_i    |  full  |   full   |   full   |

Optional training-alignment mode (cosmos_self_only_bridge=1):
              | Cosmos | Latent_i | Action_j |
  Cosmos      |  full  |    0     |    0     |
  Latent_i    |  full  |  causal  |    0     |
  Action_i    |  full  |   full   |   full   |

Inference: one-step cosmos denoising + multi-step action denoising.
  The cosmos DIT runs once to produce video features, which are cached.
  Latent/action updates then reuse cached video KV through bridge attention only.

Weight loading:
  - Latent expert: default janus LayerNorm/MLP components
  - Action expert: _action suffixed LayerNorm/MLP components when available
  - Requires last0's custom transformers on PYTHONPATH for loading dual-expert Janus model,
    OR falls back to copies of default components for training from scratch.
"""

from collections import OrderedDict
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import copy
from typing import Optional, Tuple

from janus.diffusion import ActionEmbedder, FinalLayer
from models.cosmos_janus import SlimLlamaMLP
from vae.stacked_resample import StackedDownsample2d, StackedUpsample2d
from vae.wan21_vae_encoder import DEFAULT_WAN21_VAE_CKPT, Wan21VAEEncoder


def build_token_sequence_mask(
    input_ids: torch.Tensor,
    token_ids: Optional[torch.Tensor],
    *,
    require_match: bool = False,
    name: str = "token sequence",
) -> torch.Tensor:
    """Mark the first exact occurrence of `token_ids` inside each input row."""
    if input_ids.ndim not in (1, 2):
        raise ValueError(f"input_ids must be 1D or 2D, got shape {tuple(input_ids.shape)}.")

    single_row = input_ids.ndim == 1
    rows = input_ids.unsqueeze(0) if single_row else input_ids
    mask = torch.zeros_like(rows, dtype=torch.bool)

    if token_ids is None:
        if require_match:
            raise ValueError(f"{name} token ids are required but were not provided.")
        return mask.squeeze(0) if single_row else mask

    patterns = torch.as_tensor(token_ids, device=rows.device, dtype=rows.dtype)
    if patterns.ndim == 1:
        patterns = patterns.unsqueeze(0).expand(rows.shape[0], -1)
    elif patterns.ndim != 2:
        raise ValueError(f"token_ids must be 1D or 2D, got shape {tuple(patterns.shape)}.")
    elif patterns.shape[0] != rows.shape[0]:
        raise ValueError(
            f"token_ids batch size {patterns.shape[0]} does not match input_ids batch size {rows.shape[0]}."
        )

    for batch_idx in range(rows.shape[0]):
        pattern = patterns[batch_idx]
        pattern_len = int(pattern.numel())
        if pattern_len == 0:
            if require_match:
                raise ValueError(f"{name} token ids are empty for batch item {batch_idx}.")
            continue
        if pattern_len > rows.shape[1]:
            if require_match:
                raise ValueError(
                    f"Could not locate {name}: pattern length {pattern_len} exceeds input length {rows.shape[1]}."
                )
            continue

        found = False
        for start in range(rows.shape[1] - pattern_len + 1):
            if torch.equal(rows[batch_idx, start:start + pattern_len], pattern):
                mask[batch_idx, start:start + pattern_len] = True
                found = True
                break

        if require_match and not found:
            raise ValueError(
                f"Could not locate {name} token ids in input_ids for batch item {batch_idx}."
            )

    return mask.squeeze(0) if single_row else mask


def normalize_bridge_pos_scheme(bridge_pos_scheme: object) -> str:
    normalized = str(bridge_pos_scheme).lower()
    alias_map = {
        "mrope": "mrope",
        "mrope_interleave": "mrope_interleave",
        "llama1d": "llama1d",
        "qwen": "qwen",
        "local": "mrope",
        "last0": "mrope",
    }
    if normalized not in alias_map:
        raise ValueError(
            "bridge_pos_scheme must be one of "
            "'mrope', 'mrope_interleave', 'llama1d', 'qwen', 'local', or 'last0', "
            f"got {bridge_pos_scheme!r}."
        )
    return alias_map[normalized]


def is_multimodal_bridge_pos_scheme(bridge_pos_scheme: object) -> bool:
    return normalize_bridge_pos_scheme(bridge_pos_scheme) in {"mrope", "mrope_interleave"}


@dataclass
class BridgeMRoPEBatchInfo:
    """Metadata used to build bridge rotary payloads.

    The image/video grid and image-token masks are only consumed by multimodal RoPE schemes.
    `llama1d` only uses the latent valid mask plus the sequence lengths.
    """

    image_grid_thw: Optional[torch.Tensor] = None
    latent_image_token_mask: Optional[torch.Tensor] = None
    action_image_token_mask: Optional[torch.Tensor] = None
    latent_valid_mask: Optional[torch.Tensor] = None
    latent_left_pad_lens: Optional[torch.Tensor] = None
    video_grid_thw: Optional[torch.Tensor] = None
    qwen_position_ids: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "BridgeMRoPEBatchInfo":
        return BridgeMRoPEBatchInfo(
            image_grid_thw=None
            if self.image_grid_thw is None
            else self.image_grid_thw.to(device=device, dtype=torch.long),
            latent_image_token_mask=None
            if self.latent_image_token_mask is None
            else self.latent_image_token_mask.to(device=device, dtype=torch.bool),
            action_image_token_mask=None
            if self.action_image_token_mask is None
            else self.action_image_token_mask.to(device=device, dtype=torch.bool),
            latent_valid_mask=None
            if self.latent_valid_mask is None
            else self.latent_valid_mask.to(device=device, dtype=torch.bool),
            latent_left_pad_lens=None
            if self.latent_left_pad_lens is None
            else self.latent_left_pad_lens.to(device=device, dtype=torch.long),
            video_grid_thw=None
            if self.video_grid_thw is None
            else self.video_grid_thw.to(device=device, dtype=torch.long),
            qwen_position_ids=None
            if self.qwen_position_ids is None
            else self.qwen_position_ids.to(device=device, dtype=torch.long),
        )


@dataclass
class BridgeRotaryPayload:
    """Precomputed cos/sin tensors for one bridge batch signature."""

    latent_cos: Optional[torch.Tensor] = None
    latent_sin: Optional[torch.Tensor] = None
    action_cos: Optional[torch.Tensor] = None
    action_sin: Optional[torch.Tensor] = None


class BridgeRotaryEncoder(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        head_dim = int(head_dim)
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}.")
        self.head_dim = head_dim

    @staticmethod
    def _tensor_cache_device_key(device: torch.device) -> str:
        if isinstance(device, torch.device):
            return str(device)
        return str(torch.device(device))

    @staticmethod
    def apply_precomputed_rotary(
        q: Optional[torch.Tensor],
        k: Optional[torch.Tensor],
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if q is None or k is None:
            return q, k
        return _rotate_half_apply(q, cos, sin), _rotate_half_apply(k, cos, sin)


class BridgeA1MRoPE(BridgeRotaryEncoder):
    """Cosmos-aligned 3D multimodal RoPE for bridge q/k tensors."""

    def __init__(
        self,
        head_dim: int,
        interleave_thw: bool = False,
    ):
        super().__init__(head_dim=head_dim)
        self.interleave_thw = bool(interleave_thw)
        self.base = 10_000.0
        self.h_extrapolation_ratio = 3.0
        self.w_extrapolation_ratio = 3.0
        self.t_extrapolation_ratio = 1.0

        dim_h = head_dim // 6 * 2
        dim_w = dim_h
        dim_t = head_dim - 2 * dim_h
        if dim_t <= 2 or dim_h <= 2 or dim_w <= 2:
            raise ValueError(
                "Bridge 3D RoPE requires head_dim large enough to allocate T/H/W rotary dimensions, "
                f"got head_dim={head_dim}, dim_t={dim_t}, dim_h={dim_h}, dim_w={dim_w}."
            )
        if dim_t % 2 != 0 or dim_h % 2 != 0 or dim_w % 2 != 0:
            raise ValueError(
                "Bridge 3D RoPE requires even T/H/W rotary dimensions, "
                f"got dim_t={dim_t}, dim_h={dim_h}, dim_w={dim_w}."
            )

        self.dim_t = dim_t
        self.dim_h = dim_h
        self.dim_w = dim_w
        self.half_dim = head_dim // 2

        dim_spatial_range = torch.arange(0, dim_h, 2, dtype=torch.float32)[: (dim_h // 2)] / dim_h
        dim_temporal_range = torch.arange(0, dim_t, 2, dtype=torch.float32)[: (dim_t // 2)] / dim_t

        h_ntk_factor = self._compute_ntk_factor(self.h_extrapolation_ratio, dim_h)
        w_ntk_factor = self._compute_ntk_factor(self.w_extrapolation_ratio, dim_w)
        t_ntk_factor = self._compute_ntk_factor(self.t_extrapolation_ratio, dim_t)

        h_theta = self.base * h_ntk_factor
        w_theta = self.base * w_ntk_factor
        t_theta = self.base * t_ntk_factor

        self.register_buffer(
            "temporal_freqs",
            1.0 / (t_theta ** dim_temporal_range),
            persistent=False,
        )
        self.register_buffer(
            "h_spatial_freqs",
            1.0 / (h_theta ** dim_spatial_range),
            persistent=False,
        )
        self.register_buffer(
            "w_spatial_freqs",
            1.0 / (w_theta ** dim_spatial_range),
            persistent=False,
        )
        self.register_buffer(
            "thw_interleave_half_perm",
            self._build_thw_interleave_half_perm(
                dim_t=dim_t,
                dim_h=dim_h,
                dim_w=dim_w,
            ),
            persistent=False,
        )
        self.register_buffer(
            "thw_interleave_full_perm",
            torch.cat(
                (
                    self.thw_interleave_half_perm,
                    self.thw_interleave_half_perm + self.half_dim,
                ),
                dim=0,
            ),
            persistent=False,
        )

        self._text_position_cache = {}
        self._vision_position_cache = {}
        self._batch_rotary_cache = OrderedDict()
        self._max_batch_rotary_cache_entries = 8

    @staticmethod
    def _compute_ntk_factor(extrapolation_ratio: float, dim: int) -> float:
        return float(extrapolation_ratio) ** (dim / (dim - 2))

    @staticmethod
    def _build_thw_interleave_half_perm(dim_t: int, dim_h: int, dim_w: int) -> torch.Tensor:
        t_count = int(dim_t) // 2
        h_count = int(dim_h) // 2
        w_count = int(dim_w) // 2
        h_offset = t_count
        w_offset = t_count + h_count
        perm = []
        for idx in range(max(t_count, h_count, w_count)):
            if idx < t_count:
                perm.append(idx)
            if idx < h_count:
                perm.append(h_offset + idx)
            if idx < w_count:
                perm.append(w_offset + idx)
        expected = t_count + h_count + w_count
        if len(perm) != expected or sorted(perm) != list(range(expected)):
            raise ValueError(
                "Invalid THW interleave permutation for "
                f"dim_t={dim_t}, dim_h={dim_h}, dim_w={dim_w}."
            )
        return torch.tensor(perm, dtype=torch.long)

    @staticmethod
    def _tensor_cache_device_key(device: torch.device) -> str:
        if isinstance(device, torch.device):
            return str(device)
        return str(torch.device(device))

    def _extract_uniform_grid_signature(
        self,
        grid_thw: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[Tuple[int, int, int]]]:
        if grid_thw is None:
            return None, None
        normalized = grid_thw.to(device=device, dtype=torch.long)
        if normalized.ndim == 1:
            normalized = normalized.unsqueeze(0).expand(batch_size, -1)
        elif normalized.shape[0] == 1 and batch_size != 1:
            normalized = normalized.expand(batch_size, -1)
        ref = tuple(int(x) for x in normalized[0].tolist())
        return normalized, ref

    def _get_text_position_template(self, length: int, device: torch.device) -> torch.Tensor:
        key = (self._tensor_cache_device_key(device), int(length))
        cached = self._text_position_cache.get(key)
        if cached is None:
            cached = torch.arange(length, device=device, dtype=torch.long)
            self._text_position_cache[key] = cached
        return cached

    def _get_vision_position_template(
        self,
        grid_signature: Tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        key = (self._tensor_cache_device_key(device), tuple(int(x) for x in grid_signature))
        cached = self._vision_position_cache.get(key)
        if cached is None:
            cached = _build_vision_position_ids(
                torch.tensor(grid_signature, device=device, dtype=torch.long),
                device,
            )
            self._vision_position_cache[key] = cached
        return cached

    def _extract_left_pad_lens(self, valid_mask: torch.Tensor) -> Tuple[int, ...]:
        valid_mask = valid_mask.to(dtype=torch.bool)
        left_pad_lens = (~valid_mask).sum(dim=1)
        return tuple(int(x) for x in left_pad_lens.tolist())

    def _extract_contiguous_spans(self, token_mask: Optional[torch.Tensor]) -> Tuple[Tuple[int, int], ...]:
        if token_mask is None:
            return tuple()
        token_mask = token_mask.to(dtype=torch.bool)
        token_counts = token_mask.sum(dim=1).tolist()
        spans = []
        for row, token_count in zip(token_mask, token_counts):
            image_len = int(token_count)
            if image_len == 0:
                spans.append((-1, 0))
                continue
            start = int(row.to(dtype=torch.int64).argmax().item())
            spans.append((start, image_len))
        return tuple(spans)

    def _build_batch_cache_key(
        self,
        device: torch.device,
        dtype: torch.dtype,
        latent_seq_len: int,
        action_seq_len: int,
        image_grid_signature: Optional[Tuple[int, int, int]],
        video_grid_signature: Optional[Tuple[int, int, int]],
        left_pad_lens: Tuple[int, ...],
        latent_spans: Tuple[Tuple[int, int], ...],
        action_spans: Tuple[Tuple[int, int], ...],
    ) -> Tuple[object, ...]:
        return (
            self._tensor_cache_device_key(device),
            str(dtype),
            int(latent_seq_len),
            int(action_seq_len),
            tuple(left_pad_lens),
            tuple(latent_spans),
            tuple(action_spans),
            image_grid_signature,
            video_grid_signature,
        )

    def _fill_sequence_position_ids(
        self,
        out_position_ids: torch.Tensor,
        seq_len: int,
        valid_start: int,
        image_span: Tuple[int, int],
        current_max: int,
        image_position_template: Optional[torch.Tensor],
        image_block_advance: int,
        device: torch.device,
    ) -> int:
        image_start, image_len = image_span
        if image_len == 0:
            text_len = seq_len - valid_start
            if text_len > 0:
                text_positions = self._get_text_position_template(text_len, device) + (current_max + 1)
                out_position_ids[:, valid_start:seq_len] = text_positions.unsqueeze(0)
                current_max += text_len
            return current_max

        text_before_len = image_start - valid_start
        if text_before_len > 0:
            text_before = self._get_text_position_template(text_before_len, device) + (current_max + 1)
            out_position_ids[:, valid_start:image_start] = text_before.unsqueeze(0)
            current_max += text_before_len

        out_position_ids[:, image_start:image_start + image_len] = image_position_template + (current_max + 1)
        current_max += image_block_advance

        text_after_start = image_start + image_len
        text_after_len = seq_len - text_after_start
        if text_after_len > 0:
            text_after = self._get_text_position_template(text_after_len, device) + (current_max + 1)
            out_position_ids[:, text_after_start:seq_len] = text_after.unsqueeze(0)
            current_max += text_after_len

        return current_max

    def _extract_batch_signature(
        self,
        batch_info: BridgeMRoPEBatchInfo,
        batch_size: int,
        latent_seq_len: int,
        action_seq_len: int,
        device: torch.device,
    ) -> dict:
        batch_info = batch_info.to(device)
        image_grid_thw, image_grid_signature = self._extract_uniform_grid_signature(
            batch_info.image_grid_thw,
            batch_size,
            device,
        )
        video_grid_thw, video_grid_signature = self._extract_uniform_grid_signature(
            batch_info.video_grid_thw,
            batch_size,
            device,
        )

        if batch_info.latent_valid_mask is None:
            latent_valid_mask = torch.ones((batch_size, latent_seq_len), dtype=torch.bool, device=device)
        else:
            latent_valid_mask = batch_info.latent_valid_mask.to(device=device, dtype=torch.bool)

        if batch_info.latent_image_token_mask is None:
            latent_image_token_mask = torch.zeros((batch_size, latent_seq_len), dtype=torch.bool, device=device)
        else:
            latent_image_token_mask = batch_info.latent_image_token_mask.to(device=device, dtype=torch.bool)

        if batch_info.action_image_token_mask is None:
            action_image_token_mask = torch.zeros((batch_size, action_seq_len), dtype=torch.bool, device=device)
        else:
            action_image_token_mask = batch_info.action_image_token_mask.to(device=device, dtype=torch.bool)

        if batch_info.latent_left_pad_lens is None:
            left_pad_lens = self._extract_left_pad_lens(latent_valid_mask)
        else:
            left_pad_tensor = batch_info.latent_left_pad_lens.to(device=device, dtype=torch.long).flatten()
            if left_pad_tensor.numel() != batch_size:
                raise ValueError(
                    f"latent_left_pad_lens must have {batch_size} values, got {left_pad_tensor.numel()}."
                )
            left_pad_lens = tuple(int(x) for x in left_pad_tensor.tolist())
        latent_spans = self._extract_contiguous_spans(latent_image_token_mask)
        action_spans = self._extract_contiguous_spans(action_image_token_mask) if action_seq_len > 0 else tuple()

        cache_key = self._build_batch_cache_key(
            device=device,
            dtype=torch.float32,
            latent_seq_len=latent_seq_len,
            action_seq_len=action_seq_len,
            image_grid_signature=image_grid_signature,
            video_grid_signature=video_grid_signature,
            left_pad_lens=left_pad_lens,
            latent_spans=latent_spans,
            action_spans=action_spans,
        )

        image_position_template = None
        image_block_advance = 0
        if image_grid_signature is not None:
            image_position_template = self._get_vision_position_template(image_grid_signature, device)
            image_block_advance = max(image_grid_signature)

        video_block_end = -1
        if video_grid_signature is not None:
            video_block_end = _advance_with_vision_block(
                -1,
                torch.tensor(video_grid_signature, device=device, dtype=torch.long),
            )

        return {
            "cache_key": cache_key,
            "left_pad_lens": left_pad_lens,
            "latent_spans": latent_spans,
            "action_spans": action_spans,
            "image_position_template": image_position_template,
            "image_block_advance": image_block_advance,
            "video_block_end": video_block_end,
        }

    def _build_position_ids_from_signature(
        self,
        signature: dict,
        batch_size: int,
        latent_seq_len: int,
        action_seq_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        left_pad_lens = signature["left_pad_lens"]
        latent_spans = signature["latent_spans"]
        action_spans = signature["action_spans"]
        image_position_template = signature["image_position_template"]
        image_block_advance = signature["image_block_advance"]
        video_block_end = signature["video_block_end"]

        latent_position_ids = torch.zeros((3, batch_size, latent_seq_len), dtype=torch.long, device=device)
        action_position_ids = torch.zeros((3, batch_size, action_seq_len), dtype=torch.long, device=device)

        for batch_idx in range(batch_size):
            current_max = video_block_end
            current_max = self._fill_sequence_position_ids(
                latent_position_ids[:, batch_idx],
                seq_len=latent_seq_len,
                valid_start=left_pad_lens[batch_idx],
                image_span=latent_spans[batch_idx],
                current_max=current_max,
                image_position_template=image_position_template,
                image_block_advance=image_block_advance,
                device=device,
            )
            if action_seq_len > 0:
                current_max = self._fill_sequence_position_ids(
                    action_position_ids[:, batch_idx],
                    seq_len=action_seq_len,
                    valid_start=0,
                    image_span=action_spans[batch_idx] if action_spans else (-1, 0),
                    current_max=current_max,
                    image_position_template=image_position_template,
                    image_block_advance=image_block_advance,
                    device=device,
                )

        return latent_position_ids, action_position_ids

    def _build_cosmos_3d_freqs(self, position_ids: torch.Tensor) -> torch.Tensor:
        t_position_ids = position_ids[0].float().unsqueeze(-1)
        h_position_ids = position_ids[1].float().unsqueeze(-1)
        w_position_ids = position_ids[2].float().unsqueeze(-1)

        t_freqs = t_position_ids * self.temporal_freqs[None, None, :].float()
        h_freqs = h_position_ids * self.h_spatial_freqs[None, None, :].float()
        w_freqs = w_position_ids * self.w_spatial_freqs[None, None, :].float()
        freqs = torch.cat((t_freqs, h_freqs, w_freqs), dim=-1)
        if self.interleave_thw:
            freqs = freqs.index_select(-1, self.thw_interleave_half_perm.to(device=freqs.device))
        return freqs

    @torch.no_grad()
    def _build_cos_sin(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = self._build_cosmos_3d_freqs(position_ids)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def prepare_batch_rotary(
        self,
        batch_info: BridgeMRoPEBatchInfo,
        batch_size: int,
        latent_seq_len: int,
        action_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BridgeRotaryPayload:
        signature = self._extract_batch_signature(
            batch_info=batch_info,
            batch_size=batch_size,
            latent_seq_len=latent_seq_len,
            action_seq_len=action_seq_len,
            device=device,
        )

        full_cache_key = signature["cache_key"] + (self._tensor_cache_device_key(device), str(dtype))
        cached = self._batch_rotary_cache.get(full_cache_key)
        if cached is not None:
            self._batch_rotary_cache.move_to_end(full_cache_key)
            return cached

        latent_position_ids, action_position_ids = self._build_position_ids_from_signature(
            signature=signature,
            batch_size=batch_size,
            latent_seq_len=latent_seq_len,
            action_seq_len=action_seq_len,
            device=device,
        )

        latent_cos = latent_sin = action_cos = action_sin = None
        if latent_seq_len > 0:
            dummy_latent = torch.empty((batch_size, latent_seq_len, 1, self.head_dim), device=device, dtype=dtype)
            latent_cos, latent_sin = self._build_cos_sin(dummy_latent, latent_position_ids)
            latent_cos = latent_cos.unsqueeze(2)
            latent_sin = latent_sin.unsqueeze(2)
        if action_seq_len > 0:
            dummy_action = torch.empty((batch_size, action_seq_len, 1, self.head_dim), device=device, dtype=dtype)
            action_cos, action_sin = self._build_cos_sin(dummy_action, action_position_ids)
            action_cos = action_cos.unsqueeze(2)
            action_sin = action_sin.unsqueeze(2)

        payload = BridgeRotaryPayload(
            latent_cos=latent_cos,
            latent_sin=latent_sin,
            action_cos=action_cos,
            action_sin=action_sin,
        )
        self._batch_rotary_cache[full_cache_key] = payload
        self._batch_rotary_cache.move_to_end(full_cache_key)
        while len(self._batch_rotary_cache) > self._max_batch_rotary_cache_entries:
            self._batch_rotary_cache.popitem(last=False)
        return payload

class BridgeLlama1DRoPE(BridgeRotaryEncoder):
    """Native Llama-style 1D RoPE for latent/action bridge q/k tensors."""

    def __init__(self, head_dim: int, janus_rotary_emb: nn.Module):
        super().__init__(head_dim=head_dim)
        # Keep a live reference to Janus' rotary module without re-registering it here.
        self.__dict__["janus_rotary_emb"] = janus_rotary_emb
        self._batch_rotary_cache = OrderedDict()
        self._max_batch_rotary_cache_entries = 8

    @staticmethod
    def _extract_left_pad_lens(valid_mask: torch.Tensor) -> Tuple[int, ...]:
        valid_mask = valid_mask.to(dtype=torch.bool)
        left_pad_lens = (~valid_mask).sum(dim=1)
        return tuple(int(x) for x in left_pad_lens.tolist())

    @staticmethod
    def _extract_valid_counts(valid_mask: torch.Tensor) -> Tuple[int, ...]:
        valid_counts = valid_mask.to(dtype=torch.long).sum(dim=1)
        return tuple(int(x) for x in valid_counts.tolist())

    def _extract_batch_signature(
        self,
        batch_info: BridgeMRoPEBatchInfo,
        batch_size: int,
        latent_seq_len: int,
        action_seq_len: int,
        device: torch.device,
    ) -> dict:
        batch_info = batch_info.to(device)
        if batch_info.latent_valid_mask is None:
            latent_valid_mask = torch.ones((batch_size, latent_seq_len), dtype=torch.bool, device=device)
        else:
            latent_valid_mask = batch_info.latent_valid_mask.to(device=device, dtype=torch.bool)

        if batch_info.latent_left_pad_lens is None:
            left_pad_lens = self._extract_left_pad_lens(latent_valid_mask)
        else:
            left_pad_tensor = batch_info.latent_left_pad_lens.to(device=device, dtype=torch.long).flatten()
            if left_pad_tensor.numel() != batch_size:
                raise ValueError(
                    f"latent_left_pad_lens must have {batch_size} values, got {left_pad_tensor.numel()}."
                )
            left_pad_lens = tuple(int(x) for x in left_pad_tensor.tolist())
        latent_position_counts = tuple(max(int(latent_seq_len) - left_pad, 0) for left_pad in left_pad_lens)
        cache_key = (
            self._tensor_cache_device_key(device),
            torch.float32,
            int(latent_seq_len),
            int(action_seq_len),
            tuple(left_pad_lens),
            tuple(latent_position_counts),
        )
        return {
            "cache_key": cache_key,
            "left_pad_lens": left_pad_lens,
            "latent_position_counts": latent_position_counts,
        }

    def _build_position_ids_from_signature(
        self,
        signature: dict,
        batch_size: int,
        latent_seq_len: int,
        action_seq_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        left_pad_lens = signature["left_pad_lens"]
        latent_position_counts = signature["latent_position_counts"]

        latent_position_ids = torch.zeros((batch_size, latent_seq_len), dtype=torch.long, device=device)
        if latent_seq_len > 0:
            base_positions = torch.arange(latent_seq_len, device=device, dtype=torch.long).unsqueeze(0)
            left_pad_tensor = torch.tensor(left_pad_lens, device=device, dtype=torch.long).unsqueeze(1)
            shifted_positions = base_positions - left_pad_tensor
            latent_position_ids = torch.where(
                base_positions >= left_pad_tensor,
                shifted_positions,
                torch.zeros_like(shifted_positions),
            )

        action_position_ids = torch.zeros((batch_size, action_seq_len), dtype=torch.long, device=device)
        if action_seq_len > 0:
            action_offsets = torch.tensor(latent_position_counts, device=device, dtype=torch.long).unsqueeze(1)
            action_position_ids = action_offsets + torch.arange(
                action_seq_len,
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)

        return latent_position_ids, action_position_ids

    @torch.no_grad()
    def _build_cos_sin(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.janus_rotary_emb(x, position_ids)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def prepare_batch_rotary(
        self,
        batch_info: BridgeMRoPEBatchInfo,
        batch_size: int,
        latent_seq_len: int,
        action_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BridgeRotaryPayload:
        signature = self._extract_batch_signature(
            batch_info=batch_info,
            batch_size=batch_size,
            latent_seq_len=latent_seq_len,
            action_seq_len=action_seq_len,
            device=device,
        )

        full_cache_key = signature["cache_key"] + (self._tensor_cache_device_key(device), str(dtype))
        cached = self._batch_rotary_cache.get(full_cache_key)
        if cached is not None:
            self._batch_rotary_cache.move_to_end(full_cache_key)
            return cached

        latent_position_ids, action_position_ids = self._build_position_ids_from_signature(
            signature=signature,
            batch_size=batch_size,
            latent_seq_len=latent_seq_len,
            action_seq_len=action_seq_len,
            device=device,
        )

        latent_cos = latent_sin = action_cos = action_sin = None
        if latent_seq_len > 0:
            dummy_latent = torch.empty((batch_size, latent_seq_len, 1, self.head_dim), device=device, dtype=dtype)
            latent_cos, latent_sin = self._build_cos_sin(dummy_latent, latent_position_ids)
            latent_cos = latent_cos.unsqueeze(2)
            latent_sin = latent_sin.unsqueeze(2)
        if action_seq_len > 0:
            dummy_action = torch.empty((batch_size, action_seq_len, 1, self.head_dim), device=device, dtype=dtype)
            action_cos, action_sin = self._build_cos_sin(dummy_action, action_position_ids)
            action_cos = action_cos.unsqueeze(2)
            action_sin = action_sin.unsqueeze(2)

        payload = BridgeRotaryPayload(
            latent_cos=latent_cos,
            latent_sin=latent_sin,
            action_cos=action_cos,
            action_sin=action_sin,
        )
        self._batch_rotary_cache[full_cache_key] = payload
        self._batch_rotary_cache.move_to_end(full_cache_key)
        while len(self._batch_rotary_cache) > self._max_batch_rotary_cache_entries:
            self._batch_rotary_cache.popitem(last=False)
        return payload


class BridgeQwenNativeMRoPE(BridgeRotaryEncoder):
    """Qwen3-VL native multimodal RoPE for right-branch Q/K tensors."""

    def __init__(self, head_dim: int, janus_rotary_emb: nn.Module):
        super().__init__(head_dim=head_dim)
        # Keep a live reference without registering the same rotary module twice.
        self.__dict__["janus_rotary_emb"] = janus_rotary_emb

    @staticmethod
    def _normalize_position_ids(
        position_ids: torch.Tensor,
        batch_size: int,
        total_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        position_ids = position_ids.to(device=device, dtype=torch.long)
        if position_ids.ndim == 2:
            if position_ids.shape != (batch_size, total_seq_len):
                raise ValueError(
                    "Qwen 1-D fallback position_ids must have shape "
                    f"{(batch_size, total_seq_len)}, got {tuple(position_ids.shape)}."
                )
            return position_ids
        if position_ids.ndim != 3:
            raise ValueError(
                "Qwen native position_ids must be [3,B,L] or [B,3,L], "
                f"got {tuple(position_ids.shape)}."
            )
        if position_ids.shape == (3, batch_size, total_seq_len):
            return position_ids
        if position_ids.shape == (batch_size, 3, total_seq_len):
            return position_ids.permute(1, 0, 2).contiguous()
        raise ValueError(
            "Qwen native position_ids shape mismatch: expected "
            f"{(3, batch_size, total_seq_len)} or {(batch_size, 3, total_seq_len)}, "
            f"got {tuple(position_ids.shape)}."
        )

    def _build_cos_sin(
        self,
        batch_size: int,
        seq_len: int,
        position_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dummy = torch.empty((batch_size, seq_len, 1, self.head_dim), device=device, dtype=dtype)
        cos, sin = self.janus_rotary_emb(dummy, position_ids)
        return cos.to(dtype=dtype).unsqueeze(2), sin.to(dtype=dtype).unsqueeze(2)

    def prepare_batch_rotary(
        self,
        batch_info: BridgeMRoPEBatchInfo,
        batch_size: int,
        latent_seq_len: int,
        action_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BridgeRotaryPayload:
        batch_info = batch_info.to(device)
        total_seq_len = int(latent_seq_len) + int(action_seq_len)
        if batch_info.qwen_position_ids is None:
            raise ValueError("bridge_pos_scheme='qwen' requires native Qwen position_ids.")
        position_ids = self._normalize_position_ids(
            batch_info.qwen_position_ids,
            batch_size=batch_size,
            total_seq_len=total_seq_len,
            device=device,
        )

        latent_cos = latent_sin = action_cos = action_sin = None
        if latent_seq_len > 0:
            latent_positions = position_ids[..., :latent_seq_len]
            latent_cos, latent_sin = self._build_cos_sin(
                batch_size,
                int(latent_seq_len),
                latent_positions,
                device,
                dtype,
            )
        if action_seq_len > 0:
            action_positions = position_ids[..., latent_seq_len:total_seq_len]
            action_cos, action_sin = self._build_cos_sin(
                batch_size,
                int(action_seq_len),
                action_positions,
                device,
                dtype,
            )
        return BridgeRotaryPayload(
            latent_cos=latent_cos,
            latent_sin=latent_sin,
            action_cos=action_cos,
            action_sin=action_sin,
        )


def _rotate_half_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding: x * cos + rotate_half(x) * sin."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin

def _build_vision_position_ids(grid_thw: torch.Tensor, device: torch.device) -> torch.Tensor:
    t, h, w = [int(x) for x in grid_thw.tolist()]
    t_index = torch.arange(t, device=device, dtype=torch.long).view(-1, 1).expand(-1, h * w).reshape(-1)
    h_index = torch.arange(h, device=device, dtype=torch.long).view(1, -1, 1).expand(t, -1, w).reshape(-1)
    w_index = torch.arange(w, device=device, dtype=torch.long).view(1, 1, -1).expand(t, h, -1).reshape(-1)
    return torch.stack([t_index, h_index, w_index], dim=0)


def _advance_with_vision_block(current_max: int, grid_thw: torch.Tensor) -> int:
    t, h, w = [int(x) for x in grid_thw.tolist()]
    return current_max + max(t, h, w)

class LatentNativeAttentionAdapter(nn.Module):
    """Latent branch adapter using Janus native attention projections."""

    def __init__(self, janus_layer, cosmos_num_heads, cosmos_head_dim, layer_idx: int):
        super().__init__()
        hidden_size = janus_layer.mlp.up_proj.weight.shape[1]

        self.layer_idx = int(layer_idx)
        self.branch_name = "latent"
        self.norm_qkv = janus_layer.input_layernorm
        self.norm_ffn = janus_layer.post_attention_layernorm
        self.mlp = janus_layer.mlp
        self.branch_attn = janus_layer.self_attn
        self.num_heads, self.head_dim = self._validate_attention_layout(
            self.branch_attn,
            hidden_size=hidden_size,
            expected_num_heads=cosmos_num_heads,
            expected_head_dim=cosmos_head_dim,
        )

    @staticmethod
    def _proj_out_features(proj: nn.Module) -> int:
        if hasattr(proj, "out_features"):
            return int(proj.out_features)
        if hasattr(proj, "weight"):
            return int(proj.weight.shape[0])
        raise AttributeError(f"Projection module {type(proj).__name__} has no out_features/weight.")

    def _validate_attention_layout(
        self,
        attn_module: nn.Module,
        hidden_size: int,
        expected_num_heads: int,
        expected_head_dim: int,
    ) -> Tuple[int, int]:
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if not hasattr(attn_module, proj_name):
                raise AttributeError(
                    f"Layer {self.layer_idx} {self.branch_name} attention is missing `{proj_name}`."
                )

        q_proj = getattr(attn_module, "q_proj")
        k_proj = getattr(attn_module, "k_proj")
        v_proj = getattr(attn_module, "v_proj")
        o_proj = getattr(attn_module, "o_proj")

        q_out = self._proj_out_features(q_proj)
        k_out = self._proj_out_features(k_proj)
        v_out = self._proj_out_features(v_proj)
        o_in = getattr(o_proj, "in_features", None)
        if o_in is None and hasattr(o_proj, "weight"):
            o_in = int(o_proj.weight.shape[1])
        o_in = int(o_in) if o_in is not None else None

        num_heads = getattr(attn_module, "num_heads", getattr(attn_module, "n_heads", None))
        head_dim = getattr(attn_module, "head_dim", None)

        if num_heads is None and head_dim is not None:
            if q_out % int(head_dim) != 0:
                raise ValueError(
                    f"Layer {self.layer_idx} {self.branch_name} attention q_proj out={q_out} "
                    f"is not divisible by head_dim={head_dim}."
                )
            num_heads = q_out // int(head_dim)
        if head_dim is None and num_heads is not None:
            if q_out % int(num_heads) != 0:
                raise ValueError(
                    f"Layer {self.layer_idx} {self.branch_name} attention q_proj out={q_out} "
                    f"is not divisible by num_heads={num_heads}."
                )
            head_dim = q_out // int(num_heads)

        if num_heads is None or head_dim is None:
            raise ValueError(
                f"Layer {self.layer_idx} {self.branch_name} attention layout is unknown: "
                f"num_heads={num_heads}, head_dim={head_dim}, q_proj_out={q_out}."
            )

        num_heads = int(num_heads)
        head_dim = int(head_dim)
        expected_dim = expected_num_heads * expected_head_dim

        if q_out != num_heads * head_dim:
            raise ValueError(
                f"Layer {self.layer_idx} {self.branch_name} q_proj layout mismatch: "
                f"q_proj_out={q_out}, num_heads={num_heads}, head_dim={head_dim}."
            )
        if q_out != expected_dim or k_out != expected_dim or v_out != expected_dim:
            raise ValueError(
                f"Layer {self.layer_idx} {self.branch_name} attention must match Cosmos exactly. "
                f"Janus(q={q_out}, k={k_out}, v={v_out}, num_heads={num_heads}, head_dim={head_dim}, "
                f"hidden_size={hidden_size}) vs Cosmos(num_heads={expected_num_heads}, "
                f"head_dim={expected_head_dim}, hidden_size={expected_dim})."
            )
        if num_heads != expected_num_heads or head_dim != expected_head_dim:
            raise ValueError(
                f"Layer {self.layer_idx} {self.branch_name} attention head layout mismatch: "
                f"Janus(num_heads={num_heads}, head_dim={head_dim}, hidden_size={hidden_size}) vs "
                f"Cosmos(num_heads={expected_num_heads}, head_dim={expected_head_dim}, "
                f"hidden_size={expected_dim})."
            )
        if o_in is not None and o_in != expected_dim:
            raise ValueError(
                f"Layer {self.layer_idx} {self.branch_name} o_proj input dim {o_in} does not match "
                f"Cosmos hidden size {expected_dim}."
            )
        return num_heads, head_dim

    def get_branch_qkv(self, hidden_states: torch.Tensor):
        norm_x = self.norm_qkv(hidden_states)
        B, L, _ = norm_x.shape
        q = self.branch_attn.q_proj(norm_x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.branch_attn.k_proj(norm_x).reshape(B, L, self.num_heads, self.head_dim)
        v = self.branch_attn.v_proj(norm_x).reshape(B, L, self.num_heads, self.head_dim)
        return q, k, v

    def post_attention(
        self,
        hidden_states: torch.Tensor,
        attn_out: torch.Tensor,
        token_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attn_out.dim() == 4:
            attn_out = attn_out.flatten(2, 3)
        hidden_states = hidden_states + self.branch_attn.o_proj(attn_out)
        hidden_states = hidden_states + self.mlp(self.norm_ffn(hidden_states))
        if token_valid_mask is not None:
            hidden_states = hidden_states * token_valid_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states


class ActionNativeAttentionAdapter(nn.Module):
    """Action branch adapter using Janus native attention projections."""

    def __init__(
        self,
        janus_layer,
        cosmos_num_heads,
        cosmos_head_dim,
        action_intermediate_size=None,
        layer_idx: int = 0,
    ):
        super().__init__()
        hidden_size = janus_layer.mlp.up_proj.weight.shape[1]

        self.layer_idx = int(layer_idx)
        self.branch_name = "action"
        self.norm_qkv = self._require_action_attr(janus_layer, "input_layernorm_action")
        self.norm_ffn = self._require_action_attr(janus_layer, "post_attention_layernorm_action")

        if action_intermediate_size and action_intermediate_size > 0:
            original_intermediate = janus_layer.mlp.up_proj.weight.shape[0]
            if action_intermediate_size != original_intermediate:
                self.mlp = SlimLlamaMLP(hidden_size, action_intermediate_size)
                print(
                    f"Using custom action MLP with intermediate size {action_intermediate_size} instead of "
                    f"original size {original_intermediate}."
                )
            else:
                self.mlp = self._get_action_mlp(janus_layer)
        else:
            self.mlp = self._get_action_mlp(janus_layer)

        self.branch_attn = self._require_action_attr(janus_layer, "self_attn_action")
        self.num_heads, self.head_dim = LatentNativeAttentionAdapter._validate_attention_layout(
            self,
            self.branch_attn,
            hidden_size=hidden_size,
            expected_num_heads=cosmos_num_heads,
            expected_head_dim=cosmos_head_dim,
        )

    def _require_action_attr(self, obj, attr_name: str):
        if hasattr(obj, attr_name):
            return getattr(obj, attr_name)
        raise AttributeError(
            f"Layer {self.layer_idx} action branch requires `{attr_name}` on "
            f"{type(obj).__name__}, but it is missing. Refusing to fall back to latent/shared modules."
        )

    @staticmethod
    def _get_action_mlp(janus_layer):
        if hasattr(janus_layer, 'mlp_action'):
            return janus_layer.mlp_action
        raise AttributeError(
            "Action branch requires `mlp_action` on the Janus layer, but it is missing. "
            "Refusing to fall back to the latent/shared MLP."
        )

    def _proj_out_features(self, proj: nn.Module) -> int:
        return LatentNativeAttentionAdapter._proj_out_features(proj)

    def get_branch_qkv(self, hidden_states: torch.Tensor):
        norm_x = self.norm_qkv(hidden_states)
        B, L, _ = norm_x.shape
        q = self.branch_attn.q_proj(norm_x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.branch_attn.k_proj(norm_x).reshape(B, L, self.num_heads, self.head_dim)
        v = self.branch_attn.v_proj(norm_x).reshape(B, L, self.num_heads, self.head_dim)
        return q, k, v

    def post_attention(
        self,
        hidden_states: torch.Tensor,
        attn_out: torch.Tensor,
        token_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attn_out.dim() == 4:
            attn_out = attn_out.flatten(2, 3)
        hidden_states = hidden_states + self.branch_attn.o_proj(attn_out)
        hidden_states = hidden_states + self.mlp(self.norm_ffn(hidden_states))
        if token_valid_mask is not None:
            hidden_states = hidden_states * token_valid_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states


class MoTAttentionWrapper3(nn.Module):
    """3-way MoT attention wrapper.

    Per-layer processing:
      1. Bridge attention: [video, latent, action] in cosmos attention space
         with a configurable custom mask
      2. Post-attention: residual + MLP for each expert

    Supports caching video KV for one-step cosmos + multi-step action inference.
    """

    def __init__(
        self,
        original_attn,
        latent_bridge,
        action_bridge,
        cosmos_self_only_bridge=False,
        decosmos=False,
        bridge_action_self_causal_override=None,
        interleave_video_qk=False,
        value_token_mask_video_to_value=False,
        value_token_mask_nonvalue_to_value=False,
    ):
        super().__init__()
        self.original_attn = original_attn
        self.latent_bridge = latent_bridge
        self.action_bridge = action_bridge
        self.cosmos_self_only_bridge = bool(cosmos_self_only_bridge)
        self.decosmos = bool(decosmos)
        self.value_token_mask_nonvalue_to_value = bool(value_token_mask_nonvalue_to_value)
        self.value_token_mask_video_to_value = bool(
            value_token_mask_video_to_value or self.value_token_mask_nonvalue_to_value
        )

        self.current_x_latent = None
        self.current_x_action = None
        self.current_latent_valid_mask = None
        self.current_rotary_payload = None
        self.current_mot_forward_index = 1
        self.current_cache_video_kv = False
        self.current_cache_video_kv_detach = True
        self.current_value_token_count = 0
        self.current_action_value_token_count = 0
        self.next_x_latent = None
        self.next_x_action = None
        self.detach_video_kv = False

        # For one-step cosmos inference: cache video KV
        self.cache_video_kv = False
        self.cached_k_v = None
        self.cached_v_v = None
        self.cached_video_grid_thw = None
        self.cached_value_token_count = 0

        # Inference-only override for bridge attention on action -> action.
        # None means using the default mask behavior defined by the model.
        self.bridge_action_self_causal_override = bridge_action_self_causal_override
        self.interleave_video_qk = bool(interleave_video_qk)
        if self.interleave_video_qk:
            head_dim = int(getattr(original_attn, "head_dim", latent_bridge.head_dim))
            dim_h = head_dim // 6 * 2
            dim_w = dim_h
            dim_t = head_dim - 2 * dim_h
            half_perm = BridgeA1MRoPE._build_thw_interleave_half_perm(
                dim_t=dim_t,
                dim_h=dim_h,
                dim_w=dim_w,
            )
            full_perm = torch.cat((half_perm, half_perm + head_dim // 2), dim=0)
            self.register_buffer("video_qk_interleave_perm", full_perm, persistent=False)
        else:
            self.video_qk_interleave_perm = None

    def _resolve_action_self_causal(self, default=False):
        """Default is False (bidirectional) since actions are generated in parallel via flow matching."""
        if self.bridge_action_self_causal_override is None:
            return bool(default)
        return bool(self.bridge_action_self_causal_override)

    def _store_video_kv_cache(
        self,
        k_v,
        v_v,
        batch_size: int,
        device: torch.device,
        video_size,
        detach: bool,
        value_token_count: int = 0,
    ):
        self.cached_k_v = k_v.detach() if detach else k_v
        self.cached_v_v = v_v.detach() if detach else v_v
        video_grid_thw = self._get_video_grid_thw(batch_size, device, video_size=video_size)
        self.cached_video_grid_thw = video_grid_thw.detach().clone()
        self.cached_value_token_count = int(value_token_count or 0)

    def _interleave_video_qk(self, q_v: torch.Tensor, k_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.interleave_video_qk:
            return q_v, k_v
        perm = self.video_qk_interleave_perm.to(device=q_v.device)
        return q_v.index_select(-1, perm), k_v.index_select(-1, perm)

    def _get_video_grid_thw(
        self,
        batch_size: int,
        device: torch.device,
        video_size=None,
    ) -> torch.Tensor:
        if video_size is not None:
            video_grid_thw = torch.tensor(
                [int(video_size.T), int(video_size.H), int(video_size.W)],
                device=device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)
            return video_grid_thw

        if self.cached_video_grid_thw is not None:
            cached = self.cached_video_grid_thw.to(device=device, dtype=torch.long)
            if cached.ndim == 1:
                cached = cached.unsqueeze(0).expand(batch_size, -1)
            elif cached.shape[0] == 1 and batch_size != 1:
                cached = cached.expand(batch_size, -1)
            return cached

        raise ValueError("Bridge MRoPE requires current or cached video THW, but none is available.")

    def _apply_branch_rotary(
        self,
        q_l: Optional[torch.Tensor],
        k_l: Optional[torch.Tensor],
        q_a: Optional[torch.Tensor],
        k_a: Optional[torch.Tensor],
        rotary_payload: Optional[BridgeRotaryPayload] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if q_l is None and q_a is None:
            return q_l, k_l, q_a, k_a
        if rotary_payload is None:
            raise ValueError("rotary_payload is required when latent/action bridge tokens are present.")

        if q_l is not None and k_l is not None:
            q_l, k_l = BridgeRotaryEncoder.apply_precomputed_rotary(
                q_l,
                k_l,
                rotary_payload.latent_cos,
                rotary_payload.latent_sin,
            )
        if q_a is not None and k_a is not None:
            q_a, k_a = BridgeRotaryEncoder.apply_precomputed_rotary(
                q_a,
                k_a,
                rotary_payload.action_cos,
                rotary_payload.action_sin,
            )
        return q_l, k_l, q_a, k_a

    def _query_valid_mask(self, parts, device, batch_size):
        valid_parts = []
        for part in parts:
            if part is None:
                continue
            if isinstance(part, int):
                if part > 0:
                    valid_parts.append(torch.ones((batch_size, part), device=device, dtype=torch.bool))
            else:
                valid_parts.append(part.to(device=device, dtype=torch.bool))
        if not valid_parts:
            return None
        return torch.cat(valid_parts, dim=1)

    def _value_token_range(self, S_v: int, value_token_count: int) -> Optional[Tuple[int, int]]:
        value_token_count = int(value_token_count or 0)
        if value_token_count <= 0:
            return None
        if value_token_count > S_v:
            raise ValueError(
                f"value_token_count ({value_token_count}) cannot exceed Cosmos token count ({S_v})."
            )
        return S_v - value_token_count, S_v

    def _mask_value_kv_for_nonvalue_queries(
        self,
        mask: torch.Tensor,
        S_v: int,
        S_l: int,
        S_a: int,
        value_token_count: int,
    ) -> torch.Tensor:
        value_range = self._value_token_range(S_v, value_token_count)
        if value_range is None:
            return mask

        value_start, value_end = value_range
        mask_video_queries = self.value_token_mask_video_to_value or self.value_token_mask_nonvalue_to_value
        if mask_video_queries and value_start > 0:
            mask[:value_start, value_start:value_end] = False

        if self.value_token_mask_nonvalue_to_value:
            if S_l > 0:
                mask[S_v:S_v + S_l, value_start:value_end] = False
            if S_a > 0:
                mask[S_v + S_l:S_v + S_l + S_a, value_start:value_end] = False

        return mask

    def _mask_cached_value_kv_for_nonvalue_queries(
        self,
        mask: torch.Tensor,
        S_v: int,
        value_token_count: int,
    ) -> torch.Tensor:
        if not self.value_token_mask_nonvalue_to_value:
            return mask
        value_range = self._value_token_range(S_v, value_token_count)
        if value_range is None:
            return mask
        value_start, value_end = value_range
        mask[:, value_start:value_end] = False
        return mask

    def _action_value_token_ranges(
        self,
        S_v: int,
        S_l: int,
        S_a: int,
        action_value_token_count: int,
    ) -> Optional[Tuple[int, int, int]]:
        action_value_token_count = int(action_value_token_count or 0)
        if action_value_token_count <= 0:
            return None
        if action_value_token_count > S_a:
            raise ValueError(
                f"action_value_token_count ({action_value_token_count}) cannot exceed action token count ({S_a})."
            )
        kv_start = S_v + S_l + S_a - action_value_token_count
        kv_end = S_v + S_l + S_a
        query_start = S_l + S_a - action_value_token_count
        return kv_start, kv_end, query_start

    def _mask_action_value_kv_for_nonvalue_queries(
        self,
        mask: torch.Tensor,
        S_v: int,
        S_l: int,
        S_a: int,
        action_value_token_count: int,
        *,
        cached_action_only: bool = False,
    ) -> torch.Tensor:
        ranges = self._action_value_token_ranges(S_v, S_l, S_a, action_value_token_count)
        if ranges is None:
            return mask

        kv_start, kv_end, query_start = ranges
        if cached_action_only:
            mask[S_l:query_start, kv_start:kv_end] = False
        else:
            action_query_start = S_v + S_l
            mask[action_query_start:kv_start, kv_start:kv_end] = False
        return mask

    def _build_bridge_mask(
        self,
        S_v,
        S_l,
        S_a,
        device,
        latent_valid_mask=None,
        value_token_count: int = 0,
        action_value_token_count: int = 0,
    ):
        """Build bridge attention mask.

              | Cosmos | Latent_i | Action_j |
        Cosmos |  full  |   full   |   full   |
        Lat_i  |  full  |  causal  |    0     |
        Action_i |  full  |   full   |   full   |

        If cosmos_self_only_bridge is enabled, cosmos queries attend only to
        cosmos KV while latent/action behavior stays unchanged. If decosmos is
        enabled, latent/action queries cannot attend to cosmos KV.

        Returns: [1, 1, S_total, S_total] boolean mask for SDPA (True=attend).
        """
        total = S_v + S_l + S_a
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        action_self_causal = self._resolve_action_self_causal(default=True)

        if S_v > 0:
            mask[:S_v, :S_v] = True
            if not self.cosmos_self_only_bridge:
                if S_l > 0:
                    mask[:S_v, S_v:S_v + S_l] = True
                if S_a > 0:
                    mask[:S_v, S_v + S_l:] = True

        if S_l > 0:
            if not self.decosmos:
                mask[S_v:S_v + S_l, :S_v] = True
            latent_causal = torch.tril(torch.ones(S_l, S_l, dtype=torch.bool, device=device))
            mask[S_v:S_v + S_l, S_v:S_v + S_l] = latent_causal

        if S_a > 0:
            if not self.decosmos:
                mask[S_v + S_l:, :S_v] = True
            if S_l > 0:
                mask[S_v + S_l:, S_v:S_v + S_l] = True
            if action_self_causal:
                action_mask = torch.tril(torch.ones(S_a, S_a, dtype=torch.bool, device=device))
            else:
                action_mask = torch.ones(S_a, S_a, dtype=torch.bool, device=device)
            mask[S_v + S_l:, S_v + S_l:] = action_mask

        mask = self._mask_value_kv_for_nonvalue_queries(
            mask=mask,
            S_v=S_v,
            S_l=S_l,
            S_a=S_a,
            value_token_count=value_token_count,
        )
        mask = self._mask_action_value_kv_for_nonvalue_queries(
            mask=mask,
            S_v=S_v,
            S_l=S_l,
            S_a=S_a,
            action_value_token_count=action_value_token_count,
            cached_action_only=False,
        )

        if latent_valid_mask is None:
            return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, total, total]

        latent_valid_mask = latent_valid_mask.to(device=device, dtype=torch.bool)
        batch_mask = mask.unsqueeze(0).expand(latent_valid_mask.shape[0], -1, -1).clone()
        if S_l > 0:
            batch_mask[:, :, S_v:S_v + S_l] &= latent_valid_mask[:, None, :]
        return batch_mask.unsqueeze(1)  # [B, 1, total, total]

    def _build_action_only_mask(
        self,
        S_v,
        S_l,
        S_a,
        device,
        latent_valid_mask=None,
        value_token_count: int = 0,
        action_value_token_count: int = 0,
    ):
        """Build attention mask for action-only forward (Q=[latent,action], KV=[cosmos,latent,action]).

        Q rows: [latent(S_l), action(S_a)]
        KV cols: [cosmos(S_v), latent(S_l), action(S_a)]

              | Cosmos | Latent_j | Action_j |
        Lat_i   |  full  |  causal  |    0     |
        Action_i |  full  |   full   |   full   |

        Returns: [1, 1, S_l+S_a, S_v+S_l+S_a] boolean mask for SDPA.
        """
        S_q = S_l + S_a
        S_kv = S_v + S_l + S_a
        mask = torch.ones(S_q, S_kv, dtype=torch.bool, device=device)
        action_self_causal = self._resolve_action_self_causal(default=True)

        if S_l > 0:
            # Latent Q 鈫?Latent KV: causal
            latent_causal = torch.tril(torch.ones(S_l, S_l, dtype=torch.bool, device=device))
            mask[:S_l, S_v:S_v + S_l] = latent_causal

        if S_l > 0 and S_a > 0:
            # Latent Q 鈫?Action KV: blocked
            mask[:S_l, S_v + S_l:] = False

        if S_a > 0:
            if action_self_causal:
                action_mask = torch.tril(torch.ones(S_a, S_a, dtype=torch.bool, device=device))
            else:
                action_mask = torch.ones(S_a, S_a, dtype=torch.bool, device=device)
            mask[S_l:, S_v + S_l:] = action_mask

        mask = self._mask_cached_value_kv_for_nonvalue_queries(
            mask=mask,
            S_v=S_v,
            value_token_count=value_token_count,
        )
        mask = self._mask_action_value_kv_for_nonvalue_queries(
            mask=mask,
            S_v=S_v,
            S_l=S_l,
            S_a=S_a,
            action_value_token_count=action_value_token_count,
            cached_action_only=True,
        )

        if latent_valid_mask is None:
            return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S_q, S_kv]

        latent_valid_mask = latent_valid_mask.to(device=device, dtype=torch.bool)
        batch_mask = mask.unsqueeze(0).expand(latent_valid_mask.shape[0], -1, -1).clone()
        if S_l > 0:
            batch_mask[:, :, S_v:S_v + S_l] &= latent_valid_mask[:, None, :]
        return batch_mask.unsqueeze(1)  # [B, 1, S_q, S_kv]

    def _build_latent_only_mask(self, S_v, S_l, device, latent_valid_mask=None, value_token_count: int = 0):
        """Build attention mask for latent-only forward (Q=[latent], KV=[cosmos,latent])."""
        mask = torch.ones(S_l, S_v + S_l, dtype=torch.bool, device=device)
        latent_causal = torch.tril(torch.ones(S_l, S_l, dtype=torch.bool, device=device))
        mask[:, S_v:] = latent_causal

        mask = self._mask_cached_value_kv_for_nonvalue_queries(
            mask=mask,
            S_v=S_v,
            value_token_count=value_token_count,
        )

        if latent_valid_mask is None:
            return mask.unsqueeze(0).unsqueeze(0)

        latent_valid_mask = latent_valid_mask.to(device=device, dtype=torch.bool)
        batch_mask = mask.unsqueeze(0).expand(latent_valid_mask.shape[0], -1, -1).clone()
        if S_l > 0:
            batch_mask[:, :, S_v:] &= latent_valid_mask[:, None, :]
        return batch_mask.unsqueeze(1)

    def _bridge_sdpa(self, q, k, v, attn_mask, query_valid_mask=None):
        """Run scaled dot-product attention in [B, S, H, D] format with mask."""
        # [B, S, H, D] 鈫?[B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        # [B, H, S, D] 鈫?[B, S, H*D]
        out = out.transpose(1, 2).flatten(2, 3)
        if query_valid_mask is not None:
            out = out * query_valid_mask.unsqueeze(-1).to(out.dtype)
        return out

    def forward(self, x, context=None, rope_emb=None, video_size=None, kv_cache_cfg=None):
        """Full forward: runs cosmos + latent + action through bridge attention only."""
        forward_index = int(self.current_mot_forward_index or 1)
        if forward_index < 1:
            raise ValueError(f"mot_forward_index must be >= 1, got {forward_index}.")

        use_cached_video_kv = forward_index == 2
        value_token_count = 0
        action_value_token_count = int(self.current_action_value_token_count or 0)
        if use_cached_video_kv:
            if self.cached_k_v is None or self.cached_v_v is None:
                raise RuntimeError("MoT pass 2 requires cached video KV from pass 1.")
            k_v = self.cached_k_v
            v_v = self.cached_v_v
            q_v = torch.zeros_like(k_v)
            value_token_count = int(self.cached_value_token_count or 0)
        else:
            # 1. Compute cosmos video QKV
            q_v, k_v, v_v = self.original_attn.compute_qkv(x, context, rope_emb=rope_emb)
            q_v, k_v = self._interleave_video_qk(q_v, k_v)
            value_token_count = int(self.current_value_token_count or 0)
            if self.cache_video_kv or self.current_cache_video_kv:
                cache_detach = True if self.cache_video_kv else bool(self.current_cache_video_kv_detach)
                self._store_video_kv_cache(
                    k_v=k_v,
                    v_v=v_v,
                    batch_size=q_v.shape[0],
                    device=q_v.device,
                    video_size=video_size,
                    detach=cache_detach,
                    value_token_count=value_token_count,
                )

        S_v = q_v.shape[1]

        has_latent = self.current_x_latent is not None
        has_action = self.current_x_action is not None

        if not has_latent and not has_action:
            # Cosmos-only: no mask needed (full attention)
            if value_token_count > 0 and self.value_token_mask_video_to_value:
                bridge_mask = self._build_bridge_mask(
                    S_v,
                    0,
                    0,
                    q_v.device,
                    value_token_count=value_token_count,
                    action_value_token_count=0,
                )
                result = self._bridge_sdpa(q_v, k_v, v_v, attn_mask=bridge_mask)
            else:
                result = self.original_attn.attn_op(q_v, k_v, v_v)
            out_v = self.original_attn.output_dropout(self.original_attn.output_proj(result))
            return out_v

        # 2. Shared attention: [video, latent?, action?] with custom mask
        parts_q = [q_v]
        if use_cached_video_kv:
            parts_k = [k_v]
            parts_v = [v_v]
        else:
            parts_k = [k_v.detach() if self.detach_video_kv else k_v]
            parts_v = [v_v.detach() if self.detach_video_kv else v_v]
        query_valid_parts = [S_v]
        S_l, S_a = 0, 0

        q_l = k_l = v_l = None
        q_a = k_a = v_a = None
        if has_latent:
            q_l, k_l, v_l = self.latent_bridge.get_branch_qkv(self.current_x_latent)
            S_l = q_l.shape[1]

        if has_action:
            q_a, k_a, v_a = self.action_bridge.get_branch_qkv(self.current_x_action)
            S_a = q_a.shape[1]

        q_l, k_l, q_a, k_a = self._apply_branch_rotary(
            q_l=q_l,
            k_l=k_l,
            q_a=q_a,
            k_a=k_a,
            rotary_payload=self.current_rotary_payload,
        )

        if has_latent:
            parts_q.append(q_l)
            parts_k.append(k_l)
            parts_v.append(v_l)
            query_valid_parts.append(self.current_latent_valid_mask if self.current_latent_valid_mask is not None else S_l)

        if has_action:
            parts_q.append(q_a)
            parts_k.append(k_a)
            parts_v.append(v_a)
            query_valid_parts.append(S_a)

        q = torch.cat(parts_q, dim=1)
        k = torch.cat(parts_k, dim=1)
        v = torch.cat(parts_v, dim=1)
        query_valid_mask = self._query_valid_mask(query_valid_parts, q.device, q.shape[0])

        # Build and apply custom bridge attention mask
        bridge_mask = self._build_bridge_mask(
            S_v,
            S_l,
            S_a,
            q.device,
            latent_valid_mask=self.current_latent_valid_mask,
            value_token_count=value_token_count,
            action_value_token_count=action_value_token_count if has_action else 0,
        )
        result = self._bridge_sdpa(q, k, v, attn_mask=bridge_mask, query_valid_mask=query_valid_mask)

        res_v = result[:, :S_v]
        offset = S_v
        res_l = result[:, offset:offset + S_l] if S_l > 0 else None
        offset += S_l
        res_a = result[:, offset:offset + S_a] if S_a > 0 else None

        # 3. Post-attention for each expert
        if has_latent:
            self.next_x_latent = self.latent_bridge.post_attention(
                self.current_x_latent,
                res_l,
                token_valid_mask=self.current_latent_valid_mask,
            )

        if has_action:
            self.next_x_action = self.action_bridge.post_attention(self.current_x_action, res_a)

        # 4. Video output (through cosmos output projection)
        out_v = self.original_attn.output_dropout(self.original_attn.output_proj(res_v))
        return out_v

    def forward_action_only(
        self,
        x_latent,
        x_action,
        latent_valid_mask=None,
        rotary_payload: Optional[BridgeRotaryPayload] = None,
        action_value_token_count: int = 0,
    ):
        """Action-only forward: uses cached video KV, skips cosmos DIT.

        Args:
            x_latent: [B, S_l, D_janus] latent tokens
            x_action: [B, S_a, D_janus] action tokens

        Returns:
            (next_x_latent, next_x_action)
        """
        has_cached_video = self.cached_k_v is not None and self.cached_v_v is not None
        if self.decosmos:
            has_cached_video = False
        elif not has_cached_video:
            raise RuntimeError("forward_action_only requires cached video KV unless decosmos is enabled.")
        S_v = self.cached_k_v.shape[1] if has_cached_video else 0
        value_token_count = int(self.cached_value_token_count or 0) if has_cached_video else 0

        # Shared attention: latent/action queries against cached video KV.
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

        # Q = [latent, action], KV = [cached_cosmos?, latent, action]
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

        # Post-attention
        next_x_latent = self.latent_bridge.post_attention(
            x_latent,
            res_l,
            token_valid_mask=latent_valid_mask,
        )

        next_x_action = self.action_bridge.post_attention(x_action, res_a)

        return next_x_latent, next_x_action

    def forward_latent_only(
        self,
        x_latent,
        latent_valid_mask=None,
        rotary_payload: Optional[BridgeRotaryPayload] = None,
    ):
        """Latent-only forward using cached video KV and bridge attention only.

        Args:
            x_latent: [B, S_l, D_janus] latent tokens (multimodal_embeds + generated latents)

        Returns:
            next_x_latent
        """
        has_cached_video = self.cached_k_v is not None and self.cached_v_v is not None
        if self.decosmos:
            has_cached_video = False
        elif not has_cached_video:
            raise RuntimeError("forward_latent_only requires cached video KV unless decosmos is enabled.")
        S_v = self.cached_k_v.shape[1] if has_cached_video else 0
        value_token_count = int(self.cached_value_token_count or 0) if has_cached_video else 0

        # Shared attention: latent Q against [cached_cosmos, latent] KV.
        q_l, k_l, v_l = self.latent_bridge.get_branch_qkv(x_latent)
        q_l, k_l, _, _ = self._apply_branch_rotary(
            q_l=q_l,
            k_l=k_l,
            q_a=None,
            k_a=None,
            rotary_payload=rotary_payload,
        )
        S_l = q_l.shape[1]

        q = q_l
        if has_cached_video:
            k = torch.cat([self.cached_k_v, k_l], dim=1)
            v = torch.cat([self.cached_v_v, v_l], dim=1)
        else:
            k = k_l
            v = v_l

        # Mask: latent -> cosmos: full, latent -> latent: causal.
        mask = self._build_latent_only_mask(
            S_v,
            S_l,
            q.device,
            latent_valid_mask=latent_valid_mask,
            value_token_count=value_token_count,
        )

        query_valid_mask = None if latent_valid_mask is None else latent_valid_mask
        if query_valid_mask is None:
            query_valid_mask = torch.ones((q.shape[0], S_l), device=q.device, dtype=torch.bool)
        result = self._bridge_sdpa(q, k, v, attn_mask=mask, query_valid_mask=query_valid_mask)

        # Post-attention
        next_x_latent = self.latent_bridge.post_attention(
            x_latent,
            result,
            token_valid_mask=latent_valid_mask,
        )

        return next_x_latent


class CosmosJanusMoT3Expert(nn.Module):
    """3-Expert MoT: Cosmos (video) + Janus Latent CoT + Janus Action.

    Loading from last0:
      1. Put last0 on PYTHONPATH so its custom transformers (with dual expert LlamaDecoderLayer) is used
      2. Load Janus model via AutoModelForCausalLM.from_pretrained(last0_ckpt_path, ...)
      3. The loaded model will have both latent and action expert components
      4. This class extracts them into the 3-expert MoT architecture
    """

    def __init__(self, cosmos_dit, cosmos_vae, janus_model, config):
        super().__init__()
        self.config = config
        self.dtype = torch.bfloat16
        self._video_frozen = False
        self.cosmos_self_only_bridge = bool(getattr(config, "cosmos_self_only_bridge", 0))
        self.train_embed_tokens = bool(getattr(config, "train_embed_tokens", 0))
        self.no_detach_latent_input = bool(getattr(config, "no_detach_latent_input", 0))
        self.decosmos = bool(getattr(config, "decosmos", 0))
        self.use_value_prediction = bool(getattr(config, "use_value_prediction", 0))
        self.use_action_value_prediction = bool(getattr(config, "use_action_value_prediction", 0))
        self.value_token_mask_nonvalue_to_value = bool(getattr(config, "value_token_mask_nonvalue_to_value", 0))
        self.value_token_mask_video_to_value = bool(
            getattr(config, "value_token_mask_video_to_value", 0) or self.value_token_mask_nonvalue_to_value
        )
        setattr(self.config, "use_action_value_prediction", int(self.use_action_value_prediction))
        setattr(self.config, "no_detach_latent_input", int(self.no_detach_latent_input))
        setattr(self.config, "value_token_mask_video_to_value", int(self.value_token_mask_video_to_value))
        setattr(self.config, "value_token_mask_nonvalue_to_value", int(self.value_token_mask_nonvalue_to_value))
        if self.no_detach_latent_input and not self.train_embed_tokens:
            raise ValueError("no_detach_latent_input=1 requires train_embed_tokens=1 so token embeddings keep gradients.")
        if self.use_value_prediction and self.decosmos:
            raise ValueError("use_value_prediction requires decosmos=0.")
        if self.use_value_prediction and int(getattr(cosmos_dit, "patch_temporal", 1)) != 1:
            raise ValueError("use_value_prediction requires cosmos_dit.patch_temporal == 1.")
        self.action_use_latent_prefix = bool(getattr(config, "action_use_latent_prefix", 0))
        self.state_latents_per_future = int(getattr(config, "state_latents_per_future", 0) or 0)
        self.state_encoding_mode = getattr(config, "state_encoding_mode", "token")
        if self.state_encoding_mode not in ("token", "mlp"):
            raise ValueError(
                f"state_encoding_mode must be 'token' or 'mlp', got {self.state_encoding_mode!r}."
            )
        self.janus_image_start_id = getattr(config, "janus_image_start_id", None)
        self.janus_image_end_id = getattr(config, "janus_image_end_id", None)
        self.latent_end_id = getattr(config, "latent_end_id", None)
        valid_token_vocab_size = int(getattr(config, "valid_token_vocab_size", 0) or 0)
        self.valid_token_vocab_size = valid_token_vocab_size if valid_token_vocab_size > 0 else None
        self.bridge_pos_scheme = normalize_bridge_pos_scheme(getattr(config, "bridge_pos_scheme", "mrope"))
        setattr(self.config, "bridge_pos_scheme", self.bridge_pos_scheme)
        self.use_latent_hidden_sim_loss = bool(getattr(config, "use_latent_hidden_sim_loss", 0))
        self.latent_hidden_sim_loss_mode = str(
            getattr(config, "latent_hidden_sim_loss_mode", "siglip")
        ).lower()
        if self.latent_hidden_sim_loss_mode not in ("siglip", "wan_vae"):
            raise ValueError(
                "latent_hidden_sim_loss_mode must be 'siglip' or 'wan_vae', "
                f"got {self.latent_hidden_sim_loss_mode!r}."
            )
        setattr(self.config, "latent_hidden_sim_loss_mode", self.latent_hidden_sim_loss_mode)
        self.wan21_vae_path = str(getattr(config, "wan21_vae_path", DEFAULT_WAN21_VAE_CKPT))
        setattr(self.config, "wan21_vae_path", self.wan21_vae_path)
        self.use_latent_hidden_wan_downsample_sim_loss = bool(
            getattr(config, "use_latent_hidden_wan_downsample_sim_loss", 0)
        )
        setattr(
            self.config,
            "use_latent_hidden_wan_downsample_sim_loss",
            int(self.use_latent_hidden_wan_downsample_sim_loss),
        )
        if self.use_latent_hidden_wan_downsample_sim_loss and not (
            self.use_latent_hidden_sim_loss and self.latent_hidden_sim_loss_mode == "wan_vae"
        ):
            raise ValueError(
                "use_latent_hidden_wan_downsample_sim_loss requires "
                "use_latent_hidden_sim_loss=1 and latent_hidden_sim_loss_mode='wan_vae'."
            )

        self.action_intermediate_size = getattr(config, 'action_intermediate_size', None)
        if self.action_intermediate_size is not None and self.action_intermediate_size <= 0:
            self.action_intermediate_size = None

        self.cosmos_dit = cosmos_dit
        self.cosmos_vae = cosmos_vae
        self.janus = janus_model
        self.num_cond_input_frames = max(1, int(getattr(config, "num_cond_input_frames", 1) or 1))
        self.num_cond_latent_frames = max(1, int(getattr(config, "num_cond_latent_frames", 1) or 1))
        setattr(self.config, "num_cond_input_frames", self.num_cond_input_frames)
        setattr(self.config, "num_cond_latent_frames", self.num_cond_latent_frames)

        self.janus_dim = self.janus.config.hidden_size
        self.janus_rotary_emb = getattr(self.janus.language_model.model, "rotary_emb", None)
        if self.bridge_pos_scheme == "llama1d" and self.janus_rotary_emb is None:
            raise ValueError("bridge_pos_scheme='llama1d' requires janus.language_model.model.rotary_emb.")

        self.latent_hidden_wan_upsampler = None
        if self.use_latent_hidden_sim_loss and self.latent_hidden_sim_loss_mode == "wan_vae":
            self.latent_hidden_wan_upsampler = StackedUpsample2d(
                channels=Wan21VAEEncoder.latent_channels,
                spatial_size=32,
                input_channels=self.janus_dim,
            ).to(self.dtype)
        self.latent_hidden_wan_downsampler = None
        if self.use_latent_hidden_wan_downsample_sim_loss:
            self.latent_hidden_wan_downsampler = StackedDownsample2d(
                channels=Wan21VAEEncoder.latent_channels,
                spatial_size=32,
                output_channels=self.janus_dim,
            ).to(self.dtype)
        # Keep the frozen Wan encoder out of Module registration so it is not
        # saved in MoT checkpoints. It is lazily loaded on the first Wan sim-loss
        # forward and moved to the current device there.
        self.__dict__["latent_hidden_wan_encoder"] = None

        self.state_mlp_embedder = ActionEmbedder(action_size=8, hidden_size=self.janus_dim)
        self._init_state_mlp_embedder()
        self.action_value_embedder = None
        self.action_value_final_layer = None
        if self.use_action_value_prediction:
            self.action_value_embedder = ActionEmbedder(action_size=1, hidden_size=self.janus_dim)
            self.action_value_final_layer = FinalLayer(hidden_size=self.janus_dim, out_channels=1)
            self._init_action_value_modules()

        self.train_action_self_causal_in_bridge = bool(getattr(config, "action_self_causal_in_bridge", 1))


        # Text projection (janus_dim 鈫?cosmos cross-attention dim)
        self.cosmos_crossattn_dim = self.cosmos_dit.blocks[0].cross_attn.context_dim
        if self.janus_dim != self.cosmos_crossattn_dim:
            self.text_proj = nn.Linear(self.janus_dim, self.cosmos_crossattn_dim, bias=True).to(self.dtype)
            nn.init.normal_(self.text_proj.weight, std=0.02)
            nn.init.zeros_(self.text_proj.bias)
        else:
            self.text_proj = nn.Identity()

        # Build 3-expert MoT wrappers
        self.mot_attention_wrappers = nn.ModuleList()
        self.bridge_rotary_encoder = None
        janus_layers = self.janus.language_model.model.layers

        for i, cosmos_block in enumerate(self.cosmos_dit.blocks):
            if i < len(janus_layers):
                janus_layer = janus_layers[i]

                actual_block = cosmos_block
                if hasattr(cosmos_block, '_checkpoint_wrapped_module'):
                    actual_block = cosmos_block._checkpoint_wrapped_module
                elif hasattr(cosmos_block, 'module'):
                    actual_block = cosmos_block.module

                block_head_dim = int(actual_block.self_attn.head_dim)
                if self.bridge_rotary_encoder is None:
                    if is_multimodal_bridge_pos_scheme(self.bridge_pos_scheme):
                        self.bridge_rotary_encoder = BridgeA1MRoPE(
                            head_dim=block_head_dim,
                            interleave_thw=self.bridge_pos_scheme == "mrope_interleave",
                        )
                    else:
                        self.bridge_rotary_encoder = BridgeLlama1DRoPE(
                            head_dim=block_head_dim,
                            janus_rotary_emb=self.janus_rotary_emb,
                        )
                elif self.bridge_rotary_encoder.head_dim != block_head_dim:
                    raise ValueError(
                        "All MoT bridge layers must share one head_dim for cached rotary payloads, "
                        f"but saw {block_head_dim} after locking {self.bridge_rotary_encoder.head_dim}."
                    )

                latent_bridge = LatentNativeAttentionAdapter(
                    janus_layer=janus_layer,
                    cosmos_num_heads=actual_block.self_attn.n_heads,
                    cosmos_head_dim=block_head_dim,
                    layer_idx=i,
                ).to(self.dtype)

                action_bridge = ActionNativeAttentionAdapter(
                    janus_layer=janus_layer,
                    cosmos_num_heads=actual_block.self_attn.n_heads,
                    cosmos_head_dim=block_head_dim,
                    action_intermediate_size=self.action_intermediate_size,
                    layer_idx=i,
                ).to(self.dtype)

                mot_attn = MoTAttentionWrapper3(
                    actual_block.self_attn,
                    latent_bridge,
                    action_bridge,
                    cosmos_self_only_bridge=self.cosmos_self_only_bridge,
                    decosmos=self.decosmos,
                    bridge_action_self_causal_override=self.train_action_self_causal_in_bridge,
                    interleave_video_qk=self.bridge_pos_scheme == "mrope_interleave",
                    value_token_mask_video_to_value=self.value_token_mask_video_to_value,
                    value_token_mask_nonvalue_to_value=self.value_token_mask_nonvalue_to_value,
                )
                actual_block.self_attn = mot_attn

                original_forward = actual_block.forward

                def make_new_forward(orig_fwd):
                    def new_forward(self_block, x_B_T_H_W_D, emb_B_T_D, crossattn_emb,
                                    x_latent=None, x_action=None, **kwargs):
                        x_latent_valid_mask = kwargs.pop("x_latent_valid_mask", None)
                        x_rotary_payload = kwargs.pop("x_rotary_payload", None)
                        mot_forward_index = kwargs.pop("mot_forward_index", 1)
                        mot_cache_video_kv = kwargs.pop("mot_cache_video_kv", False)
                        mot_cache_video_kv_detach = kwargs.pop("mot_cache_video_kv_detach", True)
                        mot_value_token_count = kwargs.pop("mot_value_token_count", 0)
                        mot_action_value_token_count = kwargs.pop("mot_action_value_token_count", 0)
                        self_block.self_attn.current_x_latent = x_latent
                        self_block.self_attn.current_x_action = x_action
                        self_block.self_attn.current_latent_valid_mask = x_latent_valid_mask
                        self_block.self_attn.current_rotary_payload = x_rotary_payload
                        self_block.self_attn.current_mot_forward_index = mot_forward_index
                        self_block.self_attn.current_cache_video_kv = mot_cache_video_kv
                        self_block.self_attn.current_cache_video_kv_detach = mot_cache_video_kv_detach
                        self_block.self_attn.current_value_token_count = int(mot_value_token_count or 0)
                        self_block.self_attn.current_action_value_token_count = int(mot_action_value_token_count or 0)
                        try:
                            out_video = orig_fwd(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, **kwargs)
                            out_latent = self_block.self_attn.next_x_latent
                            out_action = self_block.self_attn.next_x_action
                            return out_video, out_latent, out_action
                        finally:
                            self_block.self_attn.current_x_latent = None
                            self_block.self_attn.current_x_action = None
                            self_block.self_attn.current_latent_valid_mask = None
                            self_block.self_attn.current_rotary_payload = None
                            self_block.self_attn.current_mot_forward_index = 1
                            self_block.self_attn.current_cache_video_kv = False
                            self_block.self_attn.current_cache_video_kv_detach = True
                            self_block.self_attn.current_value_token_count = 0
                            self_block.self_attn.current_action_value_token_count = 0
                            self_block.self_attn.next_x_latent = None
                            self_block.self_attn.next_x_action = None
                    return new_forward

                actual_block.forward = types.MethodType(make_new_forward(original_forward), actual_block)
                self.mot_attention_wrappers.append(mot_attn)
            else:
                break

        if self.bridge_rotary_encoder is None:
            raise ValueError("Failed to initialize shared bridge rotary encoder because no MoT wrapper was created.")

        self._freeze_unused_janus_attention(janus_layers)

    @staticmethod
    def _freeze_unused_janus_attention(janus_layers):
        """Native Janus attentions are part of the shared attention path; keep them trainable."""
        return

    def _sample_cosmos_train_sigma(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample Cosmos 2B rectified-flow train time and shifted sigma."""
        # Match native Cosmos 2B reason RF training: u ~ logitnormal, then shift=5.
        shift = 5.0
        u = torch.sigmoid(torch.randn((batch_size,), device=device, dtype=torch.float32))
        sigma = (shift * u / (1.0 + (shift - 1.0) * u)) 
        return u.to(dtype=self.dtype), sigma.to(dtype=self.dtype)

    def _get_cached_bridge_video_grid_thw(self, batch_size: int, device: torch.device) -> torch.Tensor:
        for wrapper in self.mot_attention_wrappers:
            if wrapper.cached_video_grid_thw is not None:
                cached = wrapper.cached_video_grid_thw.to(device=device, dtype=torch.long)
                if cached.ndim == 1:
                    cached = cached.unsqueeze(0).expand(batch_size, -1)
                elif cached.shape[0] == 1 and batch_size != 1:
                    cached = cached.expand(batch_size, -1)
                return cached
        raise ValueError("No cached video grid THW is available for bridge rotary preparation.")

    def _build_bridge_rotary_payload(
        self,
        batch_info: BridgeMRoPEBatchInfo,
        latent_seq_len: int,
        action_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BridgeRotaryPayload:
        batch_size = None
        for tensor in (
            batch_info.latent_valid_mask,
            batch_info.image_grid_thw,
            batch_info.latent_image_token_mask,
            batch_info.action_image_token_mask,
            batch_info.video_grid_thw,
        ):
            if tensor is not None:
                batch_size = int(tensor.shape[0])
                break
        if batch_size is None:
            raise ValueError("Unable to infer batch_size for bridge rotary payload preparation.")
        return self.bridge_rotary_encoder.prepare_batch_rotary(
            batch_info=batch_info,
            batch_size=batch_size,
            latent_seq_len=latent_seq_len,
            action_seq_len=action_seq_len,
            device=device,
            dtype=dtype,
        )

    def _bridge_parameter_ids(self):
        """Return parameter ids for trainable latent/action bridge modules inside Cosmos blocks."""
        bridge_param_ids = set()
        for wrapper in self.mot_attention_wrappers:
            for bridge in (wrapper.latent_bridge, wrapper.action_bridge):
                bridge_param_ids.update(id(param) for param in bridge.parameters())
        return bridge_param_ids

    def _set_video_backbone_requires_grad(self, requires_grad: bool):
        """Set requires_grad for Cosmos video params while leaving bridge params unchanged."""
        bridge_param_ids = self._bridge_parameter_ids()
        for param in self.cosmos_dit.parameters():
            if id(param) in bridge_param_ids:
                continue
            param.requires_grad = requires_grad

    def _set_text_proj_requires_grad(self, requires_grad: bool):
        """Set requires_grad for text_proj when it has trainable parameters."""
        for param in self.text_proj.parameters():
            param.requires_grad = requires_grad

    def _prepare_cosmos_crossattn_emb(
        self,
        multimodal_embeds: torch.Tensor,
        cosmos_text_embeddings: Optional[torch.Tensor] = None,
        *,
        no_grad_text_proj: bool = False,
    ) -> torch.Tensor:
        """Resolve Cosmos cross-attention context from Janus or cached native Qwen text."""
        if cosmos_text_embeddings is None:
            if no_grad_text_proj:
                with torch.no_grad():
                    return self.text_proj(multimodal_embeds)
            return self.text_proj(multimodal_embeds)

        if cosmos_text_embeddings.ndim != 3:
            raise ValueError(
                "cosmos_text_embeddings must have shape [B, 512, D], "
                f"got {tuple(cosmos_text_embeddings.shape)}."
            )
        batch_size = int(multimodal_embeds.shape[0])
        seq_len = int(cosmos_text_embeddings.shape[1])
        cache_dim = int(cosmos_text_embeddings.shape[-1])
        projected_dim = int(self.cosmos_crossattn_dim)
        raw_dim = int(getattr(self.cosmos_dit, "crossattn_proj_in_channels", 100352))

        if int(cosmos_text_embeddings.shape[0]) != batch_size or seq_len != 512:
            raise ValueError(
                "cosmos_text_embeddings must have shape "
                f"[{batch_size}, 512, {projected_dim}] or [{batch_size}, 512, {raw_dim}], "
                f"got {tuple(cosmos_text_embeddings.shape)}."
            )

        cosmos_text_embeddings = cosmos_text_embeddings.to(
            device=multimodal_embeds.device,
            dtype=self.dtype,
        )
        if cache_dim == projected_dim:
            if no_grad_text_proj:
                cosmos_text_embeddings = cosmos_text_embeddings.detach()
            return cosmos_text_embeddings

        if cache_dim == raw_dim:
            if not bool(getattr(self.cosmos_dit, "use_crossattn_projection", False)):
                raise ValueError(
                    "Received raw Cosmos text embeddings with last dimension "
                    f"{raw_dim}, but cosmos_dit.use_crossattn_projection is false."
                )
            if not hasattr(self.cosmos_dit, "crossattn_proj"):
                raise AttributeError(
                    "Received raw Cosmos text embeddings with last dimension "
                    f"{raw_dim}, but cosmos_dit has no crossattn_proj module."
                )
            projected = self.cosmos_dit.crossattn_proj(cosmos_text_embeddings)
            if int(projected.shape[-1]) != projected_dim:
                raise ValueError(
                    "cosmos_dit.crossattn_proj produced last dimension "
                    f"{int(projected.shape[-1])}, expected {projected_dim}."
                )
            return projected.to(dtype=self.dtype)

        if no_grad_text_proj:
            cosmos_text_embeddings = cosmos_text_embeddings.detach()
        raise ValueError(
            "Unsupported cached Cosmos text embedding last dimension "
            f"{cache_dim}; expected projected dim {projected_dim} or raw dim {raw_dim}."
        )


    def freeze_video_backbone(self):
        """Freeze the video-side conditioning path so cached video KV stays fixed."""
        self._video_frozen = True
        # Bridge modules are registered under cosmos_dit wrappers, but should keep training after video freeze.
        self._set_video_backbone_requires_grad(False)
        self._set_text_proj_requires_grad(False)
        if hasattr(self.cosmos_vae, 'parameters'):
            for param in self.cosmos_vae.parameters():
                param.requires_grad = False
        for wrapper in self.mot_attention_wrappers:
            wrapper.detach_video_kv = True

    def unfreeze_video_backbone(self):
        self._video_frozen = False
        self._set_video_backbone_requires_grad(True)
        self._set_text_proj_requires_grad(True)
        for wrapper in self.mot_attention_wrappers:
            wrapper.detach_video_kv = False

    def set_cosmos_inference_runtime(
        self,
        sample_scheduler,
        shift=1,
        use_kerras_sigma_at_inference=False,
    ):
        """Attach inference-only Cosmos sampling state to this model instance."""
        if sample_scheduler is None:
            raise ValueError("sample_scheduler must not be None.")

        self._cosmos_inference_sample_scheduler = sample_scheduler
        self._cosmos_inference_shift = shift
        self._cosmos_inference_use_kerras_sigma_at_inference = bool(use_kerras_sigma_at_inference)
        return self

    def set_cosmos_inference_runtime_from_wrapper(self, cosmos_wrapper):
        """Extract the runtime sampling state from a loaded Cosmos wrapper."""
        sample_scheduler = getattr(cosmos_wrapper, "sample_scheduler", None)
        if sample_scheduler is None:
            raise AttributeError("cosmos_wrapper has no sample_scheduler; cannot attach inference runtime.")

        wrapper_config = getattr(cosmos_wrapper, "config", None)
        shift = getattr(wrapper_config, "shift", 1)
        use_kerras_sigma = bool(getattr(wrapper_config, "use_kerras_sigma_at_inference", False))
        return self.set_cosmos_inference_runtime(
            sample_scheduler=sample_scheduler,
            shift=shift,
            use_kerras_sigma_at_inference=use_kerras_sigma,
        )

    def _require_cosmos_inference_runtime(self):
        sample_scheduler = getattr(self, "_cosmos_inference_sample_scheduler", None)
        if sample_scheduler is None:
            raise RuntimeError(
                "Cosmos inference runtime is not attached. Call "
                "set_cosmos_inference_runtime(...) before forward_flow_joint_inference()."
            )

        shift = getattr(self, "_cosmos_inference_shift", 1)
        use_kerras_sigma = bool(getattr(self, "_cosmos_inference_use_kerras_sigma_at_inference", False))
        num_train_timesteps = float(getattr(sample_scheduler.config, "num_train_timesteps", 1000))
        return sample_scheduler, shift, use_kerras_sigma, num_train_timesteps

    def _set_bridge_action_self_causal_override(self, causal: Optional[bool]):
        for wrapper in self.mot_attention_wrappers:
            wrapper.bridge_action_self_causal_override = None if causal is None else bool(causal)

    def _clear_cached_video_kv(self):
        for wrapper in self.mot_attention_wrappers:
            wrapper.cache_video_kv = False
            wrapper.cached_k_v = None
            wrapper.cached_v_v = None
            wrapper.cached_video_grid_thw = None
            wrapper.cached_value_token_count = 0
            wrapper.current_rotary_payload = None
            wrapper.current_mot_forward_index = 1
            wrapper.current_cache_video_kv = False
            wrapper.current_cache_video_kv_detach = True
            wrapper.current_value_token_count = 0
            wrapper.current_action_value_token_count = 0

    def _latent_num_frames_for_pixels(self, pixel_frames: int) -> int:
        pixel_frames = int(pixel_frames)
        if pixel_frames < 1:
            raise ValueError(f"pixel_frames must be positive, got {pixel_frames}.")

        latent_num_frames = getattr(self.cosmos_vae, "get_latent_num_frames", None)
        if callable(latent_num_frames):
            return int(latent_num_frames(pixel_frames))

        temporal_compression_factor = getattr(self.cosmos_vae, "temporal_compression_factor", None)
        if callable(temporal_compression_factor):
            temporal_compression_factor = temporal_compression_factor()
        if temporal_compression_factor is None:
            temporal_compression_factor = 4
        temporal_compression_factor = int(temporal_compression_factor)
        if temporal_compression_factor < 1:
            raise ValueError(
                "temporal_compression_factor must be positive, "
                f"got {temporal_compression_factor}."
            )
        return 1 + (pixel_frames - 1) // temporal_compression_factor

    def _num_real_condition_latents(self, real_video_latent_frames: int) -> int:
        real_video_latent_frames = int(real_video_latent_frames)
        if real_video_latent_frames < 1:
            raise ValueError(
                "real_video_latent_frames must be positive, "
                f"got {real_video_latent_frames}."
            )
        return min(self.num_cond_latent_frames, real_video_latent_frames)

    def _value_token_count_from_x_video(self, x_video: torch.Tensor, has_value_token: bool) -> int:
        if not has_value_token:
            return 0
        if not (self.value_token_mask_video_to_value or self.value_token_mask_nonvalue_to_value):
            return 0
        if x_video.ndim != 5:
            raise ValueError(f"x_video must have shape [B, T, H, W, D], got {tuple(x_video.shape)}.")
        return int(x_video.shape[2]) * int(x_video.shape[3])

    def _infer_janus_image_grid_thw(
        self,
        batch_size: int,
        device: torch.device,
        janus_images_emb_mask: Optional[torch.Tensor] = None,
        action_image_prefix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        grid_h = None
        grid_w = None

        vision_model = getattr(self.janus, "vision_model", None)
        vision_tower = getattr(vision_model, "vision_tower", vision_model)
        patch_embed = getattr(vision_tower, "patch_embed", None)
        grid_size = getattr(patch_embed, "grid_size", None)
        if grid_size is not None:
            if isinstance(grid_size, (tuple, list)):
                if len(grid_size) == 1:
                    grid_h = grid_w = int(grid_size[0])
                elif len(grid_size) >= 2:
                    grid_h = int(grid_size[-2])
                    grid_w = int(grid_size[-1])
            else:
                grid_h = grid_w = int(grid_size)

        if grid_h is None or grid_w is None:
            num_image_tokens = None
            if janus_images_emb_mask is not None:
                num_image_tokens = int(janus_images_emb_mask.shape[-1])
            elif action_image_prefix is not None:
                prefix_len = int(action_image_prefix.shape[1])
                if prefix_len >= 2:
                    num_image_tokens = prefix_len - 2
            if num_image_tokens is None or num_image_tokens <= 0:
                num_image_tokens = 576

            grid_side = int(math.isqrt(num_image_tokens))
            grid_h = grid_w = grid_side

        return torch.tensor([1, grid_h, grid_w], device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    def _build_bridge_rotary_batch_info(
        self,
        x_latent: Optional[torch.Tensor],
        x_action: Optional[torch.Tensor],
        latent_valid_mask: Optional[torch.Tensor] = None,
        latent_left_pad_lens: Optional[torch.Tensor] = None,
        janus_images_seq_mask: Optional[torch.Tensor] = None,
        janus_images_emb_mask: Optional[torch.Tensor] = None,
        action_image_prefix: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> BridgeMRoPEBatchInfo:
        source = x_latent if x_latent is not None else x_action
        batch_size = source.shape[0]
        device = source.device
        use_mrope = is_multimodal_bridge_pos_scheme(self.bridge_pos_scheme)
        image_grid_thw = None
        if use_mrope:
            image_grid_thw = self._infer_janus_image_grid_thw(
                batch_size=batch_size,
                device=device,
                janus_images_emb_mask=janus_images_emb_mask,
                action_image_prefix=action_image_prefix,
            )

        latent_image_token_mask = None
        if use_mrope and x_latent is not None:
            latent_image_token_mask = torch.zeros((batch_size, x_latent.shape[1]), device=device, dtype=torch.bool)
            if janus_images_seq_mask is not None:
                seq_mask = janus_images_seq_mask.to(device=device, dtype=torch.bool)
                seq_width = min(seq_mask.shape[1], x_latent.shape[1])
                latent_image_token_mask[:, :seq_width] = seq_mask[:, :seq_width]

        action_image_token_mask = None
        if use_mrope and x_action is not None:
            action_image_token_mask = torch.zeros((batch_size, x_action.shape[1]), device=device, dtype=torch.bool)
            if action_image_prefix is not None:
                prefix_len = int(action_image_prefix.shape[1])
                image_token_count = max(prefix_len - 2, 0)
                if image_token_count > 0:
                    action_image_token_start = 2
                    action_image_token_mask[
                        :, action_image_token_start : action_image_token_start + image_token_count
                    ] = True

        return BridgeMRoPEBatchInfo(
            image_grid_thw=image_grid_thw,
            latent_image_token_mask=latent_image_token_mask,
            action_image_token_mask=action_image_token_mask,
            latent_valid_mask=None
            if latent_valid_mask is None
            else latent_valid_mask.to(device=device, dtype=torch.bool),
            latent_left_pad_lens=None
            if latent_left_pad_lens is None
            else latent_left_pad_lens.to(device=device, dtype=torch.long),
            video_grid_thw=None
            if video_grid_thw is None or not use_mrope
            else video_grid_thw.to(device=device, dtype=torch.long),
        )

    def _build_latent_sequence(
        self,
        multimodal_embeds: torch.Tensor,
        latent_tokens: torch.Tensor,
        janus_left_pad_lens: Optional[torch.Tensor] = None,
        janus_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Keep Janus left padding in place and append latent tokens after the context."""
        if janus_attention_mask is None and janus_left_pad_lens is None:
            return torch.cat([multimodal_embeds, latent_tokens], dim=1), None

        B, S_context, D = multimodal_embeds.shape
        N_latent = latent_tokens.shape[1]
        device = multimodal_embeds.device

        if janus_attention_mask is not None:
            context_valid = janus_attention_mask.to(device=device, dtype=torch.bool)
            if context_valid.shape != (B, S_context):
                raise ValueError(
                    f"janus_attention_mask shape {tuple(context_valid.shape)} does not match "
                    f"multimodal context shape {(B, S_context)}."
                )
        else:
            janus_left_pad_lens = janus_left_pad_lens.to(device=device, dtype=torch.long)
            context_pos = torch.arange(S_context, device=device).unsqueeze(0)
            context_valid = context_pos >= janus_left_pad_lens.unsqueeze(1)

        masked_context = multimodal_embeds * context_valid.unsqueeze(-1).to(multimodal_embeds.dtype)
        x_latent = torch.cat([masked_context, latent_tokens], dim=1)
        if N_latent > 0:
            latent_valid = torch.ones((B, N_latent), device=device, dtype=torch.bool)
            latent_valid_mask = torch.cat([context_valid, latent_valid], dim=1)
        else:
            latent_valid_mask = context_valid
        return x_latent, latent_valid_mask

    def _embed_janus_token_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        safe_input_ids = input_ids.clone()
        safe_input_ids[safe_input_ids < 0] = 0
        return self.janus.language_model.model.embed_tokens(safe_input_ids)

    def _resolve_latent_token_count(self, num_latent_tokens: Optional[int] = None) -> int:
        if num_latent_tokens is None:
            num_latent_tokens = int(getattr(self.config, "total_latent_tokens", 1) or 1)
        num_latent_tokens = int(num_latent_tokens)
        if num_latent_tokens not in (1, 2):
            raise ValueError(f"Token-latent flows only support 1 or 2 tokens in beta, got {num_latent_tokens}.")
        return num_latent_tokens

    def _normalize_latent_gt_token_ids(self, latent_gt_token_ids: torch.Tensor) -> torch.Tensor:
        if latent_gt_token_ids.ndim == 1:
            latent_gt_token_ids = latent_gt_token_ids.unsqueeze(1)
        elif latent_gt_token_ids.ndim != 2:
            raise ValueError(
                f"latent_gt_token_ids must have shape [B] or [B, N], got {tuple(latent_gt_token_ids.shape)}."
            )
        return latent_gt_token_ids

    def _require_latent_end_id(self) -> int:
        if self.latent_end_id is None:
            raise ValueError("latent_end_id must be set on the model config to build action prefixes.")

        latent_end_id = int(self.latent_end_id)
        embed_rows = int(self.janus.language_model.model.embed_tokens.weight.shape[0])
        if latent_end_id < 0 or latent_end_id >= embed_rows:
            raise ValueError(
                f"Invalid latent_end_id {latent_end_id}; embed_tokens has {embed_rows} rows."
            )
        return latent_end_id

    def _init_state_mlp_embedder(self):
        nn.init.normal_(self.state_mlp_embedder.mlp.fc1.weight, std=0.02)
        nn.init.normal_(self.state_mlp_embedder.mlp.fc2.weight, std=0.02)
        nn.init.constant_(self.state_mlp_embedder.mlp.fc1.bias, 0)
        nn.init.constant_(self.state_mlp_embedder.mlp.fc2.bias, 0)

    def _init_action_value_modules(self):
        if self.action_value_embedder is None or self.action_value_final_layer is None:
            return
        nn.init.normal_(self.action_value_embedder.mlp.fc1.weight, std=0.02)
        nn.init.normal_(self.action_value_embedder.mlp.fc2.weight, std=0.02)
        nn.init.constant_(self.action_value_embedder.mlp.fc1.bias, 0)
        nn.init.constant_(self.action_value_embedder.mlp.fc2.bias, 0)
        nn.init.normal_(self.action_value_final_layer.mlp.fc1.weight, std=0.02)
        nn.init.constant_(self.action_value_final_layer.mlp.fc1.bias, 0)
        nn.init.constant_(self.action_value_final_layer.mlp.fc2.weight, 0)
        nn.init.constant_(self.action_value_final_layer.mlp.fc2.bias, 0)

    def encode_state_values(self, state_values: torch.Tensor) -> torch.Tensor:
        if state_values.ndim not in (2, 3):
            raise ValueError(
                f"state_values must have shape [B, 8] or [B, N, 8], got {tuple(state_values.shape)}."
            )
        if state_values.shape[-1] != 8:
            raise ValueError(f"state_values last dimension must be 8, got {state_values.shape[-1]}.")

        values = state_values.to(device=self.state_mlp_embedder.mlp.fc1.weight.device, dtype=self.dtype)
        if values.ndim == 2:
            return self.state_mlp_embedder(values).unsqueeze(1)

        B, N = values.shape[:2]
        flat = values.reshape(B * N, values.shape[-1])
        embeds = self.state_mlp_embedder(flat).reshape(B, N, 1, self.janus_dim)
        return embeds

    def _janus_vision_dtype(self) -> torch.dtype:
        try:
            return next(self.janus.vision_model.parameters()).dtype
        except StopIteration:
            return self.dtype

    def _encode_janus_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError(
                f"Janus pixel values must have shape [B, N, C, H, W], got {tuple(pixel_values.shape)}."
        )

        B, N = pixel_values.shape[:2]
        flat = pixel_values.reshape(B * N, *pixel_values.shape[2:]).to(dtype=self._janus_vision_dtype())
        image_embeds = self.janus.aligner(self.janus.vision_model(flat))
        return image_embeds.reshape(B, N, image_embeds.shape[1], image_embeds.shape[2])

    def _require_janus_image_boundary_ids(self) -> Tuple[int, int]:
        if self.janus_image_start_id is None or self.janus_image_end_id is None:
            raise ValueError(
                "janus_image_start_id and janus_image_end_id must be set on the model config."
            )

        start_id = int(self.janus_image_start_id)
        end_id = int(self.janus_image_end_id)
        if start_id < 0 or end_id < 0:
            raise ValueError(
                f"Invalid Janus image boundary ids: start={start_id}, end={end_id}."
            )
        return start_id, end_id

    def prepare_multimodal_embeds_with_state_action(
        self,
        janus_input_ids: torch.Tensor,
        now_state: Optional[torch.Tensor],
        janus_pixel_values: torch.Tensor,
        janus_action_pixel_values: torch.Tensor,
        janus_images_seq_mask: torch.Tensor,
        janus_state_seq_mask: Optional[torch.Tensor],
        janus_images_emb_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Embed text placeholders, main-view image tokens, current state, and wrist image prefix."""
        B = janus_input_ids.shape[0]
        device = janus_input_ids.device

        multimodal_embeds = self._embed_janus_token_ids(janus_input_ids)

        main_image_embeds = self._encode_janus_pixel_values(janus_pixel_values)
        _, n_images, n_image_tokens, D = main_image_embeds.shape
        flat_main_image_embeds = main_image_embeds.reshape(B, n_images * n_image_tokens, D)

        image_seq_mask = janus_images_seq_mask.to(device=device, dtype=torch.bool)
        image_emb_mask = janus_images_emb_mask.to(device=device, dtype=torch.bool).reshape(B, -1)
        if int(image_seq_mask.sum().item()) != int(image_emb_mask.sum().item()):
            raise ValueError(
                "janus_images_seq_mask token count does not match janus_images_emb_mask token count."
            )
        if flat_main_image_embeds.shape[:2] != image_emb_mask.shape:
            raise ValueError(
                f"Main image embeds shape {tuple(flat_main_image_embeds.shape[:2])} does not match "
                f"image emb mask shape {tuple(image_emb_mask.shape)}."
            )
        multimodal_embeds[image_seq_mask] = flat_main_image_embeds[image_emb_mask]

        state_embeds = None
        state_seq_mask = None
        if janus_state_seq_mask is not None:
            state_seq_mask = janus_state_seq_mask.to(device=device, dtype=torch.bool)

        if now_state is not None:
            if state_seq_mask is None:
                raise ValueError("now_state was provided but janus_state_seq_mask is None.")
            if now_state.shape[0] != B:
                raise ValueError(
                    f"now_state batch size {now_state.shape[0]} does not match janus_input_ids batch size {B}."
                )

            if self.state_encoding_mode == "mlp":
                state_embeds = self.encode_state_values(now_state)
            else:
                state_embeds = self._embed_janus_token_ids(now_state)
            state_token_count = state_embeds.shape[1]
            state_counts = state_seq_mask.sum(dim=1)
            if not torch.all(state_counts == state_token_count):
                raise ValueError(
                    "Each row of janus_state_seq_mask must contain exactly "
                    f"{state_token_count} state placeholder tokens; got {state_counts.tolist()}."
                )
            multimodal_embeds[state_seq_mask] = state_embeds.reshape(B * state_token_count, D)
        elif state_seq_mask is not None and bool(state_seq_mask.any().item()):
            raise ValueError("janus_state_seq_mask contains placeholders but now_state is None.")

        if janus_action_pixel_values.ndim != 5 or janus_action_pixel_values.shape[1] != 1:
            raise ValueError(
                "janus_action_pixel_values must have shape [B, 1, C, H, W], "
                f"got {tuple(janus_action_pixel_values.shape)}."
            )
        action_image_embeds = self._encode_janus_pixel_values(janus_action_pixel_values).squeeze(1)
        start_id, end_id = self._require_janus_image_boundary_ids()
        boundary_ids = torch.tensor([[start_id, end_id]], device=device, dtype=torch.long)
        boundary_embeds = self._embed_janus_token_ids(boundary_ids)
        image_start_embed = boundary_embeds[:, 0:1, :].expand(B, -1, -1)
        image_end_embed = boundary_embeds[:, 1:2, :].expand(B, -1, -1)
        action_image_prefix = torch.cat(
            [image_start_embed, action_image_embeds, image_end_embed],
            dim=1,
        )

        return multimodal_embeds, state_embeds, action_image_prefix

    def _extract_action_image_prefix(
        self,
        multimodal_embeds: torch.Tensor,
        janus_images_seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract `[image_start, image_tokens..., image_end]` from multimodal embeddings."""
        mask = janus_images_seq_mask.to(device=multimodal_embeds.device, dtype=torch.bool)
        prefixes = []

        for batch_idx in range(multimodal_embeds.shape[0]):
            true_positions = torch.nonzero(mask[batch_idx], as_tuple=False).flatten()
            first_true = int(true_positions[0].item())
            last_true = int(true_positions[-1].item())
            prefix_start = first_true - 1
            prefix_end = last_true + 1
            prefix = multimodal_embeds[batch_idx, prefix_start:prefix_end + 1, :]
            prefixes.append(prefix)

        return torch.stack(prefixes, dim=0)

    def _extract_action_state_prefix(
        self,
        multimodal_embeds: torch.Tensor,
        janus_state_seq_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Average-pool current state token embeddings into one action prefix token."""
        if janus_state_seq_mask is None:
            raise ValueError(
                "state_latents_per_future is enabled, but janus_state_seq_mask was not provided."
            )

        mask = janus_state_seq_mask.to(device=multimodal_embeds.device, dtype=torch.bool)
        if mask.shape[:2] != multimodal_embeds.shape[:2]:
            raise ValueError(
                f"janus_state_seq_mask shape {tuple(mask.shape)} does not match "
                f"multimodal_embeds shape prefix {tuple(multimodal_embeds.shape[:2])}."
            )

        prefixes = []
        for batch_idx in range(multimodal_embeds.shape[0]):
            true_positions = torch.nonzero(mask[batch_idx], as_tuple=False).flatten()
            if true_positions.numel() == 0:
                raise ValueError(
                    f"janus_state_seq_mask has no current-state tokens for batch item {batch_idx}."
                )
            prefix = multimodal_embeds[batch_idx, true_positions, :].mean(dim=0, keepdim=True)
            prefixes.append(prefix)

        return torch.stack(prefixes, dim=0)

    def _build_action_sequence(
        self,
        action_latent: torch.Tensor,
        timestep_act: torch.Tensor,
        action_value_latent: Optional[torch.Tensor] = None,
        action_image_prefix: Optional[torch.Tensor] = None,
        action_state_prefix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build action tokens as `[latent_end?, image_prefix, state_prefix, time_token, action_chunk, value?]`."""
        time_tokens = self.janus.t_embedder(timestep_act).unsqueeze(1)
        action_tokens = self.janus.x_embedder(action_latent)
        action_value_tokens = None
        if action_value_latent is not None:
            if not self.use_action_value_prediction:
                raise ValueError("action_value_latent was provided but use_action_value_prediction=0.")
            if self.action_value_embedder is None:
                raise RuntimeError("action_value_embedder is not initialized.")
            action_value_tokens = self.action_value_embedder(
                action_value_latent.to(device=action_tokens.device, dtype=action_tokens.dtype)
            )

        pieces = []
        has_action_prefix = action_image_prefix is not None or action_state_prefix is not None
        if has_action_prefix:
            latent_end_ids = torch.full(
                (action_tokens.shape[0], 1),
                self._require_latent_end_id(),
                device=action_tokens.device,
                dtype=torch.long,
            )
            if self.train_embed_tokens:
                latent_end_embed = self._embed_janus_token_ids(latent_end_ids)
            else:
                with torch.no_grad():
                    latent_end_embed = self._embed_janus_token_ids(latent_end_ids)
            pieces.append(latent_end_embed.to(
                device=action_tokens.device,
                dtype=action_tokens.dtype,
            ))

        if action_image_prefix is not None:
            pieces.append(action_image_prefix.to(
                device=action_tokens.device,
                dtype=action_tokens.dtype,
            ))
        if action_state_prefix is not None:
            pieces.append(action_state_prefix.to(
                device=action_tokens.device,
                dtype=action_tokens.dtype,
            ))

        pieces.extend([time_tokens, action_tokens])
        if action_value_tokens is not None:
            pieces.append(action_value_tokens)
        return torch.cat(pieces, dim=1)

    def _decode_value_prediction(self, value_logits_volume: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(value_logits_volume.to(torch.float32)).mean(dim=(1, 2, 3, 4))

    def _build_value_logit_latent(
        self,
        value_targets: torch.Tensor,
        reference_latent: torch.Tensor,
    ) -> torch.Tensor:
        B, C, _, H, W = reference_latent.shape
        value_targets = value_targets.to(device=reference_latent.device, dtype=torch.float32).view(B)
        value_logits = torch.logit(value_targets.clamp(1e-6, 1.0 - 1e-6)).to(dtype=reference_latent.dtype)
        return value_logits.view(B, 1, 1, 1, 1).expand(B, C, 1, H, W)

    def _get_latent_hidden_wan_encoder(self, device: torch.device) -> Wan21VAEEncoder:
        encoder = self.__dict__.get("latent_hidden_wan_encoder", None)
        if encoder is None:
            encoder = Wan21VAEEncoder(
                vae_pth=self.wan21_vae_path,
                dtype=torch.float32,
                device=device,
                freeze=True,
                normalize_latents=True,
            )
            self.__dict__["latent_hidden_wan_encoder"] = encoder
            return encoder

        try:
            encoder_device = next(encoder.parameters()).device
        except StopIteration:
            encoder_device = device
        if encoder_device != device:
            encoder.to(device=device)
        encoder.eval()
        return encoder

    def _encode_latent_hidden_wan_target(
        self,
        future_pixel_values: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if future_pixel_values.ndim == 5 and future_pixel_values.shape[1] == 1:
            future_pixel_values = future_pixel_values[:, 0]
        elif future_pixel_values.ndim != 4:
            raise ValueError(
                "wan_vae latent hidden sim expects future pixels shaped "
                f"[B, 1, 3, 256, 256] or [B, 3, 256, 256], got {tuple(future_pixel_values.shape)}."
            )

        if future_pixel_values.shape != (batch_size, 3, 256, 256):
            raise ValueError(
                "wan_vae latent hidden sim expects future pixels shaped "
                f"{(batch_size, 3, 256, 256)}, got {tuple(future_pixel_values.shape)}."
            )

        encoder = self._get_latent_hidden_wan_encoder(device)
        future_for_vae = future_pixel_values.to(device=device, dtype=torch.float32)
        future_for_vae = future_for_vae * 2.0 - 1.0

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=False):
                target_latent = encoder.encode(future_for_vae)

        if target_latent.ndim != 5 or target_latent.shape[2] != 1:
            raise ValueError(
                "Wan encoder must return [B, 16, 1, 32, 32] for single-frame targets, "
                f"got {tuple(target_latent.shape)}."
            )
        target_latent = target_latent[:, :, 0]
        expected_shape = (batch_size, Wan21VAEEncoder.latent_channels, 32, 32)
        if target_latent.shape != expected_shape:
            raise ValueError(
                "Wan encoder produced unexpected target shape "
                f"{tuple(target_latent.shape)}, expected {expected_shape}."
            )
        return target_latent

    def _compute_latent_hidden_wan_vae_loss(
        self,
        latent_anchor_hidden: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> torch.Tensor:
        if self.latent_hidden_wan_upsampler is None:
            raise RuntimeError(
                "latent_hidden_sim_loss_mode='wan_vae' requires latent_hidden_wan_upsampler."
            )

        B = latent_anchor_hidden.shape[0]
        hidden_image = latent_anchor_hidden.to(dtype=self.dtype).reshape(B, self.janus_dim, 1, 1)
        pred_latent = self.latent_hidden_wan_upsampler(hidden_image)
        expected_shape = (B, Wan21VAEEncoder.latent_channels, 32, 32)
        if pred_latent.shape != expected_shape:
            raise ValueError(
                "Wan upsampler produced unexpected shape "
                f"{tuple(pred_latent.shape)}, expected {expected_shape}."
            )
        if target_latent.shape != pred_latent.shape:
            raise ValueError(
                "Wan latent target shape does not match prediction: "
                f"target={tuple(target_latent.shape)}, pred={tuple(pred_latent.shape)}."
            )

        return F.mse_loss(pred_latent.to(torch.float32), target_latent.to(torch.float32))

    def _compute_latent_hidden_wan_downsample_sim_loss(
        self,
        latent_anchor_hidden: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> torch.Tensor:
        if self.latent_hidden_wan_downsampler is None:
            raise RuntimeError(
                "use_latent_hidden_wan_downsample_sim_loss requires latent_hidden_wan_downsampler."
            )

        B = latent_anchor_hidden.shape[0]
        target_hidden_image = self.latent_hidden_wan_downsampler(target_latent.to(dtype=self.dtype))
        expected_shape = (B, self.janus_dim, 1, 1)
        if target_hidden_image.shape != expected_shape:
            raise ValueError(
                "Wan downsampler produced unexpected shape "
                f"{tuple(target_hidden_image.shape)}, expected {expected_shape}."
            )
        target_hidden = target_hidden_image.reshape(B, self.janus_dim)
        similarity = F.cosine_similarity(
            latent_anchor_hidden.to(torch.float32),
            target_hidden.to(torch.float32),
            dim=-1,
        ).mean()
        return 1.0 - similarity

    def joint_denoise_step(
        self,
        video_latent: torch.Tensor,
        action_latent: torch.Tensor,
        latent_tokens: torch.Tensor,
        multimodal_embeds: torch.Tensor,
        timestep_vid: torch.Tensor,
        timestep_act: torch.Tensor,
        action_value_latent: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        janus_left_pad_lens: Optional[torch.Tensor] = None,
        janus_attention_mask: Optional[torch.Tensor] = None,
        action_image_prefix: Optional[torch.Tensor] = None,
        action_state_prefix: Optional[torch.Tensor] = None,
        janus_images_seq_mask: Optional[torch.Tensor] = None,
        janus_images_emb_mask: Optional[torch.Tensor] = None,
        cosmos_text_embeddings: Optional[torch.Tensor] = None,
        forward_pass_index: int = 1,
        run_action_branch: bool = True,
        detach_latent_input: bool = False,
        detach_latent_context: bool = False,
        latent_trainable_tail_tokens: int = 0,
        cache_video_kv: bool = False,
        cache_video_kv_detach: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Single denoising step for all 3 experts.

        Args:
            video_latent: [B, C, T, H, W] noisy video latent
            action_latent: [B, action_chunk, action_dim] noisy action
            latent_tokens: [B, S_latent, D] latent CoT token embeddings
            multimodal_embeds: [B, S_context, D] janus multimodal embeddings (image+text)
            janus_images_seq_mask: [B, S_context] mask of Janus image tokens inside the multimodal context
            janus_images_emb_mask: [B, N, S_img] Janus image embedding mask used to infer image HW
            timestep_vid: [B] video timestep
            timestep_act: [B] action timestep
            fps: [B] frames per second
            janus_left_pad_lens: [B] left-pad lengths used to build latent/action rotary positions.
            janus_attention_mask: [B, S_context] mask controlling which Janus context tokens are visible.
            detach_latent_input: if True, detach the assembled x_latent before MoT blocks.
            detach_latent_context: if True with detach_latent_input, detach only the context prefix.
            latent_trainable_tail_tokens: number of final latent tokens to keep trainable.

        Returns:
            video_v: predicted video velocity
            action_v: predicted action velocity
            action_value_v: predicted action-branch scalar value velocity
            latent_hidden: latent CoT hidden states (for similarity loss)
        """
        if forward_pass_index < 1:
            raise ValueError(f"forward_pass_index must be >= 1, got {forward_pass_index}.")

        video_latent = video_latent.to(self.dtype)
        action_latent = action_latent.to(self.dtype)
        latent_tokens = latent_tokens.to(self.dtype)
        multimodal_embeds = multimodal_embeds.to(self.dtype)

        B = video_latent.shape[0]
        device = video_latent.device
        if fps is None:
            fps = torch.full((B,), 10.0, device=device, dtype=self.dtype)
        elif not isinstance(fps, torch.Tensor):
            fps = torch.full((B,), float(fps), device=device, dtype=self.dtype)

        # --- Prepare action tokens ---
        x_action = None
        if run_action_branch:
            x_action = self._build_action_sequence(
                action_latent=action_latent,
                timestep_act=timestep_act,
                action_value_latent=action_value_latent,
                action_image_prefix=action_image_prefix,
                action_state_prefix=action_state_prefix,
            )

        # --- Prepare latent tokens ---
        # x_latent = [left_pad, real multimodal context, latent_cot_tokens]
        x_latent, latent_valid_mask = self._build_latent_sequence(
            multimodal_embeds,
            latent_tokens,
            janus_left_pad_lens=janus_left_pad_lens,
            janus_attention_mask=janus_attention_mask,
        )
        if detach_latent_input:
            tail_count = max(0, int(latent_trainable_tail_tokens or 0))
            if detach_latent_context and tail_count > 0:
                if tail_count > x_latent.shape[1]:
                    raise ValueError(
                        f"latent_trainable_tail_tokens={tail_count} exceeds "
                        f"latent sequence length {x_latent.shape[1]}."
                    )
                context_len = x_latent.shape[1] - tail_count
                x_latent = torch.cat([x_latent[:, :context_len].detach(), x_latent[:, context_len:]], dim=1)
            else:
                x_latent = x_latent.detach()

        # --- Prepare cosmos DIT input ---
        scale = getattr(self.cosmos_dit, 'timestep_scale', 1.0)
        cosmos_t_scaled = timestep_vid * 1000.0 * scale
        cosmos_t_unsqueeze = cosmos_t_scaled.unsqueeze(1) if cosmos_t_scaled.ndim == 1 else cosmos_t_scaled

        use_wan_fp32 = getattr(self.cosmos_dit, 'use_wan_fp32_strategy', False)
        if use_wan_fp32:
            cosmos_t_unsqueeze = cosmos_t_unsqueeze.to(torch.float32)
        else:
            cosmos_t_unsqueeze = cosmos_t_unsqueeze.to(self.dtype)

        with torch.amp.autocast('cuda', enabled=use_wan_fp32, dtype=torch.float32):
            t_emb, adaln_lora = self.cosmos_dit.t_embedder(cosmos_t_unsqueeze)
            t_emb = self.cosmos_dit.t_embedding_norm(t_emb)

        _, _, T_vid, H_vid, W_vid = video_latent.shape
        value_latent_count = 1 if self.use_value_prediction else 0
        real_video_latent_frames = T_vid - value_latent_count
        n_cond = self._num_real_condition_latents(real_video_latent_frames)
        condition_mask = torch.zeros((B, 1, T_vid, H_vid, W_vid), device=device, dtype=video_latent.dtype)
        condition_mask[:, :, :n_cond, :, :] = 1.0

        video_latent_with_mask = torch.cat([video_latent, condition_mask], dim=1)
        padding_mask = torch.zeros((B, 1, H_vid, W_vid), device=device, dtype=video_latent.dtype)

        x_video, rope_emb, extra_pos = self.cosmos_dit.prepare_embedded_sequence(
            video_latent_with_mask, fps=fps, padding_mask=padding_mask
        )
        value_token_count = self._value_token_count_from_x_video(
            x_video,
            has_value_token=self.use_value_prediction,
        )
        video_grid_thw = None
        if is_multimodal_bridge_pos_scheme(self.bridge_pos_scheme):
            video_grid_thw = torch.tensor(
                [int(x_video.shape[1]), int(x_video.shape[2]), int(x_video.shape[3])],
                device=device,
                dtype=torch.long,
            ).unsqueeze(0).expand(B, -1)
        rotary_batch_info = self._build_bridge_rotary_batch_info(
            x_latent=x_latent,
            x_action=x_action,
            latent_valid_mask=latent_valid_mask,
            latent_left_pad_lens=janus_left_pad_lens,
            janus_images_seq_mask=janus_images_seq_mask,
            janus_images_emb_mask=janus_images_emb_mask,
            action_image_prefix=action_image_prefix,
            video_grid_thw=video_grid_thw,
        )
        rotary_payload = self._build_bridge_rotary_payload(
            batch_info=rotary_batch_info,
            latent_seq_len=x_latent.shape[1],
            action_seq_len=0 if x_action is None else x_action.shape[1],
            device=device,
            dtype=x_latent.dtype,
        )

        # Cross-attention embedding for Cosmos.
        crossattn_emb = self._prepare_cosmos_crossattn_emb(
            multimodal_embeds,
            cosmos_text_embeddings=cosmos_text_embeddings,
            no_grad_text_proj=bool(self.decosmos),
        )

        # --- Run through MoT blocks ---
        for i, block in enumerate(self.cosmos_dit.blocks):
            if i < len(self.mot_attention_wrappers):
                x_video, x_latent, x_action = block(
                    x_B_T_H_W_D=x_video,
                    emb_B_T_D=t_emb,
                    crossattn_emb=crossattn_emb,
                    x_latent=x_latent,
                    x_action=x_action,
                    x_latent_valid_mask=latent_valid_mask,
                    x_rotary_payload=rotary_payload,
                    mot_forward_index=forward_pass_index,
                    mot_cache_video_kv=cache_video_kv,
                    mot_cache_video_kv_detach=cache_video_kv_detach,
                    mot_value_token_count=value_token_count,
                    mot_action_value_token_count=1 if (run_action_branch and action_value_latent is not None) else 0,
                    rope_emb_L_1_1_D=rope_emb,
                    adaln_lora_B_T_3D=adaln_lora,
                    extra_per_block_pos_emb=extra_pos,
                )
            else:
                if self._video_frozen:
                    with torch.no_grad():
                        x_video = block(
                            x_B_T_H_W_D=x_video, emb_B_T_D=t_emb,
                            crossattn_emb=crossattn_emb,
                            rope_emb_L_1_1_D=rope_emb,
                            adaln_lora_B_T_3D=adaln_lora,
                            extra_per_block_pos_emb=extra_pos,
                        )
                else:
                    x_video = block(
                        x_B_T_H_W_D=x_video, emb_B_T_D=t_emb,
                        crossattn_emb=crossattn_emb,
                        rope_emb_L_1_1_D=rope_emb,
                        adaln_lora_B_T_3D=adaln_lora,
                        extra_per_block_pos_emb=extra_pos,
                    )

        # --- Extract outputs ---
        # Video output is only consumed from pass 1 during training.
        if forward_pass_index == 2:
            video_v = video_latent.new_empty(0)
        elif self._video_frozen:
            with torch.no_grad():
                x_video_patch = self.cosmos_dit.final_layer(x_video, t_emb, adaln_lora_B_T_3D=adaln_lora)
                video_v = self.cosmos_dit.unpatchify(x_video_patch)
        else:
            x_video_patch = self.cosmos_dit.final_layer(x_video, t_emb, adaln_lora_B_T_3D=adaln_lora)
            video_v = self.cosmos_dit.unpatchify(x_video_patch)

        action_v = None
        action_value_v = None
        if run_action_branch:
            if x_action is None:
                raise RuntimeError("run_action_branch=True but x_action is None after MoT blocks.")
            if not hasattr(self.janus.language_model.model, 'norm_action'):
                raise AttributeError(
                    "Action branch requires `janus.language_model.model.norm_action`, but it is missing. "
                    "Refusing to fall back to the latent/shared final norm."
                )
            action_norm = self.janus.language_model.model.norm_action
            x_action_norm = action_norm(x_action)
            chunk_size = action_latent.shape[1]
            if action_value_latent is not None:
                if self.action_value_final_layer is None:
                    raise RuntimeError("action_value_final_layer is not initialized.")
                action_out = x_action_norm[:, -(chunk_size + 1):-1, :]
                action_value_out = x_action_norm[:, -1:, :]
                action_value_v = self.action_value_final_layer(action_value_out)
            else:
                action_out = x_action_norm[:, -chunk_size:, :]
            action_v = self.janus.final_layer(action_out)

        # Latent CoT hidden states (for similarity loss)
        latent_norm = self.janus.language_model.model.norm
        latent_hidden = latent_norm(x_latent)

        return video_v, action_v, action_value_v, latent_hidden

    @torch.no_grad()
    def run_cosmos_once(
        self,
        video_latent: torch.Tensor,
        multimodal_embeds: torch.Tensor,
        timestep_vid: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        cosmos_text_embeddings: Optional[torch.Tensor] = None,
        has_value_token: bool = False,
        num_condition_latent_frames: Optional[int] = None,
    ):
        """Run cosmos DIT once and cache Cosmos KV at each MoT block.

        After this call, each MoT wrapper has cached (K_video, V_video) for use
        in generate_latent_cot and action_denoise_step.

        Args:
            video_latent: [B, C, T, H, W] noisy video latent
            multimodal_embeds: [B, S_context, D] for cosmos cross-attention
            timestep_vid: [B] video timestep
            fps: [B] frames per second
            num_condition_latent_frames: actual number of leading real-video latent frames to mask as condition

        Returns:
            video_v: predicted video velocity from single pass
        """
        video_latent = video_latent.to(self.dtype)
        multimodal_embeds = multimodal_embeds.to(self.dtype)

        B = video_latent.shape[0]
        device = video_latent.device
        if fps is None:
            fps = torch.full((B,), 10.0, device=device, dtype=self.dtype)
        elif not isinstance(fps, torch.Tensor):
            fps = torch.full((B,), float(fps), device=device, dtype=self.dtype)

        # Prepare cosmos DIT input
        scale = getattr(self.cosmos_dit, 'timestep_scale', 1.0)
        cosmos_t_scaled = timestep_vid * 1000.0 * scale
        cosmos_t_unsqueeze = cosmos_t_scaled.unsqueeze(1) if cosmos_t_scaled.ndim == 1 else cosmos_t_scaled

        use_wan_fp32 = getattr(self.cosmos_dit, 'use_wan_fp32_strategy', False)
        if use_wan_fp32:
            cosmos_t_unsqueeze = cosmos_t_unsqueeze.to(torch.float32)
        else:
            cosmos_t_unsqueeze = cosmos_t_unsqueeze.to(self.dtype)

        with torch.amp.autocast('cuda', enabled=use_wan_fp32, dtype=torch.float32):
            t_emb, adaln_lora = self.cosmos_dit.t_embedder(cosmos_t_unsqueeze)
            t_emb = self.cosmos_dit.t_embedding_norm(t_emb)

        _, _, T_vid, H_vid, W_vid = video_latent.shape
        value_latent_count = 1 if has_value_token else 0
        real_video_latent_frames = T_vid - value_latent_count
        if num_condition_latent_frames is None:
            n_cond = self._num_real_condition_latents(real_video_latent_frames)
        else:
            n_cond = max(1, min(int(num_condition_latent_frames), real_video_latent_frames))
        condition_mask = torch.zeros((B, 1, T_vid, H_vid, W_vid), device=device, dtype=self.dtype)
        condition_mask[:, :, :n_cond, :, :] = 1.0

        video_latent_with_mask = torch.cat([video_latent, condition_mask], dim=1)
        padding_mask = torch.zeros((B, 1, H_vid, W_vid), device=device, dtype=self.dtype)

        x_video, rope_emb, extra_pos = self.cosmos_dit.prepare_embedded_sequence(
            video_latent_with_mask, fps=fps, padding_mask=padding_mask
        )
        value_token_count = self._value_token_count_from_x_video(
            x_video,
            has_value_token=has_value_token,
        )

        crossattn_emb = self._prepare_cosmos_crossattn_emb(
            multimodal_embeds,
            cosmos_text_embeddings=cosmos_text_embeddings,
        )

        # Enable caching on all MoT wrappers, clear janus tokens (cosmos-only)
        for wrapper in self.mot_attention_wrappers:
            wrapper.cache_video_kv = True
            wrapper.current_x_latent = None
            wrapper.current_x_action = None
            wrapper.current_latent_valid_mask = None
            wrapper.current_rotary_payload = None
            wrapper.current_value_token_count = value_token_count
            wrapper.current_action_value_token_count = 0

        # Run through all cosmos blocks (MoT wrappers will run cosmos-only path and cache KV)
        for i, block in enumerate(self.cosmos_dit.blocks):
            if i < len(self.mot_attention_wrappers):
                # The monkey-patched forward expects x_latent/x_action kwargs;
                # passing None triggers cosmos-only path in MoT wrapper
                x_video, _, _ = block(
                    x_B_T_H_W_D=x_video,
                    emb_B_T_D=t_emb,
                    crossattn_emb=crossattn_emb,
                    x_latent=None,
                    x_action=None,
                    mot_value_token_count=value_token_count,
                    rope_emb_L_1_1_D=rope_emb,
                    adaln_lora_B_T_3D=adaln_lora,
                    extra_per_block_pos_emb=extra_pos,
                )
            else:
                x_video = block(
                    x_B_T_H_W_D=x_video, emb_B_T_D=t_emb,
                    crossattn_emb=crossattn_emb,
                    rope_emb_L_1_1_D=rope_emb,
                    adaln_lora_B_T_3D=adaln_lora,
                    extra_per_block_pos_emb=extra_pos,
                )

        # Disable caching
        for wrapper in self.mot_attention_wrappers:
            wrapper.cache_video_kv = False

        # Video output
        x_video_patch = self.cosmos_dit.final_layer(x_video, t_emb, adaln_lora_B_T_3D=adaln_lora)
        video_v = self.cosmos_dit.unpatchify(x_video_patch)

        return video_v

    @torch.no_grad()
    def action_denoise_step(
        self,
        action_latent: torch.Tensor,
        latent_tokens: torch.Tensor,
        multimodal_embeds: torch.Tensor,
        timestep_act: torch.Tensor,
        action_value_latent: Optional[torch.Tensor] = None,
        janus_left_pad_lens: Optional[torch.Tensor] = None,
        janus_attention_mask: Optional[torch.Tensor] = None,
        action_image_prefix: Optional[torch.Tensor] = None,
        action_state_prefix: Optional[torch.Tensor] = None,
        janus_images_seq_mask: Optional[torch.Tensor] = None,
        janus_images_emb_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Run action denoising step using cached cosmos video features.

        Must be called after run_cosmos_once() which populates the video KV cache.

        Args:
            action_latent: [B, action_chunk, action_dim] noisy action
            latent_tokens: [B, S_latent, D] latent CoT embeddings
            multimodal_embeds: [B, S_context, D] janus multimodal embeddings
            janus_images_seq_mask: [B, S_context] mask of Janus image tokens inside the multimodal context
            janus_images_emb_mask: [B, N, S_img] Janus image embedding mask used to infer image HW
            timestep_act: [B] action timestep
            janus_left_pad_lens: [B] left-pad lengths used to keep latent padding masked on the left
            janus_attention_mask: [B, S_context] mask controlling which Janus context tokens are visible

        Returns:
            action_v: predicted action velocity
            action_value_v: predicted action-branch scalar value velocity
            latent_hidden: latent hidden states
        """
        action_latent = action_latent.to(self.dtype)
        latent_tokens = latent_tokens.to(self.dtype)
        multimodal_embeds = multimodal_embeds.to(self.dtype)

        x_action = self._build_action_sequence(
            action_latent=action_latent,
            timestep_act=timestep_act,
            action_value_latent=action_value_latent,
            action_image_prefix=action_image_prefix,
            action_state_prefix=action_state_prefix,
        )
        x_latent, latent_valid_mask = self._build_latent_sequence(
            multimodal_embeds,
            latent_tokens,
            janus_left_pad_lens=janus_left_pad_lens,
            janus_attention_mask=janus_attention_mask,
        )
        cached_video_grid_thw = None
        if is_multimodal_bridge_pos_scheme(self.bridge_pos_scheme) and not self.decosmos:
            cached_video_grid_thw = self._get_cached_bridge_video_grid_thw(x_latent.shape[0], x_latent.device)
        rotary_batch_info = self._build_bridge_rotary_batch_info(
            x_latent=x_latent,
            x_action=x_action,
            latent_valid_mask=latent_valid_mask,
            latent_left_pad_lens=janus_left_pad_lens,
            janus_images_seq_mask=janus_images_seq_mask,
            janus_images_emb_mask=janus_images_emb_mask,
            action_image_prefix=action_image_prefix,
            video_grid_thw=cached_video_grid_thw,
        )
        rotary_payload = self._build_bridge_rotary_payload(
            batch_info=rotary_batch_info,
            latent_seq_len=x_latent.shape[1],
            action_seq_len=x_action.shape[1],
            device=x_latent.device,
            dtype=x_latent.dtype,
        )

        # Run through MoT blocks using cached video KV (no cosmos DIT)
        for wrapper in self.mot_attention_wrappers:
            x_latent, x_action = wrapper.forward_action_only(
                x_latent,
                x_action,
                latent_valid_mask=latent_valid_mask,
                rotary_payload=rotary_payload,
                action_value_token_count=1 if action_value_latent is not None else 0,
            )

        # Action output
        if not hasattr(self.janus.language_model.model, 'norm_action'):
            raise AttributeError(
                "Action branch requires `janus.language_model.model.norm_action`, but it is missing. "
                "Refusing to fall back to the latent/shared final norm."
            )
        action_norm = self.janus.language_model.model.norm_action
        x_action_norm = action_norm(x_action)
        chunk_size = action_latent.shape[1]
        action_value_v = None
        if action_value_latent is not None:
            if self.action_value_final_layer is None:
                raise RuntimeError("action_value_final_layer is not initialized.")
            action_out = x_action_norm[:, -(chunk_size + 1):-1, :]
            action_value_out = x_action_norm[:, -1:, :]
            action_value_v = self.action_value_final_layer(action_value_out)
        else:
            action_out = x_action_norm[:, -chunk_size:, :]
        action_v = self.janus.final_layer(action_out)

        # Latent hidden states
        latent_norm = self.janus.language_model.model.norm
        latent_hidden = latent_norm(x_latent)

        return action_v, action_value_v, latent_hidden

    def forward(
        self,
        first_frame: torch.Tensor,
        video_frames: torch.Tensor,
        actions: torch.Tensor,
        janus_input_ids: torch.Tensor,
        janus_pixel_values: torch.Tensor,
        janus_action_pixel_values: torch.Tensor,
        janus_images_seq_mask: torch.Tensor,
        janus_images_emb_mask: torch.Tensor,
        fps,
        latent_gt_token_ids: torch.Tensor,
        janus_state_seq_mask: Optional[torch.Tensor] = None,
        now_state: Optional[torch.Tensor] = None,
        janus_left_pad_lens: Optional[torch.Tensor] = None,
        janus_attention_mask: Optional[torch.Tensor] = None,
        cosmos_text_embeddings: Optional[torch.Tensor] = None,
        value_targets: Optional[torch.Tensor] = None,
        latent_hidden_sim_pixel_values: Optional[torch.Tensor] = None,
        loss_weights: tuple = (0.2, 1.0, 1.0),
    ):
        """Training forward pass with one- or two-token latent CE supervision.

        Args:
            first_frame, video_frames, actions: same as 2-expert model
            janus_*: standard janus multimodal inputs for the current observation
            fps: frames per second
            latent_gt_token_ids: [B] or [B, N] ground truth tokenizer ids for 1 or 2 latent tokens.
            janus_left_pad_lens: [B] number of left-pad tokens in janus_input_ids for rotary positions.
            janus_attention_mask: [B, S_context] mask for Janus context attention visibility.
            latent_hidden_sim_pixel_values: optional [B, 1, C, H, W] future image pixels for hidden sim loss.
            loss_weights: (video_weight, action_weight, latent_weight, cosmos_value_weight?, action_value_weight?, latent_hidden_sim_weight?, latent_hidden_wan_downsample_sim_weight?)

        Returns:
            total_loss, video_loss (float), action_loss (float), latent_ce_loss (float),
            aux_metrics (dict[str, float])
        """
        B = video_frames.shape[0]
        device = video_frames.device
        aux_metrics = {}

        # Normalize large training inputs once so downstream math can reuse them
        # without repeating no-op casts on the same device/dtype.
        actions = actions.to(self.dtype)
        janus_pixel_values = janus_pixel_values.to(self.dtype)
        janus_action_pixel_values = janus_action_pixel_values.to(self.dtype)
        latent_gt_token_ids = self._normalize_latent_gt_token_ids(
            latent_gt_token_ids.to(device=device, dtype=torch.long)
        )
        if latent_gt_token_ids.shape[0] != B:
            raise ValueError(
                f"latent_gt_token_ids batch size {latent_gt_token_ids.shape[0]} does not match B={B}."
            )
        latent_token_count = self._resolve_latent_token_count(latent_gt_token_ids.shape[1])
        configured_latent_token_count = self._resolve_latent_token_count()
        if latent_token_count != configured_latent_token_count:
            raise ValueError(
                "latent_gt_token_ids token count does not match config.total_latent_tokens: "
                f"got {latent_token_count}, expected {configured_latent_token_count}."
            )
        if latent_hidden_sim_pixel_values is not None:
            latent_hidden_sim_pixel_values = latent_hidden_sim_pixel_values.to(device=device, dtype=self.dtype)

        if not isinstance(fps, torch.Tensor):
            fps = torch.full((B,), float(fps), device=device, dtype=self.dtype)

        # VAE encode
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
        video_norm = (video_frames * 2.0 - 1.0)
        full_video = torch.cat([first_frame_norm, video_norm], dim=2)

        with torch.no_grad():
            clean_video_latent = self.cosmos_vae.encode(full_video.to(self.dtype)).to(self.dtype)
            n_cond = self._num_real_condition_latents(clean_video_latent.shape[2])
            condition_latent = clean_video_latent[:, :, :n_cond]

        value_targets_float = None
        clean_value_latent = None
        uses_value_target = self.use_value_prediction or self.use_action_value_prediction
        if uses_value_target:
            if value_targets is None:
                raise ValueError(
                    "value_targets must be provided when use_value_prediction=1 "
                    "or use_action_value_prediction=1."
                )
            value_targets_float = value_targets.to(device=device, dtype=torch.float32).view(B)
        if self.use_value_prediction:
            clean_value_latent = self._build_value_logit_latent(value_targets_float, clean_video_latent)

        if self.train_embed_tokens:
            multimodal_embeds, state_embeds, wrist_action_image_prefix = (
                self.prepare_multimodal_embeds_with_state_action(
                    janus_input_ids=janus_input_ids,
                    now_state=now_state,
                    janus_pixel_values=janus_pixel_values,
                    janus_action_pixel_values=janus_action_pixel_values,
                    janus_images_seq_mask=janus_images_seq_mask,
                    janus_state_seq_mask=janus_state_seq_mask,
                    janus_images_emb_mask=janus_images_emb_mask,
                )
            )
        else:
            with torch.no_grad():
                multimodal_embeds, state_embeds, wrist_action_image_prefix = (
                    self.prepare_multimodal_embeds_with_state_action(
                        janus_input_ids=janus_input_ids,
                        now_state=now_state,
                        janus_pixel_values=janus_pixel_values,
                        janus_action_pixel_values=janus_action_pixel_values,
                        janus_images_seq_mask=janus_images_seq_mask,
                        janus_state_seq_mask=janus_state_seq_mask,
                        janus_images_emb_mask=janus_images_emb_mask,
                    )
                )

        multimodal_embeds = multimodal_embeds.to(self.dtype)
        if state_embeds is not None:
            state_embeds = state_embeds.to(self.dtype)
        wrist_action_image_prefix = wrist_action_image_prefix.to(self.dtype)

        action_image_prefix = None
        action_state_prefix = None
        if self.action_use_latent_prefix:
            action_image_prefix = wrist_action_image_prefix
            if state_embeds is not None:
                action_state_prefix = state_embeds

        # Flow matching noise. Video follows native Cosmos RF train-time sampling.
        u_vid, t_vid = self._sample_cosmos_train_sigma(B, device)
        aux_metrics["cosmos_train_u_mean"] = u_vid.detach().to(torch.float32).mean().item()
        aux_metrics["cosmos_train_sigma_mean"] = t_vid.detach().to(torch.float32).mean().item()
        # Bias action flow-matching timesteps slightly toward noisier states.
        t_act = torch.distributions.Beta(
            torch.tensor(1.5, device=device, dtype=torch.float32),
            torch.tensor(1.0, device=device, dtype=torch.float32),
        ).sample((B,)).to(dtype=self.dtype)

        t_expanded_vid = t_vid.view(B, 1, 1, 1, 1)
        t_expanded_act = t_act.view(B, 1, 1)

        video_noise = torch.randn_like(clean_video_latent)
        action_noise = torch.randn_like(actions, dtype=self.dtype)

        noisy_video_latent = t_expanded_vid * video_noise + (1.0 - t_expanded_vid) * clean_video_latent
        target_value_v = None
        if self.use_value_prediction:
            value_noise = torch.randn_like(clean_value_latent)
            noisy_value_latent = t_expanded_vid * value_noise + (1.0 - t_expanded_vid) * clean_value_latent
            noisy_video_latent = torch.cat([noisy_video_latent, noisy_value_latent], dim=2)
            target_value_v = value_noise - clean_value_latent
        noisy_action = t_expanded_act * action_noise + (1.0 - t_expanded_act) * actions
        noisy_action_value = None
        target_action_value_v = None
        if self.use_action_value_prediction:
            clean_action_value = value_targets_float.to(device=device, dtype=self.dtype).view(B, 1, 1)
            action_value_noise = torch.randn_like(clean_action_value)
            noisy_action_value = (
                t_expanded_act * action_value_noise
                + (1.0 - t_expanded_act) * clean_action_value
            )
            target_action_value_v = action_value_noise - clean_action_value

        noisy_video_latent[:, :, :n_cond] = condition_latent

        target_video_v = video_noise - clean_video_latent
        target_video_v[:, :, :n_cond] = 0.0
        target_action_v = action_noise - actions

        latent_gt_embeds = self._embed_janus_token_ids(latent_gt_token_ids).to(self.dtype)

        self._clear_cached_video_kv()
        try:
            # Single pass: supervise next latent tokens with CE and condition action
            # on the GT token embeddings.
            # Keep training passes stateless because gradient checkpoint recomputation
            # cannot safely depend on module-local KV caches created by a previous pass.
            pred_video_v, pred_action_v, pred_action_value_v, latent_hidden = self.joint_denoise_step(
                video_latent=noisy_video_latent,
                action_latent=noisy_action,
                action_value_latent=noisy_action_value,
                latent_tokens=latent_gt_embeds,
                multimodal_embeds=multimodal_embeds,
                janus_images_seq_mask=janus_images_seq_mask,
                janus_images_emb_mask=janus_images_emb_mask,
                timestep_vid=t_vid,
                timestep_act=t_act,
                fps=fps,
                janus_left_pad_lens=janus_left_pad_lens,
                janus_attention_mask=janus_attention_mask,
                cosmos_text_embeddings=cosmos_text_embeddings,
                action_image_prefix=action_image_prefix,
                action_state_prefix=action_state_prefix,
                forward_pass_index=1,
                run_action_branch=True,
                detach_latent_input=not self.no_detach_latent_input,
                detach_latent_context=not self.no_detach_latent_input,
                latent_trainable_tail_tokens=0 if self.no_detach_latent_input else latent_gt_embeds.shape[1],
                cache_video_kv=False,
            )

            # Losses
            if len(loss_weights) == 3:
                video_weight, action_weight, latent_weight = loss_weights
                cosmos_value_weight = 0.0
                action_value_weight = 0.0
                latent_hidden_sim_weight = 0.0
                latent_hidden_wan_downsample_sim_weight = 0.0
            elif len(loss_weights) == 4:
                video_weight, action_weight, latent_weight, cosmos_value_weight = loss_weights
                action_value_weight = 0.0
                latent_hidden_sim_weight = 0.0
                latent_hidden_wan_downsample_sim_weight = 0.0
            elif len(loss_weights) == 5:
                video_weight, action_weight, latent_weight, cosmos_value_weight, action_value_weight = loss_weights
                latent_hidden_sim_weight = 0.0
                latent_hidden_wan_downsample_sim_weight = 0.0
            elif len(loss_weights) == 6:
                (
                    video_weight,
                    action_weight,
                    latent_weight,
                    cosmos_value_weight,
                    action_value_weight,
                    latent_hidden_sim_weight,
                ) = loss_weights
                latent_hidden_wan_downsample_sim_weight = 0.0
            elif len(loss_weights) == 7:
                (
                    video_weight,
                    action_weight,
                    latent_weight,
                    cosmos_value_weight,
                    action_value_weight,
                    latent_hidden_sim_weight,
                    latent_hidden_wan_downsample_sim_weight,
                ) = loss_weights
            else:
                raise ValueError(f"loss_weights must have 3, 4, 5, 6, or 7 entries, got {len(loss_weights)}.")
            if self.use_value_prediction and len(loss_weights) < 4:
                raise ValueError("use_value_prediction requires loss_weights to include cosmos_value_weight.")
            if self.use_action_value_prediction and len(loss_weights) < 5:
                raise ValueError("use_action_value_prediction requires loss_weights to include action_value_weight.")
            if self.use_latent_hidden_wan_downsample_sim_loss and len(loss_weights) < 7:
                raise ValueError(
                    "use_latent_hidden_wan_downsample_sim_loss requires loss_weights to include "
                    "latent_hidden_wan_downsample_sim_weight."
                )
            loss_cosmos_value = torch.zeros((), device=device, dtype=torch.float32)
            loss_action_value = torch.zeros((), device=device, dtype=torch.float32)
            loss_latent_hidden_sim = torch.zeros((), device=device, dtype=torch.float32)
            loss_latent_hidden_wan_downsample_sim = torch.zeros((), device=device, dtype=torch.float32)
            if self.decosmos:
                loss_video = torch.zeros((), device=device, dtype=torch.float32)
            else:
                if self.use_value_prediction:
                    pred_value_v = pred_video_v[:, :, -1:]
                    pred_video_v = pred_video_v[:, :, :-1]
                    loss_cosmos_value = F.mse_loss(pred_value_v, target_value_v)
                    aux_metrics["cosmos_value_loss"] = loss_cosmos_value.detach().item()
                if n_cond >= target_video_v.shape[2]:
                    loss_video = torch.zeros((), device=device, dtype=torch.float32)
                else:
                    loss_video = F.mse_loss(pred_video_v[:, :, n_cond:], target_video_v[:, :, n_cond:])

            if pred_action_v is None:
                raise RuntimeError("Token CE training must produce an action prediction.")

            latent_anchor_hiddens = latent_hidden[:, -(latent_token_count + 1):-1, :]
            if latent_anchor_hiddens.shape[:2] != (B, latent_token_count):
                raise ValueError(
                    "Latent hidden sequence is too short for next-token supervision: "
                    f"anchor_shape={tuple(latent_anchor_hiddens.shape)}, token_count={latent_token_count}."
                )
            latent_logits = self.janus.language_model.lm_head(latent_anchor_hiddens)
            latent_ce_per_item = F.cross_entropy(
                latent_logits.reshape(B * latent_token_count, -1).to(torch.float32),
                latent_gt_token_ids.reshape(B * latent_token_count),
                reduction="none",
            ).view(B, latent_token_count)
            latent_ce_per_token = latent_ce_per_item.mean(dim=0)
            loss_latent = latent_ce_per_token.mean()
            aux_metrics["latent_ce_loss"] = loss_latent.detach().item()
            if latent_token_count > 1:
                for token_idx in range(latent_token_count):
                    aux_metrics[f"latent_ce_loss_{token_idx}"] = (
                        latent_ce_per_token[token_idx].detach().item()
                    )
            if latent_hidden_sim_pixel_values is not None:
                if self.latent_hidden_sim_loss_mode == "wan_vae":
                    target_wan_latent = self._encode_latent_hidden_wan_target(
                        future_pixel_values=latent_hidden_sim_pixel_values,
                        batch_size=B,
                        device=latent_anchor_hiddens.device,
                    )
                    wan_vae_losses = []
                    for token_idx in range(latent_token_count):
                        token_loss = self._compute_latent_hidden_wan_vae_loss(
                            latent_anchor_hidden=latent_anchor_hiddens[:, token_idx, :],
                            target_latent=target_wan_latent,
                        )
                        wan_vae_losses.append(token_loss)
                        if latent_token_count > 1:
                            aux_metrics[f"latent_hidden_wan_vae_mse_loss_{token_idx}"] = (
                                token_loss.detach().item()
                            )
                    loss_latent_hidden_sim = torch.stack(wan_vae_losses).mean()
                    aux_metrics["latent_hidden_wan_vae_mse_loss"] = loss_latent_hidden_sim.detach().item()
                    if self.use_latent_hidden_wan_downsample_sim_loss:
                        wan_downsample_losses = []
                        for token_idx in range(latent_token_count):
                            token_downsample_loss = (
                                self._compute_latent_hidden_wan_downsample_sim_loss(
                                    latent_anchor_hidden=latent_anchor_hiddens[:, token_idx, :],
                                    target_latent=target_wan_latent,
                                )
                            )
                            wan_downsample_losses.append(token_downsample_loss)
                            if latent_token_count > 1:
                                aux_metrics[f"latent_hidden_wan_downsample_sim_loss_{token_idx}"] = (
                                    token_downsample_loss.detach().item()
                                )
                        loss_latent_hidden_wan_downsample_sim = torch.stack(wan_downsample_losses).mean()
                        aux_metrics["latent_hidden_wan_downsample_sim_loss"] = (
                            loss_latent_hidden_wan_downsample_sim.detach().item()
                        )
                else:
                    if latent_hidden_sim_pixel_values.ndim != 5 or latent_hidden_sim_pixel_values.shape[1] != 1:
                        raise ValueError(
                            "latent_hidden_sim_pixel_values must have shape [B, 1, C, H, W], "
                            f"got {tuple(latent_hidden_sim_pixel_values.shape)}."
                        )
                    with torch.no_grad():
                        latent_hidden_sim_gt = self._encode_janus_pixel_values(
                            latent_hidden_sim_pixel_values
                        ).mean(dim=2)
                    if latent_hidden_sim_gt.shape != (B, 1, latent_anchor_hiddens.shape[-1]):
                        raise ValueError(
                            "Encoded latent hidden sim target shape "
                            f"{tuple(latent_hidden_sim_gt.shape)} does not match expected "
                            f"{(B, 1, latent_anchor_hiddens.shape[-1])}."
                        )
                    latent_hidden_sim_target = latent_hidden_sim_gt[:, 0, :]
                    latent_hidden_sim_target = latent_hidden_sim_target[:, None, :].expand(
                        B,
                        latent_token_count,
                        latent_anchor_hiddens.shape[-1],
                    )
                    similarity = F.cosine_similarity(
                        latent_anchor_hiddens.to(torch.float32),
                        latent_hidden_sim_target.to(torch.float32),
                        dim=-1,
                    )
                    latent_hidden_sim_per_token = 1.0 - similarity.mean(dim=0)
                    loss_latent_hidden_sim = latent_hidden_sim_per_token.mean()
                    if latent_token_count > 1:
                        for token_idx in range(latent_token_count):
                            aux_metrics[f"latent_hidden_sim_loss_{token_idx}"] = (
                                latent_hidden_sim_per_token[token_idx].detach().item()
                            )
                aux_metrics["latent_hidden_sim_loss"] = loss_latent_hidden_sim.detach().item()

            loss_action = F.mse_loss(pred_action_v, target_action_v)
            if self.use_action_value_prediction:
                if pred_action_value_v is None:
                    raise RuntimeError("Action-value prediction is enabled but no action-value output was produced.")
                loss_action_value = F.mse_loss(pred_action_value_v, target_action_value_v)
                aux_metrics["action_value_loss"] = loss_action_value.detach().item()
        finally:
            self._clear_cached_video_kv()

        if self._video_frozen:
            video_weight = 0.0

        total_loss = (
            video_weight * loss_video
            + action_weight * loss_action
            + latent_weight * loss_latent
            + cosmos_value_weight * loss_cosmos_value
            + action_value_weight * loss_action_value
            + latent_hidden_sim_weight * loss_latent_hidden_sim
            + latent_hidden_wan_downsample_sim_weight * loss_latent_hidden_wan_downsample_sim
        )
        if self.decosmos:
            return total_loss, loss_video, loss_action.item(), loss_latent.item(), aux_metrics

        return total_loss, loss_video.item(), loss_action.item(), loss_latent.item(), aux_metrics

    @torch.no_grad()
    def forward_flow_joint_inference(
        self,
        janus_input_ids, janus_pixel_values, janus_images_seq_mask, janus_images_emb_mask,
        first_frame,
        action_denoise_steps=10,
        cosmos_denoise_steps=1,
        num_latent_tokens=None,
        fps=None,
        action_self_causal_in_bridge=False,
        janus_left_pad_lens: Optional[torch.Tensor] = None,
        janus_state_seq_mask: Optional[torch.Tensor] = None,
        janus_action_pixel_values: Optional[torch.Tensor] = None,
        now_state: Optional[torch.Tensor] = None,
        janus_attention_mask: Optional[torch.Tensor] = None,
        cosmos_text_embeddings: Optional[torch.Tensor] = None,
        return_value_prediction: Optional[bool] = None,
        return_action_value_prediction: Optional[bool] = None,
        return_latent_visual_debug: bool = False,
    ):
        """Inference: one-step cosmos 鈫?autoregressive CoT 鈫?multi-step action denoising.

        Phase 1: Run cosmos DIT once from pure noise, caching video KV at each MoT block.
        Phase 2: Autoregressively generate latent CoT tokens using cached video features.
        Phase 3: Multi-step action denoising using cached video features + generated CoT.

        Args:
            janus_*: standard janus inputs
            first_frame: [B, C, H, W] or [B, C, T_cond, H, W] conditioning frame(s), normalized to [0,1]
            action_denoise_steps: number of action denoising steps (default 10)
            num_latent_tokens: number of latent CoT tokens to generate. Defaults to
                config.total_latent_tokens when omitted.
            fps: frames per second
            action_self_causal_in_bridge: whether action tokens use causal
                self-visibility inside inference-time bridge attention. Default False
                (bidirectional) since actions are generated in parallel via flow matching.
            janus_left_pad_lens: [B] left-pad lengths for Janus inputs, if batched inputs are left-padded.
            janus_action_pixel_values: [B, 1, C, H, W] wrist/action-view image pixel values.
            now_state: [B, S_state] token ids or [B, 8] normalized float state values.
            janus_attention_mask: [B, S_context] mask controlling visible Janus context tokens.

        Returns:
            pred_video: [B, C, T, H, W] generated video (rough quality, in [-1, 1])
            pred_action: [B, action_chunk, action_dim] predicted actions
        """
        if cosmos_denoise_steps <= 0:
            raise ValueError("cosmos_denoise_steps must be positive.")

        num_latent_tokens = self._resolve_latent_token_count(num_latent_tokens)

        B = first_frame.shape[0]
        device = first_frame.device
        decosmos = bool(getattr(self, "decosmos", False) or getattr(self.config, "decosmos", False))
        if return_value_prediction is None:
            return_value_prediction = self.use_value_prediction
        return_value_prediction = bool(return_value_prediction)
        if return_action_value_prediction is None:
            return_action_value_prediction = self.use_action_value_prediction
        return_action_value_prediction = bool(return_action_value_prediction)
        if return_value_prediction and decosmos:
            raise ValueError("return_value_prediction requires decosmos=0.")
        if return_value_prediction and not self.use_value_prediction:
            raise ValueError("return_value_prediction=True requires use_value_prediction=1.")
        if return_action_value_prediction and not self.use_action_value_prediction:
            raise ValueError(
                "return_action_value_prediction=True requires use_action_value_prediction=1."
            )
        sample_scheduler = shift = use_kerras_sigma = num_train_timesteps = None

        if fps is None:
            fps = torch.full((B,), 10.0, device=device, dtype=self.dtype)
        elif not isinstance(fps, torch.Tensor):
            fps = torch.full((B,), float(fps), device=device, dtype=self.dtype)
        if janus_left_pad_lens is not None:
            janus_left_pad_lens = janus_left_pad_lens.to(device=device, dtype=torch.long)
        if janus_attention_mask is not None:
            janus_attention_mask = janus_attention_mask.to(device=device, dtype=torch.bool)
        if janus_action_pixel_values is None:
            raise ValueError(
                "janus_action_pixel_values is required for 3-expert CoT inference "
                "so the action branch can use the wrist/action-view image prefix."
            )

        janus_pixel_values = janus_pixel_values.to(self.dtype)
        janus_action_pixel_values = janus_action_pixel_values.to(self.dtype)

        condition_latent = None
        if not decosmos:
            if first_frame.ndim == 4:
                first_frame_for_cosmos = (first_frame * 2.0 - 1.0).unsqueeze(2)
            elif first_frame.ndim == 5:
                first_frame_for_cosmos = first_frame * 2.0 - 1.0
            else:
                raise ValueError(
                    "first_frame must have shape [B, C, H, W] or [B, C, T_cond, H, W], "
                    f"got {tuple(first_frame.shape)}."
                )
            condition_latent = self.cosmos_vae.encode(first_frame_for_cosmos.to(self.dtype)).to(self.dtype)

        multimodal_embeds, state_embeds, wrist_action_image_prefix = (
            self.prepare_multimodal_embeds_with_state_action(
                janus_input_ids=janus_input_ids,
                now_state=now_state,
                janus_pixel_values=janus_pixel_values,
                janus_action_pixel_values=janus_action_pixel_values,
                janus_images_seq_mask=janus_images_seq_mask,
                janus_state_seq_mask=janus_state_seq_mask,
                janus_images_emb_mask=janus_images_emb_mask,
            )
        )
        multimodal_embeds = multimodal_embeds.to(self.dtype)
        if state_embeds is not None:
            state_embeds = state_embeds.to(self.dtype)
        wrist_action_image_prefix = wrist_action_image_prefix.to(self.dtype)

        action_image_prefix = None
        action_state_prefix = None
        if self.action_use_latent_prefix:
            action_image_prefix = wrist_action_image_prefix
            if state_embeds is not None:
                action_state_prefix = state_embeds

        action_chunk = getattr(self.config, 'action_chunk', 16)
        action_dim = getattr(self.config, 'action_dim', 7)

        if not decosmos:
            T_cond = condition_latent.shape[2]
            C_lat = condition_latent.shape[1]
            H_lat, W_lat = condition_latent.shape[3], condition_latent.shape[4]
            T_total = self._latent_num_frames_for_pixels(getattr(self.config, 'video_frames', 16))
            if T_total < T_cond:
                raise ValueError(
                    "Conditioning history encodes to more latent frames than the configured "
                    f"video window: T_cond={T_cond}, T_total={T_total}, "
                    f"video_frames={getattr(self.config, 'video_frames', 16)}."
                )

            video_noise = torch.randn(B, C_lat, T_total, H_lat, W_lat, device=device, dtype=self.dtype)
            video_noise[:, :, :T_cond] = condition_latent
            if return_value_prediction or T_total > T_cond:
                sample_scheduler, shift, use_kerras_sigma, num_train_timesteps = (
                    self._require_cosmos_inference_runtime()
                )

        self._set_bridge_action_self_causal_override(action_self_causal_in_bridge)
        try:
            # === Phase 1: Scheduler-driven cosmos DIT (caches final-step video KV) ===
            x_vid = None
            predicted_value = None
            if not decosmos:
                x_vid = video_noise
                x_value = None
                if return_value_prediction:
                    x_value = torch.randn(B, C_lat, 1, H_lat, W_lat, device=device, dtype=self.dtype)
                if T_total == T_cond and not return_value_prediction:
                    t_vid = torch.zeros((B,), device=device, dtype=self.dtype)
                    _ = self.run_cosmos_once(
                        video_latent=x_vid,
                        multimodal_embeds=multimodal_embeds,
                        timestep_vid=t_vid,
                        fps=fps,
                        cosmos_text_embeddings=cosmos_text_embeddings,
                        has_value_token=False,
                        num_condition_latent_frames=T_cond,
                    )
                else:
                    sample_scheduler.set_timesteps(
                        cosmos_denoise_steps,
                        device=device,
                        shift=shift,
                        use_kerras_sigma=use_kerras_sigma,
                    )
                    timesteps = sample_scheduler.timesteps

                    for raw_t in timesteps:
                        raw_t = raw_t.to(device=device)
                        t_vid = torch.ones((B,), device=device, dtype=self.dtype)
                        t_vid = t_vid * (raw_t.to(dtype=self.dtype) / num_train_timesteps)

                        x_cosmos = torch.cat([x_vid, x_value], dim=2) if return_value_prediction else x_vid
                        pred_cosmos_v = self.run_cosmos_once(
                            video_latent=x_cosmos,
                            multimodal_embeds=multimodal_embeds,
                            timestep_vid=t_vid,
                            fps=fps,
                            cosmos_text_embeddings=cosmos_text_embeddings,
                            has_value_token=return_value_prediction,
                            num_condition_latent_frames=T_cond,
                        )
                        if return_value_prediction:
                            video_v = pred_cosmos_v[:, :, :-1]
                            x_cosmos = sample_scheduler.step(
                                pred_cosmos_v.to(torch.float32),
                                raw_t,
                                x_cosmos.to(torch.float32),
                                return_dict=False,
                            )[0].to(self.dtype)
                            x_vid = x_cosmos[:, :, :-1]
                            x_value = x_cosmos[:, :, -1:]
                            predicted_value = self._decode_value_prediction(x_value)
                        else:
                            video_v = pred_cosmos_v
                            x_vid = sample_scheduler.step(
                                video_v.to(torch.float32),
                                raw_t,
                                x_vid.to(torch.float32),
                                return_dict=False,
                            )[0].to(self.dtype)
                        x_vid[:, :, :T_cond] = condition_latent

            # === Phase 2: Autoregressive CoT generation (uses cached video KV) ===
            latent_embeds = self.generate_latent_cot(
                multimodal_embeds=multimodal_embeds,
                janus_images_seq_mask=janus_images_seq_mask,
                janus_images_emb_mask=janus_images_emb_mask,
                num_latent_tokens=num_latent_tokens,
                janus_left_pad_lens=janus_left_pad_lens,
                janus_attention_mask=janus_attention_mask,
                return_latent_visual_debug=return_latent_visual_debug,
            )
            latent_visual_debug = None
            if return_latent_visual_debug:
                latent_embeds, latent_visual_debug = latent_embeds

            # === Phase 3: Multi-step action denoising (uses cached video KV + generated CoT) ===
            x_act = torch.randn(B, action_chunk, action_dim, device=device, dtype=self.dtype)
            x_action_value = None
            predicted_action_value = None
            if return_action_value_prediction:
                x_action_value = torch.randn(B, 1, 1, device=device, dtype=self.dtype)
            dt = 1.0 / action_denoise_steps

            for step in range(action_denoise_steps):
                t = 1.0 - step * dt
                t_act = torch.full((B,), t, device=device, dtype=self.dtype)

                pred_act_v, pred_action_value_v, _ = self.action_denoise_step(
                    action_latent=x_act,
                    latent_tokens=latent_embeds,
                    multimodal_embeds=multimodal_embeds,
                    action_value_latent=x_action_value,
                    janus_images_seq_mask=janus_images_seq_mask,
                    janus_images_emb_mask=janus_images_emb_mask,
                    timestep_act=t_act,
                    janus_left_pad_lens=janus_left_pad_lens,
                    janus_attention_mask=janus_attention_mask,
                    action_image_prefix=action_image_prefix,
                    action_state_prefix=action_state_prefix,
                )

                x_act = x_act - dt * pred_act_v
                if return_action_value_prediction:
                    if pred_action_value_v is None:
                        raise RuntimeError("Action-value inference requested but model returned no value velocity.")
                    x_action_value = x_action_value - dt * pred_action_value_v

            pred_video = None if decosmos else self.cosmos_vae.decode(x_vid.to(self.dtype))
            pred_action = x_act
            if return_action_value_prediction:
                predicted_action_value = x_action_value.to(torch.float32).view(B)

            outputs = [pred_video, pred_action]
            if return_action_value_prediction:
                outputs.extend([predicted_value, predicted_action_value])
            elif return_value_prediction:
                outputs.append(predicted_value)
            if return_latent_visual_debug:
                outputs.append(latent_visual_debug)
            return tuple(outputs)
        finally:
            self._clear_cached_video_kv()
            self._set_bridge_action_self_causal_override(None)

    @torch.no_grad()
    def generate_latent_cot(
        self,
        multimodal_embeds,
        janus_images_seq_mask: Optional[torch.Tensor] = None,
        janus_images_emb_mask: Optional[torch.Tensor] = None,
        num_latent_tokens=None,
        janus_left_pad_lens: Optional[torch.Tensor] = None,
        janus_attention_mask: Optional[torch.Tensor] = None,
        return_latent_visual_debug: bool = False,
    ):
        """Phase 2 inference: generate 1 or 2 latent vocabulary tokens autoregressively.

        Must be called after run_cosmos_once() which populates video KV cache at each
        MoT wrapper. Each newly generated latent token is fed back into the latent
        branch before predicting the next token.

        Args:
            multimodal_embeds: [B, S_context, D] janus multimodal embeddings (image+text)
            janus_images_seq_mask: [B, S_context] mask of Janus image tokens inside the multimodal context
            janus_images_emb_mask: [B, N, S_img] Janus image embedding mask used to infer image HW
            num_latent_tokens: must be 1 or 2 for beta token-latent inference.
            janus_left_pad_lens: [B] left-pad lengths used to keep latent padding masked on the left
            janus_attention_mask: [B, S_context] mask controlling visible Janus context tokens.

        Returns:
            latent_embeds: [B, N, D] predicted latent token embeddings
        """
        num_latent_tokens = self._resolve_latent_token_count(num_latent_tokens)

        janus_norm = self.janus.language_model.model.norm

        # Preserve the original left-padding layout in the Janus context.
        multimodal_embeds = multimodal_embeds.to(self.dtype)
        B = multimodal_embeds.shape[0]
        device = multimodal_embeds.device
        latent_dim = multimodal_embeds.shape[-1]
        if janus_left_pad_lens is not None:
            janus_left_pad_lens = janus_left_pad_lens.to(device=device, dtype=torch.long)
        if janus_attention_mask is not None:
            janus_attention_mask = janus_attention_mask.to(device=device, dtype=torch.bool)
        cached_video_grid_thw = None
        if is_multimodal_bridge_pos_scheme(self.bridge_pos_scheme) and not self.decosmos:
            cached_video_grid_thw = self._get_cached_bridge_video_grid_thw(B, device)

        latent_embeds_parts = []
        last_anchor_hidden = None
        last_token_ids = None
        for _step_idx in range(num_latent_tokens):
            prev_latents = (
                torch.cat(latent_embeds_parts, dim=1)
                if latent_embeds_parts
                else multimodal_embeds.new_zeros(B, 0, latent_dim)
            )
            x_latent, latent_valid_mask = self._build_latent_sequence(
                multimodal_embeds,
                prev_latents,
                janus_left_pad_lens=janus_left_pad_lens,
                janus_attention_mask=janus_attention_mask,
            )
            rotary_batch_info = self._build_bridge_rotary_batch_info(
                x_latent=x_latent,
                x_action=None,
                latent_valid_mask=latent_valid_mask,
                latent_left_pad_lens=janus_left_pad_lens,
                janus_images_seq_mask=janus_images_seq_mask,
                janus_images_emb_mask=janus_images_emb_mask,
                video_grid_thw=cached_video_grid_thw,
            )
            rotary_payload = self._build_bridge_rotary_payload(
                batch_info=rotary_batch_info,
                latent_seq_len=x_latent.shape[1],
                action_seq_len=0,
                device=device,
                dtype=x_latent.dtype,
            )

            hidden = x_latent
            for wrapper in self.mot_attention_wrappers:
                hidden = wrapper.forward_latent_only(
                    hidden,
                    latent_valid_mask=latent_valid_mask,
                    rotary_payload=rotary_payload,
                )

            anchor_hidden = janus_norm(hidden[:, -1, :])
            latent_logits = self.janus.language_model.lm_head(anchor_hidden)
            if self.valid_token_vocab_size is not None and self.valid_token_vocab_size < latent_logits.shape[-1]:
                latent_logits = latent_logits[..., :self.valid_token_vocab_size]
            latent_token_ids = latent_logits.argmax(dim=-1)
            latent_step_embeds = self._embed_janus_token_ids(latent_token_ids.unsqueeze(1)).to(self.dtype)
            latent_embeds_parts.append(latent_step_embeds)
            last_anchor_hidden = anchor_hidden
            last_token_ids = latent_token_ids

        latent_embeds = torch.cat(latent_embeds_parts, dim=1)
        if return_latent_visual_debug:
            return latent_embeds, {
                "latent_anchor_hidden": last_anchor_hidden.detach(),
                "latent_token_ids": last_token_ids.detach(),
            }
        return latent_embeds
