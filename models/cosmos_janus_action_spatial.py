"""2-MoT Cosmos + Janus-action model with spatial-token supervision.

This module keeps Cosmos video conditioning unchanged and removes the old
middle latent expert.  The Janus branch follows a Fast-in-Slow layout:

    slow 24-layer prefix: [image + prompt context] + [spatial token(s)]
    fast 4-layer suffix:  slow hidden A + [time token] + [action tokens]

The spatial tokens are supervised immediately after the slow Janus/Cosmos
layers.  The final action tokens are appended only for the fast AE layers and
supervised with flow matching.
"""

from typing import Optional, Sequence, Tuple
import math
import types
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp

from janus.diffusion import ActionEmbedder
from vae.stacked_resample import StackedDownsample2d, StackedUpsample2d
from vae.wan21_vae_encoder import DEFAULT_WAN21_VAE_CKPT, Wan21VAEEncoder

from models.cosmos_janus_cot import (
    BridgeA1MRoPE,
    BridgeLlama1DRoPE,
    BridgeMRoPEBatchInfo,
    BridgeRotaryEncoder,
    BridgeRotaryPayload,
    LatentNativeAttentionAdapter,
    is_multimodal_bridge_pos_scheme,
    normalize_bridge_pos_scheme,
)


class ActionStandardAttentionAdapter(LatentNativeAttentionAdapter):
    """Action adapter that uses standard Janus layer weights.

    The action pretrain checkpoint stores normal Llama layer names, not the
    old three-expert `_action` suffixed names.  This adapter deliberately uses
    input_layernorm/post_attention_layernorm/self_attn/mlp directly.
    """

    def __init__(self, janus_layer, cosmos_num_heads, cosmos_head_dim, layer_idx: int):
        super().__init__(
            janus_layer=janus_layer,
            cosmos_num_heads=cosmos_num_heads,
            cosmos_head_dim=cosmos_head_dim,
            layer_idx=layer_idx,
        )
        self.branch_name = "action"


class MoTAttentionWrapper2(nn.Module):
    """Two-way MoT bridge attention for Cosmos video + one Janus action branch."""

    def __init__(self, original_attn, action_bridge, interleave_video_qk=False):
        super().__init__()
        self.original_attn = original_attn
        self.action_bridge = action_bridge
        self.interleave_video_qk = bool(interleave_video_qk)

        self.current_x_action = None
        self.current_action_valid_mask = None
        self.current_rotary_payload = None
        self.current_cache_video_kv = False
        self.current_cache_video_kv_detach = True
        self.current_action_tail_token_count = 0
        self.next_x_action = None

        self.cache_video_kv = False
        self.cached_k_v = None
        self.cached_v_v = None
        self.cached_video_grid_thw = None
        self.cached_k_a_prefix = None
        self.cached_v_a_prefix = None
        self.cached_action_prefix_valid_mask = None
        self.detach_video_kv = False
        self.detach_video_kv_for_action = False

        if self.interleave_video_qk:
            head_dim = int(getattr(original_attn, "head_dim", action_bridge.head_dim))
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

    def _interleave_video_qk(self, q_v: torch.Tensor, k_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.interleave_video_qk:
            return q_v, k_v
        perm = self.video_qk_interleave_perm.to(device=q_v.device)
        return q_v.index_select(-1, perm), k_v.index_select(-1, perm)

    def _get_video_grid_thw(self, batch_size: int, device: torch.device, video_size=None) -> torch.Tensor:
        if video_size is not None:
            return torch.tensor(
                [int(video_size.T), int(video_size.H), int(video_size.W)],
                device=device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)

        if self.cached_video_grid_thw is not None:
            cached = self.cached_video_grid_thw.to(device=device, dtype=torch.long)
            if cached.ndim == 1:
                cached = cached.unsqueeze(0).expand(batch_size, -1)
            elif cached.shape[0] == 1 and batch_size != 1:
                cached = cached.expand(batch_size, -1)
            return cached

        raise ValueError("Bridge rotary needs video THW, but no current or cached grid is available.")

    def _store_video_kv_cache(self, k_v, v_v, batch_size: int, device: torch.device, video_size, detach: bool):
        self.cached_k_v = k_v.detach() if detach else k_v
        self.cached_v_v = v_v.detach() if detach else v_v
        self.cached_video_grid_thw = self._get_video_grid_thw(
            batch_size=batch_size,
            device=device,
            video_size=video_size,
        ).detach().clone()

    @staticmethod
    def _query_valid_mask(parts, device, batch_size):
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

    @staticmethod
    def _bridge_sdpa(q, k, v, attn_mask, query_valid_mask=None):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).flatten(2, 3)
        if query_valid_mask is not None:
            out = out * query_valid_mask.unsqueeze(-1).to(out.dtype)
        return out

    @staticmethod
    def _build_action_self_mask(S_a: int, device: torch.device, action_tail_token_count: int = 0) -> torch.Tensor:
        mask = torch.tril(torch.ones(S_a, S_a, dtype=torch.bool, device=device))
        tail = max(0, int(action_tail_token_count or 0))
        if tail > 1:
            if tail > S_a:
                raise ValueError(f"action_tail_token_count={tail} exceeds action sequence length {S_a}.")
            tail_start = S_a - tail
            mask[tail_start:, tail_start:] = True
        return mask

    def _build_bridge_mask(
        self,
        S_v: int,
        S_a: int,
        device: torch.device,
        action_valid_mask: Optional[torch.Tensor] = None,
        action_tail_token_count: int = 0,
    ):
        total = S_v + S_a
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        if S_v > 0:
            mask[:S_v, :S_v] = True
        if S_a > 0:
            mask[S_v:, :S_v] = True
            mask[S_v:, S_v:] = self._build_action_self_mask(
                S_a,
                device,
                action_tail_token_count=action_tail_token_count,
            )
        if action_valid_mask is None:
            return mask.unsqueeze(0).unsqueeze(0)

        action_valid_mask = action_valid_mask.to(device=device, dtype=torch.bool)
        batch_mask = mask.unsqueeze(0).expand(action_valid_mask.shape[0], -1, -1).clone()
        batch_mask[:, :, S_v:] &= action_valid_mask[:, None, :]
        return batch_mask.unsqueeze(1)

    def _build_action_only_mask(
        self,
        S_v: int,
        S_a: int,
        device: torch.device,
        action_valid_mask: Optional[torch.Tensor] = None,
        action_tail_token_count: int = 0,
    ):
        mask = torch.ones(S_a, S_v + S_a, dtype=torch.bool, device=device)
        mask[:, S_v:] = self._build_action_self_mask(
            S_a,
            device,
            action_tail_token_count=action_tail_token_count,
        )
        if action_valid_mask is None:
            return mask.unsqueeze(0).unsqueeze(0)

        action_valid_mask = action_valid_mask.to(device=device, dtype=torch.bool)
        batch_mask = mask.unsqueeze(0).expand(action_valid_mask.shape[0], -1, -1).clone()
        batch_mask[:, :, S_v:] &= action_valid_mask[:, None, :]
        return batch_mask.unsqueeze(1)

    def _build_action_cached_suffix_mask(
        self,
        S_v: int,
        S_prefix: int,
        S_suffix: int,
        device: torch.device,
        prefix_valid_mask: Optional[torch.Tensor] = None,
        suffix_valid_mask: Optional[torch.Tensor] = None,
        action_tail_token_count: int = 0,
    ):
        total_action = S_prefix + S_suffix
        full_self = self._build_action_self_mask(
            total_action,
            device,
            action_tail_token_count=action_tail_token_count,
        )
        suffix_self = full_self[S_prefix:, :]
        mask = torch.ones(S_suffix, S_v + total_action, dtype=torch.bool, device=device)
        mask[:, S_v:] = suffix_self

        batch_size = None
        if prefix_valid_mask is not None:
            batch_size = int(prefix_valid_mask.shape[0])
        if suffix_valid_mask is not None:
            batch_size = int(suffix_valid_mask.shape[0]) if batch_size is None else batch_size
        if batch_size is None:
            return mask.unsqueeze(0).unsqueeze(0)

        batch_mask = mask.unsqueeze(0).expand(batch_size, -1, -1).clone()
        if prefix_valid_mask is not None and S_prefix > 0:
            batch_mask[:, :, S_v:S_v + S_prefix] &= prefix_valid_mask.to(device=device, dtype=torch.bool)[:, None, :]
        if suffix_valid_mask is not None and S_suffix > 0:
            batch_mask[:, :, S_v + S_prefix:] &= suffix_valid_mask.to(device=device, dtype=torch.bool)[:, None, :]
        return batch_mask.unsqueeze(1)

    @staticmethod
    def _apply_action_rotary(q_a, k_a, rotary_payload: Optional[BridgeRotaryPayload]):
        if q_a is None or k_a is None:
            return q_a, k_a
        if rotary_payload is None:
            raise ValueError("rotary_payload is required when action bridge tokens are present.")
        cos = rotary_payload.action_cos if rotary_payload.action_cos is not None else rotary_payload.latent_cos
        sin = rotary_payload.action_sin if rotary_payload.action_sin is not None else rotary_payload.latent_sin
        return BridgeRotaryEncoder.apply_precomputed_rotary(q_a, k_a, cos, sin)

    def forward(self, x, context=None, rope_emb=None, video_size=None, kv_cache_cfg=None):
        q_v, k_v, v_v = self.original_attn.compute_qkv(x, context, rope_emb=rope_emb)
        q_v, k_v = self._interleave_video_qk(q_v, k_v)
        if self.cache_video_kv or self.current_cache_video_kv:
            cache_detach = True if self.cache_video_kv else bool(self.current_cache_video_kv_detach)
            self._store_video_kv_cache(
                k_v=k_v,
                v_v=v_v,
                batch_size=q_v.shape[0],
                device=q_v.device,
                video_size=video_size,
                detach=cache_detach,
            )

        if self.current_x_action is None:
            result = self.original_attn.attn_op(q_v, k_v, v_v)
            return self.original_attn.output_dropout(self.original_attn.output_proj(result))

        q_a, k_a, v_a = self.action_bridge.get_branch_qkv(self.current_x_action)
        q_a, k_a = self._apply_action_rotary(q_a, k_a, self.current_rotary_payload)

        S_v = q_v.shape[1]
        S_a = q_a.shape[1]
        if self.detach_video_kv_for_action and not self.detach_video_kv:
            res_v = self.original_attn.attn_op(q_v, k_v, v_v)
            k = torch.cat([k_v.detach(), k_a], dim=1)
            v = torch.cat([v_v.detach(), v_a], dim=1)
            action_mask = self._build_action_only_mask(
                S_v,
                S_a,
                q_a.device,
                action_valid_mask=self.current_action_valid_mask,
                action_tail_token_count=self.current_action_tail_token_count,
            )
            action_query_valid_mask = self.current_action_valid_mask
            if action_query_valid_mask is None:
                action_query_valid_mask = torch.ones((q_a.shape[0], S_a), device=q_a.device, dtype=torch.bool)
            res_a = self._bridge_sdpa(
                q_a,
                k,
                v,
                attn_mask=action_mask,
                query_valid_mask=action_query_valid_mask,
            )
        else:
            k_video = k_v.detach() if self.detach_video_kv else k_v
            v_video = v_v.detach() if self.detach_video_kv else v_v
            q = torch.cat([q_v, q_a], dim=1)
            k = torch.cat([k_video, k_a], dim=1)
            v = torch.cat([v_video, v_a], dim=1)

            query_valid_mask = self._query_valid_mask(
                [S_v, self.current_action_valid_mask if self.current_action_valid_mask is not None else S_a],
                q.device,
                q.shape[0],
            )
            bridge_mask = self._build_bridge_mask(
                S_v,
                S_a,
                q.device,
                action_valid_mask=self.current_action_valid_mask,
                action_tail_token_count=self.current_action_tail_token_count,
            )
            result = self._bridge_sdpa(q, k, v, attn_mask=bridge_mask, query_valid_mask=query_valid_mask)

            res_v = result[:, :S_v]
            res_a = result[:, S_v:]
        self.next_x_action = self.action_bridge.post_attention(
            self.current_x_action,
            res_a,
            token_valid_mask=self.current_action_valid_mask,
        )
        return self.original_attn.output_dropout(self.original_attn.output_proj(res_v))

    def forward_action_only(
        self,
        x_action: torch.Tensor,
        action_valid_mask: Optional[torch.Tensor] = None,
        rotary_payload: Optional[BridgeRotaryPayload] = None,
        action_tail_token_count: int = 0,
    ) -> torch.Tensor:
        if self.cached_k_v is None or self.cached_v_v is None:
            raise RuntimeError("forward_action_only requires cached Cosmos KV from run_cosmos_once().")

        q_a, k_a, v_a = self.action_bridge.get_branch_qkv(x_action)
        q_a, k_a = self._apply_action_rotary(q_a, k_a, rotary_payload)

        S_v = self.cached_k_v.shape[1]
        S_a = q_a.shape[1]
        k = torch.cat([self.cached_k_v, k_a], dim=1)
        v = torch.cat([self.cached_v_v, v_a], dim=1)
        mask = self._build_action_only_mask(
            S_v,
            S_a,
            q_a.device,
            action_valid_mask=action_valid_mask,
            action_tail_token_count=action_tail_token_count,
        )
        query_valid_mask = action_valid_mask
        if query_valid_mask is None:
            query_valid_mask = torch.ones((q_a.shape[0], S_a), device=q_a.device, dtype=torch.bool)
        result = self._bridge_sdpa(q_a, k, v, attn_mask=mask, query_valid_mask=query_valid_mask)
        return self.action_bridge.post_attention(x_action, result, token_valid_mask=action_valid_mask)

    def forward_action_prefix_and_cache(
        self,
        x_action: torch.Tensor,
        action_valid_mask: Optional[torch.Tensor] = None,
        rotary_payload: Optional[BridgeRotaryPayload] = None,
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

    def forward_action_suffix_only(
        self,
        x_action_suffix: torch.Tensor,
        suffix_valid_mask: Optional[torch.Tensor] = None,
        rotary_payload: Optional[BridgeRotaryPayload] = None,
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
        result = self._bridge_sdpa(q_a, k, v, attn_mask=mask, query_valid_mask=query_valid_mask)
        return self.action_bridge.post_attention(x_action_suffix, result, token_valid_mask=suffix_valid_mask)


class CosmosJanusActionSpatialMoT2Expert(nn.Module):
    """Cosmos video branch + Janus action branch with in-branch spatial tokens."""

    def __init__(self, cosmos_dit, cosmos_vae, janus_model, config):
        super().__init__()
        self.config = config
        self.dtype = torch.bfloat16
        self._video_frozen = False
        self.train_embed_tokens = True
        self.valid_token_vocab_size = int(getattr(config, "valid_token_vocab_size", 0) or 0) or None
        self.bridge_pos_scheme = normalize_bridge_pos_scheme(getattr(config, "bridge_pos_scheme", "llama1d"))
        setattr(self.config, "bridge_pos_scheme", self.bridge_pos_scheme)

        self.janus_image_start_id = getattr(config, "janus_image_start_id", None)
        self.janus_image_end_id = getattr(config, "janus_image_end_id", None)
        self.action_detach_slow_prefix = bool(int(getattr(config, "action_detach_slow_prefix", 0) or 0))
        self.action_detach_video_branch = bool(int(getattr(config, "action_detach_video_branch", 0) or 0))
        self.action_detach_slow_video_branch = bool(
            int(getattr(config, "action_detach_slow_video_branch", 0) or 0)
        )
        setattr(self.config, "action_detach_slow_prefix", int(self.action_detach_slow_prefix))
        setattr(self.config, "action_detach_video_branch", int(self.action_detach_video_branch))
        setattr(self.config, "action_detach_slow_video_branch", int(self.action_detach_slow_video_branch))
        self.state_encoding_mode = str(getattr(config, "state_encoding_mode", "mlp")).lower()
        if self.state_encoding_mode not in ("token", "mlp"):
            raise ValueError(f"state_encoding_mode must be 'token' or 'mlp', got {self.state_encoding_mode!r}.")
        self.state_dim = int(getattr(config, "state_dim", 8) or 8)
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {self.state_dim}.")
        setattr(self.config, "state_dim", self.state_dim)

        self.use_spatial_hidden_sim_loss = bool(
            getattr(config, "use_spatial_hidden_sim_loss", getattr(config, "use_latent_hidden_sim_loss", 0))
        )
        self.spatial_hidden_sim_loss_mode = str(
            getattr(config, "spatial_hidden_sim_loss_mode", getattr(config, "latent_hidden_sim_loss_mode", "siglip"))
        ).lower()
        if self.spatial_hidden_sim_loss_mode not in ("siglip", "wan_vae"):
            raise ValueError(
                "spatial_hidden_sim_loss_mode must be 'siglip' or 'wan_vae', "
                f"got {self.spatial_hidden_sim_loss_mode!r}."
            )
        self.spatial_hidden_sim_pool_mode = str(
            getattr(config, "spatial_hidden_sim_pool_mode", getattr(config, "latent_hidden_sim_pool_mode", "pool"))
        ).lower()
        if self.spatial_hidden_sim_pool_mode not in ("pool", "one_mlp", "mlp"):
            raise ValueError(
                "spatial_hidden_sim_pool_mode must be 'pool', 'one_mlp', or 'mlp', "
                f"got {self.spatial_hidden_sim_pool_mode!r}."
            )
        setattr(self.config, "spatial_hidden_sim_pool_mode", self.spatial_hidden_sim_pool_mode)
        self.use_spatial_hidden_wan_downsample_sim_loss = bool(
            getattr(
                config,
                "use_spatial_hidden_wan_downsample_sim_loss",
                getattr(config, "use_latent_hidden_wan_downsample_sim_loss", 0),
            )
        )
        self.wan21_vae_path = str(getattr(config, "wan21_vae_path", DEFAULT_WAN21_VAE_CKPT))

        self.cosmos_dit = cosmos_dit
        self.cosmos_vae = cosmos_vae
        self.janus = janus_model
        self.janus_dim = self.janus.config.hidden_size
        self.janus_rotary_emb = getattr(self.janus.language_model.model, "rotary_emb", None)
        if self.bridge_pos_scheme == "llama1d" and self.janus_rotary_emb is None:
            raise ValueError("bridge_pos_scheme='llama1d' requires janus.language_model.model.rotary_emb.")

        self.special_token_vocab = list(getattr(config, "special_token_vocab", []) or [])
        if not self.special_token_vocab:
            raise ValueError("config.special_token_vocab must be provided for independent spatial token training.")
        setattr(self.config, "special_token_vocab", list(self.special_token_vocab))
        self.special_token_embedding = nn.Embedding(len(self.special_token_vocab), self.janus_dim).to(self.dtype)
        self.special_token_lm_head = nn.Linear(self.janus_dim, len(self.special_token_vocab), bias=False).to(self.dtype)
        self._init_special_token_modules(getattr(config, "special_token_init_ids", None))

        self.num_cond_input_frames = max(1, int(getattr(config, "num_cond_input_frames", 1) or 1))
        self.num_cond_latent_frames = max(1, int(getattr(config, "num_cond_latent_frames", 1) or 1))
        setattr(self.config, "num_cond_input_frames", self.num_cond_input_frames)
        setattr(self.config, "num_cond_latent_frames", self.num_cond_latent_frames)

        self.state_mlp_embedder = ActionEmbedder(action_size=self.state_dim, hidden_size=self.janus_dim)
        self._init_state_mlp_embedder()
        self.spatial_hidden_sim_score_mlps = nn.ModuleList()
        self.spatial_hidden_sim_proj_mlps = nn.ModuleList()
        if (
            self.use_spatial_hidden_sim_loss
            and self.spatial_hidden_sim_loss_mode == "siglip"
            and self.spatial_hidden_sim_pool_mode != "pool"
        ):
            mlp_count = 1 if self.spatial_hidden_sim_pool_mode == "one_mlp" else self._resolve_spatial_token_count(
                getattr(config, "total_spatial_tokens", getattr(config, "total_latent_tokens", 1))
            )
            self.spatial_hidden_sim_score_mlps = nn.ModuleList(
                [self._make_spatial_hidden_sim_mlp(self.janus_dim, 1) for _ in range(mlp_count)]
            )
            self.spatial_hidden_sim_proj_mlps = nn.ModuleList(
                [self._make_spatial_hidden_sim_mlp(self.janus_dim, self.janus_dim) for _ in range(mlp_count)]
            )

        self.spatial_hidden_wan_upsampler = None
        if self.use_spatial_hidden_sim_loss and self.spatial_hidden_sim_loss_mode == "wan_vae":
            self.spatial_hidden_wan_upsampler = StackedUpsample2d(
                channels=Wan21VAEEncoder.latent_channels,
                spatial_size=32,
                input_channels=self.janus_dim,
            ).to(self.dtype)
        self.spatial_hidden_wan_downsampler = None
        if self.use_spatial_hidden_wan_downsample_sim_loss:
            self.spatial_hidden_wan_downsampler = StackedDownsample2d(
                channels=Wan21VAEEncoder.latent_channels,
                spatial_size=32,
                output_channels=self.janus_dim,
            ).to(self.dtype)
        self.__dict__["spatial_hidden_wan_encoder"] = None

        self.cosmos_crossattn_dim = self.cosmos_dit.blocks[0].cross_attn.context_dim
        if self.janus_dim != self.cosmos_crossattn_dim:
            self.text_proj = nn.Linear(self.janus_dim, self.cosmos_crossattn_dim, bias=True).to(self.dtype)
            nn.init.normal_(self.text_proj.weight, std=0.02)
            nn.init.zeros_(self.text_proj.bias)
        else:
            self.text_proj = nn.Identity()

        self.mot_attention_wrappers = nn.ModuleList()
        self.slow_mot_attention_wrappers = nn.ModuleList()
        self.fast_mot_attention_wrappers = nn.ModuleList()
        self.fast_action_layers = nn.ModuleList()
        self.bridge_rotary_encoder = None
        cosmos_blocks = list(self.cosmos_dit.blocks)
        janus_layers = self.janus.language_model.model.layers
        self.slow_mot_layer_count = min(len(cosmos_blocks), len(janus_layers))
        self.fast_mot_layer_count = max(0, len(cosmos_blocks) - self.slow_mot_layer_count)
        if self.fast_mot_layer_count > len(janus_layers):
            raise ValueError(
                "FiS fast layer count exceeds available Janus layers: "
                f"fast={self.fast_mot_layer_count}, janus={len(janus_layers)}."
            )
        if self.fast_mot_layer_count > 0:
            fast_start = len(janus_layers) - self.fast_mot_layer_count
            self.fast_action_layers = nn.ModuleList(
                [copy.deepcopy(janus_layers[fast_start + j]) for j in range(self.fast_mot_layer_count)]
            )

        for i, cosmos_block in enumerate(cosmos_blocks):
            if i < self.slow_mot_layer_count:
                janus_layer = janus_layers[i]
                wrapper_list = self.slow_mot_attention_wrappers
                detach_video_kv_for_action = self.action_detach_slow_video_branch
            else:
                fast_idx = i - self.slow_mot_layer_count
                if fast_idx >= self.fast_mot_layer_count:
                    break
                janus_layer = self.fast_action_layers[fast_idx]
                wrapper_list = self.fast_mot_attention_wrappers
                detach_video_kv_for_action = self.action_detach_video_branch
            actual_block = cosmos_block
            if hasattr(cosmos_block, "_checkpoint_wrapped_module"):
                actual_block = cosmos_block._checkpoint_wrapped_module
            elif hasattr(cosmos_block, "module"):
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
                    "All MoT bridge layers must share one head_dim, "
                    f"got {block_head_dim} after {self.bridge_rotary_encoder.head_dim}."
                )

            action_bridge = ActionStandardAttentionAdapter(
                janus_layer=janus_layer,
                cosmos_num_heads=actual_block.self_attn.n_heads,
                cosmos_head_dim=block_head_dim,
                layer_idx=i,
            ).to(self.dtype)
            mot_attn = MoTAttentionWrapper2(
                actual_block.self_attn,
                action_bridge,
                interleave_video_qk=self.bridge_pos_scheme == "mrope_interleave",
            )
            mot_attn.detach_video_kv_for_action = detach_video_kv_for_action
            actual_block.self_attn = mot_attn
            original_forward = actual_block.forward

            def make_new_forward(orig_fwd):
                def new_forward(self_block, x_B_T_H_W_D, emb_B_T_D, crossattn_emb, x_action=None, **kwargs):
                    x_action_valid_mask = kwargs.pop("x_action_valid_mask", None)
                    x_rotary_payload = kwargs.pop("x_rotary_payload", None)
                    mot_cache_video_kv = kwargs.pop("mot_cache_video_kv", False)
                    mot_cache_video_kv_detach = kwargs.pop("mot_cache_video_kv_detach", True)
                    mot_action_tail_token_count = kwargs.pop("mot_action_tail_token_count", 0)
                    self_block.self_attn.current_x_action = x_action
                    self_block.self_attn.current_action_valid_mask = x_action_valid_mask
                    self_block.self_attn.current_rotary_payload = x_rotary_payload
                    self_block.self_attn.current_cache_video_kv = mot_cache_video_kv
                    self_block.self_attn.current_cache_video_kv_detach = mot_cache_video_kv_detach
                    self_block.self_attn.current_action_tail_token_count = int(mot_action_tail_token_count or 0)
                    try:
                        out_video = orig_fwd(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, **kwargs)
                        out_action = self_block.self_attn.next_x_action
                        return out_video, out_action
                    finally:
                        self_block.self_attn.current_x_action = None
                        self_block.self_attn.current_action_valid_mask = None
                        self_block.self_attn.current_rotary_payload = None
                        self_block.self_attn.current_cache_video_kv = False
                        self_block.self_attn.current_cache_video_kv_detach = True
                        self_block.self_attn.current_action_tail_token_count = 0
                        self_block.self_attn.next_x_action = None
                return new_forward

            actual_block.forward = types.MethodType(make_new_forward(original_forward), actual_block)
            self.mot_attention_wrappers.append(mot_attn)
            wrapper_list.append(mot_attn)

        if self.bridge_rotary_encoder is None:
            raise ValueError("Failed to initialize MoT wrappers.")
        if self.fast_mot_layer_count == 0:
            raise ValueError(
                "FiS action architecture requires extra Cosmos layers after Janus slow layers, "
                f"got cosmos={len(cosmos_blocks)}, janus={len(janus_layers)}."
            )

    def _sample_cosmos_train_sigma(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        shift = 5.0
        u = torch.sigmoid(torch.randn((batch_size,), device=device, dtype=torch.float32))
        sigma = shift * u / (1.0 + (shift - 1.0) * u)
        return u.to(dtype=self.dtype), sigma.to(dtype=self.dtype)

    def _init_state_mlp_embedder(self):
        nn.init.normal_(self.state_mlp_embedder.mlp.fc1.weight, std=0.02)
        nn.init.normal_(self.state_mlp_embedder.mlp.fc2.weight, std=0.02)
        nn.init.constant_(self.state_mlp_embedder.mlp.fc1.bias, 0)
        nn.init.constant_(self.state_mlp_embedder.mlp.fc2.bias, 0)

    def _make_spatial_hidden_sim_mlp(self, in_features: int, out_features: int) -> nn.Module:
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        return Mlp(
            in_features=int(in_features),
            hidden_features=self.janus_dim,
            out_features=int(out_features),
            act_layer=approx_gelu,
            drop=0,
        ).to(self.dtype)

    def _init_special_token_modules(self, source_ids_by_token: Optional[Sequence[Sequence[int]]]):
        if source_ids_by_token is None:
            raise ValueError("config.special_token_init_ids must be provided for special token initialization.")
        if len(source_ids_by_token) != len(self.special_token_vocab):
            raise ValueError(
                "special_token_init_ids length must match special_token_vocab length: "
                f"{len(source_ids_by_token)} != {len(self.special_token_vocab)}."
            )
        base_embed = self.janus.language_model.model.embed_tokens.weight
        base_lm_head = getattr(self.janus.language_model, "lm_head", None)
        base_lm_head_weight = None if base_lm_head is None else getattr(base_lm_head, "weight", None)
        with torch.no_grad():
            for row_idx, (token_text, source_ids) in enumerate(zip(self.special_token_vocab, source_ids_by_token)):
                ids = [int(token_id) for token_id in source_ids]
                if not ids:
                    raise ValueError(f"Special token {token_text!r} has no source token ids.")
                if min(ids) < 0 or max(ids) >= int(base_embed.shape[0]):
                    raise ValueError(
                        f"Source ids for {token_text!r} exceed base embedding rows: "
                        f"ids={ids}, rows={int(base_embed.shape[0])}."
                    )
                embed_ids = torch.tensor(ids, device=base_embed.device, dtype=torch.long)
                embed_init = base_embed.index_select(0, embed_ids).mean(dim=0)
                self.special_token_embedding.weight[row_idx].copy_(
                    embed_init.to(device=self.special_token_embedding.weight.device, dtype=self.special_token_embedding.weight.dtype)
                )

                if base_lm_head_weight is not None:
                    if max(ids) >= int(base_lm_head_weight.shape[0]):
                        raise ValueError(
                            f"Source ids for {token_text!r} exceed base lm_head rows: "
                            f"ids={ids}, rows={int(base_lm_head_weight.shape[0])}."
                        )
                    head_ids = torch.tensor(ids, device=base_lm_head_weight.device, dtype=torch.long)
                    head_init = base_lm_head_weight.index_select(0, head_ids).mean(dim=0)
                else:
                    head_init = embed_init
                self.special_token_lm_head.weight[row_idx].copy_(
                    head_init.to(device=self.special_token_lm_head.weight.device, dtype=self.special_token_lm_head.weight.dtype)
                )

    def _bridge_parameter_ids(self):
        ids = set()
        for wrapper in self.mot_attention_wrappers:
            ids.update(id(param) for param in wrapper.action_bridge.parameters())
        return ids

    def _set_video_backbone_requires_grad(self, requires_grad: bool):
        bridge_ids = self._bridge_parameter_ids()
        for param in self.cosmos_dit.parameters():
            if id(param) in bridge_ids:
                continue
            param.requires_grad = requires_grad

    def _set_text_proj_requires_grad(self, requires_grad: bool):
        for param in self.text_proj.parameters():
            param.requires_grad = requires_grad

    def freeze_video_backbone(self):
        self._video_frozen = True
        self._set_video_backbone_requires_grad(False)
        self._set_text_proj_requires_grad(False)
        if hasattr(self.cosmos_vae, "parameters"):
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

    def set_cosmos_inference_runtime(self, sample_scheduler, shift=1, use_kerras_sigma_at_inference=False):
        if sample_scheduler is None:
            raise ValueError("sample_scheduler must not be None.")
        self._cosmos_inference_sample_scheduler = sample_scheduler
        self._cosmos_inference_shift = shift
        self._cosmos_inference_use_kerras_sigma_at_inference = bool(use_kerras_sigma_at_inference)
        return self

    def set_cosmos_inference_runtime_from_wrapper(self, cosmos_wrapper):
        sample_scheduler = getattr(cosmos_wrapper, "sample_scheduler", None)
        if sample_scheduler is None:
            raise AttributeError("cosmos_wrapper has no sample_scheduler.")
        wrapper_config = getattr(cosmos_wrapper, "config", None)
        shift = getattr(wrapper_config, "shift", 1)
        use_kerras_sigma = bool(getattr(wrapper_config, "use_kerras_sigma_at_inference", False))
        return self.set_cosmos_inference_runtime(sample_scheduler, shift, use_kerras_sigma)

    def _require_cosmos_inference_runtime(self):
        sample_scheduler = getattr(self, "_cosmos_inference_sample_scheduler", None)
        if sample_scheduler is None:
            raise RuntimeError("Call set_cosmos_inference_runtime(...) before inference.")
        shift = getattr(self, "_cosmos_inference_shift", 1)
        use_kerras_sigma = bool(getattr(self, "_cosmos_inference_use_kerras_sigma_at_inference", False))
        num_train_timesteps = float(getattr(sample_scheduler.config, "num_train_timesteps", 1000))
        return sample_scheduler, shift, use_kerras_sigma, num_train_timesteps

    def _clear_cached_video_kv(self):
        for wrapper in self.mot_attention_wrappers:
            wrapper.cache_video_kv = False
            wrapper.cached_k_v = None
            wrapper.cached_v_v = None
            wrapper.cached_video_grid_thw = None
            wrapper.cached_k_a_prefix = None
            wrapper.cached_v_a_prefix = None
            wrapper.cached_action_prefix_valid_mask = None
            wrapper.current_rotary_payload = None
            wrapper.current_cache_video_kv = False
            wrapper.current_cache_video_kv_detach = True
            wrapper.current_action_tail_token_count = 0

    def _clear_cached_action_prefix_kv(self):
        for wrapper in self.mot_attention_wrappers:
            wrapper.cached_k_a_prefix = None
            wrapper.cached_v_a_prefix = None
            wrapper.cached_action_prefix_valid_mask = None

    def _get_cached_action_prefix_valid_mask(self) -> torch.Tensor:
        for wrapper in self.mot_attention_wrappers:
            if wrapper.cached_action_prefix_valid_mask is not None:
                return wrapper.cached_action_prefix_valid_mask
        raise RuntimeError("No cached action prefix KV is available.")

    def _get_cached_action_prefix_len(self) -> int:
        for wrapper in self.mot_attention_wrappers:
            if wrapper.cached_k_a_prefix is not None:
                return int(wrapper.cached_k_a_prefix.shape[1])
        return 0

    def _get_cached_fast_prefix_valid_mask(self) -> torch.Tensor:
        for wrapper in self.fast_mot_attention_wrappers:
            if wrapper.cached_action_prefix_valid_mask is not None:
                return wrapper.cached_action_prefix_valid_mask
        raise RuntimeError("No cached fast action prefix KV is available.")

    def _get_cached_fast_prefix_len(self) -> int:
        for wrapper in self.fast_mot_attention_wrappers:
            if wrapper.cached_k_a_prefix is not None:
                return int(wrapper.cached_k_a_prefix.shape[1])
        return 0

    def _clear_cached_fast_prefix_kv(self):
        for wrapper in self.fast_mot_attention_wrappers:
            wrapper.cached_k_a_prefix = None
            wrapper.cached_v_a_prefix = None
            wrapper.cached_action_prefix_valid_mask = None

    def _slow_forward_prefix(
        self,
        x_prefix: torch.Tensor,
        prefix_valid_mask: torch.Tensor,
        rotary_payload: BridgeRotaryPayload,
        cache_prefix_kv: bool = False,
    ) -> torch.Tensor:
        hidden = x_prefix
        for wrapper in self.slow_mot_attention_wrappers:
            hidden = wrapper.forward_action_prefix_and_cache(
                hidden,
                action_valid_mask=prefix_valid_mask,
                rotary_payload=rotary_payload,
                action_tail_token_count=0,
                append_to_cache=cache_prefix_kv,
            )
        return hidden

    def _fast_forward_full(
        self,
        x_fast: torch.Tensor,
        fast_valid_mask: torch.Tensor,
        rotary_payload: BridgeRotaryPayload,
        action_tail_token_count: int,
        cache_prefix_kv: bool = False,
    ) -> torch.Tensor:
        hidden = x_fast
        for wrapper in self.fast_mot_attention_wrappers:
            hidden = wrapper.forward_action_prefix_and_cache(
                hidden,
                action_valid_mask=fast_valid_mask,
                rotary_payload=rotary_payload,
                action_tail_token_count=action_tail_token_count,
                append_to_cache=cache_prefix_kv,
            )
        return hidden

    def _fast_forward_suffix_cached(
        self,
        suffix: torch.Tensor,
        suffix_valid_mask: torch.Tensor,
        suffix_rotary_payload: BridgeRotaryPayload,
        action_tail_token_count: int,
    ) -> torch.Tensor:
        hidden = suffix
        for wrapper in self.fast_mot_attention_wrappers:
            hidden = wrapper.forward_action_suffix_only(
                hidden,
                suffix_valid_mask=suffix_valid_mask,
                rotary_payload=suffix_rotary_payload,
                action_tail_token_count=action_tail_token_count,
            )
        return hidden

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
        temporal_compression_factor = int(temporal_compression_factor or 4)
        return 1 + (pixel_frames - 1) // temporal_compression_factor

    def _num_real_condition_latents(self, real_video_latent_frames: int) -> int:
        real_video_latent_frames = int(real_video_latent_frames)
        if real_video_latent_frames < 1:
            raise ValueError(f"real_video_latent_frames must be positive, got {real_video_latent_frames}.")
        return min(self.num_cond_latent_frames, real_video_latent_frames)

    def _prepare_cosmos_crossattn_emb(
        self,
        multimodal_embeds: torch.Tensor,
        cosmos_text_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cosmos_text_embeddings is None:
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
                f"cosmos_text_embeddings must have batch={batch_size} and seq=512, "
                f"got {tuple(cosmos_text_embeddings.shape)}."
            )
        cosmos_text_embeddings = cosmos_text_embeddings.to(
            device=multimodal_embeds.device,
            dtype=self.dtype,
        )
        if cache_dim == projected_dim:
            return cosmos_text_embeddings
        if cache_dim == raw_dim:
            if not bool(getattr(self.cosmos_dit, "use_crossattn_projection", False)):
                raise ValueError("Received raw Cosmos text embeddings but crossattn projection is disabled.")
            projected = self.cosmos_dit.crossattn_proj(cosmos_text_embeddings)
            return projected.to(dtype=self.dtype)
        raise ValueError(f"Unsupported cached Cosmos text embedding dim {cache_dim}.")

    def _embed_janus_token_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        safe_ids = input_ids.clone()
        safe_ids[safe_ids < 0] = 0
        return self.janus.language_model.model.embed_tokens(safe_ids)

    def _embed_special_token_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.numel() > 0:
            min_id = int(input_ids.min().item())
            max_id = int(input_ids.max().item())
            if min_id < 0 or max_id >= len(self.special_token_vocab):
                raise ValueError(
                    f"Special token ids must be in [0, {len(self.special_token_vocab) - 1}], "
                    f"got min={min_id}, max={max_id}."
                )
        return self.special_token_embedding(input_ids)

    def _special_token_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.special_token_lm_head(hidden_states)

    def _janus_vision_dtype(self) -> torch.dtype:
        try:
            return next(self.janus.vision_model.parameters()).dtype
        except StopIteration:
            return self.dtype

    def _encode_janus_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError(f"Janus pixel values must have shape [B, N, C, H, W], got {tuple(pixel_values.shape)}.")
        B, N = pixel_values.shape[:2]
        flat = pixel_values.reshape(B * N, *pixel_values.shape[2:]).to(dtype=self._janus_vision_dtype())
        image_embeds = self.janus.aligner(self.janus.vision_model(flat))
        return image_embeds.reshape(B, N, image_embeds.shape[1], image_embeds.shape[2])

    def _build_spatial_hidden_siglip_targets(
        self,
        future_pixel_values: torch.Tensor,
        batch_size: int,
        spatial_token_count: int,
        hidden_dim: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            siglip_tokens = self._encode_janus_pixel_values(future_pixel_values)
        if siglip_tokens.ndim != 4 or siglip_tokens.shape[0] != batch_size or siglip_tokens.shape[1] != 1:
            raise ValueError(
                "SigLIP hidden sim expects encoded future image tokens shaped [B, 1, T, D], "
                f"got {tuple(siglip_tokens.shape)}."
            )
        siglip_tokens = siglip_tokens[:, 0, :, :].to(dtype=self.dtype)
        if siglip_tokens.shape[-1] != hidden_dim:
            raise ValueError(
                f"SigLIP token dim must match spatial hidden dim {hidden_dim}, got {siglip_tokens.shape[-1]}."
            )

        if self.spatial_hidden_sim_pool_mode == "pool":
            pooled = siglip_tokens.mean(dim=1)
            return pooled[:, None, :].expand(batch_size, spatial_token_count, hidden_dim)

        if len(self.spatial_hidden_sim_score_mlps) == 0 or len(self.spatial_hidden_sim_proj_mlps) == 0:
            raise RuntimeError(
                f"spatial_hidden_sim_pool_mode={self.spatial_hidden_sim_pool_mode!r} requires initialized MLP modules."
            )
        if self.spatial_hidden_sim_pool_mode == "mlp" and spatial_token_count > len(self.spatial_hidden_sim_score_mlps):
            raise ValueError(
                f"Need {spatial_token_count} per-token SigLIP pooling MLPs, "
                f"but only {len(self.spatial_hidden_sim_score_mlps)} were initialized."
            )

        targets = []
        for token_idx in range(spatial_token_count):
            mlp_idx = 0 if self.spatial_hidden_sim_pool_mode == "one_mlp" else token_idx
            scores = self.spatial_hidden_sim_score_mlps[mlp_idx](siglip_tokens).squeeze(-1)
            weights = torch.softmax(scores.to(torch.float32), dim=1).to(dtype=siglip_tokens.dtype)
            pooled = torch.sum(siglip_tokens * weights.unsqueeze(-1), dim=1)
            targets.append(self.spatial_hidden_sim_proj_mlps[mlp_idx](pooled))
        return torch.stack(targets, dim=1)

    def encode_state_values(self, state_values: torch.Tensor) -> torch.Tensor:
        if state_values.ndim not in (2, 3):
            raise ValueError(f"state_values must have shape [B, 8] or [B, N, 8], got {tuple(state_values.shape)}.")
        if state_values.shape[-1] != self.state_dim:
            raise ValueError(
                f"state_values last dimension must be {self.state_dim}, got {state_values.shape[-1]}."
            )
        values = state_values.to(device=self.state_mlp_embedder.mlp.fc1.weight.device, dtype=self.dtype)
        if values.ndim == 2:
            return self.state_mlp_embedder(values).unsqueeze(1)
        B, N = values.shape[:2]
        flat = values.reshape(B * N, values.shape[-1])
        return self.state_mlp_embedder(flat).reshape(B, N, 1, self.janus_dim)

    def prepare_action_context_embeds(
        self,
        janus_input_ids: torch.Tensor,
        now_state: Optional[torch.Tensor],
        janus_pixel_values: torch.Tensor,
        janus_images_seq_mask: torch.Tensor,
        janus_state_seq_mask: Optional[torch.Tensor],
        janus_images_emb_mask: torch.Tensor,
    ) -> torch.Tensor:
        B = janus_input_ids.shape[0]
        device = janus_input_ids.device
        context_embeds = self._embed_janus_token_ids(janus_input_ids)

        image_embeds = self._encode_janus_pixel_values(janus_pixel_values)
        _, n_images, n_image_tokens, D = image_embeds.shape
        flat_image_embeds = image_embeds.reshape(B, n_images * n_image_tokens, D)
        image_seq_mask = janus_images_seq_mask.to(device=device, dtype=torch.bool)
        image_emb_mask = janus_images_emb_mask.to(device=device, dtype=torch.bool).reshape(B, -1)
        if int(image_seq_mask.sum().item()) != int(image_emb_mask.sum().item()):
            raise ValueError("janus_images_seq_mask token count does not match janus_images_emb_mask token count.")
        context_embeds[image_seq_mask] = flat_image_embeds[image_emb_mask]

        if janus_state_seq_mask is not None:
            state_seq_mask = janus_state_seq_mask.to(device=device, dtype=torch.bool)
        else:
            state_seq_mask = None
        if now_state is not None:
            if state_seq_mask is None:
                raise ValueError("now_state was provided but janus_state_seq_mask is None.")
            if self.state_encoding_mode == "mlp":
                state_embeds = self.encode_state_values(now_state)
            else:
                state_embeds = self._embed_janus_token_ids(now_state)
            state_token_count = state_embeds.shape[1]
            state_counts = state_seq_mask.sum(dim=1)
            if not torch.all(state_counts == state_token_count):
                raise ValueError(
                    "Each row of janus_state_seq_mask must contain exactly "
                    f"{state_token_count} state tokens; got {state_counts.tolist()}."
                )
            context_embeds[state_seq_mask] = state_embeds.reshape(B * state_token_count, D)
        elif state_seq_mask is not None and bool(state_seq_mask.any().item()):
            raise ValueError("janus_state_seq_mask contains placeholders but now_state is None.")
        return context_embeds

    def _build_action_sequence(
        self,
        context_embeds: torch.Tensor,
        spatial_token_embeds: Optional[torch.Tensor] = None,
        action_latent: Optional[torch.Tensor] = None,
        timestep_act: Optional[torch.Tensor] = None,
        janus_left_pad_lens: Optional[torch.Tensor] = None,
        janus_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S_context, _ = context_embeds.shape
        device = context_embeds.device
        pieces = []
        valid_pieces = []

        if janus_attention_mask is not None:
            context_valid = janus_attention_mask.to(device=device, dtype=torch.bool)
            if context_valid.shape != (B, S_context):
                raise ValueError(
                    f"janus_attention_mask shape {tuple(context_valid.shape)} does not match {(B, S_context)}."
                )
        elif janus_left_pad_lens is not None:
            janus_left_pad_lens = janus_left_pad_lens.to(device=device, dtype=torch.long)
            context_pos = torch.arange(S_context, device=device).unsqueeze(0)
            context_valid = context_pos >= janus_left_pad_lens.unsqueeze(1)
        else:
            context_valid = torch.ones((B, S_context), device=device, dtype=torch.bool)

        pieces.append(context_embeds * context_valid.unsqueeze(-1).to(context_embeds.dtype))
        valid_pieces.append(context_valid)

        if spatial_token_embeds is not None and spatial_token_embeds.shape[1] > 0:
            spatial_token_embeds = spatial_token_embeds.to(device=device, dtype=context_embeds.dtype)
            pieces.append(spatial_token_embeds)
            valid_pieces.append(torch.ones((B, spatial_token_embeds.shape[1]), device=device, dtype=torch.bool))

        if action_latent is not None:
            if timestep_act is None:
                raise ValueError("timestep_act is required when action_latent is provided.")
            time_tokens = self.janus.t_embedder(timestep_act).unsqueeze(1)
            action_tokens = self.janus.x_embedder(action_latent.to(dtype=context_embeds.dtype))
            pieces.extend([time_tokens, action_tokens])
            valid_pieces.append(torch.ones((B, 1), device=device, dtype=torch.bool))
            valid_pieces.append(torch.ones((B, action_tokens.shape[1]), device=device, dtype=torch.bool))

        return torch.cat(pieces, dim=1), torch.cat(valid_pieces, dim=1)

    @staticmethod
    def _last_valid_context_indices(action_valid_mask: torch.Tensor, context_len: int) -> torch.Tensor:
        context_valid = action_valid_mask[:, :context_len].to(dtype=torch.bool)
        if not bool(context_valid.any(dim=1).all().item()):
            raise ValueError("Each batch item must have at least one valid Janus context token.")
        positions = torch.arange(context_len, device=action_valid_mask.device, dtype=torch.long).unsqueeze(0)
        return torch.where(context_valid, positions, torch.zeros_like(positions)).max(dim=1).values

    def _spatial_anchor_indices(
        self,
        action_valid_mask: torch.Tensor,
        context_len: int,
        spatial_token_count: int,
    ) -> torch.Tensor:
        B = action_valid_mask.shape[0]
        device = action_valid_mask.device
        anchors = torch.empty((B, spatial_token_count), device=device, dtype=torch.long)
        anchors[:, 0] = self._last_valid_context_indices(action_valid_mask, context_len)
        for token_idx in range(1, spatial_token_count):
            anchors[:, token_idx] = context_len + token_idx - 1
        return anchors

    def _gather_spatial_anchor_hiddens(
        self,
        action_hidden_norm: torch.Tensor,
        action_valid_mask: torch.Tensor,
        context_len: int,
        spatial_token_count: int,
    ) -> torch.Tensor:
        anchors = self._spatial_anchor_indices(action_valid_mask, context_len, spatial_token_count)
        return action_hidden_norm.gather(
            1,
            anchors.unsqueeze(-1).expand(-1, -1, action_hidden_norm.shape[-1]),
        )

    def _infer_janus_image_grid_thw(
        self,
        batch_size: int,
        device: torch.device,
        janus_images_emb_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        grid_h = grid_w = None
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
            num_image_tokens = int(janus_images_emb_mask.shape[-1]) if janus_images_emb_mask is not None else 576
            side = int(math.isqrt(num_image_tokens))
            grid_h = grid_w = side
        return torch.tensor([1, grid_h, grid_w], device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    def _build_action_rotary_batch_info(
        self,
        x_action: torch.Tensor,
        action_valid_mask: Optional[torch.Tensor],
        janus_left_pad_lens: Optional[torch.Tensor] = None,
        janus_images_seq_mask: Optional[torch.Tensor] = None,
        janus_images_emb_mask: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> BridgeMRoPEBatchInfo:
        batch_size = x_action.shape[0]
        device = x_action.device
        use_mrope = is_multimodal_bridge_pos_scheme(self.bridge_pos_scheme)
        image_grid_thw = None
        latent_image_token_mask = None
        if use_mrope:
            image_grid_thw = self._infer_janus_image_grid_thw(
                batch_size=batch_size,
                device=device,
                janus_images_emb_mask=janus_images_emb_mask,
            )
            latent_image_token_mask = torch.zeros((batch_size, x_action.shape[1]), device=device, dtype=torch.bool)
            if janus_images_seq_mask is not None:
                seq_mask = janus_images_seq_mask.to(device=device, dtype=torch.bool)
                seq_width = min(seq_mask.shape[1], x_action.shape[1])
                latent_image_token_mask[:, :seq_width] = seq_mask[:, :seq_width]

        return BridgeMRoPEBatchInfo(
            image_grid_thw=image_grid_thw,
            latent_image_token_mask=latent_image_token_mask,
            action_image_token_mask=None,
            latent_valid_mask=None if action_valid_mask is None else action_valid_mask.to(device=device, dtype=torch.bool),
            latent_left_pad_lens=None
            if janus_left_pad_lens is None
            else janus_left_pad_lens.to(device=device, dtype=torch.long),
            video_grid_thw=None
            if video_grid_thw is None or not use_mrope
            else video_grid_thw.to(device=device, dtype=torch.long),
        )

    def _build_bridge_rotary_payload(
        self,
        batch_info: BridgeMRoPEBatchInfo,
        action_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BridgeRotaryPayload:
        batch_size = None
        for tensor in (batch_info.latent_valid_mask, batch_info.latent_image_token_mask, batch_info.video_grid_thw):
            if tensor is not None:
                batch_size = int(tensor.shape[0])
                break
        if batch_size is None:
            raise ValueError("Unable to infer batch size for bridge rotary payload.")
        return self.bridge_rotary_encoder.prepare_batch_rotary(
            batch_info=batch_info,
            batch_size=batch_size,
            latent_seq_len=action_seq_len,
            action_seq_len=0,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _slice_bridge_rotary_payload(
        rotary_payload: BridgeRotaryPayload,
        start: int,
        end: int,
    ) -> BridgeRotaryPayload:
        start = int(start)
        end = int(end)

        def maybe_slice(tensor):
            if tensor is None:
                return None
            return tensor[:, start:end]

        return BridgeRotaryPayload(
            latent_cos=maybe_slice(rotary_payload.latent_cos),
            latent_sin=maybe_slice(rotary_payload.latent_sin),
            action_cos=maybe_slice(rotary_payload.action_cos),
            action_sin=maybe_slice(rotary_payload.action_sin),
        )

    def _build_action_rotary_payload_for_length(
        self,
        batch_size: int,
        total_action_len: int,
        action_valid_mask: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        janus_left_pad_lens: Optional[torch.Tensor] = None,
        janus_images_seq_mask: Optional[torch.Tensor] = None,
        janus_images_emb_mask: Optional[torch.Tensor] = None,
    ) -> BridgeRotaryPayload:
        dummy_action = torch.empty(
            (batch_size, int(total_action_len), self.janus_dim),
            device=device,
            dtype=dtype,
        )
        cached_video_grid_thw = None
        if is_multimodal_bridge_pos_scheme(self.bridge_pos_scheme):
            cached_video_grid_thw = self._get_cached_bridge_video_grid_thw(batch_size, device)
        rotary_info = self._build_action_rotary_batch_info(
            x_action=dummy_action,
            action_valid_mask=action_valid_mask,
            janus_left_pad_lens=janus_left_pad_lens,
            janus_images_seq_mask=janus_images_seq_mask,
            janus_images_emb_mask=janus_images_emb_mask,
            video_grid_thw=cached_video_grid_thw,
        )
        return self._build_bridge_rotary_payload(
            rotary_info,
            action_seq_len=int(total_action_len),
            device=device,
            dtype=dtype,
        )

    def _get_cached_bridge_video_grid_thw(self, batch_size: int, device: torch.device) -> torch.Tensor:
        for wrapper in self.mot_attention_wrappers:
            if wrapper.cached_video_grid_thw is not None:
                cached = wrapper.cached_video_grid_thw.to(device=device, dtype=torch.long)
                if cached.ndim == 1:
                    cached = cached.unsqueeze(0).expand(batch_size, -1)
                elif cached.shape[0] == 1 and batch_size != 1:
                    cached = cached.expand(batch_size, -1)
                return cached
        raise ValueError("No cached video grid THW is available.")

    def joint_denoise_step(
        self,
        video_latent: torch.Tensor,
        action_latent: torch.Tensor,
        spatial_token_embeds: torch.Tensor,
        context_embeds: torch.Tensor,
        timestep_vid: torch.Tensor,
        timestep_act: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        janus_left_pad_lens: Optional[torch.Tensor] = None,
        janus_attention_mask: Optional[torch.Tensor] = None,
        janus_images_seq_mask: Optional[torch.Tensor] = None,
        janus_images_emb_mask: Optional[torch.Tensor] = None,
        cosmos_text_embeddings: Optional[torch.Tensor] = None,
        cosmos_context_embeds: Optional[torch.Tensor] = None,
        cache_video_kv: bool = False,
        cache_video_kv_detach: bool = True,
    ):
        video_latent = video_latent.to(self.dtype)
        action_latent = action_latent.to(self.dtype)
        spatial_token_embeds = spatial_token_embeds.to(self.dtype)
        context_embeds = context_embeds.to(self.dtype)

        B = video_latent.shape[0]
        device = video_latent.device
        if fps is None:
            fps = torch.full((B,), 10.0, device=device, dtype=self.dtype)
        elif not isinstance(fps, torch.Tensor):
            fps = torch.full((B,), float(fps), device=device, dtype=self.dtype)

        x_slow, slow_valid_mask = self._build_action_sequence(
            context_embeds=context_embeds,
            spatial_token_embeds=spatial_token_embeds,
            janus_left_pad_lens=janus_left_pad_lens,
            janus_attention_mask=janus_attention_mask,
        )

        scale = getattr(self.cosmos_dit, "timestep_scale", 1.0)
        cosmos_t_scaled = timestep_vid * 1000.0 * scale
        cosmos_t_unsqueeze = cosmos_t_scaled.unsqueeze(1) if cosmos_t_scaled.ndim == 1 else cosmos_t_scaled
        use_wan_fp32 = getattr(self.cosmos_dit, "use_wan_fp32_strategy", False)
        cosmos_t_unsqueeze = cosmos_t_unsqueeze.to(torch.float32 if use_wan_fp32 else self.dtype)
        with torch.amp.autocast("cuda", enabled=use_wan_fp32, dtype=torch.float32):
            t_emb, adaln_lora = self.cosmos_dit.t_embedder(cosmos_t_unsqueeze)
            t_emb = self.cosmos_dit.t_embedding_norm(t_emb)

        _, _, T_vid, H_vid, W_vid = video_latent.shape
        n_cond = self._num_real_condition_latents(T_vid)
        condition_mask = torch.zeros((B, 1, T_vid, H_vid, W_vid), device=device, dtype=video_latent.dtype)
        condition_mask[:, :, :n_cond] = 1.0
        video_latent_with_mask = torch.cat([video_latent, condition_mask], dim=1)
        padding_mask = torch.zeros((B, 1, H_vid, W_vid), device=device, dtype=video_latent.dtype)
        x_video, rope_emb, extra_pos = self.cosmos_dit.prepare_embedded_sequence(
            video_latent_with_mask,
            fps=fps,
            padding_mask=padding_mask,
        )

        video_grid_thw = None
        if is_multimodal_bridge_pos_scheme(self.bridge_pos_scheme):
            video_grid_thw = torch.tensor(
                [int(x_video.shape[1]), int(x_video.shape[2]), int(x_video.shape[3])],
                device=device,
                dtype=torch.long,
            ).unsqueeze(0).expand(B, -1)
        slow_rotary_info = self._build_action_rotary_batch_info(
            x_action=x_slow,
            action_valid_mask=slow_valid_mask,
            janus_left_pad_lens=janus_left_pad_lens,
            janus_images_seq_mask=janus_images_seq_mask,
            janus_images_emb_mask=janus_images_emb_mask,
            video_grid_thw=video_grid_thw,
        )
        slow_rotary_payload = self._build_bridge_rotary_payload(
            slow_rotary_info,
            action_seq_len=x_slow.shape[1],
            device=device,
            dtype=x_slow.dtype,
        )
        crossattn_source_embeds = context_embeds if cosmos_context_embeds is None else cosmos_context_embeds.to(self.dtype)
        crossattn_emb = self._prepare_cosmos_crossattn_emb(
            crossattn_source_embeds,
            cosmos_text_embeddings=cosmos_text_embeddings,
        )

        for i, block in enumerate(self.cosmos_dit.blocks):
            if i < self.slow_mot_layer_count:
                x_video, x_slow = block(
                    x_B_T_H_W_D=x_video,
                    emb_B_T_D=t_emb,
                    crossattn_emb=crossattn_emb,
                    x_action=x_slow,
                    x_action_valid_mask=slow_valid_mask,
                    x_rotary_payload=slow_rotary_payload,
                    mot_cache_video_kv=cache_video_kv,
                    mot_cache_video_kv_detach=cache_video_kv_detach,
                    mot_action_tail_token_count=0,
                    rope_emb_L_1_1_D=rope_emb,
                    adaln_lora_B_T_3D=adaln_lora,
                    extra_per_block_pos_emb=extra_pos,
                )
            else:
                break

        slow_norm = self.janus.language_model.model.norm
        x_slow_norm = slow_norm(x_slow)
        spatial_anchor_hiddens = self._gather_spatial_anchor_hiddens(
            x_slow_norm,
            slow_valid_mask,
            context_len=context_embeds.shape[1],
            spatial_token_count=spatial_token_embeds.shape[1],
        )

        time_tokens = self.janus.t_embedder(timestep_act).unsqueeze(1).to(dtype=self.dtype)
        action_tokens = self.janus.x_embedder(action_latent.to(dtype=self.dtype))
        x_fast_prefix = x_slow.detach() if self.action_detach_slow_prefix else x_slow
        x_fast = torch.cat([x_fast_prefix, time_tokens, action_tokens], dim=1)
        suffix_valid_mask = torch.ones((B, 1 + action_tokens.shape[1]), device=device, dtype=torch.bool)
        fast_valid_mask = torch.cat([slow_valid_mask, suffix_valid_mask], dim=1)
        fast_rotary_info = self._build_action_rotary_batch_info(
            x_action=x_fast,
            action_valid_mask=fast_valid_mask,
            janus_left_pad_lens=janus_left_pad_lens,
            janus_images_seq_mask=janus_images_seq_mask,
            janus_images_emb_mask=janus_images_emb_mask,
            video_grid_thw=video_grid_thw,
        )
        fast_rotary_payload = self._build_bridge_rotary_payload(
            fast_rotary_info,
            action_seq_len=x_fast.shape[1],
            device=device,
            dtype=x_fast.dtype,
        )

        for i, block in enumerate(self.cosmos_dit.blocks):
            if i < self.slow_mot_layer_count:
                continue
            if i < self.slow_mot_layer_count + self.fast_mot_layer_count:
                x_video, x_fast = block(
                    x_B_T_H_W_D=x_video,
                    emb_B_T_D=t_emb,
                    crossattn_emb=crossattn_emb,
                    x_action=x_fast,
                    x_action_valid_mask=fast_valid_mask,
                    x_rotary_payload=fast_rotary_payload,
                    mot_cache_video_kv=cache_video_kv,
                    mot_cache_video_kv_detach=cache_video_kv_detach,
                    mot_action_tail_token_count=action_latent.shape[1],
                    rope_emb_L_1_1_D=rope_emb,
                    adaln_lora_B_T_3D=adaln_lora,
                    extra_per_block_pos_emb=extra_pos,
                )
            else:
                if self._video_frozen:
                    with torch.no_grad():
                        x_video = block(
                            x_B_T_H_W_D=x_video,
                            emb_B_T_D=t_emb,
                            crossattn_emb=crossattn_emb,
                            rope_emb_L_1_1_D=rope_emb,
                            adaln_lora_B_T_3D=adaln_lora,
                            extra_per_block_pos_emb=extra_pos,
                        )
                else:
                    x_video = block(
                        x_B_T_H_W_D=x_video,
                        emb_B_T_D=t_emb,
                        crossattn_emb=crossattn_emb,
                        rope_emb_L_1_1_D=rope_emb,
                        adaln_lora_B_T_3D=adaln_lora,
                        extra_per_block_pos_emb=extra_pos,
                    )

        if self._video_frozen:
            with torch.no_grad():
                x_video_patch = self.cosmos_dit.final_layer(x_video, t_emb, adaln_lora_B_T_3D=adaln_lora)
                video_v = self.cosmos_dit.unpatchify(x_video_patch)
        else:
            x_video_patch = self.cosmos_dit.final_layer(x_video, t_emb, adaln_lora_B_T_3D=adaln_lora)
            video_v = self.cosmos_dit.unpatchify(x_video_patch)

        x_fast_norm = slow_norm(x_fast)
        action_out = x_fast_norm[:, -action_latent.shape[1]:, :]
        action_v = self.janus.final_layer(action_out)
        return video_v, action_v, spatial_anchor_hiddens

    @torch.no_grad()
    def run_cosmos_once(
        self,
        video_latent: torch.Tensor,
        context_embeds: torch.Tensor,
        timestep_vid: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        cosmos_text_embeddings: Optional[torch.Tensor] = None,
        cosmos_context_embeds: Optional[torch.Tensor] = None,
        num_condition_latent_frames: Optional[int] = None,
    ):
        video_latent = video_latent.to(self.dtype)
        context_embeds = context_embeds.to(self.dtype)
        B = video_latent.shape[0]
        device = video_latent.device
        if fps is None:
            fps = torch.full((B,), 10.0, device=device, dtype=self.dtype)
        elif not isinstance(fps, torch.Tensor):
            fps = torch.full((B,), float(fps), device=device, dtype=self.dtype)

        scale = getattr(self.cosmos_dit, "timestep_scale", 1.0)
        cosmos_t_scaled = timestep_vid * 1000.0 * scale
        cosmos_t_unsqueeze = cosmos_t_scaled.unsqueeze(1) if cosmos_t_scaled.ndim == 1 else cosmos_t_scaled
        use_wan_fp32 = getattr(self.cosmos_dit, "use_wan_fp32_strategy", False)
        cosmos_t_unsqueeze = cosmos_t_unsqueeze.to(torch.float32 if use_wan_fp32 else self.dtype)
        with torch.amp.autocast("cuda", enabled=use_wan_fp32, dtype=torch.float32):
            t_emb, adaln_lora = self.cosmos_dit.t_embedder(cosmos_t_unsqueeze)
            t_emb = self.cosmos_dit.t_embedding_norm(t_emb)

        _, _, T_vid, H_vid, W_vid = video_latent.shape
        if num_condition_latent_frames is None:
            n_cond = self._num_real_condition_latents(T_vid)
        else:
            n_cond = max(1, min(int(num_condition_latent_frames), T_vid))
        condition_mask = torch.zeros((B, 1, T_vid, H_vid, W_vid), device=device, dtype=self.dtype)
        condition_mask[:, :, :n_cond] = 1.0
        video_latent_with_mask = torch.cat([video_latent, condition_mask], dim=1)
        padding_mask = torch.zeros((B, 1, H_vid, W_vid), device=device, dtype=self.dtype)
        x_video, rope_emb, extra_pos = self.cosmos_dit.prepare_embedded_sequence(
            video_latent_with_mask,
            fps=fps,
            padding_mask=padding_mask,
        )
        crossattn_source_embeds = context_embeds if cosmos_context_embeds is None else cosmos_context_embeds.to(self.dtype)
        crossattn_emb = self._prepare_cosmos_crossattn_emb(
            crossattn_source_embeds,
            cosmos_text_embeddings=cosmos_text_embeddings,
        )

        for wrapper in self.mot_attention_wrappers:
            wrapper.cache_video_kv = True
            wrapper.current_x_action = None
            wrapper.current_action_valid_mask = None
            wrapper.current_rotary_payload = None

        for i, block in enumerate(self.cosmos_dit.blocks):
            if i < len(self.mot_attention_wrappers):
                x_video, _ = block(
                    x_B_T_H_W_D=x_video,
                    emb_B_T_D=t_emb,
                    crossattn_emb=crossattn_emb,
                    x_action=None,
                    rope_emb_L_1_1_D=rope_emb,
                    adaln_lora_B_T_3D=adaln_lora,
                    extra_per_block_pos_emb=extra_pos,
                )
            else:
                x_video = block(
                    x_B_T_H_W_D=x_video,
                    emb_B_T_D=t_emb,
                    crossattn_emb=crossattn_emb,
                    rope_emb_L_1_1_D=rope_emb,
                    adaln_lora_B_T_3D=adaln_lora,
                    extra_per_block_pos_emb=extra_pos,
                )
        for wrapper in self.mot_attention_wrappers:
            wrapper.cache_video_kv = False

        x_video_patch = self.cosmos_dit.final_layer(x_video, t_emb, adaln_lora_B_T_3D=adaln_lora)
        return self.cosmos_dit.unpatchify(x_video_patch)

    @torch.no_grad()
    def generate_spatial_tokens(
        self,
        context_embeds: torch.Tensor,
        num_spatial_tokens: Optional[int] = None,
        janus_left_pad_lens: Optional[torch.Tensor] = None,
        janus_attention_mask: Optional[torch.Tensor] = None,
        janus_images_seq_mask: Optional[torch.Tensor] = None,
        janus_images_emb_mask: Optional[torch.Tensor] = None,
        return_spatial_debug: bool = False,
    ):
        num_spatial_tokens = self._resolve_spatial_token_count(num_spatial_tokens)
        context_embeds = context_embeds.to(self.dtype)
        B = context_embeds.shape[0]
        device = context_embeds.device
        if janus_left_pad_lens is not None:
            janus_left_pad_lens = janus_left_pad_lens.to(device=device, dtype=torch.long)
        if janus_attention_mask is not None:
            janus_attention_mask = janus_attention_mask.to(device=device, dtype=torch.bool)

        cached_video_grid_thw = None
        if is_multimodal_bridge_pos_scheme(self.bridge_pos_scheme):
            cached_video_grid_thw = self._get_cached_bridge_video_grid_thw(B, device)

        spatial_parts = []
        last_anchor_hidden = None
        last_token_ids = None
        for _ in range(num_spatial_tokens):
            prev = (
                torch.cat(spatial_parts, dim=1)
                if spatial_parts
                else context_embeds.new_zeros(B, 0, context_embeds.shape[-1])
            )
            x_slow, slow_valid_mask = self._build_action_sequence(
                context_embeds=context_embeds,
                spatial_token_embeds=prev,
                janus_left_pad_lens=janus_left_pad_lens,
                janus_attention_mask=janus_attention_mask,
            )
            rotary_info = self._build_action_rotary_batch_info(
                x_action=x_slow,
                action_valid_mask=slow_valid_mask,
                janus_left_pad_lens=janus_left_pad_lens,
                janus_images_seq_mask=janus_images_seq_mask,
                janus_images_emb_mask=janus_images_emb_mask,
                video_grid_thw=cached_video_grid_thw,
            )
            rotary_payload = self._build_bridge_rotary_payload(
                rotary_info,
                action_seq_len=x_slow.shape[1],
                device=device,
                dtype=x_slow.dtype,
            )
            hidden = self._slow_forward_prefix(
                x_slow,
                slow_valid_mask,
                rotary_payload,
                cache_prefix_kv=False,
            )
            hidden_norm = self.janus.language_model.model.norm(hidden)
            if prev.shape[1] == 0:
                anchor_idx = self._last_valid_context_indices(slow_valid_mask, context_embeds.shape[1])
            else:
                anchor_idx = torch.full((B,), context_embeds.shape[1] + prev.shape[1] - 1, device=device, dtype=torch.long)
            anchor_hidden = hidden_norm[torch.arange(B, device=device), anchor_idx]
            logits = self._special_token_logits(anchor_hidden)
            token_ids = logits.argmax(dim=-1)
            spatial_parts.append(self._embed_special_token_ids(token_ids.unsqueeze(1)).to(self.dtype))
            last_anchor_hidden = anchor_hidden
            last_token_ids = token_ids

        spatial_embeds = torch.cat(spatial_parts, dim=1)
        if return_spatial_debug:
            return spatial_embeds, {
                "spatial_anchor_hidden": last_anchor_hidden.detach(),
                "spatial_token_ids": last_token_ids.detach(),
            }
        return spatial_embeds

    @torch.no_grad()
    def generate_spatial_tokens_cached(
        self,
        context_embeds: torch.Tensor,
        num_spatial_tokens: Optional[int] = None,
        janus_left_pad_lens: Optional[torch.Tensor] = None,
        janus_attention_mask: Optional[torch.Tensor] = None,
        janus_images_seq_mask: Optional[torch.Tensor] = None,
        janus_images_emb_mask: Optional[torch.Tensor] = None,
        return_spatial_debug: bool = False,
    ):
        num_spatial_tokens = self._resolve_spatial_token_count(num_spatial_tokens)
        self._clear_cached_action_prefix_kv()
        context_embeds = context_embeds.to(self.dtype)
        B = context_embeds.shape[0]
        device = context_embeds.device
        if janus_left_pad_lens is not None:
            janus_left_pad_lens = janus_left_pad_lens.to(device=device, dtype=torch.long)
        if janus_attention_mask is not None:
            janus_attention_mask = janus_attention_mask.to(device=device, dtype=torch.bool)

        x_context, context_valid_mask = self._build_action_sequence(
            context_embeds=context_embeds,
            janus_left_pad_lens=janus_left_pad_lens,
            janus_attention_mask=janus_attention_mask,
        )
        context_rotary_payload = self._build_action_rotary_payload_for_length(
            batch_size=B,
            total_action_len=x_context.shape[1],
            action_valid_mask=context_valid_mask,
            device=device,
            dtype=x_context.dtype,
            janus_left_pad_lens=janus_left_pad_lens,
            janus_images_seq_mask=janus_images_seq_mask,
            janus_images_emb_mask=janus_images_emb_mask,
        )
        hidden = self._slow_forward_prefix(
            x_context,
            context_valid_mask,
            context_rotary_payload,
            cache_prefix_kv=False,
        )
        hidden_norm = self.janus.language_model.model.norm(hidden)
        anchor_idx = self._last_valid_context_indices(context_valid_mask, context_embeds.shape[1])
        last_anchor_hidden = hidden_norm[torch.arange(B, device=device), anchor_idx]

        spatial_parts = []
        last_prediction_anchor_hidden = last_anchor_hidden
        last_token_ids = None
        for _ in range(num_spatial_tokens):
            last_prediction_anchor_hidden = last_anchor_hidden
            logits = self._special_token_logits(last_anchor_hidden)
            token_ids = logits.argmax(dim=-1)
            token_embeds = self._embed_special_token_ids(token_ids.unsqueeze(1)).to(self.dtype)
            spatial_parts.append(token_embeds)
            last_token_ids = token_ids

            x_next = torch.cat([context_embeds, *spatial_parts], dim=1)
            next_valid_mask = torch.cat(
                [
                    context_valid_mask,
                    torch.ones((B, len(spatial_parts)), device=device, dtype=torch.bool),
                ],
                dim=1,
            )
            next_rotary_payload = self._build_action_rotary_payload_for_length(
                batch_size=B,
                total_action_len=x_next.shape[1],
                action_valid_mask=next_valid_mask,
                device=device,
                dtype=x_next.dtype,
                janus_left_pad_lens=janus_left_pad_lens,
                janus_images_seq_mask=janus_images_seq_mask,
                janus_images_emb_mask=janus_images_emb_mask,
            )
            next_hidden = self._slow_forward_prefix(
                x_next,
                next_valid_mask,
                next_rotary_payload,
                cache_prefix_kv=False,
            )
            next_hidden_norm = self.janus.language_model.model.norm(next_hidden)
            last_anchor_hidden = next_hidden_norm[:, -1, :]

        spatial_embeds = torch.cat(spatial_parts, dim=1)
        slow_prefix = torch.cat([context_embeds, spatial_embeds], dim=1)
        slow_prefix_valid_mask = torch.cat(
            [
                context_valid_mask,
                torch.ones((B, spatial_embeds.shape[1]), device=device, dtype=torch.bool),
            ],
            dim=1,
        )
        slow_prefix_rotary_payload = self._build_action_rotary_payload_for_length(
            batch_size=B,
            total_action_len=slow_prefix.shape[1],
            action_valid_mask=slow_prefix_valid_mask,
            device=device,
            dtype=slow_prefix.dtype,
            janus_left_pad_lens=janus_left_pad_lens,
            janus_images_seq_mask=janus_images_seq_mask,
            janus_images_emb_mask=janus_images_emb_mask,
        )
        slow_A = self._slow_forward_prefix(
            slow_prefix,
            slow_prefix_valid_mask,
            slow_prefix_rotary_payload,
            cache_prefix_kv=False,
        )
        fast_prefix_rotary_payload = self._build_action_rotary_payload_for_length(
            batch_size=B,
            total_action_len=slow_A.shape[1],
            action_valid_mask=slow_prefix_valid_mask,
            device=device,
            dtype=slow_A.dtype,
            janus_left_pad_lens=janus_left_pad_lens,
            janus_images_seq_mask=janus_images_seq_mask,
            janus_images_emb_mask=janus_images_emb_mask,
        )
        self._fast_forward_full(
            slow_A,
            slow_prefix_valid_mask,
            fast_prefix_rotary_payload,
            action_tail_token_count=0,
            cache_prefix_kv=True,
        )

        if return_spatial_debug:
            return spatial_embeds, {
                "spatial_anchor_hidden": last_prediction_anchor_hidden.detach(),
                "spatial_token_ids": last_token_ids.detach(),
            }
        return spatial_embeds

    @torch.no_grad()
    def action_denoise_step(
        self,
        action_latent: torch.Tensor,
        spatial_token_embeds: torch.Tensor,
        context_embeds: torch.Tensor,
        timestep_act: torch.Tensor,
        janus_left_pad_lens: Optional[torch.Tensor] = None,
        janus_attention_mask: Optional[torch.Tensor] = None,
        janus_images_seq_mask: Optional[torch.Tensor] = None,
        janus_images_emb_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_latent = action_latent.to(self.dtype)
        spatial_token_embeds = spatial_token_embeds.to(self.dtype)
        context_embeds = context_embeds.to(self.dtype)
        B = context_embeds.shape[0]
        device = context_embeds.device
        if janus_left_pad_lens is not None:
            janus_left_pad_lens = janus_left_pad_lens.to(device=device, dtype=torch.long)
        if janus_attention_mask is not None:
            janus_attention_mask = janus_attention_mask.to(device=device, dtype=torch.bool)

        x_slow, slow_valid_mask = self._build_action_sequence(
            context_embeds=context_embeds,
            spatial_token_embeds=spatial_token_embeds,
            janus_left_pad_lens=janus_left_pad_lens,
            janus_attention_mask=janus_attention_mask,
        )
        cached_video_grid_thw = None
        if is_multimodal_bridge_pos_scheme(self.bridge_pos_scheme):
            cached_video_grid_thw = self._get_cached_bridge_video_grid_thw(B, device)
        slow_rotary_info = self._build_action_rotary_batch_info(
            x_action=x_slow,
            action_valid_mask=slow_valid_mask,
            janus_left_pad_lens=janus_left_pad_lens,
            janus_images_seq_mask=janus_images_seq_mask,
            janus_images_emb_mask=janus_images_emb_mask,
            video_grid_thw=cached_video_grid_thw,
        )
        slow_rotary_payload = self._build_bridge_rotary_payload(
            slow_rotary_info,
            action_seq_len=x_slow.shape[1],
            device=device,
            dtype=x_slow.dtype,
        )
        slow_A = self._slow_forward_prefix(
            x_slow,
            slow_valid_mask,
            slow_rotary_payload,
            cache_prefix_kv=False,
        )

        time_tokens = self.janus.t_embedder(timestep_act).unsqueeze(1).to(dtype=self.dtype)
        action_tokens = self.janus.x_embedder(action_latent.to(dtype=self.dtype))
        suffix = torch.cat([time_tokens, action_tokens], dim=1)
        suffix_valid_mask = torch.ones((B, suffix.shape[1]), device=device, dtype=torch.bool)
        x_fast = torch.cat([slow_A, suffix], dim=1)
        fast_valid_mask = torch.cat([slow_valid_mask, suffix_valid_mask], dim=1)
        fast_rotary_info = self._build_action_rotary_batch_info(
            x_action=x_fast,
            action_valid_mask=fast_valid_mask,
            janus_left_pad_lens=janus_left_pad_lens,
            janus_images_seq_mask=janus_images_seq_mask,
            janus_images_emb_mask=janus_images_emb_mask,
            video_grid_thw=cached_video_grid_thw,
        )
        fast_rotary_payload = self._build_bridge_rotary_payload(
            fast_rotary_info,
            action_seq_len=x_fast.shape[1],
            device=device,
            dtype=x_fast.dtype,
        )
        hidden = self._fast_forward_full(
            x_fast,
            fast_valid_mask,
            fast_rotary_payload,
            action_tail_token_count=action_latent.shape[1],
            cache_prefix_kv=False,
        )
        hidden_norm = self.janus.language_model.model.norm(hidden)
        action_out = hidden_norm[:, -action_latent.shape[1]:, :]
        return self.janus.final_layer(action_out)

    @torch.no_grad()
    def action_denoise_step_cached(
        self,
        action_latent: torch.Tensor,
        timestep_act: torch.Tensor,
        janus_left_pad_lens: Optional[torch.Tensor] = None,
        janus_images_seq_mask: Optional[torch.Tensor] = None,
        janus_images_emb_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_latent = action_latent.to(self.dtype)
        B = action_latent.shape[0]
        device = action_latent.device
        prefix_len = self._get_cached_fast_prefix_len()
        if prefix_len <= 0:
            raise RuntimeError("action_denoise_step_cached requires cached FiS fast prefix KV.")
        if janus_left_pad_lens is not None:
            janus_left_pad_lens = janus_left_pad_lens.to(device=device, dtype=torch.long)

        time_tokens = self.janus.t_embedder(timestep_act).unsqueeze(1).to(dtype=self.dtype)
        action_tokens = self.janus.x_embedder(action_latent.to(dtype=self.dtype))
        suffix = torch.cat([time_tokens, action_tokens], dim=1)
        suffix_valid_mask = torch.ones((B, suffix.shape[1]), device=device, dtype=torch.bool)
        full_valid_mask = torch.cat(
            [
                self._get_cached_fast_prefix_valid_mask().to(device=device, dtype=torch.bool),
                suffix_valid_mask,
            ],
            dim=1,
        )
        total_len = prefix_len + suffix.shape[1]
        full_rotary_payload = self._build_action_rotary_payload_for_length(
            batch_size=B,
            total_action_len=total_len,
            action_valid_mask=full_valid_mask,
            device=device,
            dtype=suffix.dtype,
            janus_left_pad_lens=janus_left_pad_lens,
            janus_images_seq_mask=janus_images_seq_mask,
            janus_images_emb_mask=janus_images_emb_mask,
        )
        suffix_rotary_payload = self._slice_bridge_rotary_payload(
            full_rotary_payload,
            prefix_len,
            total_len,
        )
        hidden = self._fast_forward_suffix_cached(
            suffix,
            suffix_valid_mask=suffix_valid_mask,
            suffix_rotary_payload=suffix_rotary_payload,
            action_tail_token_count=action_latent.shape[1],
        )
        hidden_norm = self.janus.language_model.model.norm(hidden)
        action_out = hidden_norm[:, -action_latent.shape[1]:, :]
        return self.janus.final_layer(action_out)

    def _resolve_spatial_token_count(self, num_spatial_tokens: Optional[int] = None) -> int:
        if num_spatial_tokens is None:
            num_spatial_tokens = int(getattr(self.config, "total_spatial_tokens", getattr(self.config, "total_latent_tokens", 1)) or 1)
        num_spatial_tokens = int(num_spatial_tokens)
        if num_spatial_tokens not in (1, 2):
            raise ValueError(f"Spatial token training supports 1 or 2 tokens, got {num_spatial_tokens}.")
        return num_spatial_tokens

    @staticmethod
    def _normalize_spatial_gt_token_ids(spatial_gt_token_ids: torch.Tensor) -> torch.Tensor:
        if spatial_gt_token_ids.ndim == 1:
            spatial_gt_token_ids = spatial_gt_token_ids.unsqueeze(1)
        elif spatial_gt_token_ids.ndim != 2:
            raise ValueError(
                f"spatial_gt_token_ids must have shape [B] or [B, N], got {tuple(spatial_gt_token_ids.shape)}."
            )
        return spatial_gt_token_ids

    def _get_spatial_hidden_wan_encoder(self, device: torch.device) -> Wan21VAEEncoder:
        encoder = self.__dict__.get("spatial_hidden_wan_encoder", None)
        if encoder is None:
            encoder = Wan21VAEEncoder(
                vae_pth=self.wan21_vae_path,
                dtype=torch.float32,
                device=device,
                freeze=True,
                normalize_latents=True,
            )
            self.__dict__["spatial_hidden_wan_encoder"] = encoder
            return encoder
        try:
            encoder_device = next(encoder.parameters()).device
        except StopIteration:
            encoder_device = device
        if encoder_device != device:
            encoder.to(device=device)
        encoder.eval()
        return encoder

    def _encode_spatial_hidden_wan_target(
        self,
        future_pixel_values: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if future_pixel_values.ndim == 5 and future_pixel_values.shape[1] == 1:
            future_pixel_values = future_pixel_values[:, 0]
        elif future_pixel_values.ndim != 4:
            raise ValueError(
                "wan_vae spatial hidden sim expects future pixels shaped "
                f"[B, 1, 3, 256, 256] or [B, 3, 256, 256], got {tuple(future_pixel_values.shape)}."
            )
        if future_pixel_values.shape != (batch_size, 3, 256, 256):
            raise ValueError(
                "wan_vae spatial hidden sim expects future pixels shaped "
                f"{(batch_size, 3, 256, 256)}, got {tuple(future_pixel_values.shape)}."
            )
        encoder = self._get_spatial_hidden_wan_encoder(device)
        future_for_vae = future_pixel_values.to(device=device, dtype=torch.float32) * 2.0 - 1.0
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=False):
                target_latent = encoder.encode(future_for_vae)
        if target_latent.ndim != 5 or target_latent.shape[2] != 1:
            raise ValueError(f"Wan encoder must return [B, 16, 1, 32, 32], got {tuple(target_latent.shape)}.")
        return target_latent[:, :, 0]

    def _compute_spatial_hidden_wan_vae_loss(self, spatial_anchor_hidden: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
        if self.spatial_hidden_wan_upsampler is None:
            raise RuntimeError("wan_vae spatial sim requires spatial_hidden_wan_upsampler.")
        B = spatial_anchor_hidden.shape[0]
        hidden_image = spatial_anchor_hidden.to(dtype=self.dtype).reshape(B, self.janus_dim, 1, 1)
        pred_latent = self.spatial_hidden_wan_upsampler(hidden_image)
        return F.mse_loss(pred_latent.to(torch.float32), target_latent.to(torch.float32))

    def _compute_spatial_hidden_wan_downsample_sim_loss(
        self,
        spatial_anchor_hidden: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> torch.Tensor:
        if self.spatial_hidden_wan_downsampler is None:
            raise RuntimeError("spatial hidden Wan downsample sim requires spatial_hidden_wan_downsampler.")
        B = spatial_anchor_hidden.shape[0]
        target_hidden_image = self.spatial_hidden_wan_downsampler(target_latent.to(dtype=self.dtype))
        target_hidden = target_hidden_image.reshape(B, self.janus_dim)
        similarity = F.cosine_similarity(
            spatial_anchor_hidden.to(torch.float32),
            target_hidden.to(torch.float32),
            dim=-1,
        ).mean()
        return 1.0 - similarity

    def forward(
        self,
        first_frame: torch.Tensor,
        video_frames: torch.Tensor,
        actions: torch.Tensor,
        janus_input_ids: torch.Tensor,
        janus_pixel_values: torch.Tensor,
        janus_images_seq_mask: torch.Tensor,
        janus_images_emb_mask: torch.Tensor,
        fps,
        janus_action_pixel_values: Optional[torch.Tensor] = None,
        spatial_gt_token_ids: Optional[torch.Tensor] = None,
        latent_gt_token_ids: Optional[torch.Tensor] = None,
        janus_state_seq_mask: Optional[torch.Tensor] = None,
        now_state: Optional[torch.Tensor] = None,
        janus_left_pad_lens: Optional[torch.Tensor] = None,
        janus_attention_mask: Optional[torch.Tensor] = None,
        cosmos_text_embeddings: Optional[torch.Tensor] = None,
        cosmos_janus_input_ids: Optional[torch.Tensor] = None,
        cosmos_janus_images_seq_mask: Optional[torch.Tensor] = None,
        cosmos_janus_state_seq_mask: Optional[torch.Tensor] = None,
        cosmos_janus_images_emb_mask: Optional[torch.Tensor] = None,
        value_targets: Optional[torch.Tensor] = None,
        spatial_hidden_sim_pixel_values: Optional[torch.Tensor] = None,
        latent_hidden_sim_pixel_values: Optional[torch.Tensor] = None,
        loss_weights: tuple = (0.2, 1.0, 1.0),
    ):
        B = video_frames.shape[0]
        device = video_frames.device
        aux_metrics = {}
        actions = actions.to(self.dtype)
        janus_pixel_values = janus_pixel_values.to(self.dtype)
        if spatial_gt_token_ids is None:
            spatial_gt_token_ids = latent_gt_token_ids
        if spatial_gt_token_ids is None:
            raise ValueError("spatial_gt_token_ids must be provided.")
        spatial_gt_token_ids = self._normalize_spatial_gt_token_ids(
            spatial_gt_token_ids.to(device=device, dtype=torch.long)
        )
        spatial_token_count = self._resolve_spatial_token_count(spatial_gt_token_ids.shape[1])
        if spatial_hidden_sim_pixel_values is None:
            spatial_hidden_sim_pixel_values = latent_hidden_sim_pixel_values
        if spatial_hidden_sim_pixel_values is not None:
            spatial_hidden_sim_pixel_values = spatial_hidden_sim_pixel_values.to(device=device, dtype=self.dtype)
        if not isinstance(fps, torch.Tensor):
            fps = torch.full((B,), float(fps), device=device, dtype=self.dtype)

        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
        video_norm = video_frames * 2.0 - 1.0
        full_video = torch.cat([first_frame_norm, video_norm], dim=2)
        with torch.no_grad():
            clean_video_latent = self.cosmos_vae.encode(full_video.to(self.dtype)).to(self.dtype)
            n_cond = self._num_real_condition_latents(clean_video_latent.shape[2])
            condition_latent = clean_video_latent[:, :, :n_cond]

        context_embeds = self.prepare_action_context_embeds(
            janus_input_ids=janus_input_ids,
            now_state=now_state,
            janus_pixel_values=janus_pixel_values,
            janus_images_seq_mask=janus_images_seq_mask,
            janus_state_seq_mask=janus_state_seq_mask,
            janus_images_emb_mask=janus_images_emb_mask,
        ).to(self.dtype)
        cosmos_context_embeds = None
        if cosmos_text_embeddings is None and cosmos_janus_input_ids is not None:
            if cosmos_janus_images_seq_mask is None or cosmos_janus_images_emb_mask is None:
                raise ValueError(
                    "cosmos_janus_images_seq_mask and cosmos_janus_images_emb_mask are required "
                    "when cosmos_janus_input_ids is provided."
                )
            cosmos_context_embeds = self.prepare_action_context_embeds(
                janus_input_ids=cosmos_janus_input_ids,
                now_state=now_state,
                janus_pixel_values=janus_pixel_values,
                janus_images_seq_mask=cosmos_janus_images_seq_mask,
                janus_state_seq_mask=cosmos_janus_state_seq_mask,
                janus_images_emb_mask=cosmos_janus_images_emb_mask,
            ).to(self.dtype)

        u_vid, t_vid = self._sample_cosmos_train_sigma(B, device)
        aux_metrics["cosmos_train_u_mean"] = u_vid.detach().to(torch.float32).mean().item()
        aux_metrics["cosmos_train_sigma_mean"] = t_vid.detach().to(torch.float32).mean().item()
        t_act = torch.distributions.Beta(
            torch.tensor(1.5, device=device, dtype=torch.float32),
            torch.tensor(1.0, device=device, dtype=torch.float32),
        ).sample((B,)).to(dtype=self.dtype)

        video_noise = torch.randn_like(clean_video_latent)
        action_noise = torch.randn_like(actions, dtype=self.dtype)
        noisy_video_latent = t_vid.view(B, 1, 1, 1, 1) * video_noise + (1.0 - t_vid.view(B, 1, 1, 1, 1)) * clean_video_latent
        noisy_action = t_act.view(B, 1, 1) * action_noise + (1.0 - t_act.view(B, 1, 1)) * actions
        noisy_video_latent[:, :, :n_cond] = condition_latent
        target_video_v = video_noise - clean_video_latent
        target_video_v[:, :, :n_cond] = 0.0
        target_action_v = action_noise - actions
        spatial_gt_embeds = self._embed_special_token_ids(spatial_gt_token_ids).to(self.dtype)

        self._clear_cached_video_kv()
        try:
            pred_video_v, pred_action_v, spatial_anchor_hiddens = self.joint_denoise_step(
                video_latent=noisy_video_latent,
                action_latent=noisy_action,
                spatial_token_embeds=spatial_gt_embeds,
                context_embeds=context_embeds,
                timestep_vid=t_vid,
                timestep_act=t_act,
                fps=fps,
                janus_left_pad_lens=janus_left_pad_lens,
                janus_attention_mask=janus_attention_mask,
                janus_images_seq_mask=janus_images_seq_mask,
                janus_images_emb_mask=janus_images_emb_mask,
                cosmos_text_embeddings=cosmos_text_embeddings,
                cosmos_context_embeds=cosmos_context_embeds,
                cache_video_kv=False,
            )

            if len(loss_weights) == 3:
                video_weight, action_weight, spatial_weight = loss_weights
                spatial_hidden_sim_weight = 0.0
                spatial_hidden_wan_downsample_sim_weight = 0.0
            elif len(loss_weights) == 4:
                video_weight, action_weight, spatial_weight, spatial_hidden_sim_weight = loss_weights
                spatial_hidden_wan_downsample_sim_weight = 0.0
            elif len(loss_weights) == 5:
                (
                    video_weight,
                    action_weight,
                    spatial_weight,
                    spatial_hidden_sim_weight,
                    spatial_hidden_wan_downsample_sim_weight,
                ) = loss_weights
            else:
                raise ValueError(f"loss_weights must have 3, 4, or 5 entries, got {len(loss_weights)}.")

            if n_cond >= target_video_v.shape[2]:
                loss_video = torch.zeros((), device=device, dtype=torch.float32)
            else:
                loss_video = F.mse_loss(pred_video_v[:, :, n_cond:], target_video_v[:, :, n_cond:])
            loss_action = F.mse_loss(pred_action_v, target_action_v)

            spatial_logits = self._special_token_logits(spatial_anchor_hiddens)
            spatial_ce_per_item = F.cross_entropy(
                spatial_logits.reshape(B * spatial_token_count, -1).to(torch.float32),
                spatial_gt_token_ids.reshape(B * spatial_token_count),
                reduction="none",
            ).view(B, spatial_token_count)
            spatial_ce_per_token = spatial_ce_per_item.mean(dim=0)
            loss_spatial = spatial_ce_per_token.mean()
            aux_metrics["spatial_ce_loss"] = loss_spatial.detach().item()
            if spatial_token_count > 1:
                for token_idx in range(spatial_token_count):
                    aux_metrics[f"spatial_ce_loss_{token_idx}"] = spatial_ce_per_token[token_idx].detach().item()

            loss_spatial_hidden_sim = torch.zeros((), device=device, dtype=torch.float32)
            loss_spatial_hidden_wan_downsample_sim = torch.zeros((), device=device, dtype=torch.float32)
            if spatial_hidden_sim_pixel_values is not None:
                if self.spatial_hidden_sim_loss_mode == "wan_vae":
                    target_wan_latent = self._encode_spatial_hidden_wan_target(
                        future_pixel_values=spatial_hidden_sim_pixel_values,
                        batch_size=B,
                        device=spatial_anchor_hiddens.device,
                    )
                    wan_losses = []
                    for token_idx in range(spatial_token_count):
                        token_loss = self._compute_spatial_hidden_wan_vae_loss(
                            spatial_anchor_hidden=spatial_anchor_hiddens[:, token_idx, :],
                            target_latent=target_wan_latent,
                        )
                        wan_losses.append(token_loss)
                        if spatial_token_count > 1:
                            aux_metrics[f"spatial_hidden_wan_vae_mse_loss_{token_idx}"] = token_loss.detach().item()
                    loss_spatial_hidden_sim = torch.stack(wan_losses).mean()
                    aux_metrics["spatial_hidden_wan_vae_mse_loss"] = loss_spatial_hidden_sim.detach().item()
                    if self.use_spatial_hidden_wan_downsample_sim_loss:
                        down_losses = []
                        for token_idx in range(spatial_token_count):
                            token_loss = self._compute_spatial_hidden_wan_downsample_sim_loss(
                                spatial_anchor_hidden=spatial_anchor_hiddens[:, token_idx, :],
                                target_latent=target_wan_latent,
                            )
                            down_losses.append(token_loss)
                            if spatial_token_count > 1:
                                aux_metrics[f"spatial_hidden_wan_downsample_sim_loss_{token_idx}"] = token_loss.detach().item()
                        loss_spatial_hidden_wan_downsample_sim = torch.stack(down_losses).mean()
                        aux_metrics["spatial_hidden_wan_downsample_sim_loss"] = (
                            loss_spatial_hidden_wan_downsample_sim.detach().item()
                        )
                else:
                    if spatial_hidden_sim_pixel_values.ndim != 5 or spatial_hidden_sim_pixel_values.shape[1] != 1:
                        raise ValueError(
                            "spatial_hidden_sim_pixel_values must have shape [B, 1, C, H, W], "
                            f"got {tuple(spatial_hidden_sim_pixel_values.shape)}."
                        )
                    sim_target = self._build_spatial_hidden_siglip_targets(
                        future_pixel_values=spatial_hidden_sim_pixel_values,
                        batch_size=B,
                        spatial_token_count=spatial_token_count,
                        hidden_dim=spatial_anchor_hiddens.shape[-1],
                    )
                    similarity = F.cosine_similarity(
                        spatial_anchor_hiddens.to(torch.float32),
                        sim_target.to(torch.float32),
                        dim=-1,
                    )
                    per_token = 1.0 - similarity.mean(dim=0)
                    loss_spatial_hidden_sim = per_token.mean()
                    if spatial_token_count > 1:
                        for token_idx in range(spatial_token_count):
                            aux_metrics[f"spatial_hidden_sim_loss_{token_idx}"] = per_token[token_idx].detach().item()
                aux_metrics["spatial_hidden_sim_loss"] = loss_spatial_hidden_sim.detach().item()
        finally:
            self._clear_cached_video_kv()

        if self._video_frozen:
            video_weight = 0.0
        total_loss = (
            video_weight * loss_video
            + action_weight * loss_action
            + spatial_weight * loss_spatial
            + spatial_hidden_sim_weight * loss_spatial_hidden_sim
            + spatial_hidden_wan_downsample_sim_weight * loss_spatial_hidden_wan_downsample_sim
        )
        return total_loss, loss_video.item(), loss_action.item(), loss_spatial.item(), aux_metrics

    @torch.no_grad()
    def forward_flow_joint_inference(
        self,
        janus_input_ids,
        janus_pixel_values,
        janus_images_seq_mask,
        janus_images_emb_mask,
        first_frame,
        action_denoise_steps=10,
        cosmos_denoise_steps=1,
        num_spatial_tokens=None,
        num_latent_tokens=None,
        fps=None,
        action_self_causal_in_bridge=None,
        janus_left_pad_lens: Optional[torch.Tensor] = None,
        janus_state_seq_mask: Optional[torch.Tensor] = None,
        janus_action_pixel_values: Optional[torch.Tensor] = None,
        now_state: Optional[torch.Tensor] = None,
        janus_attention_mask: Optional[torch.Tensor] = None,
        cosmos_text_embeddings: Optional[torch.Tensor] = None,
        cosmos_janus_input_ids: Optional[torch.Tensor] = None,
        cosmos_janus_images_seq_mask: Optional[torch.Tensor] = None,
        cosmos_janus_state_seq_mask: Optional[torch.Tensor] = None,
        cosmos_janus_images_emb_mask: Optional[torch.Tensor] = None,
        return_spatial_debug: bool = False,
        return_latent_visual_debug: bool = False,
        **unused_kwargs,
    ):
        if cosmos_denoise_steps <= 0:
            raise ValueError("cosmos_denoise_steps must be positive.")
        if num_spatial_tokens is None:
            num_spatial_tokens = num_latent_tokens
        num_spatial_tokens = self._resolve_spatial_token_count(num_spatial_tokens)

        B = first_frame.shape[0]
        device = first_frame.device
        if fps is None:
            fps = torch.full((B,), 10.0, device=device, dtype=self.dtype)
        elif not isinstance(fps, torch.Tensor):
            fps = torch.full((B,), float(fps), device=device, dtype=self.dtype)
        if janus_left_pad_lens is not None:
            janus_left_pad_lens = janus_left_pad_lens.to(device=device, dtype=torch.long)
        if janus_attention_mask is not None:
            janus_attention_mask = janus_attention_mask.to(device=device, dtype=torch.bool)

        janus_pixel_values = janus_pixel_values.to(self.dtype)
        context_embeds = self.prepare_action_context_embeds(
            janus_input_ids=janus_input_ids,
            now_state=now_state,
            janus_pixel_values=janus_pixel_values,
            janus_images_seq_mask=janus_images_seq_mask,
            janus_state_seq_mask=janus_state_seq_mask,
            janus_images_emb_mask=janus_images_emb_mask,
        ).to(self.dtype)
        cosmos_context_embeds = None
        if cosmos_text_embeddings is None and cosmos_janus_input_ids is not None:
            if cosmos_janus_images_seq_mask is None or cosmos_janus_images_emb_mask is None:
                raise ValueError(
                    "cosmos_janus_images_seq_mask and cosmos_janus_images_emb_mask are required "
                    "when cosmos_janus_input_ids is provided."
                )
            cosmos_context_embeds = self.prepare_action_context_embeds(
                janus_input_ids=cosmos_janus_input_ids,
                now_state=now_state,
                janus_pixel_values=janus_pixel_values,
                janus_images_seq_mask=cosmos_janus_images_seq_mask,
                janus_state_seq_mask=cosmos_janus_state_seq_mask,
                janus_images_emb_mask=cosmos_janus_images_emb_mask,
            ).to(self.dtype)

        if first_frame.ndim == 4:
            first_frame_for_cosmos = (first_frame * 2.0 - 1.0).unsqueeze(2)
        elif first_frame.ndim == 5:
            first_frame_for_cosmos = first_frame * 2.0 - 1.0
        else:
            raise ValueError(f"first_frame must be [B,C,H,W] or [B,C,T,H,W], got {tuple(first_frame.shape)}.")
        condition_latent = self.cosmos_vae.encode(first_frame_for_cosmos.to(self.dtype)).to(self.dtype)

        action_chunk = int(getattr(self.config, "action_chunk", 16))
        action_dim = int(getattr(self.config, "action_dim", 7))
        T_cond = condition_latent.shape[2]
        C_lat = condition_latent.shape[1]
        H_lat, W_lat = condition_latent.shape[3], condition_latent.shape[4]
        T_total = self._latent_num_frames_for_pixels(getattr(self.config, "video_frames", 16))
        if T_total < T_cond:
            raise ValueError(
                f"Conditioning history encodes to {T_cond} latent frames, exceeding total {T_total}."
            )
        video_noise = torch.randn(B, C_lat, T_total, H_lat, W_lat, device=device, dtype=self.dtype)
        video_noise[:, :, :T_cond] = condition_latent

        sample_scheduler, shift, use_kerras_sigma, num_train_timesteps = self._require_cosmos_inference_runtime()
        self._clear_cached_video_kv()
        try:
            x_vid = video_noise
            if T_total == T_cond:
                t_vid = torch.zeros((B,), device=device, dtype=self.dtype)
                _ = self.run_cosmos_once(
                    video_latent=x_vid,
                    context_embeds=context_embeds,
                    timestep_vid=t_vid,
                    fps=fps,
                    cosmos_text_embeddings=cosmos_text_embeddings,
                    cosmos_context_embeds=cosmos_context_embeds,
                    num_condition_latent_frames=T_cond,
                )
            else:
                sample_scheduler.set_timesteps(
                    cosmos_denoise_steps,
                    device=device,
                    shift=shift,
                    use_kerras_sigma=use_kerras_sigma,
                )
                for raw_t in sample_scheduler.timesteps:
                    raw_t = raw_t.to(device=device)
                    t_vid = torch.ones((B,), device=device, dtype=self.dtype)
                    t_vid = t_vid * (raw_t.to(dtype=self.dtype) / num_train_timesteps)
                    pred_video_v = self.run_cosmos_once(
                        video_latent=x_vid,
                        context_embeds=context_embeds,
                        timestep_vid=t_vid,
                        fps=fps,
                        cosmos_text_embeddings=cosmos_text_embeddings,
                        cosmos_context_embeds=cosmos_context_embeds,
                        num_condition_latent_frames=T_cond,
                    )
                    x_vid = sample_scheduler.step(
                        pred_video_v.to(torch.float32),
                        raw_t,
                        x_vid.to(torch.float32),
                        return_dict=False,
                    )[0].to(self.dtype)
                    x_vid[:, :, :T_cond] = condition_latent

            spatial_result = self.generate_spatial_tokens_cached(
                context_embeds=context_embeds,
                num_spatial_tokens=num_spatial_tokens,
                janus_left_pad_lens=janus_left_pad_lens,
                janus_attention_mask=janus_attention_mask,
                janus_images_seq_mask=janus_images_seq_mask,
                janus_images_emb_mask=janus_images_emb_mask,
                return_spatial_debug=return_spatial_debug or return_latent_visual_debug,
            )
            spatial_debug = None
            if return_spatial_debug or return_latent_visual_debug:
                spatial_embeds, spatial_debug = spatial_result
            else:
                spatial_embeds = spatial_result

            x_act = torch.randn(B, action_chunk, action_dim, device=device, dtype=self.dtype)
            dt = 1.0 / action_denoise_steps
            for step in range(action_denoise_steps):
                t = 1.0 - step * dt
                t_act = torch.full((B,), t, device=device, dtype=self.dtype)
                pred_act_v = self.action_denoise_step_cached(
                    action_latent=x_act,
                    timestep_act=t_act,
                    janus_left_pad_lens=janus_left_pad_lens,
                    janus_images_seq_mask=janus_images_seq_mask,
                    janus_images_emb_mask=janus_images_emb_mask,
                )
                x_act = x_act - dt * pred_act_v

            pred_video = self.cosmos_vae.decode(x_vid.to(self.dtype))
            outputs = [pred_video, x_act]
            if return_spatial_debug or return_latent_visual_debug:
                outputs.append(spatial_debug)
            return tuple(outputs)
        finally:
            self._clear_cached_video_kv()
