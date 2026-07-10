import json
import os
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen_vla import Qwen3VLVLAModel


def resolve_trex_checkpoint_path(path: str) -> str:
    """Return the concrete T-Rex checkpoint directory containing model.pt."""
    path = os.path.abspath(os.path.expanduser(str(path)))
    if os.path.isfile(os.path.join(path, "model.pt")):
        return path
    candidate = os.path.join(path, "checkpoint-0-610000")
    if os.path.isfile(os.path.join(candidate, "model.pt")):
        return candidate
    checkpoints = []
    if os.path.isdir(path):
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if name.startswith("checkpoint-") and os.path.isfile(os.path.join(full, "model.pt")):
                checkpoints.append(full)
    if checkpoints:
        return sorted(checkpoints)[-1]
    raise FileNotFoundError(f"Could not find T-Rex model.pt under {path}")


class TiedLMHead(nn.Module):
    """LM head tied to Qwen/T-Rex input embeddings."""

    def __init__(self, embed_tokens: nn.Embedding):
        super().__init__()
        self.embed_tokens = embed_tokens

    @property
    def weight(self):
        return self.embed_tokens.weight

    def set_embed_tokens(self, embed_tokens: nn.Embedding):
        self.embed_tokens = embed_tokens

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.embed_tokens.weight)


class TrexActionModel(nn.Module):
    """Compatibility wrapper exposing T-Rex action expert through Janus-like names."""

    is_trex_action_model = True

    def __init__(self, vla: Qwen3VLVLAModel):
        super().__init__()
        self.vla = vla
        self.config = SimpleNamespace(
            hidden_size=int(vla.config.hidden_size),
            vocab_size=int(vla.model.embed_tokens.weight.shape[0]),
            image_token_id=int(vla.image_token_id),
        )
        self.t_embedder = vla.t_embedder
        self.x_embedder = vla.x_embedder
        self.final_layer = vla.final_layer
        self.lm_head = TiedLMHead(vla.model.embed_tokens)
        self._refresh_language_model_proxy()

    def _refresh_language_model_proxy(self):
        text_proxy = SimpleNamespace(
            layers=self.vla.model.layers,
            norm=self.vla.model.norm_action,
            norm_action=self.vla.model.norm_action,
            embed_tokens=self.vla.model.embed_tokens,
            rotary_emb=self.vla.model.rotary_emb,
        )
        language_proxy = SimpleNamespace(
            model=text_proxy,
            lm_head=self.lm_head,
            get_input_embeddings=lambda: self.vla.model.embed_tokens,
            resize_token_embeddings=self.resize_token_embeddings,
        )
        object.__setattr__(self, "language_model", language_proxy)

    @property
    def image_token_id(self) -> int:
        return int(self.vla.image_token_id)

    @property
    def visual(self):
        return self.vla.visual

    @staticmethod
    def _load_config(config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            full_cfg = json.load(f)
        text_cfg_dict = full_cfg.get("text_config", full_cfg)
        try:
            from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

            text_config = Qwen3VLTextConfig(**text_cfg_dict)
        except Exception:
            text_config = SimpleNamespace(**text_cfg_dict)
        return full_cfg, text_config

    @staticmethod
    def _build_visual(full_cfg: dict):
        vision_cfg = full_cfg.get("vision_config", {})
        model_type = full_cfg.get("model_type", "qwen3_vl")
        try:
            if model_type == "qwen3_vl":
                from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
                from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

                cfg = Qwen3VLVisionConfig(**{k: v for k, v in vision_cfg.items() if k != "model_type"})
                return Qwen3VLVisionModel(cfg)
            from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLVisionModel

            cfg = Qwen2VLVisionConfig(**{k: v for k, v in vision_cfg.items() if k != "model_type"})
            return Qwen2VLVisionModel(cfg)
        except Exception as exc:
            raise RuntimeError(f"Failed to build Qwen visual tower from T-Rex config: {exc}") from exc

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        action_dim: int,
        action_chunk: int,
        torch_dtype=torch.bfloat16,
        use_robot_state: bool = False,
        verbose: bool = True,
    ) -> Tuple["TrexActionModel", dict]:
        ckpt_dir = resolve_trex_checkpoint_path(checkpoint_path)
        full_cfg, text_config = cls._load_config(os.path.join(ckpt_dir, "config.json"))
        image_token_id = int(full_cfg.get("image_token_id", 151655))
        vla = Qwen3VLVLAModel(
            config=text_config,
            action_dim=int(action_dim),
            action_chunk=int(action_chunk),
            use_robot_state=bool(use_robot_state),
            image_token_id=image_token_id,
            use_tactile_deform=False,
            use_tactile_code=False,
            use_tactile_vqvae=False,
            n_flare_tokens_per_frame=0,
            n_flare_steps=0,
        )
        vla.visual = cls._build_visual(full_cfg)
        vla.initialize_vla_weights()

        ckpt_file = os.path.join(ckpt_dir, "model.pt")
        state = torch.load(ckpt_file, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model_state = vla.state_dict()
        filtered = {}
        skipped = []
        for key, value in state.items():
            if key not in model_state:
                continue
            if tuple(model_state[key].shape) != tuple(value.shape):
                skipped.append((key, tuple(value.shape), tuple(model_state[key].shape)))
                continue
            filtered[key] = value
        missing, unexpected = vla.load_state_dict(filtered, strict=False)
        vla.to(dtype=torch_dtype)
        wrapped = cls(vla)
        info = {
            "checkpoint_dir": ckpt_dir,
            "loaded": len(filtered),
            "skipped_mismatch": skipped,
            "missing": missing,
            "unexpected": unexpected,
        }
        if verbose:
            print(
                "[TrexActionModel] loaded "
                f"{len(filtered)} tensors from {ckpt_dir}; "
                f"skipped_mismatch={len(skipped)}, missing={len(missing)}, unexpected={len(unexpected)}"
            )
            if skipped:
                print(f"  first skipped mismatch: {skipped[0]}")
        return wrapped, info

    def resize_token_embeddings(self, target_vocab_size: int):
        target_vocab_size = int(target_vocab_size)
        old_embed = self.vla.model.embed_tokens
        old_weight = old_embed.weight
        old_vocab, hidden = old_weight.shape
        if target_vocab_size == old_vocab:
            return old_embed
        new_embed = nn.Embedding(
            target_vocab_size,
            hidden,
            padding_idx=getattr(old_embed, "padding_idx", None),
            device=old_weight.device,
            dtype=old_weight.dtype,
        )
        std = float(getattr(self.vla.config, "initializer_range", 0.02))
        nn.init.normal_(new_embed.weight, mean=0.0, std=std)
        copy_rows = min(old_vocab, target_vocab_size)
        with torch.no_grad():
            new_embed.weight[:copy_rows].copy_(old_weight[:copy_rows])
        self.vla.model.embed_tokens = new_embed
        self.lm_head.set_embed_tokens(new_embed)
        self.config.vocab_size = target_vocab_size
        self.vla.config.vocab_size = target_vocab_size
        self.vla.model.config.vocab_size = target_vocab_size
        self._refresh_language_model_proxy()
        return new_embed

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.vla.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    def visual_mean_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        return self.visual_token_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            batch_size=batch_size,
        ).mean(dim=1)

    def visual_token_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if pixel_values is None or image_grid_thw is None:
            raise ValueError("pixel_values and image_grid_thw are required for T-Rex visual features.")
        dtype = self.vla.model.embed_tokens.weight.dtype
        device = self.vla.model.embed_tokens.weight.device
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        image_grid_thw = image_grid_thw.to(device=device, dtype=torch.long)
        out = self.vla.visual(pixel_values, grid_thw=image_grid_thw)
        features = out[0] if isinstance(out, (tuple, list)) else out
        merge = int(getattr(self.vla.visual, "spatial_merge_size", 2) or 2)
        counts = [
            int(g[0].item() * (g[1].item() // merge) * (g[2].item() // merge))
            for g in image_grid_thw
        ]
        if len(counts) != int(batch_size):
            raise ValueError(
                f"Expected one image per batch item for visual_mean_features, "
                f"got grids={len(counts)} batch={batch_size}."
            )
        if len(set(counts)) != 1:
            raise ValueError(f"Expected equal visual token counts per batch item, got {counts}.")
        chunks = torch.split(features, counts, dim=0)
        return torch.stack(list(chunks), dim=0)
