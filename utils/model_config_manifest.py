"""Save and validate the model-defining configuration beside checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Any, Optional


MODEL_CONFIG_FILENAME = "model_config.json"
MODEL_CONFIG_SCHEMA_VERSION = 1

MODEL_CONFIG_FIELDS = (
    "model_variant",
    "qwen3vl2b_model_path",
    "action_model_path",
    "cosmos_model_path",
    "cosmos_experiment_name",
    "cosmos_text_cache_path",
    "video_h",
    "video_w",
    "video_frames",
    "num_cond_input_frames",
    "fps",
    "action_dim",
    "action_chunk",
    "total_latent_tokens",
    "latent_token_mode",
    "latent_token_fields",
    "special_token_vocab",
    "special_token_weight_tied",
    "robot_state",
    "state_placeholder_tokens",
    "state_dim",
    "state_encoding_mode",
    "bridge_pos_scheme",
    "right_single_attn_position",
    "detach_action_cosmos_kv",
)

# These fields change tensor shapes or forward semantics and must never be
# silently overridden when evaluating a self-describing checkpoint.
STRICT_RUNTIME_FIELDS = (
    "video_h",
    "video_w",
    "video_frames",
    "num_cond_input_frames",
    "action_dim",
    "action_chunk",
    "total_latent_tokens",
    "latent_token_mode",
    "special_token_vocab",
    "special_token_weight_tied",
    "robot_state",
    "state_placeholder_tokens",
    "state_dim",
    "state_encoding_mode",
    "bridge_pos_scheme",
    "right_single_attn_position",
)


def _get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def canonical_model_config(config: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field in MODEL_CONFIG_FIELDS:
        if field == "action_model_path":
            value = _get(config, field, None) or _get(config, "action_expert_path", None)
        elif field == "model_variant":
            value = _get(config, field, "mot2_action_spatial")
        else:
            value = _get(config, field, None)
        if value is not None:
            values[field] = _json_value(value)
    return values


def _git_source_state(project_root: Optional[str]) -> dict[str, Any]:
    if not project_root:
        return {}
    root = os.path.abspath(project_root)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--", "."],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {
        "git_commit": commit,
        "git_dirty": bool(status.strip()),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def build_model_manifest(config: Any, *, project_root: Optional[str] = None) -> dict[str, Any]:
    return {
        "schema_version": MODEL_CONFIG_SCHEMA_VERSION,
        "model_config": canonical_model_config(config),
        "source": _git_source_state(project_root),
    }


def save_model_manifest(
    checkpoint_dir: str,
    config: Any,
    *,
    project_root: Optional[str] = None,
) -> str:
    path = os.path.join(checkpoint_dir, MODEL_CONFIG_FILENAME)
    payload = build_model_manifest(config, project_root=project_root)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
    return path


def load_model_manifest(checkpoint_dir: str) -> Optional[dict[str, Any]]:
    path = os.path.join(str(checkpoint_dir), MODEL_CONFIG_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("schema_version", -1)) != MODEL_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported model config schema in {path}: "
            f"{payload.get('schema_version')!r}."
        )
    if not isinstance(payload.get("model_config"), dict):
        raise ValueError(f"Invalid model config manifest: {path}")
    return payload


def validate_runtime_model_config(
    runtime_config: Any,
    manifest: dict[str, Any],
    *,
    fields: tuple[str, ...] = STRICT_RUNTIME_FIELDS,
) -> None:
    saved = manifest["model_config"]
    runtime = canonical_model_config(runtime_config)
    mismatches = []
    for field in fields:
        if field not in saved or field not in runtime:
            continue
        if runtime[field] != saved[field]:
            mismatches.append(
                f"{field}: checkpoint={saved[field]!r}, runtime={runtime[field]!r}"
            )
    if mismatches:
        raise ValueError(
            "Runtime model configuration does not match checkpoint manifest: "
            + "; ".join(mismatches)
        )
