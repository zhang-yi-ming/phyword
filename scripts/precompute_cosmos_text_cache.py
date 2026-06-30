#!/usr/bin/env python3
"""Precompute raw Cosmos Reason1/Qwen text embeddings for training JSON files."""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

from utils.cosmos_text_cache import (
    CosmosQwenTextEmbedder,
    CosmosTextEmbeddingCache,
    RAW_COSMOS_TEXT_SHAPE,
    canonicalize_prompt,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read a VLA training JSON, deduplicate input_prompt values, and cache native "
            f"Cosmos Reason1/Qwen raw FULL_CONCAT text embeddings shaped {RAW_COSMOS_TEXT_SHAPE}."
        )
    )
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_cache_path", type=str, required=True)
    parser.add_argument("--cosmos_model_path", type=str, required=True)
    parser.add_argument(
        "--cosmos_experiment_name",
        type=str,
        required=True,
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_unique_prompts(input_json: str) -> list[str]:
    with open(input_json, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected {input_json} to contain a JSON list, got {type(data)!r}.")

    seen = set()
    prompts = []
    for index, sample in enumerate(data):
        if "input_prompt" not in sample:
            raise KeyError(f"Sample {index} is missing required field 'input_prompt'.")
        prompt = canonicalize_prompt(sample["input_prompt"])
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
    return prompts


def load_cosmos_text_config(args):
    # Keep Cosmos wrapper loading lightweight by blocking online text encoder construction.
    import cosmos_predict2._src.predict2.models.text2world_model_rectified_flow as t2w_module
    from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint

    class DummyTextEncoder(nn.Module):
        def __init__(self, *unused_args, **unused_kwargs):
            super().__init__()

    t2w_module.TextEncoder = DummyTextEncoder

    experiment_opts = ["data_train=mock", "data_val=mock"]
    cosmos_wrapper, cosmos_config = load_model_from_checkpoint(
        experiment_name=args.cosmos_experiment_name,
        s3_checkpoint_dir=args.cosmos_model_path,
        config_file="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        load_ema_to_reg=True,
        to_device=args.device,
        experiment_opts=experiment_opts,
    )

    text_encoder_config = getattr(cosmos_config.model.config, "text_encoder_config", None)
    if text_encoder_config is None:
        raise ValueError(
            "The selected Cosmos experiment has no text_encoder_config; cannot compute native Qwen embeddings."
        )
    if not bool(getattr(text_encoder_config, "compute_online", False)):
        raise ValueError(
            "The selected Cosmos experiment does not use online text encoding; "
            "this cache script expects a Reason1/Qwen online encoder."
        )
    cosmos_wrapper.net.eval()
    return cosmos_wrapper.net, text_encoder_config


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    prompts = load_unique_prompts(args.input_json)
    cache = CosmosTextEmbeddingCache(args.output_cache_path, create=True)

    pending = [
        prompt
        for prompt in prompts
        if args.overwrite or not cache.contains(prompt)
    ]

    print(f"Loaded {len(prompts)} unique prompts from {args.input_json}.")
    print(f"Cache path: {os.path.abspath(args.output_cache_path)}")
    print(f"Pending prompts: {len(pending)} (overwrite={int(args.overwrite)})")
    if not pending:
        return

    cosmos_dit, text_encoder_config = load_cosmos_text_config(args)
    embedder = CosmosQwenTextEmbedder(
        cosmos_dit=cosmos_dit,
        text_encoder_config=text_encoder_config,
        device=args.device,
    )

    total_saved = 0
    for start in range(0, len(pending), args.batch_size):
        batch_prompts = pending[start : start + args.batch_size]
        embeddings = embedder.compute_batch(batch_prompts)
        cache.save_many(
            list(zip(batch_prompts, embeddings)),
            overwrite=args.overwrite,
            metadata={
                "source": "cosmos_reason1p1_qwen_raw_full_concat",
                "embedding_format": "raw_qwen_full_concat",
                "cosmos_experiment_name": args.cosmos_experiment_name,
            },
        )
        total_saved += len(batch_prompts)
        print(f"Saved {total_saved}/{len(pending)} cached embeddings.")

    del embedder, cosmos_dit
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
