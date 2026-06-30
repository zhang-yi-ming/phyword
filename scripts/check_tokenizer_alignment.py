"""
Check whether the processor tokenizer loaded from `model_path` is aligned with
the Janus model loaded from `mot_model_path`.

What this script checks:
  1. Processor tokenizer length vs. model input embedding rows.
  2. Processor tokenizer length vs. model lm_head rows.
  3. Whether the appended special tokens sit in the expected tail layout:
       <action_0> ... <action_255> <|latent_start|> <|latent_pad|> <|latent_end|>
  4. Whether ActionTokenizer's reserved tail range matches the action tokens.
  5. Optional: whether a tokenizer loadable from `mot_model_path` matches the
     processor tokenizer on length and key special token ids.

Usage:
python scripts/check_tokenizer_alignment.py \
    --model_path /media/liuzhuoyang/LCoT_VLA/Janus-Pro-1B \
    --mot_model_path /media/liuzhuoyang/LCoT_VLA/exp_pretrain/action_only_flow/janus_pro_siglip_encoder_1B_no_state_lr_2e-5_flow_1217/checkpoint-4-5530345/tfmr
      

JANUS_MODEL_PATH="/media/liuzhuoyang/LCoT_VLA/Janus-Pro-1B"

MOT_MODEL_PATH="/media/liuzhuoyang/LCoT_VLA/exp_pretrain/action_only_flow/janus_pro_siglip_encoder_1B_no_state_lr_2e-5_flow_1217/checkpoint-4-5530345/tfmr"

COSMOS_PT_PATH="/media/liuzhuoyang/cosmos_mot/ckpts/Cosmos-Predict2.5-2B/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt"
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from janus.models import ActionTokenizer, VLChatProcessor


def get_input_embedding_rows(model) -> int:
    return int(model.language_model.model.embed_tokens.weight.shape[0])


def get_output_embedding_rows(model) -> Optional[int]:
    lm_head = getattr(model.language_model, "lm_head", None)
    if lm_head is None or getattr(lm_head, "weight", None) is None:
        return None
    return int(lm_head.weight.shape[0])


def token_id_or_none(tokenizer, token: str) -> Optional[int]:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        return None
    if isinstance(token_id, int) and token_id < 0:
        return None
    return int(token_id)


def collect_special_token_ids(tokenizer) -> Dict[str, Optional[int]]:
    keys = [
        "<image_placeholder>",
        "<begin_of_image>",
        "<end_of_image>",
        "<action_0>",
        "<action_255>",
        "<|latent_start|>",
        "<|latent_pad|>",
        "<|latent_end|>",
    ]
    return {key: token_id_or_none(tokenizer, key) for key in keys}


def check_tail_layout(tokenizer, need_to_sub: int) -> Tuple[List[str], Dict[str, object]]:
    issues: List[str] = []
    tokenizer_len = len(tokenizer)
    action_ids = [token_id_or_none(tokenizer, f"<action_{i}>") for i in range(256)]
    latent_start = token_id_or_none(tokenizer, "<|latent_start|>")
    latent_pad = token_id_or_none(tokenizer, "<|latent_pad|>")
    latent_end = token_id_or_none(tokenizer, "<|latent_end|>")

    summary: Dict[str, object] = {
        "tokenizer_len": tokenizer_len,
        "action_id_min": min(x for x in action_ids if x is not None) if all(x is not None for x in action_ids) else None,
        "action_id_max": max(x for x in action_ids if x is not None) if all(x is not None for x in action_ids) else None,
        "latent_start_id": latent_start,
        "latent_pad_id": latent_pad,
        "latent_end_id": latent_end,
    }

    if not all(x is not None for x in action_ids):
        issues.append("Some <action_i> tokens are missing from the tokenizer.")
    else:
        expected_action_ids = list(range(tokenizer_len - need_to_sub - 256, tokenizer_len - need_to_sub))
        if action_ids != expected_action_ids:
            issues.append(
                "Action token ids are not laid out as the 256 tokens immediately before the latent special tokens."
            )

    expected_latent_ids = [tokenizer_len - 3, tokenizer_len - 2, tokenizer_len - 1]
    actual_latent_ids = [latent_start, latent_pad, latent_end]
    if actual_latent_ids != expected_latent_ids:
        issues.append(
            f"Latent special token ids are not the final 3 tokenizer ids. "
            f"Expected {expected_latent_ids}, got {actual_latent_ids}."
        )

    action_tokenizer = ActionTokenizer(tokenizer, need_to_sub=need_to_sub)
    reserved_ids = list(range(action_tokenizer.my_vocab_size - action_tokenizer.vocab_size, action_tokenizer.my_vocab_size))
    summary["action_tokenizer_reserved_min"] = reserved_ids[0]
    summary["action_tokenizer_reserved_max"] = reserved_ids[-1]
    if all(x is not None for x in action_ids) and set(action_ids) != set(reserved_ids):
        issues.append(
            "ActionTokenizer reserved id range does not match the ids assigned to <action_0> ... <action_255>."
        )

    return issues, summary


def compare_two_tokenizers(ref_tokenizer, other_tokenizer) -> List[str]:
    issues: List[str] = []
    if len(ref_tokenizer) != len(other_tokenizer):
        issues.append(
            f"Tokenizer length mismatch: model_path tokenizer={len(ref_tokenizer)}, "
            f"mot_model_path tokenizer={len(other_tokenizer)}."
        )

    for token in [
        "<action_0>",
        "<action_255>",
        "<|latent_start|>",
        "<|latent_pad|>",
        "<|latent_end|>",
    ]:
        ref_id = token_id_or_none(ref_tokenizer, token)
        other_id = token_id_or_none(other_tokenizer, token)
        if ref_id != other_id:
            issues.append(
                f"Token id mismatch for {token}: model_path tokenizer={ref_id}, mot_model_path tokenizer={other_id}."
            )
    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path used to load VLChatProcessor")
    parser.add_argument("--mot_model_path", type=str, required=True, help="Path used to load Janus model weights")
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--need_to_sub", type=int, default=3, help="ActionTokenizer need_to_sub value")
    args = parser.parse_args()

    print("=== Loading processor/tokenizer from model_path ===")
    processor = VLChatProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    print(f"processor.original_tokenizer_len = {getattr(processor, 'original_tokenizer_len', 'N/A')}")
    print(f"processor.num_add_tokens        = {getattr(processor, 'num_add_tokens', 'N/A')}")
    print(f"len(processor.tokenizer)       = {len(tokenizer)}")

    print("\n=== Loading Janus model from mot_model_path ===")
    model = AutoModelForCausalLM.from_pretrained(
        args.mot_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        flow=True,
        action_dim=args.action_dim,
        ignore_mismatched_sizes=True,
    )
    input_rows = get_input_embedding_rows(model)
    output_rows = get_output_embedding_rows(model)
    print(f"model input embedding rows     = {input_rows}")
    print(f"model lm_head rows             = {output_rows}")

    issues: List[str] = []

    if len(tokenizer) != input_rows:
        issues.append(
            f"Tokenizer length ({len(tokenizer)}) != input embedding rows ({input_rows})."
        )
    if output_rows is not None and len(tokenizer) != output_rows:
        issues.append(
            f"Tokenizer length ({len(tokenizer)}) != lm_head rows ({output_rows})."
        )

    special_ids = collect_special_token_ids(tokenizer)
    print("\n=== Key token ids from model_path tokenizer ===")
    for key, value in special_ids.items():
        print(f"{key:<20} -> {value}")

    for token, token_id in special_ids.items():
        if token_id is None:
            issues.append(f"Special token missing from tokenizer: {token}")
        elif token_id >= input_rows:
            issues.append(
                f"Token id out of embedding range: {token} has id {token_id}, but input embeddings only have {input_rows} rows."
            )

    tail_issues, tail_summary = check_tail_layout(tokenizer, need_to_sub=args.need_to_sub)
    issues.extend(tail_issues)
    print("\n=== Tail layout summary ===")
    for key, value in tail_summary.items():
        print(f"{key:<30} = {value}")

    print("\n=== Optional tokenizer load from mot_model_path ===")
    try:
        mot_tokenizer = AutoTokenizer.from_pretrained(
            args.mot_model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        print(f"len(mot_model_path tokenizer)   = {len(mot_tokenizer)}")
        issues.extend(compare_two_tokenizers(tokenizer, mot_tokenizer))
    except Exception as exc:
        print(f"Could not load tokenizer from mot_model_path: {exc}")

    print("\n=== Verdict ===")
    if issues:
        for item in issues:
            print(f"[FAIL] {item}")
        sys.exit(1)

    print("[PASS] model_path tokenizer, action/latent tail layout, and mot_model_path embeddings are aligned.")


if __name__ == "__main__":
    main()
