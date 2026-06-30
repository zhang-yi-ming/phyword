"""Debug VLACotDataset batches without loading the full training model.

This script mirrors the dataset construction used by train_cot.py:
  - load VLChatProcessor
  - build VLACotDataset
  - collate samples with VLACotDataset.collate_fn

It prints token, mask, padding, and tensor summaries so the dataset can be
inspected before running the expensive model forward.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from janus.models import VLChatProcessor  # noqa: E402
from train_cot import VLACotDataset  # noqa: E402


class PrintAccelerator:
    """Small stand-in for Accelerate's print interface used by VLACotDataset."""

    @staticmethod
    def print(*args, **kwargs):
        print(*args, **kwargs)


def parse_indices(indices: str) -> Optional[list[int]]:
    if not indices:
        return None
    parsed = []
    for piece in indices.split(","):
        piece = piece.strip()
        if not piece:
            continue
        parsed.append(int(piece))
    return parsed or None


def as_int_list(values: torch.Tensor) -> list[int]:
    return [int(x) for x in values.detach().cpu().reshape(-1).tolist()]


def token_for_id(tokenizer, token_id: int) -> str:
    try:
        token = tokenizer.convert_ids_to_tokens(int(token_id))
    except Exception:
        token = None
    return str(token)


def tokens_for_ids(tokenizer, token_ids: Iterable[int]) -> list[str]:
    ids = [int(x) for x in token_ids]
    try:
        tokens = tokenizer.convert_ids_to_tokens(ids)
    except Exception:
        tokens = [token_for_id(tokenizer, x) for x in ids]
    if isinstance(tokens, str):
        tokens = [tokens]
    return [str(x) for x in tokens]


def tensor_summary(name: str, value) -> None:
    if value is None:
        print(f"{name}: None")
        return
    if not torch.is_tensor(value):
        print(f"{name}: {type(value).__name__} = {value}")
        return

    shape = tuple(value.shape)
    summary = f"{name}: shape={shape}, dtype={value.dtype}"
    if value.numel() == 0:
        print(summary + ", numel=0")
        return

    detached = value.detach()
    if detached.dtype == torch.bool:
        true_count = int(detached.sum().item())
        summary += f", true={true_count}, false={detached.numel() - true_count}"
    elif detached.is_floating_point():
        summary += (
            f", min={float(detached.min().item()):.6g}, "
            f"max={float(detached.max().item()):.6g}, "
            f"mean={float(detached.float().mean().item()):.6g}"
        )
    elif detached.dtype in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.long,
    ):
        summary += f", min={int(detached.min().item())}, max={int(detached.max().item())}"
    print(summary)


def collate_pad_id(tokenizer) -> int:
    return int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0


def print_tokenizer_info(processor, tokenizer) -> None:
    pad_id = tokenizer.pad_token_id
    processor_pad_id = getattr(processor, "pad_id", None)
    latent_start_id = tokenizer.convert_tokens_to_ids("<|latent_start|>")
    image_id = getattr(processor, "image_id", None)

    print("\n=== Tokenizer / Processor ===")
    print(f"len(tokenizer): {len(tokenizer)}")
    print(f"tokenizer.pad_token: {getattr(tokenizer, 'pad_token', None)}")
    print(f"tokenizer.pad_token_id: {pad_id}")
    print(f"processor.pad_id: {processor_pad_id}")
    print(f"collate pad id used by train_cot.py: {collate_pad_id(tokenizer)}")
    print(f"id 0 token: {token_for_id(tokenizer, 0)}")
    if pad_id is not None:
        print(f"pad id token: {token_for_id(tokenizer, int(pad_id))}")
    if processor_pad_id is not None:
        print(f"processor pad id token: {token_for_id(tokenizer, int(processor_pad_id))}")
    print(f"<|latent_start|> id: {latent_start_id}")
    print(f"processor.image_id: {image_id}")
    if image_id is not None:
        print(f"image token: {token_for_id(tokenizer, int(image_id))}")


def find_positions(ids: torch.Tensor, token_id: int) -> list[int]:
    if token_id is None or token_id < 0:
        return []
    positions = (ids == int(token_id)).nonzero(as_tuple=False).reshape(-1)
    return as_int_list(positions)


def effective_non_pad(ids: torch.Tensor, pad_id: int) -> tuple[int, int, int]:
    non_pad = ids != int(pad_id)
    non_pad_count = int(non_pad.sum().item())
    pad_count = int(ids.numel() - non_pad_count)
    if non_pad_count == 0:
        return 0, pad_count, -1
    last_non_pad = int(non_pad.nonzero(as_tuple=False).reshape(-1)[-1].item())
    return non_pad_count, pad_count, last_non_pad


def print_tail_tokens(
    tokenizer,
    ids: torch.Tensor,
    title: str,
    tail_tokens: int,
    decode: bool,
    pad_id: int,
) -> None:
    ids = ids.detach().cpu()
    tail = ids[-min(tail_tokens, ids.numel()) :]
    tail_ids = as_int_list(tail)
    tail_token_strings = tokens_for_ids(tokenizer, tail_ids)
    print(f"{title} tail ids: {tail_ids}")
    print(f"{title} tail tokens: {tail_token_strings}")
    if decode:
        try:
            decoded_all = tokenizer.decode(as_int_list(ids), skip_special_tokens=False)
            decoded_no_pad = tokenizer.decode(
                [x for x in as_int_list(ids) if x != int(pad_id)],
                skip_special_tokens=False,
            )
            print(f"{title} decoded all: {decoded_all}")
            print(f"{title} decoded without collate pad id: {decoded_no_pad}")
        except Exception as exc:
            print(f"{title} decode failed: {exc}")


def print_sample(sample: dict, sample_name: str, processor, tokenizer, args) -> None:
    pad_id = collate_pad_id(tokenizer)
    latent_start_id = tokenizer.convert_tokens_to_ids("<|latent_start|>")

    print(f"\n=== Single Sample: {sample_name} ===")
    for key, value in sample.items():
        tensor_summary(key, value)

    ids = sample["janus_input_ids"]
    seq_mask = sample["janus_images_seq_mask"]
    state_mask = sample["janus_state_seq_mask"]
    emb_mask = sample["janus_images_emb_mask"]
    latent_positions = find_positions(ids, latent_start_id)
    non_pad_count, pad_count, last_non_pad = effective_non_pad(ids, pad_id)

    print(f"janus_input_ids length: {ids.numel()}")
    print(f"non-pad count by collate pad id ({pad_id}): {non_pad_count}")
    print(f"pad count by collate pad id ({pad_id}): {pad_count}")
    print(f"last non-pad index: {last_non_pad}")
    print(f"<|latent_start|> positions: {latent_positions}")
    print(f"janus_images_seq_mask.sum(): {int(seq_mask.sum().item())}")
    print(f"janus_state_seq_mask.sum(): {int(state_mask.sum().item())}")
    print(f"janus_images_emb_mask.sum(): {int(emb_mask.sum().item())}")
    print(
        "image mask count match: "
        f"{int(seq_mask.sum().item()) == int(emb_mask.sum().item())}"
    )
    print_tail_tokens(tokenizer, ids, sample_name, args.tail_tokens, args.decode, pad_id)


def print_batch(batch: dict, batch_name: str, processor, tokenizer, args) -> None:
    pad_id = collate_pad_id(tokenizer)
    latent_start_id = tokenizer.convert_tokens_to_ids("<|latent_start|>")

    print(f"\n=== Batch: {batch_name} ===")
    for key in (
        "janus_input_ids",
        "janus_images_seq_mask",
        "janus_state_seq_mask",
        "janus_images_emb_mask",
        "janus_pixel_values",
        "videos",
        "actions",
        "gt_latent_token_ids",
        "future_pixel_values",
        "future_state_ids",
    ):
        tensor_summary(key, batch.get(key))

    ids_batch = batch["janus_input_ids"]
    seq_mask_batch = batch["janus_images_seq_mask"]
    state_mask_batch = batch["janus_state_seq_mask"]
    emb_mask_batch = batch["janus_images_emb_mask"]
    batch_size, s_context = ids_batch.shape
    total_latent_tokens = int(getattr(args, "total_latent_tokens", 0))
    latent_pred_indices = list(range(s_context - 1, s_context - 1 + total_latent_tokens))

    print(f"S_context = janus_input_ids.shape[1] = {s_context}")
    print(
        "simulated train_cot latent_pred_indices "
        f"(total_latent_tokens={total_latent_tokens}): {latent_pred_indices}"
    )

    seq_counts = seq_mask_batch.reshape(batch_size, -1).sum(dim=1)
    state_counts = state_mask_batch.reshape(batch_size, -1).sum(dim=1)
    emb_counts = emb_mask_batch.reshape(batch_size, -1).sum(dim=1)

    for i in range(batch_size):
        ids = ids_batch[i]
        non_pad_count, pad_count, last_non_pad = effective_non_pad(ids, pad_id)
        latent_positions = find_positions(ids, latent_start_id)
        last_latent = latent_positions[-1] if latent_positions else -1
        pad_after_latent = False
        tokens_after_latent = []
        if last_latent >= 0 and last_latent + 1 < ids.numel():
            after = ids[last_latent + 1 :]
            pad_after_latent = bool((after == int(pad_id)).any().item())
            tokens_after_latent = as_int_list(after[: args.tail_tokens])

        print(f"\n--- Batch sample {i} ---")
        print(f"non-pad count: {non_pad_count}")
        print(f"pad count: {pad_count}")
        print(f"last non-pad index: {last_non_pad}")
        print(f"<|latent_start|> positions: {latent_positions}")
        print(f"image seq mask count: {int(seq_counts[i].item())}")
        print(f"state seq mask count: {int(state_counts[i].item())}")
        print(f"image emb mask count: {int(emb_counts[i].item())}")
        print(f"image mask count match: {int(seq_counts[i].item()) == int(emb_counts[i].item())}")
        print(f"pad appears after <|latent_start|>: {pad_after_latent}")
        if tokens_after_latent:
            print(f"ids after last <|latent_start|> (clipped): {tokens_after_latent}")
            print(
                "tokens after last <|latent_start|> (clipped): "
                f"{tokens_for_ids(tokenizer, tokens_after_latent)}"
            )
        print_tail_tokens(
            tokenizer,
            ids,
            f"{batch_name} sample {i}",
            args.tail_tokens,
            args.decode,
            pad_id,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print VLACotDataset samples and batches for debugging.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/data/zhangxuheng/data/libero_training_data_last05/libero_cosmos_janus/train.json",
    )
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/data/zhangxuheng/ckpt/pretrained/Janus-Pro-1B",
    )
    parser.add_argument("--video_h", type=int, default=256)
    parser.add_argument("--video_w", type=int, default=256)
    parser.add_argument("--video_frames", type=int, default=16)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--action_chunk", type=int, default=16)
    parser.add_argument("--robot_state", type=int, default=0)
    parser.add_argument("--total_latent_tokens", type=int, default=1)
    parser.add_argument("--img_latents_per_future", type=int, default=0)
    parser.add_argument("--state_latents_per_future", type=int, default=0)
    parser.add_argument("--num_future_frames", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--indices",
        type=str,
        default="",
        help="Comma-separated dataset indices, e.g. '0,1'. If set, collates exactly these samples.",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle DataLoader when --indices is not set.")
    parser.add_argument("--decode", action="store_true", help="Decode full token ids for each printed sample.")
    parser.add_argument("--tail_tokens", type=int, default=32)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    selected_indices = parse_indices(args.indices)

    print("=== Building VLChatProcessor ===")
    processor = VLChatProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    print_tokenizer_info(processor, tokenizer)

    print("\n=== Building VLACotDataset ===")
    dataset = VLACotDataset(args, processor, PrintAccelerator())
    print(f"dataset length: {len(dataset)}")

    if selected_indices is not None:
        print(f"\n=== Debugging explicit indices: {selected_indices} ===")
        samples = []
        for index in selected_indices:
            sample = dataset[index]
            samples.append(sample)
            print_sample(sample, f"dataset[{index}]", processor, tokenizer, args)
        batch = dataset.collate_fn(samples)
        print_batch(batch, "explicit indices", processor, tokenizer, args)
        return

    print("\n=== Debugging dataset[0] before DataLoader ===")
    first_sample = dataset[0]
    print_sample(first_sample, "dataset[0]", processor, tokenizer, args)

    print("\n=== Building DataLoader ===")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break
        print_batch(batch, f"dataloader batch {batch_idx}", processor, tokenizer, args)


if __name__ == "__main__":
    main()
