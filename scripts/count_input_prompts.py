#!/usr/bin/env python3
"""Count unique input_prompt values in a training JSON file."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DATA_PATH = (
    "/mnt/nas/zhangyiming/database/data/libero_training_data_last05_lastest/"
    "libero_spatial_20hz_224_dual/train.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count and print unique input_prompt values from a training JSON file."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to training JSON. Defaults to {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="input_prompt",
        help="Field name to count. Defaults to input_prompt.",
    )
    parser.add_argument(
        "--count_only",
        action="store_true",
        help="Only print summary counts, not the unique prompt values.",
    )
    parser.add_argument(
        "--strip",
        action="store_true",
        help="Strip leading/trailing whitespace before counting prompts.",
    )
    return parser.parse_args()


def load_samples(data_path: Path) -> list[dict[str, Any]]:
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        for key in ("data", "samples", "train"):
            value = data.get(key)
            if isinstance(value, list):
                samples = value
                break
        else:
            raise ValueError(
                f"{data_path} is a JSON object, but none of data/samples/train is a list."
            )
    else:
        raise ValueError(f"{data_path} must contain a JSON list or object, got {type(data).__name__}.")

    bad_indices = [idx for idx, sample in enumerate(samples) if not isinstance(sample, dict)]
    if bad_indices:
        preview = ", ".join(str(idx) for idx in bad_indices[:10])
        raise ValueError(f"All samples must be JSON objects. Bad sample indices: {preview}")

    return samples


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    samples = load_samples(data_path)

    prompts = []
    missing = 0
    for sample in samples:
        if args.field not in sample:
            missing += 1
            continue
        prompt = sample[args.field]
        if not isinstance(prompt, str):
            prompt = str(prompt)
        if args.strip:
            prompt = prompt.strip()
        prompts.append(prompt)

    counts = Counter(prompts)

    print(f"data_path: {data_path}")
    print(f"total_samples: {len(samples)}")
    print(f"samples_with_{args.field}: {len(prompts)}")
    print(f"missing_{args.field}: {missing}")
    print(f"unique_{args.field}: {len(counts)}")

    if not args.count_only:
        print(f"\nunique {args.field} values:")
        for index, (prompt, count) in enumerate(sorted(counts.items()), start=1):
            print(f"[{index}] count={count} | {prompt}")


if __name__ == "__main__":
    main()
