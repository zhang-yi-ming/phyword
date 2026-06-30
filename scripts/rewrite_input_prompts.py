#!/usr/bin/env python3
"""Rewrite selected input_prompt values in a training JSON file."""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DATA_PATH = (
    "/mnt/nas/zhangyiming/database/data/libero_training_data_last05_lastest/"
    "libero_spatial_20hz_224_dual/train.json"
)


PROMPT_REPLACEMENTS = {
    "pick up the black bowl between the plate and the ramekin and place it on the plate": (
        "task 1: grasp the black bowl located in the narrow gap between the plate and the ramekin, "
        "then set that bowl down on top of the plate"
    ),
    "pick up the black bowl from table center and place it on the plate": (
        "task 2: take the black bowl resting at the center of the table surface and move it onto the plate"
    ),
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": (
        "task 3: retrieve the black bowl from inside the top drawer of the wooden cabinet, "
        "then place it onto the plate"
    ),
    "pick up the black bowl next to the cookie box and place it on the plate": (
        "task 4: grab the black bowl positioned beside the cookie box and transfer it onto the plate"
    ),
    "pick up the black bowl next to the plate and place it on the plate": (
        "task 5: pick the black bowl that is immediately adjacent to the plate and put it directly onto the plate"
    ),
    "pick up the black bowl next to the ramekin and place it on the plate": (
        "task 6: lift the black bowl sitting beside the ramekin and carry it over to the plate"
    ),
    "pick up the black bowl on the cookie box and place it on the plate": (
        "task 7: remove the black bowl from the top of the cookie box and place the bowl on the plate"
    ),
    "pick up the black bowl on the ramekin and place it on the plate": (
        "task 8: take the black bowl that is stacked on the ramekin and set it onto the plate"
    ),
    "pick up the black bowl on the stove and place it on the plate": (
        "task 9: retrieve the black bowl from the stove area and put it down on the plate"
    ),
    "pick up the black bowl on the wooden cabinet and place it on the plate": (
        "task 10: lift the black bowl from the top surface of the wooden cabinet and relocate it onto the plate"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a training JSON, replace known input_prompt strings with more distinct "
            "semantic-preserving variants, and write a new JSON file."
        )
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to source training JSON. Defaults to {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Path to output JSON. Defaults to <data_path stem>_rewritten_prompts.json.",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="input_prompt",
        help="Field name to rewrite. Defaults to input_prompt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any prompt value is not in PROMPT_REPLACEMENTS.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Pretty-print JSON with this indent. Defaults to compact JSON.",
    )
    return parser.parse_args()


def load_json(data_path: Path) -> Any:
    with data_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_samples(data: Any, data_path: Path) -> list[dict[str, Any]]:
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


def default_output_path(data_path: Path) -> Path:
    return data_path.with_name(f"{data_path.stem}_rewritten_prompts{data_path.suffix}")


def rewrite_prompts(samples: list[dict[str, Any]], field: str, strict: bool) -> Counter:
    stats = Counter()
    unknown_prompts = Counter()

    for sample in samples:
        if field not in sample:
            stats["missing_field"] += 1
            continue

        prompt = sample[field]
        if prompt in PROMPT_REPLACEMENTS:
            sample[field] = PROMPT_REPLACEMENTS[prompt]
            stats["rewritten"] += 1
        else:
            stats["unchanged"] += 1
            unknown_prompts[prompt] += 1

    if strict and unknown_prompts:
        preview = "\n".join(
            f"count={count} | {prompt}" for prompt, count in unknown_prompts.most_common(20)
        )
        raise ValueError(f"Found prompt values without replacements:\n{preview}")

    return stats


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output_path = Path(args.output_path) if args.output_path else default_output_path(data_path)

    if output_path.resolve() == data_path.resolve():
        raise ValueError("output_path must be different from data_path so the source JSON stays unchanged.")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    data = load_json(data_path)
    rewritten_data = copy.deepcopy(data)
    samples = extract_samples(rewritten_data, data_path)
    stats = rewrite_prompts(samples, field=args.field, strict=args.strict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rewritten_data, f, ensure_ascii=False, indent=args.indent)
        f.write("\n")

    print(f"source_json: {data_path}")
    print(f"output_json: {output_path}")
    print(f"total_samples: {len(samples)}")
    print(f"rewritten: {stats['rewritten']}")
    print(f"unchanged: {stats['unchanged']}")
    print(f"missing_{args.field}: {stats['missing_field']}")
    print(f"replacement_prompts: {len(PROMPT_REPLACEMENTS)}")


if __name__ == "__main__":
    main()
