import argparse
import os
import tempfile
from collections import Counter, defaultdict

import numpy as np


REQUIRED_KEYS = ("image_primary", "state", "action", "language_instruction")


def natural_episode_key(path):
    name = os.path.basename(path)
    stem, _ = os.path.splitext(name)
    parts = stem.split("_")
    if parts and parts[-1].isdigit():
        return int(parts[-1])
    return stem


def resolve_task_dir(args):
    if args.task_dir:
        return args.task_dir
    if not args.data_root or not args.task:
        raise ValueError("Either --task_dir or both --data_root and --task must be provided.")
    return os.path.join(args.data_root, args.task)


def as_step_dict(step):
    if isinstance(step, dict):
        return step
    if hasattr(step, "item"):
        try:
            item = step.item()
            if isinstance(item, dict):
                return item
        except Exception:
            pass
    return None


def normalize_image_array(img):
    arr = np.asarray(img)
    while arr.ndim > 3:
        arr = arr[0]
    return arr


def choose_step_indices(length, sample_steps):
    if sample_steps <= 0 or sample_steps >= length:
        return list(range(length))
    if sample_steps == 1:
        return [0]
    indices = np.linspace(0, length - 1, sample_steps, dtype=int).tolist()
    return sorted(set(indices))


def validate_episode(path, sample_steps):
    errors = []
    warnings = []
    info = {
        "length": 0,
        "state_dims": Counter(),
        "action_dims": Counter(),
        "image_shapes": Counter(),
        "image_dtypes": Counter(),
        "language_examples": [],
    }

    try:
        episode = np.load(path, allow_pickle=True)
    except Exception as exc:
        return info, [f"failed to load npy: {exc}"], warnings

    try:
        length = len(episode)
    except TypeError:
        return info, ["loaded object has no length; expected an episode sequence"], warnings

    info["length"] = length
    if length <= 0:
        return info, ["empty episode"], warnings

    step_indices = choose_step_indices(length, sample_steps)
    for i in step_indices:
        step = as_step_dict(episode[i])
        if step is None:
            errors.append(f"step {i}: expected dict-like step, got {type(episode[i]).__name__}")
            continue

        missing = [key for key in REQUIRED_KEYS if key not in step]
        if missing:
            errors.append(f"step {i}: missing required keys {missing}")
            continue

        img = normalize_image_array(step["image_primary"])
        info["image_shapes"][tuple(img.shape)] += 1
        info["image_dtypes"][str(img.dtype)] += 1
        if img.ndim != 3:
            errors.append(f"step {i}: image_primary after squeeze must be HWC, got shape {img.shape}")
        elif img.shape[-1] not in (3, 4):
            errors.append(f"step {i}: image_primary channel dim should be 3 or 4, got shape {img.shape}")
        if img.size == 0:
            errors.append(f"step {i}: image_primary is empty")
        if not np.issubdtype(img.dtype, np.integer):
            warnings.append(f"step {i}: image_primary dtype is {img.dtype}, expected uint8-like")
        elif img.min() < 0 or img.max() > 255:
            errors.append(f"step {i}: image_primary values out of [0, 255], min={img.min()}, max={img.max()}")

        state = np.asarray(step["state"], dtype=np.float32).reshape(-1)
        action = np.asarray(step["action"], dtype=np.float32).reshape(-1)
        info["state_dims"][int(state.shape[0])] += 1
        info["action_dims"][int(action.shape[0])] += 1
        if state.shape[0] == 0:
            errors.append(f"step {i}: state is empty")
        if action.shape[0] == 0:
            errors.append(f"step {i}: action is empty")
        if not np.isfinite(state).all():
            errors.append(f"step {i}: state contains NaN or Inf")
        if not np.isfinite(action).all():
            errors.append(f"step {i}: action contains NaN or Inf")

        instruction = step["language_instruction"]
        if not isinstance(instruction, str):
            errors.append(f"step {i}: language_instruction must be str, got {type(instruction).__name__}")
        elif not instruction.strip():
            errors.append(f"step {i}: language_instruction is empty")
        elif len(info["language_examples"]) < 3 and instruction not in info["language_examples"]:
            info["language_examples"].append(instruction)

    return info, errors, warnings


def check_video_write(first_valid_path, fps):
    try:
        import imageio
    except Exception as exc:
        return False, f"imageio import failed: {exc}"

    try:
        episode = np.load(first_valid_path, allow_pickle=True)
        frames = []
        for i in range(min(len(episode), 8)):
            step = as_step_dict(episode[i])
            if step is None or "image_primary" not in step:
                continue
            frames.append(normalize_image_array(step["image_primary"]).astype(np.uint8))
        if not frames:
            return False, "no valid image_primary frames found for video write test"
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            imageio.mimwrite(tmp.name, frames, fps=fps, macro_block_size=1)
        return True, "ok"
    except Exception as exc:
        return False, f"video write test failed: {exc}"


def main():
    parser = argparse.ArgumentParser(
        description="Validate LIBERO .npy episodes for last05_beta/utils/gen_libero_video_json_stat.py."
    )
    parser.add_argument("--data_root", type=str, default="", help="Root containing task subdirectories.")
    parser.add_argument("--task", type=str, default="", help="Task directory name, e.g. libero_spatial_no_noops.")
    parser.add_argument("--task_dir", type=str, default="", help="Direct path to a task directory of episode_*.npy files.")
    parser.add_argument("--max_files", type=int, default=0, help="Max files to validate. 0 means all files.")
    parser.add_argument("--sample_steps", type=int, default=0, help="Steps sampled per episode. 0 means all steps.")
    parser.add_argument("--expected_action_dim", type=int, default=7)
    parser.add_argument("--expected_state_dim", type=int, default=0, help="0 means infer only.")
    parser.add_argument("--fps", type=int, default=20, help="FPS used only for optional mp4 write test.")
    parser.add_argument("--check_video_write", action="store_true", help="Also test whether imageio can write mp4.")
    args = parser.parse_args()

    task_dir = resolve_task_dir(args)
    print(f"[validate] task_dir: {task_dir}")

    if not os.path.isdir(task_dir):
        raise SystemExit(f"[FAIL] task_dir does not exist or is not a directory: {task_dir}")

    npy_files = [
        os.path.join(task_dir, name)
        for name in os.listdir(task_dir)
        if name.endswith(".npy")
    ]
    npy_files.sort(key=natural_episode_key)
    if args.max_files > 0:
        npy_files = npy_files[: args.max_files]

    if not npy_files:
        raise SystemExit("[FAIL] no .npy files found")

    total_errors = 0
    total_warnings = 0
    total_steps = 0
    lengths = []
    state_dims = Counter()
    action_dims = Counter()
    image_shapes = Counter()
    image_dtypes = Counter()
    per_file_errors = defaultdict(list)
    per_file_warnings = defaultdict(list)
    first_valid_path = None

    for idx, path in enumerate(npy_files, start=1):
        info, errors, warnings = validate_episode(path, args.sample_steps)
        lengths.append(info["length"])
        total_steps += info["length"]
        state_dims.update(info["state_dims"])
        action_dims.update(info["action_dims"])
        image_shapes.update(info["image_shapes"])
        image_dtypes.update(info["image_dtypes"])

        if errors:
            per_file_errors[path].extend(errors[:20])
            total_errors += len(errors)
        else:
            first_valid_path = first_valid_path or path

        if warnings:
            per_file_warnings[path].extend(warnings[:20])
            total_warnings += len(warnings)

        if idx % 50 == 0:
            print(f"[validate] checked {idx}/{len(npy_files)} files...")

    print("\n===== Summary =====")
    print(f"files checked: {len(npy_files)}")
    print(f"total episode steps: {total_steps}")
    print(f"episode length: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.2f}")
    print(f"action dims: {dict(action_dims)}")
    print(f"state dims: {dict(state_dims)}")
    print(f"image shapes: {dict(image_shapes.most_common(8))}")
    print(f"image dtypes: {dict(image_dtypes)}")

    if args.expected_action_dim > 0 and set(action_dims) != {args.expected_action_dim}:
        total_errors += 1
        print(f"[FAIL] expected action dim {args.expected_action_dim}, got {dict(action_dims)}")
    if args.expected_state_dim > 0 and set(state_dims) != {args.expected_state_dim}:
        total_errors += 1
        print(f"[FAIL] expected state dim {args.expected_state_dim}, got {dict(state_dims)}")

    if args.check_video_write:
        if first_valid_path is None:
            total_errors += 1
            print("[FAIL] skipped mp4 write test because no valid episode was found")
        else:
            ok, msg = check_video_write(first_valid_path, args.fps)
            print(f"mp4 write test at fps={args.fps}: {msg}")
            if not ok:
                total_errors += 1

    if per_file_errors:
        print("\n===== Example Errors =====")
        for path, errors in list(per_file_errors.items())[:10]:
            print(f"{os.path.basename(path)}:")
            for err in errors[:10]:
                print(f"  - {err}")

    if per_file_warnings:
        print("\n===== Example Warnings =====")
        for path, warnings in list(per_file_warnings.items())[:5]:
            print(f"{os.path.basename(path)}:")
            for warning in warnings[:5]:
                print(f"  - {warning}")

    if total_errors:
        raise SystemExit(f"\n[FAIL] validation failed with {total_errors} errors and {total_warnings} warnings.")

    print(f"\n[OK] dataset looks compatible with gen_libero_video_json_stat.py. warnings={total_warnings}")
    print("Next: set DATA_ROOT to the parent directory, TASK_LISTS to this task, and VIDEO_FPS to 20.")


if __name__ == "__main__":
    main()
