import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np


PRIMARY_IMAGE_KEYS = (
    "obs/agentview_rgb",
    "obs/agentview_image",
    "obs/agentview",
)
WRIST_IMAGE_KEYS = (
    "obs/eye_in_hand_rgb",
    "obs/robot0_eye_in_hand_rgb",
    "obs/robot0_eye_in_hand_image",
    "obs/eye_in_hand_image",
)


def natural_key(path):
    name = Path(path).stem
    parts = name.replace("-", "_").split("_")
    nums = [int(p) for p in parts if p.isdigit()]
    return (parts, nums)


def h5_attr_to_str(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_json_attr(group, name):
    if name not in group.attrs:
        return None
    try:
        return json.loads(h5_attr_to_str(group.attrs[name]))
    except Exception:
        return None


def extract_language(h5_file, hdf5_path):
    data = h5_file["data"]
    problem_info = load_json_attr(data, "problem_info")
    if isinstance(problem_info, dict) and "language_instruction" in problem_info:
        lang = problem_info["language_instruction"]
        if isinstance(lang, list):
            lang = " ".join(str(x) for x in lang)
        return str(lang).strip().strip('"')

    stem = Path(hdf5_path).stem
    if stem.endswith("_demo"):
        stem = stem[:-5]
    return stem.replace("_", " ")


def extract_control_freq(h5_file):
    data = h5_file["data"]
    env_args = load_json_attr(data, "env_args")
    if not isinstance(env_args, dict):
        env_args = load_json_attr(data, "env_info")
    if not isinstance(env_args, dict):
        return None

    if "control_freq" in env_args:
        return env_args["control_freq"]
    env_kwargs = env_args.get("env_kwargs")
    if isinstance(env_kwargs, dict):
        return env_kwargs.get("control_freq")
    return None


def get_dataset(group, candidates, explicit_key=""):
    keys = (explicit_key,) if explicit_key else candidates
    for key in keys:
        if key and key in group:
            return group[key][()]
    raise KeyError(f"none of these datasets exist: {list(keys)}")


def ensure_t_h_w_c(images):
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError(f"image sequence must be 4D, got shape {images.shape}")
    if images.shape[-1] in (3, 4):
        return images
    if images.shape[1] in (3, 4):
        return np.transpose(images, (0, 2, 3, 1))
    raise ValueError(f"cannot infer image channel axis from shape {images.shape}")


def maybe_rotate_180(images, rotate):
    if rotate:
        return images[:, ::-1, ::-1, :]
    return images


def quat_to_axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64)
    xyz = quat[..., :3]
    w = np.clip(quat[..., 3:4], -1.0, 1.0)
    den = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    angle = 2.0 * np.arccos(w)
    axisangle = np.zeros_like(xyz)
    valid = den[..., 0] > 1e-8
    axisangle[valid] = xyz[valid] * (angle[valid] / den[valid])
    return axisangle.astype(np.float32)


def build_state(demo):
    if "obs/ee_pos" in demo and "obs/ee_ori" in demo and "obs/gripper_states" in demo:
        ee_pos = np.asarray(demo["obs/ee_pos"][()], dtype=np.float32)
        ee_ori = np.asarray(demo["obs/ee_ori"][()], dtype=np.float32)
        gripper = np.asarray(demo["obs/gripper_states"][()], dtype=np.float32)
        if gripper.ndim == 1:
            gripper = gripper[:, None]
        return np.concatenate([ee_pos, ee_ori, gripper], axis=-1)

    if "obs/ee_states" in demo and "obs/gripper_states" in demo:
        ee_states = np.asarray(demo["obs/ee_states"][()], dtype=np.float32)
        gripper = np.asarray(demo["obs/gripper_states"][()], dtype=np.float32)
        if gripper.ndim == 1:
            gripper = gripper[:, None]
        return np.concatenate([ee_states, gripper], axis=-1)

    if "robot_states" in demo:
        robot_states = np.asarray(demo["robot_states"][()], dtype=np.float32)
        if robot_states.shape[-1] >= 9:
            gripper = robot_states[..., :2]
            ee_pos = robot_states[..., 2:5]
            ee_quat = robot_states[..., 5:9]
            ee_axisangle = quat_to_axisangle(ee_quat)
            return np.concatenate([ee_pos, ee_axisangle, gripper], axis=-1)
        return robot_states

    raise KeyError("cannot build state; expected obs/ee_pos+obs/ee_ori+obs/gripper_states, obs/ee_states+obs/gripper_states, or robot_states")


def convert_gripper(actions, mode):
    actions = np.asarray(actions, dtype=np.float32).copy()
    if actions.shape[-1] < 1 or mode == "raw":
        return actions

    gripper = actions[..., -1]
    should_convert = mode == "zero_one" or (mode == "auto" and np.nanmin(gripper) < -0.1)
    if should_convert:
        actions[..., -1] = (gripper > 0).astype(np.float32)
    return actions


def is_noop(action, prev_kept_action=None, threshold=1e-4):
    if prev_kept_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    return np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_kept_action[-1]


def demo_is_success(demo):
    if "dones" in demo:
        dones = np.asarray(demo["dones"][()])
        return bool(dones.size > 0 and dones[-1])
    if "rewards" in demo:
        rewards = np.asarray(demo["rewards"][()])
        return bool(rewards.size > 0 and rewards[-1] > 0)
    return None


def list_demos(data_group):
    demos = [key for key in data_group.keys() if key.startswith("demo_")]
    return sorted(demos, key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x)


def convert_one_hdf5(path, output_dir, args, start_episode_id):
    converted = 0
    skipped_empty = 0
    skipped_failure = 0
    with h5py.File(path, "r") as h5_file:
        if "data" not in h5_file:
            raise KeyError(f"{path} has no top-level 'data' group")

        data = h5_file["data"]
        language = extract_language(h5_file, path)
        control_freq = extract_control_freq(h5_file)
        if control_freq is not None:
            print(f"[info] {Path(path).name}: control_freq={control_freq}")
        else:
            print(f"[info] {Path(path).name}: control_freq not found in hdf5 attrs")

        for demo_name in list_demos(data):
            demo = data[demo_name]
            success = demo_is_success(demo)
            if args.only_success:
                if success is False:
                    skipped_failure += 1
                    continue
                if success is None:
                    print(f"[warn] {Path(path).name}/{demo_name}: no dones/rewards found; keeping because success is unknown")

            actions_raw = np.asarray(demo["actions"][()], dtype=np.float32)
            actions = convert_gripper(actions_raw, args.gripper_format)
            primary = ensure_t_h_w_c(get_dataset(demo, PRIMARY_IMAGE_KEYS, args.primary_key))
            wrist = None
            try:
                wrist = ensure_t_h_w_c(get_dataset(demo, WRIST_IMAGE_KEYS, args.wrist_key))
            except KeyError:
                if args.require_wrist:
                    raise
            states = build_state(demo)

            length = min(len(actions), len(primary), len(states))
            if wrist is not None:
                length = min(length, len(wrist))

            episode = []
            prev_kept_action = None
            for i in range(length):
                if args.skip_noops and is_noop(actions_raw[i], prev_kept_action, args.noop_threshold):
                    continue
                step = {
                    "image_primary": maybe_rotate_180(primary[i : i + 1], args.rotate_180)[0].astype(np.uint8),
                    "state": states[i].astype(np.float32),
                    "action": actions[i].astype(np.float32),
                    "language_instruction": language,
                }
                if wrist is not None:
                    step["image_wrist"] = maybe_rotate_180(wrist[i : i + 1], args.rotate_180)[0].astype(np.uint8)
                episode.append(step)
                prev_kept_action = actions_raw[i]

            if not episode:
                skipped_empty += 1
                continue

            episode_id = start_episode_id + converted
            out_path = output_dir / f"episode_{episode_id}.npy"
            if out_path.exists() and not args.overwrite:
                raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it")
            np.save(out_path, np.array(episode, dtype=object), allow_pickle=True)
            converted += 1

    return converted, skipped_empty, skipped_failure


def main():
    parser = argparse.ArgumentParser(description="Convert LIBERO hdf5 demos to last05-style .npy episodes.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing *_demo.hdf5 files.")
    parser.add_argument("--output_root", type=str, required=True, help="Parent output directory for npy task folder.")
    parser.add_argument("--output_task", type=str, default="", help="Output task folder name. Defaults to input dir basename.")
    parser.add_argument("--primary_key", type=str, default="", help="Explicit hdf5 dataset path for primary image.")
    parser.add_argument("--wrist_key", type=str, default="", help="Explicit hdf5 dataset path for wrist image.")
    parser.add_argument("--require_wrist", action="store_true")
    parser.add_argument("--no_rotate_180", dest="rotate_180", action="store_false", help="Do not rotate images by 180 degrees.")
    parser.set_defaults(rotate_180=True)
    parser.add_argument("--skip_noops", action="store_true", help="Filter no-op actions, matching *_no_noops style data.")
    parser.add_argument("--only_success", action="store_true",
                        help="Keep only successful demos when hdf5 has dones/rewards.")
    parser.add_argument("--noop_threshold", type=float, default=1e-4)
    parser.add_argument("--gripper_format", choices=["auto", "raw", "zero_one"], default="auto",
                        help="Convert gripper action to 0/1 if needed. auto converts when negative values are detected.")
    parser.add_argument("--start_index", type=int, default=1)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"input_dir is not a directory: {input_dir}")

    output_task = args.output_task or input_dir.name
    output_dir = Path(args.output_root) / output_task
    output_dir.mkdir(parents=True, exist_ok=True)

    hdf5_files = sorted(
        [p for p in input_dir.iterdir() if p.suffix in (".hdf5", ".h5")],
        key=natural_key,
    )
    if args.max_files > 0:
        hdf5_files = hdf5_files[: args.max_files]
    if not hdf5_files:
        raise SystemExit(f"no .hdf5/.h5 files found in {input_dir}")

    print(f"[convert] input_dir={input_dir}")
    print(f"[convert] output_dir={output_dir}")
    print(f"[convert] files={len(hdf5_files)}, rotate_180={args.rotate_180}, skip_noops={args.skip_noops}, only_success={args.only_success}")

    next_episode_id = args.start_index
    total_converted = 0
    total_skipped_empty = 0
    total_skipped_failure = 0
    for hdf5_path in hdf5_files:
        print(f"[convert] {hdf5_path.name}")
        converted, skipped_empty, skipped_failure = convert_one_hdf5(hdf5_path, output_dir, args, next_episode_id)
        next_episode_id += converted
        total_converted += converted
        total_skipped_empty += skipped_empty
        total_skipped_failure += skipped_failure
        print(f"  wrote {converted} episodes, skipped_empty={skipped_empty}, skipped_failure={skipped_failure}")

    print(
        f"[done] wrote {total_converted} npy episodes to {output_dir}; "
        f"skipped_empty={total_skipped_empty}, skipped_failure={total_skipped_failure}"
    )


if __name__ == "__main__":
    main()
