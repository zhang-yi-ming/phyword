"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import time

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
# import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def get_libero_env(task, model_family, resolution=256, seed=0, control_freq=10):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "control_freq": control_freq,
    }
    if control_freq == 0:
        env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution
        }
        
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def _to_rgb_uint8(img):
    frame = np.asarray(img)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    return frame.astype(np.uint8)


VALUE_CURVE_COLORS = {
    "cosmos value": (78, 201, 176),
    "action value": (245, 158, 11),
    "value score": (78, 201, 176),
}


def _value_labels(value_scores):
    labels = []
    if value_scores is None:
        return labels
    for item in value_scores:
        if isinstance(item, dict):
            for label in ("cosmos value", "action value", "value score"):
                if label in item and label not in labels:
                    labels.append(label)
            for label in item:
                if label not in labels:
                    labels.append(label)
        elif item is not None and "value score" not in labels:
            labels.append("value score")
    return labels


def _value_at(value_scores, idx, label=None):
    if value_scores is None or idx >= len(value_scores):
        return None
    value = value_scores[idx]
    if value is None:
        return None
    if isinstance(value, dict):
        if label is None:
            return None
        value = value.get(label)
        if value is None:
            return None
    return float(value)


def _draw_value_rollout_frame(img, value_scores, idx):
    frame = _to_rgb_uint8(img)
    labels = _value_labels(value_scores)
    if not labels:
        return frame
    header_height = 82 + 16 * max(0, len(labels) - 1)
    pil_frame = Image.fromarray(frame, mode="RGB")
    canvas = Image.new("RGB", (pil_frame.width, pil_frame.height + header_height), (18, 22, 28))
    canvas.paste(pil_frame, (0, header_height))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for label_idx, label in enumerate(labels):
        current_value = _value_at(value_scores, idx, label=label)
        color = VALUE_CURVE_COLORS.get(label, (245, 248, 252))
        if current_value is None:
            score_text = f"{label}: n/a"
        else:
            score_text = f"{label}: {current_value:.4f}"
        draw.text((8, 8 + 16 * label_idx), score_text, fill=color, font=font)

    x0, y0 = 8, 34 + 16 * max(0, len(labels) - 1)
    x1, y1 = max(9, pil_frame.width - 8), header_height - 10
    draw.rectangle((x0, y0, x1, y1), outline=(87, 96, 111))
    draw.line((x0, y1, x1, y1), fill=(87, 96, 111))

    total = max(1, len(value_scores) if value_scores is not None else idx + 1)
    for label in labels:
        color = VALUE_CURVE_COLORS.get(label, (245, 248, 252))
        points = []
        for frame_idx in range(idx + 1):
            value = _value_at(value_scores, frame_idx, label=label)
            if value is None:
                continue
            value = min(1.0, max(0.0, value))
            x = x0 + int((x1 - x0) * frame_idx / max(1, total - 1))
            y = y1 - int((y1 - y0) * value)
            points.append((x, y))

        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        for point in points[-16:]:
            x, y = point
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

    return np.asarray(canvas, dtype=np.uint8)


def save_rollout_video(
    rollout_images,
    idx,
    success,
    task_description,
    log_file=None,
    cosmos_denoise_steps=None,
    rollout_dir=None,
    value_scores=None,
    fps=30,
):
    """Saves an MP4 replay of an episode."""
    if not rollout_dir:
        # rollout_dir = f"/mnt/data/chenhao_save/vis/lcot_doublerl/libero/rollouts/{DATE}"
        rollout_dir = f"../experiments/rollouts/{DATE}"
        if cosmos_denoise_steps is not None:
            rollout_dir = f"{rollout_dir}_cosmos_denoise_steps_{cosmos_denoise_steps}"
    success_dir = "success_true" if bool(success) else "success_false"
    rollout_dir = os.path.join(rollout_dir, success_dir)
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--openvla_oft--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=fps, macro_block_size=1)
    for frame_idx, img in enumerate(rollout_images):
        if value_scores is not None:
            img = _draw_value_rollout_frame(img, value_scores, frame_idx)
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
