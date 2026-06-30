"""Utilities for drawing end-effector history trajectories on LIBERO images."""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from PIL import Image, ImageDraw


DEFAULT_HISTORY_TRAJECTORY_CAMERA_CONFIG = str(Path(__file__).with_name("libero_camera_params.yaml"))


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_history_trajectory_camera_config(
    task_suite_name: str,
    config_path: Optional[str] = None,
) -> dict[str, Any]:
    config_path = str(config_path or DEFAULT_HISTORY_TRAJECTORY_CAMERA_CONFIG)
    with open(config_path, "r") as f:
        all_params = yaml.safe_load(f) or {}

    suite_params = all_params.get(str(task_suite_name))
    if suite_params is None:
        raise KeyError(f"No camera params for task suite {task_suite_name!r} in {config_path}.")

    camera_params = suite_params.get("primary")
    if camera_params is None:
        raise KeyError(f"No primary camera params for {task_suite_name!r} in {config_path}.")

    cam_pos = np.asarray(camera_params["camera_position"], dtype=np.float64)
    cam_quat = np.asarray(camera_params["camera_quaternion_wxyz"], dtype=np.float64)
    source_image_hw = int(camera_params.get("source_image_hw", 128))

    if cam_pos.shape != (3,):
        raise ValueError(f"camera_position must have 3 values, got {cam_pos.shape}.")
    if cam_quat.shape != (4,):
        raise ValueError(f"camera_quaternion_wxyz must have 4 values, got {cam_quat.shape}.")
    if source_image_hw <= 0:
        raise ValueError(f"source_image_hw must be positive, got {source_image_hw}.")

    return {
        "cam_pos": cam_pos,
        "r_wc": quat_wxyz_to_rotmat(cam_quat),
        "fovy_deg": float(camera_params["fovy_deg"]),
        "source_image_hw": source_image_hw,
        "mirror_x": bool(camera_params.get("mirror_x", True)),
        "point_radius": int(camera_params.get("point_radius", 2)),
        "line_width": int(camera_params.get("line_width", 1)),
        "point_color": tuple(int(x) for x in camera_params.get("point_color_rgb", [220, 30, 30])),
        "line_color": tuple(int(x) for x in camera_params.get("line_color_rgb", [220, 80, 80])),
    }


def project_world_to_eval_pixel(p_world, camera_config: dict[str, Any], out_h: int, out_w: int):
    p_cam = camera_config["r_wc"].T @ (
        np.asarray(p_world, dtype=np.float64) - camera_config["cam_pos"]
    )
    z = -p_cam[2]
    if z <= 1e-6:
        return None

    source_hw = float(camera_config["source_image_hw"])
    f = 0.5 * source_hw / np.tan(np.deg2rad(camera_config["fovy_deg"]) / 2)
    u = source_hw / 2.0 + (p_cam[0] / z) * f
    v = source_hw / 2.0 - (p_cam[1] / z) * f

    sx = out_w / source_hw
    sy = out_h / source_hw
    if camera_config["mirror_x"]:
        u_out = ((source_hw - 1 - u) + 0.5) * sx - 0.5
    else:
        u_out = (u + 0.5) * sx - 0.5
    v_out = (v + 0.5) * sy - 0.5

    return float(u_out), float(v_out)


def draw_history_trajectory_on_image(
    pil_img: Image.Image,
    ee_history,
    camera_config: dict[str, Any],
) -> Image.Image:
    if not ee_history:
        return pil_img

    draw = ImageDraw.Draw(pil_img)
    out_w, out_h = pil_img.size
    pixels = [
        project_world_to_eval_pixel(point, camera_config, out_h=out_h, out_w=out_w)
        for point in ee_history
    ]

    prev = None
    for uv in pixels:
        if uv is None:
            prev = None
            continue
        if prev is not None:
            draw.line(
                [prev, uv],
                fill=camera_config["line_color"],
                width=camera_config["line_width"],
            )
        prev = uv

    radius = camera_config["point_radius"]
    for uv in pixels:
        if uv is None:
            continue
        u, v = uv
        draw.ellipse(
            [u - radius, v - radius, u + radius, v + radius],
            fill=camera_config["point_color"],
            outline=camera_config["point_color"],
        )

    return pil_img
