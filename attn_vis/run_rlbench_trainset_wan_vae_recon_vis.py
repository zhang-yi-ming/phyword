#!/usr/bin/env python
"""Visualize Wan2.1 VAE reconstruction quality on RLBench train images.

This script reads the RLBench train JSON, selects the first N episodes per
task, then runs front-camera images through Wan21VAEEncoder -> Wan21VAEDecoder.
It saves side-by-side visualizations with the original input on the left and
the reconstruction on the right.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from vae.wan21_vae_decoder import DEFAULT_WAN21_VAE_CKPT, Wan21VAEDecoder
from vae.wan21_vae_encoder import Wan21VAEEncoder


DEFAULT_DATA_JSON = "/mnt/nas/zhangyiming/database/rlbench/train/json/train_action_chunk1_sumpos_lastrot.json"
DEFAULT_OUTPUT_ROOT = "/mnt/nas/zhangyiming/last05_beta/experiments_rlbench/wan_vae_recon_vis"


@dataclass
class Config:
    data_path: str
    output_dir: str
    vae_path: str
    mode: str
    task_names: str
    num_trajectories_per_task: int
    max_tasks: int
    max_frames_per_episode: int
    frame_stride: int
    image_size: int
    image_batch_size: int
    fps: int
    device: str
    dtype: str
    video_codec: str
    dry_run: bool


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Run Wan2.1 VAE encode/decode reconstruction visualization on RLBench train images."
    )
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_JSON)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--vae_path", type=str, default=DEFAULT_WAN21_VAE_CKPT)
    parser.add_argument("--mode", type=str, choices=("image", "video", "both"), default="both")
    parser.add_argument("--task_names", type=str, default="", help="Comma-separated task filter. Empty means all tasks.")
    parser.add_argument("--num_trajectories_per_task", type=int, default=1)
    parser.add_argument("--max_tasks", type=int, default=0, help="Optional task cap for quick checks. 0 means no cap.")
    parser.add_argument("--max_frames_per_episode", type=int, default=0, help="Optional frame cap per episode. 0 means all.")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--image_batch_size", type=int, default=4)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', 'cuda', or 'cuda:N'.")
    parser.add_argument("--dtype", type=str, choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument(
        "--video_codec",
        type=str,
        default="",
        help="Optional imageio/ffmpeg codec. Empty matches RLBench eval defaults, usually H.264.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.num_trajectories_per_task <= 0:
        raise ValueError("--num_trajectories_per_task must be positive.")
    if args.max_tasks < 0:
        raise ValueError("--max_tasks must be >= 0.")
    if args.max_frames_per_episode < 0:
        raise ValueError("--max_frames_per_episode must be >= 0.")
    if args.frame_stride <= 0:
        raise ValueError("--frame_stride must be positive.")
    if args.image_size <= 0 or args.image_size % Wan21VAEEncoder.spatial_compression != 0:
        raise ValueError("--image_size must be positive and divisible by 8.")
    if args.image_batch_size <= 0:
        raise ValueError("--image_batch_size must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(DEFAULT_OUTPUT_ROOT, time.strftime("%Y_%m_%d-%H_%M_%S"))

    return Config(
        data_path=args.data_path,
        output_dir=output_dir,
        vae_path=args.vae_path,
        mode=args.mode,
        task_names=args.task_names,
        num_trajectories_per_task=int(args.num_trajectories_per_task),
        max_tasks=int(args.max_tasks),
        max_frames_per_episode=int(args.max_frames_per_episode),
        frame_stride=int(args.frame_stride),
        image_size=int(args.image_size),
        image_batch_size=int(args.image_batch_size),
        fps=int(args.fps),
        device=args.device,
        dtype=args.dtype,
        video_codec=str(args.video_codec or "").strip(),
        dry_run=bool(args.dry_run),
    )


def resolve_device(device_arg: str) -> torch.device:
    device_arg = str(device_arg).strip().lower()
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        return torch.device("cuda:0")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_arg]


def load_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected train JSON list at {path}, got {type(data).__name__}.")
    return data


def parse_front_pic_index(path: str) -> int:
    match = re.search(r"front_(\d+)\.[^.]+$", os.path.basename(str(path)))
    if match is None:
        raise ValueError(f"Could not parse RLBench front image index from path: {path}")
    return int(match.group(1))


def resolve_episode_key(sample: dict[str, Any]) -> tuple[str, int]:
    if "task_name" in sample:
        task = str(sample["task_name"])
    elif "task" in sample:
        task = str(sample["task"])
    else:
        front_pic = str(sample["front_pic"])
        episode_dir = os.path.dirname(front_pic)
        task_dir = os.path.dirname(episode_dir)
        task = os.path.basename(task_dir) or os.path.basename(episode_dir)
    return task, int(sample["episode_index"])


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(name)).strip("_") or "unknown"


def select_episodes(cfg: Config, data: list[dict[str, Any]]) -> list[tuple[str, int, dict[str, Any]]]:
    requested_tasks = [task.strip() for task in str(cfg.task_names or "").split(",") if task.strip()]
    requested_task_set = set(requested_tasks)
    grouped: OrderedDict[str, OrderedDict[int, dict[str, Any]]] = OrderedDict()

    for sample in data:
        task_name, episode_index = resolve_episode_key(sample)
        if requested_task_set and task_name not in requested_task_set:
            continue
        grouped.setdefault(task_name, OrderedDict()).setdefault(int(episode_index), sample)

    if requested_task_set:
        missing = sorted(requested_task_set - set(grouped))
        if missing:
            raise ValueError(f"Requested task_names not found in train JSON: {missing}")

    selected: list[tuple[str, int, dict[str, Any]]] = []
    task_names = sorted(grouped)
    if cfg.max_tasks > 0:
        task_names = task_names[: cfg.max_tasks]

    for task_name in task_names:
        episodes = sorted(grouped[task_name].items(), key=lambda item: item[0])
        episodes = episodes[: cfg.num_trajectories_per_task]
        log(
            f"Selected task={task_name}: episodes={[episode for episode, _ in episodes]} "
            f"(num_trajectories_per_task={cfg.num_trajectories_per_task})"
        )
        for episode_index, sample in episodes:
            selected.append((task_name, int(episode_index), sample))
    return selected


def list_episode_front_images(sample: dict[str, Any], cfg: Config) -> list[Path]:
    front_pic = Path(str(sample["front_pic"]))
    episode_dir = front_pic.parent
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode image directory does not exist: {episode_dir}")

    candidates: list[Path] = []
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        candidates.extend(episode_dir.glob(f"front_{suffix}"))

    def sort_key(path: Path) -> tuple[int, str]:
        try:
            return parse_front_pic_index(str(path)), path.name
        except ValueError:
            return 10**12, path.name

    candidates = sorted(set(candidates), key=sort_key)
    if not candidates:
        candidates = [front_pic]

    candidates = candidates[:: cfg.frame_stride]
    if cfg.max_frames_per_episode > 0:
        candidates = candidates[: cfg.max_frames_per_episode]
    if not candidates:
        raise ValueError(f"No front images selected for episode directory: {episode_dir}")
    return candidates


def resize_center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {image.size}")
    if width < height:
        new_width = size
        new_height = int(round(height * size / width))
    else:
        new_height = size
        new_width = int(round(width * size / height))
    resample = getattr(Image, "Resampling", Image).BILINEAR
    image = image.resize((new_width, new_height), resample=resample)
    left = max(0, (new_width - size) // 2)
    top = max(0, (new_height - size) // 2)
    return image.crop((left, top, left + size, top + size))


def load_frame_uint8(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = resize_center_crop(image, image_size)
    return np.asarray(image, dtype=np.uint8)


def frames_to_vae_tensor(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    array = np.stack(frames, axis=0)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous().float()
    tensor = tensor / 127.5 - 1.0
    return tensor.to(device=device, dtype=torch.float32, non_blocking=True)


def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().to(device="cpu", dtype=torch.float32).clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) * 127.5
    tensor = tensor.round().clamp(0, 255).to(dtype=torch.uint8)
    if tensor.ndim == 4:
        return tensor.permute(0, 2, 3, 1).numpy()
    if tensor.ndim == 5:
        return tensor.permute(0, 2, 3, 4, 1).numpy()
    raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(tensor.shape)}")


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError(f"Side-by-side frames must have equal shape, got {left.shape} and {right.shape}")
    return np.concatenate([left, right], axis=1)


def next_wan_legal_frame_count(frame_count: int) -> int:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    remainder = (frame_count - 1) % Wan21VAEEncoder.temporal_compression
    if remainder == 0:
        return frame_count
    return frame_count + (Wan21VAEEncoder.temporal_compression - remainder)


def write_video(path: Path, frames_rgb: list[np.ndarray], fps: int, codec: str) -> None:
    if not frames_rgb:
        raise ValueError("Cannot write an empty video.")
    height, width = frames_rgb[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    # Match the RLBench eval writer. imageio uses its bundled ffmpeg and writes
    # browser-friendly H.264 MP4s in this environment, while OpenCV mp4v/FMP4
    # files can fail in remote previewers.
    writer_kwargs = {"fps": float(fps), "macro_block_size": 1}
    if codec:
        writer_kwargs["codec"] = codec
    writer = imageio.get_writer(str(path), **writer_kwargs)
    try:
        for frame in frames_rgb:
            if frame.shape[:2] != (height, width):
                raise ValueError(f"Video frame shape mismatch: expected {(height, width)}, got {frame.shape[:2]}")
            writer.append_data(np.asarray(frame).astype(np.uint8))
    finally:
        writer.close()


def build_vae(cfg: Config, device: torch.device) -> tuple[Wan21VAEEncoder, Wan21VAEDecoder]:
    dtype = resolve_dtype(cfg.dtype)
    log(f"Loading Wan2.1 VAE from {cfg.vae_path}")
    encoder = Wan21VAEEncoder(
        vae_pth=cfg.vae_path,
        dtype=dtype,
        device=device,
        freeze=True,
        normalize_latents=True,
    )
    decoder = Wan21VAEDecoder(
        vae_pth=cfg.vae_path,
        dtype=dtype,
        device=device,
        freeze=True,
        normalized_latents=True,
        clamp_output=True,
    )
    encoder.eval()
    decoder.eval()
    return encoder, decoder


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_image_mode(
    cfg: Config,
    encoder: Wan21VAEEncoder,
    decoder: Wan21VAEDecoder,
    device: torch.device,
    task_name: str,
    episode_index: int,
    image_paths: list[Path],
    episode_dir: Path,
) -> dict[str, Any]:
    output_dir = episode_dir / "image"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    latent_shapes: list[tuple[int, ...]] = []

    for start in range(0, len(image_paths), cfg.image_batch_size):
        batch_paths = image_paths[start : start + cfg.image_batch_size]
        original_frames = [load_frame_uint8(path, cfg.image_size) for path in batch_paths]
        batch = frames_to_vae_tensor(original_frames, device=device)
        with torch.inference_mode():
            latent = encoder.encode(batch)
            recon = decoder.decode(latent)
        if recon.ndim != 5 or recon.shape[2] != 1:
            raise ValueError(f"Expected image reconstruction shape [B,3,1,H,W], got {tuple(recon.shape)}")
        recon_frames = tensor_to_uint8(recon[:, :, 0])
        latent_shapes.append(tuple(int(x) for x in latent.shape))

        for path, original, reconstructed in zip(batch_paths, original_frames, recon_frames):
            frame_index = parse_front_pic_index(str(path))
            panel = side_by_side(original, reconstructed)
            out_path = output_dir / f"task={sanitize_name(task_name)}_episode={episode_index:04d}_front={frame_index:06d}_wan_recon.png"
            Image.fromarray(panel).save(out_path)
            outputs.append(str(out_path))

    return {
        "mode": "image",
        "num_outputs": len(outputs),
        "outputs": outputs,
        "latent_shapes": [list(shape) for shape in latent_shapes],
    }


def run_video_mode(
    cfg: Config,
    encoder: Wan21VAEEncoder,
    decoder: Wan21VAEDecoder,
    device: torch.device,
    task_name: str,
    episode_index: int,
    image_paths: list[Path],
    episode_dir: Path,
) -> dict[str, Any]:
    output_dir = episode_dir / "video"
    output_dir.mkdir(parents=True, exist_ok=True)
    original_frames = [load_frame_uint8(path, cfg.image_size) for path in image_paths]
    original_count = len(original_frames)
    padded_count = next_wan_legal_frame_count(original_count)
    pad_frames = padded_count - original_count
    if pad_frames:
        original_frames = original_frames + [original_frames[-1].copy() for _ in range(pad_frames)]

    video = frames_to_vae_tensor(original_frames, device=device).permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    with torch.inference_mode():
        latent = encoder.encode(video)
        recon = decoder.decode(latent)
    if recon.ndim != 5:
        raise ValueError(f"Expected video reconstruction shape [B,3,T,H,W], got {tuple(recon.shape)}")
    if recon.shape[0] != 1 or recon.shape[2] != padded_count:
        raise ValueError(
            "Wan decoder returned unexpected video length: "
            f"shape={tuple(recon.shape)}, expected batch=1 and T={padded_count}."
        )
    recon_frames = tensor_to_uint8(recon)[0]
    panels = [side_by_side(original, reconstructed) for original, reconstructed in zip(original_frames, recon_frames)]

    out_path = output_dir / f"task={sanitize_name(task_name)}_episode={episode_index:04d}_wan_recon_side_by_side_paddedT={padded_count}.mp4"
    write_video(out_path, panels, cfg.fps, cfg.video_codec)

    first_panel_path = output_dir / f"task={sanitize_name(task_name)}_episode={episode_index:04d}_wan_recon_first_frame.png"
    Image.fromarray(panels[0]).save(first_panel_path)

    return {
        "mode": "video",
        "output": str(out_path),
        "first_frame_preview": str(first_panel_path),
        "original_frame_count": original_count,
        "padded_frame_count": padded_count,
        "pad_frames": pad_frames,
        "latent_shape": list(int(x) for x in latent.shape),
        "reconstruction_shape": list(int(x) for x in recon.shape),
    }


def process_episode(
    cfg: Config,
    encoder: Wan21VAEEncoder,
    decoder: Wan21VAEDecoder,
    device: torch.device,
    task_name: str,
    episode_index: int,
    sample: dict[str, Any],
) -> dict[str, Any]:
    image_paths = list_episode_front_images(sample, cfg)
    task_dir = Path(cfg.output_dir) / f"task={sanitize_name(task_name)}"
    episode_dir = task_dir / f"episode={episode_index:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    log(
        f"Processing task={task_name} episode={episode_index} frames={len(image_paths)} "
        f"mode={cfg.mode}"
    )
    result: dict[str, Any] = {
        "task_name": task_name,
        "episode_index": episode_index,
        "source_episode_dir": str(Path(str(sample["front_pic"])).parent),
        "frame_paths": [str(path) for path in image_paths],
        "image_size": cfg.image_size,
        "modes": {},
    }

    if cfg.mode in ("image", "both"):
        result["modes"]["image"] = run_image_mode(
            cfg, encoder, decoder, device, task_name, episode_index, image_paths, episode_dir
        )
    if cfg.mode in ("video", "both"):
        result["modes"]["video"] = run_video_mode(
            cfg, encoder, decoder, device, task_name, episode_index, image_paths, episode_dir
        )

    save_json(episode_dir / "metadata.json", result)
    return result


def main() -> None:
    cfg = parse_args()
    data_path = Path(cfg.data_path)
    vae_path = Path(cfg.vae_path)
    output_dir = Path(cfg.output_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Train JSON not found: {data_path}")
    if not cfg.dry_run and not vae_path.exists():
        raise FileNotFoundError(f"Wan2.1 VAE checkpoint not found: {vae_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(output_dir / "run_config.json", asdict(cfg))
    log(f"Output dir: {output_dir}")
    log(f"Reading train JSON: {data_path}")
    data = load_json(str(data_path))
    selected = select_episodes(cfg, data)
    log(f"Selected {len(selected)} episodes.")
    if cfg.dry_run:
        dry_payload = [
            {
                "task_name": task_name,
                "episode_index": episode_index,
                "num_front_images": len(list_episode_front_images(sample, cfg)),
                "front_pic": sample.get("front_pic"),
            }
            for task_name, episode_index, sample in selected
        ]
        save_json(output_dir / "dry_run_selection.json", {"selected": dry_payload})
        log("Dry run complete; VAE was not loaded.")
        return

    device = resolve_device(cfg.device)
    log(f"Using device={device}, dtype={cfg.dtype}")
    encoder, decoder = build_vae(cfg, device)

    all_results = []
    for task_name, episode_index, sample in selected:
        all_results.append(process_episode(cfg, encoder, decoder, device, task_name, episode_index, sample))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_json(output_dir / "summary.json", {"config": asdict(cfg), "episodes": all_results})
    log(f"Finished. Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
