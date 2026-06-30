import argparse
import json
from pathlib import Path

import h5py


def copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def update_env_attrs(attrs, image_resolution, control_freq):
    for attr_name in ("env_args", "env_info"):
        if attr_name not in attrs:
            continue
        try:
            env_meta = json.loads(attrs[attr_name])
        except Exception:
            continue

        env_kwargs = env_meta.get("env_kwargs")
        if isinstance(env_kwargs, dict):
            env_kwargs["camera_heights"] = image_resolution
            env_kwargs["camera_widths"] = image_resolution
            env_kwargs["control_freq"] = control_freq
        else:
            env_meta["camera_heights"] = image_resolution
            env_meta["camera_widths"] = image_resolution
            env_meta["control_freq"] = control_freq

        attrs[attr_name] = json.dumps(env_meta)


def repair_one(raw_path, target_path, image_resolution, control_freq):
    with h5py.File(raw_path, "r") as raw_file, h5py.File(target_path, "r+") as target_file:
        raw_data = raw_file["data"]
        target_data = target_file["data"]

        copy_attrs(raw_data, target_data)
        update_env_attrs(target_data.attrs, image_resolution, control_freq)

        for demo_name in target_data.keys():
            if not demo_name.startswith("demo_"):
                continue
            demo = target_data[demo_name]
            if "actions" in demo:
                demo.attrs["num_samples"] = demo["actions"].shape[0]
            if demo_name in raw_data and "model_file" in raw_data[demo_name].attrs:
                demo.attrs["model_file"] = raw_data[demo_name].attrs["model_file"]


def main():
    parser = argparse.ArgumentParser(description="Repair attrs for regenerated LIBERO hdf5 files.")
    parser.add_argument("--raw_dir", required=True, help="Original LIBERO hdf5 directory.")
    parser.add_argument("--target_dir", required=True, help="Regenerated hdf5 directory to repair in place.")
    parser.add_argument("--image_resolution", type=int, default=224)
    parser.add_argument("--control_freq", type=int, default=20)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    target_dir = Path(args.target_dir)
    target_files = sorted(target_dir.glob("*.hdf5"))
    if not target_files:
        raise SystemExit(f"No .hdf5 files found in target_dir: {target_dir}")

    repaired = 0
    for target_path in target_files:
        raw_path = raw_dir / target_path.name
        if not raw_path.exists():
            print(f"[warn] raw file not found for {target_path.name}, skipping")
            continue
        repair_one(raw_path, target_path, args.image_resolution, args.control_freq)
        repaired += 1
        print(f"[ok] repaired {target_path.name}")

    print(f"[done] repaired {repaired}/{len(target_files)} files")


if __name__ == "__main__":
    main()
