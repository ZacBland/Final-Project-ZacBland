#!/usr/bin/env python3
"""
Download and resize Nutrition5k images from Google Cloud Storage.

Downloads images selectively (no tar.gz) and resizes them immediately to save
disk space. Supports resuming interrupted downloads.

Optionally downloads side-angle videos, extracts frames with ffmpeg, resizes
them, and deletes the original video to conserve disk space.

Prerequisites:
    - gcloud CLI installed (https://cloud.google.com/sdk/docs/install)
    - pip install Pillow
    - ffmpeg installed (only needed with --include_side_angles)

Usage:
    python download_nutrition5k.py --output_dir ./data --resolution 224
    python download_nutrition5k.py --output_dir ./data --resolution 224 --max_dishes 5
    python download_nutrition5k.py --output_dir ./data --resolution 224 --include_side_angles
    python download_nutrition5k.py --output_dir ./data --resolution 224 --include_side_angles --frame_interval 10
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

GCS_BUCKET = "gs://nutrition5k_dataset/nutrition5k_dataset"

SIDE_ANGLE_CAMERAS = ["camera_A", "camera_B", "camera_C", "camera_D"]

METADATA_FILES = [
    "metadata/dish_metadata_cafe1.csv",
    "metadata/dish_metadata_cafe2.csv",
    "metadata/ingredients_metadata.csv",
]

SPLIT_FILES = [
    "dish_ids/splits/rgb_train_ids.txt",
    "dish_ids/splits/rgb_test_ids.txt",
]


def gcloud_cp(src: str, dst: str) -> bool:
    """Download a file from GCS using gcloud storage. Returns True on success."""
    try:
        subprocess.run(
            ["gcloud", "storage", "cp", src, dst],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Failed to download {src}: {e.stderr.decode().strip()}")
        return False
    except FileNotFoundError:
        print("ERROR: gcloud CLI not found. Install the Google Cloud SDK first.")
        print("  https://cloud.google.com/sdk/docs/install")
        sys.exit(1)


def gcloud_ls(gcs_path: str) -> list[str]:
    """List objects in a GCS path. Returns list of GCS URIs."""
    try:
        result = subprocess.run(
            ["gcloud", "storage", "ls", gcs_path],
            check=True,
            capture_output=True,
            text=True,
        )
        return [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    except subprocess.CalledProcessError:
        return []
    except FileNotFoundError:
        print("ERROR: gcloud CLI not found. Install the Google Cloud SDK first.")
        print("  https://cloud.google.com/sdk/docs/install")
        sys.exit(1)


def download_metadata(output_dir: Path) -> None:
    """Download metadata CSVs and train/test split files."""
    print("=== Downloading metadata ===")
    for rel_path in METADATA_FILES + SPLIT_FILES:
        dst = output_dir / rel_path
        if dst.exists():
            print(f"  [skip] {rel_path} (already exists)")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = f"{GCS_BUCKET}/{rel_path}"
        print(f"  Downloading {rel_path} ...")
        if gcloud_cp(src, str(dst)):
            print(f"  [ok]   {rel_path}")
    print()


def get_dish_ids_from_metadata(output_dir: Path) -> list[str]:
    """Extract unique dish IDs from the metadata CSVs."""
    dish_ids = set()
    for csv_name in ["metadata/dish_metadata_cafe1.csv", "metadata/dish_metadata_cafe2.csv"]:
        csv_path = output_dir / csv_name
        if not csv_path.exists():
            print(f"  Warning: {csv_name} not found, skipping")
            continue
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].startswith("dish_"):
                    dish_ids.add(row[0].strip())
    return sorted(dish_ids)


def get_overhead_dish_ids() -> list[str]:
    """List dish IDs that have overhead images in the GCS bucket."""
    print("  Listing available overhead dishes from GCS bucket...")
    entries = gcloud_ls(f"{GCS_BUCKET}/imagery/realsense_overhead/")
    dish_ids = []
    for entry in entries:
        # Entries look like: gs://.../realsense_overhead/dish_XXXXXXXXXX/
        name = entry.rstrip("/").split("/")[-1]
        if name.startswith("dish_"):
            dish_ids.append(name)
    return sorted(dish_ids)


def download_and_resize_image(
    dish_id: str, output_dir: Path, resolution: int
) -> str:
    """
    Download a single overhead RGB image and resize it.
    Returns: 'ok', 'skip', or 'fail'
    """
    final_path = output_dir / "imagery" / "realsense_overhead" / dish_id / "rgb.png"

    if final_path.exists():
        return "skip"

    final_path.parent.mkdir(parents=True, exist_ok=True)
    gcs_path = f"{GCS_BUCKET}/imagery/realsense_overhead/{dish_id}/rgb.png"

    # Download to a temp file, resize, then save to final location
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if not gcloud_cp(gcs_path, tmp_path):
            return "fail"

        # Resize and save
        with Image.open(tmp_path) as img:
            img_resized = img.resize((resolution, resolution), Image.LANCZOS)
            img_resized.save(final_path, "PNG")

        return "ok"
    except Exception as e:
        print(f"  Error processing {dish_id}: {e}")
        if final_path.exists():
            final_path.unlink()
        return "fail"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def extract_and_resize_frames(
    dish_id: str,
    output_dir: Path,
    resolution: int,
    frame_interval: int,
    cameras: list[str],
) -> dict[str, int]:
    """
    Download side-angle videos for a dish, extract every Nth frame,
    resize to target resolution, and delete the original video.

    Returns dict with counts: {'ok': N, 'skip': N, 'fail': N}
    """
    counts = {"ok": 0, "skip": 0, "fail": 0}

    for camera in cameras:
        frames_dir = output_dir / "imagery" / "side_angles" / dish_id / camera
        # Check if frames already exist (resume support)
        if frames_dir.exists() and any(frames_dir.iterdir()):
            counts["skip"] += 1
            continue

        gcs_path = f"{GCS_BUCKET}/imagery/side_angles/{dish_id}/{camera}.h264"

        # Download video to a temp directory
        tmp_dir = tempfile.mkdtemp()
        tmp_video = os.path.join(tmp_dir, f"{camera}.h264")
        tmp_frames_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(tmp_frames_dir, exist_ok=True)

        try:
            if not gcloud_cp(gcs_path, tmp_video):
                counts["fail"] += 1
                continue

            # Extract every Nth frame using ffmpeg
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", tmp_video,
                "-vf", f"select=not(mod(n\\,{frame_interval}))",
                "-vsync", "vfr",
                "-q:v", "2",
                os.path.join(tmp_frames_dir, "frame_%04d.png"),
            ]
            result = subprocess.run(
                ffmpeg_cmd, capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"  ffmpeg error for {dish_id}/{camera}: {result.stderr[-200:]}")
                counts["fail"] += 1
                continue

            # Delete the video immediately to free space
            os.unlink(tmp_video)

            # Resize each extracted frame and move to final location
            frames_dir.mkdir(parents=True, exist_ok=True)
            frame_files = sorted(glob.glob(os.path.join(tmp_frames_dir, "*.png")))

            for frame_path in frame_files:
                frame_name = os.path.basename(frame_path)
                final_frame = frames_dir / frame_name
                with Image.open(frame_path) as img:
                    img_resized = img.resize(
                        (resolution, resolution), Image.LANCZOS
                    )
                    img_resized.save(final_frame, "PNG")
                os.unlink(frame_path)

            counts["ok"] += 1

        except Exception as e:
            print(f"  Error processing {dish_id}/{camera}: {e}")
            if frames_dir.exists() and not any(frames_dir.iterdir()):
                frames_dir.rmdir()
            counts["fail"] += 1
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Download and resize Nutrition5k images."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Directory to save downloaded data (default: ./data)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="Target image size NxN in pixels (default: 224)",
    )
    parser.add_argument(
        "--max_dishes",
        type=int,
        default=None,
        help="Limit number of dishes to download (for testing)",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Re-download and overwrite existing images",
    )
    parser.add_argument(
        "--include_side_angles",
        action="store_true",
        help="Also download side-angle videos, extract frames, and resize them",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=5,
        help="Extract every Nth frame from side-angle videos (default: 5, matching the paper)",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=SIDE_ANGLE_CAMERAS,
        choices=SIDE_ANGLE_CAMERAS,
        help="Which cameras to extract frames from (default: all four A-D)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir.resolve()}")
    print(f"Target resolution: {args.resolution}x{args.resolution}")
    if args.include_side_angles:
        print(f"Side angles:      enabled (every {args.frame_interval}th frame, cameras: {args.cameras})")
    print()

    # Step 1: Download metadata
    download_metadata(output_dir)

    # Step 2: Get dish IDs that actually have overhead images
    print("=== Collecting dish IDs ===")
    overhead_dish_ids = get_overhead_dish_ids()
    if not overhead_dish_ids:
        print("ERROR: No overhead dish IDs found in GCS bucket.")
        sys.exit(1)
    print(f"  Found {len(overhead_dish_ids)} dishes with overhead images")

    # Also get all dish IDs from metadata for side angles
    all_dish_ids = get_dish_ids_from_metadata(output_dir)
    print(f"  Found {len(all_dish_ids)} total dishes in metadata")

    if args.max_dishes:
        overhead_dish_ids = overhead_dish_ids[: args.max_dishes]
        all_dish_ids = all_dish_ids[: args.max_dishes]
        print(f"  Limited to first {args.max_dishes} dishes")
    print()

    # Step 3: Download and resize overhead images
    print("=== Downloading and resizing overhead RGB images ===")
    counts = {"ok": 0, "skip": 0, "fail": 0}
    total = len(overhead_dish_ids)

    for i, dish_id in enumerate(overhead_dish_ids, 1):
        status = download_and_resize_image(dish_id, output_dir, args.resolution)

        if status == "skip" and not args.no_skip_existing:
            counts["skip"] += 1
            if i % 100 == 0 or i == total:
                print(f"  [{i}/{total}] Progress update - {counts['skip']} skipped so far")
        elif status == "ok":
            counts["ok"] += 1
            print(f"  [{i}/{total}] {dish_id} - resized to {args.resolution}x{args.resolution}")
        else:
            counts["fail"] += 1

    # Step 4: Side-angle video frames (optional)
    side_counts = {"ok": 0, "skip": 0, "fail": 0}
    if args.include_side_angles:
        side_dish_ids = all_dish_ids
        if args.max_dishes:
            side_dish_ids = side_dish_ids[: args.max_dishes]
        side_total = len(side_dish_ids)

        print()
        print(f"=== Downloading side-angle videos & extracting frames (every {args.frame_interval}th) ===")
        for i, dish_id in enumerate(side_dish_ids, 1):
            result = extract_and_resize_frames(
                dish_id, output_dir, args.resolution,
                args.frame_interval, args.cameras,
            )
            side_counts["ok"] += result["ok"]
            side_counts["skip"] += result["skip"]
            side_counts["fail"] += result["fail"]

            if result["ok"] > 0:
                print(f"  [{i}/{side_total}] {dish_id} - extracted frames from {result['ok']} camera(s)")
            elif i % 100 == 0 or i == side_total:
                print(f"  [{i}/{side_total}] Progress update")

    # Summary
    print()
    print("=== Summary: Overhead RGB ===")
    print(f"  Downloaded & resized: {counts['ok']}")
    print(f"  Skipped (existing):   {counts['skip']}")
    print(f"  Failed:               {counts['fail']}")
    print(f"  Total dishes:         {total}")

    if args.include_side_angles:
        print()
        print("=== Summary: Side-Angle Frames ===")
        print(f"  Cameras processed:  {side_counts['ok']}")
        print(f"  Cameras skipped:    {side_counts['skip']}")
        print(f"  Cameras failed:     {side_counts['fail']}")
        total_cameras = len(side_dish_ids) * len(args.cameras)
        print(f"  Total camera/dish:  {total_cameras}")


if __name__ == "__main__":
    main()
