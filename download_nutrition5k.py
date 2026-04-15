#!/usr/bin/env python3
"""
Download and resize Nutrition5k images from Google Cloud Storage.

Downloads images selectively (no tar.gz) and resizes them immediately to save
disk space. Supports resuming interrupted downloads.

Optionally downloads side-angle videos, extracts frames with ffmpeg, resizes
them, and deletes the original video to conserve disk space.

Prerequisites:
    - gsutil (Google Cloud SDK) installed and on PATH
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
    "metadata/ingredient_metadata.csv",
]

SPLIT_FILES = [
    "dish_ids/splits/rgb_train_ids.txt",
    "dish_ids/splits/rgb_test_ids.txt",
]


def gsutil_cp(src: str, dst: str) -> bool:
    """Download a file from GCS. Returns True on success."""
    try:
        subprocess.run(
            ["gsutil", "-q", "cp", src, dst],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Failed to download {src}: {e.stderr.decode().strip()}")
        return False
    except FileNotFoundError:
        print("ERROR: gsutil not found. Install the Google Cloud SDK first.")
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
        if gsutil_cp(src, str(dst)):
            print(f"  [ok]   {rel_path}")
        # If it fails, try alternate split file names
    print()


def get_dish_ids(output_dir: Path) -> list[str]:
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
        if not gsutil_cp(gcs_path, tmp_path):
            return "fail"

        # Resize and save
        with Image.open(tmp_path) as img:
            img_resized = img.resize((resolution, resolution), Image.LANCZOS)
            img_resized.save(final_path, "PNG")

        return "ok"
    except Exception as e:
        print(f"  Error processing {dish_id}: {e}")
        # Clean up partial output
        if final_path.exists():
            final_path.unlink()
        return "fail"
    finally:
        # Always clean up temp file
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
            if not gsutil_cp(gcs_path, tmp_video):
                counts["fail"] += 1
                continue

            # Extract every Nth frame using ffmpeg
            # -vsync vfr avoids duplicate frames; select filter picks every Nth
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
            # Clean up partial output
            if frames_dir.exists() and not any(frames_dir.iterdir()):
                frames_dir.rmdir()
            counts["fail"] += 1
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Download and resize Nutrition5k overhead RGB images."
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

    # Step 2: Get dish IDs
    print("=== Collecting dish IDs from metadata ===")
    dish_ids = get_dish_ids(output_dir)
    if not dish_ids:
        print("ERROR: No dish IDs found. Check that metadata downloaded correctly.")
        sys.exit(1)
    print(f"  Found {len(dish_ids)} dishes")

    if args.max_dishes:
        dish_ids = dish_ids[: args.max_dishes]
        print(f"  Limited to first {args.max_dishes} dishes")
    print()

    # Step 3: Download and resize images
    print("=== Downloading and resizing overhead RGB images ===")
    counts = {"ok": 0, "skip": 0, "fail": 0}
    total = len(dish_ids)

    for i, dish_id in enumerate(dish_ids, 1):
        status = download_and_resize_image(dish_id, output_dir, args.resolution)

        if status == "skip" and not args.no_skip_existing:
            counts["skip"] += 1
            # Print skip status less frequently to reduce noise
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
        print()
        print(f"=== Downloading side-angle videos & extracting frames (every {args.frame_interval}th) ===")
        for i, dish_id in enumerate(dish_ids, 1):
            result = extract_and_resize_frames(
                dish_id, output_dir, args.resolution,
                args.frame_interval, args.cameras,
            )
            side_counts["ok"] += result["ok"]
            side_counts["skip"] += result["skip"]
            side_counts["fail"] += result["fail"]

            # Progress for non-skipped
            if result["ok"] > 0:
                print(f"  [{i}/{total}] {dish_id} - extracted frames from {result['ok']} camera(s)")
            elif i % 100 == 0 or i == total:
                print(f"  [{i}/{total}] Progress update")

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
        total_cameras = total * len(args.cameras)
        print(f"  Total camera/dish:  {total_cameras}")


if __name__ == "__main__":
    main()
