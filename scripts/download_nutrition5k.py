#!/usr/bin/env python3
"""
Download and resize Nutrition5k images from Google Cloud Storage.

Downloads images selectively (no tar.gz) and resizes them immediately to save
disk space. Supports resuming interrupted downloads.

Also downloads side-angle videos, extracts frames with ffmpeg, resizes and
flips them, and deletes the original video to conserve disk space.

Prerequisites:
    - gcloud CLI installed (https://cloud.google.com/sdk/docs/install)
    - pip install Pillow
    - ffmpeg installed (for side-angle video frame extraction)

Configure all settings via the global variables below, then run:
    python3 scripts/download_nutrition5k.py
"""

import concurrent.futures
import csv
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image

# =====================================================================
# CONFIGURATION — edit these variables to control downloading
# =====================================================================

OUTPUT_DIR = "./data"
RESOLUTION = 320              # target image size NxN in pixels
MAX_DISHES = None             # set to an int (e.g. 10) for testing
INCLUDE_SIDE_ANGLES = True    # download side-angle videos and extract frames
FRAME_INTERVAL = 10           # extract every Nth frame from videos
CAMERAS = ["camera_B", "camera_C"]
WORKERS = 8                   # number of parallel download threads

# =====================================================================

GCS_BUCKET = "gs://nutrition5k_dataset/nutrition5k_dataset"

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
        name = entry.rstrip("/").split("/")[-1]
        if name.startswith("dish_"):
            dish_ids.append(name)
    return sorted(dish_ids)


def format_eta(elapsed, completed, total):
    """Format an ETA string from elapsed time and progress."""
    rate = completed / elapsed if elapsed > 0 else 0
    eta_s = (total - completed) / rate if rate > 0 else 0
    eta_m, eta_sec = divmod(int(eta_s), 60)
    eta_h, eta_m = divmod(eta_m, 60)
    return f"{eta_h}h{eta_m:02d}m{eta_sec:02d}s" if eta_h else f"{eta_m}m{eta_sec:02d}s"


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

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if not gcloud_cp(gcs_path, tmp_path):
            return "fail"

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
    resize to target resolution, flip vertically, and delete the original video.

    Returns dict with counts: {'ok': N, 'skip': N, 'fail': N}
    """
    counts = {"ok": 0, "skip": 0, "fail": 0}

    for camera in cameras:
        frames_dir = output_dir / "imagery" / "side_angles" / dish_id / camera
        if frames_dir.exists() and any(frames_dir.iterdir()):
            counts["skip"] += 1
            continue

        gcs_path = f"{GCS_BUCKET}/imagery/side_angles/{dish_id}/{camera}.h264"

        tmp_dir = tempfile.mkdtemp()
        tmp_video = os.path.join(tmp_dir, f"{camera}.h264")
        tmp_frames_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(tmp_frames_dir, exist_ok=True)

        try:
            if not gcloud_cp(gcs_path, tmp_video):
                counts["fail"] += 1
                continue

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

            os.unlink(tmp_video)

            frames_dir.mkdir(parents=True, exist_ok=True)
            frame_files = sorted(glob.glob(os.path.join(tmp_frames_dir, "*.png")))

            for frame_path in frame_files:
                frame_name = os.path.basename(frame_path)
                final_frame = frames_dir / frame_name
                with Image.open(frame_path) as img:
                    # Side-angle frames are captured upside down — flip vertically
                    img_flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
                    img_resized = img_flipped.resize(
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
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir.resolve()}")
    print(f"Target resolution: {RESOLUTION}x{RESOLUTION}")
    if INCLUDE_SIDE_ANGLES:
        print(f"Side angles:      enabled (every {FRAME_INTERVAL}th frame, cameras: {CAMERAS})")
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

    all_dish_ids = get_dish_ids_from_metadata(output_dir)
    print(f"  Found {len(all_dish_ids)} total dishes in metadata")

    if MAX_DISHES:
        overhead_dish_ids = overhead_dish_ids[:MAX_DISHES]
        all_dish_ids = all_dish_ids[:MAX_DISHES]
        print(f"  Limited to first {MAX_DISHES} dishes")
    print()

    # Step 3: Download and resize overhead images
    print(f"=== Downloading and resizing overhead RGB images ({WORKERS} threads) ===")
    counts = {"ok": 0, "skip": 0, "fail": 0}
    total = len(overhead_dish_ids)
    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_dish = {
            executor.submit(download_and_resize_image, dish_id, output_dir, RESOLUTION): dish_id
            for dish_id in overhead_dish_ids
        }
        for i, future in enumerate(concurrent.futures.as_completed(future_to_dish), 1):
            status = future.result()

            if status == "skip":
                counts["skip"] += 1
            elif status == "ok":
                counts["ok"] += 1
            else:
                counts["fail"] += 1

            eta_str = format_eta(time.time() - t_start, i, total)
            print(f"\r  [{i}/{total}] ok:{counts['ok']} skip:{counts['skip']} fail:{counts['fail']} | ETA: {eta_str}   ", end="", flush=True)

    print()  # newline after overhead completes

    # Step 4: Side-angle video frames
    side_counts = {"ok": 0, "skip": 0, "fail": 0}
    if INCLUDE_SIDE_ANGLES:
        side_dish_ids = all_dish_ids
        if MAX_DISHES:
            side_dish_ids = side_dish_ids[:MAX_DISHES]
        side_total = len(side_dish_ids)

        print()
        print(f"=== Downloading side-angle videos & extracting frames (every {FRAME_INTERVAL}th, {WORKERS} threads) ===")
        t_side_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
            future_to_dish = {
                executor.submit(
                    extract_and_resize_frames,
                    dish_id, output_dir, RESOLUTION,
                    FRAME_INTERVAL, CAMERAS,
                ): dish_id
                for dish_id in side_dish_ids
            }
            for i, future in enumerate(concurrent.futures.as_completed(future_to_dish), 1):
                result = future.result()
                side_counts["ok"] += result["ok"]
                side_counts["skip"] += result["skip"]
                side_counts["fail"] += result["fail"]

                eta_str = format_eta(time.time() - t_side_start, i, side_total)
                print(f"\r  [{i}/{side_total}] ok:{side_counts['ok']} skip:{side_counts['skip']} fail:{side_counts['fail']} | ETA: {eta_str}   ", end="", flush=True)

        print()  # newline after side-angles complete

    # Summary
    print()
    print("=== Summary: Overhead RGB ===")
    print(f"  Downloaded & resized: {counts['ok']}")
    print(f"  Skipped (existing):   {counts['skip']}")
    print(f"  Failed:               {counts['fail']}")
    print(f"  Total dishes:         {total}")

    if INCLUDE_SIDE_ANGLES:
        print()
        print("=== Summary: Side-Angle Frames ===")
        print(f"  Cameras processed:  {side_counts['ok']}")
        print(f"  Cameras skipped:    {side_counts['skip']}")
        print(f"  Cameras failed:     {side_counts['fail']}")
        total_cameras = len(side_dish_ids) * len(CAMERAS)
        print(f"  Total camera/dish:  {total_cameras}")


if __name__ == "__main__":
    main()
