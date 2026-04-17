#!/usr/bin/env python3
"""
Verify integrity of all downloaded images and delete corrupt ones.

Scans all .png files under DATA_DIR/imagery/, attempts to open and decode
each image with PIL, and optionally deletes any that are corrupt or truncated.

Uses multithreading for fast scanning.

Usage:
    python3 scripts/verify_images.py
"""

import concurrent.futures
import glob
import os
import sys
import time

from PIL import Image

# =====================================================================
# CONFIGURATION — edit these variables
# =====================================================================

DATA_DIR = "./data"
WORKERS = 8          # number of parallel verification threads
DRY_RUN = False      # True = report only, False = delete corrupt files

# =====================================================================


def format_eta(elapsed, completed, total):
    """Format an ETA string from elapsed time and progress."""
    rate = completed / elapsed if elapsed > 0 else 0
    eta_s = (total - completed) / rate if rate > 0 else 0
    eta_m, eta_sec = divmod(int(eta_s), 60)
    eta_h, eta_m = divmod(eta_m, 60)
    return f"{eta_h}h{eta_m:02d}m{eta_sec:02d}s" if eta_h else f"{eta_m}m{eta_sec:02d}s"


def verify_image(path):
    """
    Verify a single image file.
    Returns the path if corrupt, None if OK.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        # verify() doesn't catch all issues — re-open and fully decode
        with Image.open(path) as img:
            img.convert("RGB")
        return None
    except (OSError, SyntaxError, Exception):
        return path


def main():
    imagery_dir = os.path.join(DATA_DIR, "imagery")

    if not os.path.isdir(imagery_dir):
        print(f"ERROR: imagery directory not found: {imagery_dir}")
        sys.exit(1)

    # Collect all .png files
    print(f"Scanning for images in {os.path.abspath(imagery_dir)} ...")
    all_images = glob.glob(os.path.join(imagery_dir, "**", "*.png"), recursive=True)
    total = len(all_images)

    if total == 0:
        print("No .png files found.")
        return

    print(f"Found {total} images to verify (workers={WORKERS}, dry_run={DRY_RUN})")
    print()

    corrupt_files = []
    checked = 0
    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(verify_image, path): path for path in all_images}

        for future in concurrent.futures.as_completed(futures):
            checked += 1
            result = future.result()
            if result is not None:
                corrupt_files.append(result)

            eta_str = format_eta(time.time() - t_start, checked, total)
            print(
                f"\r  [{checked}/{total}] corrupt: {len(corrupt_files)} | ETA: {eta_str}   ",
                end="",
                flush=True,
            )

    elapsed = time.time() - t_start
    print()
    print()

    if not corrupt_files:
        print(f"All {total} images are valid. ({elapsed:.1f}s)")
        return

    # Report corrupt files
    print(f"Found {len(corrupt_files)} corrupt image(s):")
    for path in sorted(corrupt_files):
        print(f"  {path}")
    print()

    if DRY_RUN:
        print(f"DRY_RUN is enabled — no files were deleted.")
        print(f"Set DRY_RUN = False and re-run to delete them.")
    else:
        for path in corrupt_files:
            os.remove(path)
        print(f"Deleted {len(corrupt_files)} corrupt file(s).")

    print()
    print(f"=== Summary ===")
    print(f"  Total scanned:  {total}")
    print(f"  Valid:           {total - len(corrupt_files)}")
    print(f"  Corrupt:         {len(corrupt_files)}")
    print(f"  Time:            {elapsed:.1f}s")


if __name__ == "__main__":
    main()
