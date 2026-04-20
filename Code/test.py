#!/usr/bin/env python3
"""
Evaluate a trained Nutrition5k model on the test split.

Computes per-nutrient MAE and MAE% (matching the paper's evaluation protocol)
and optionally saves a predictions CSV compatible with compute_statistics.py.

Usage:
    python Code/test.py
    python Code/test.py --checkpoint models/baseline_cnn/checkpoints/best_model.pth --output_csv models/baseline_cnn/predictions.csv
    python Code/test.py --checkpoint models/MoE/checkpoints/best_model.pth --output_csv models/MoE/predictions.csv --image_sources overhead side_angles
"""

import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Import from train.py
from train import (
    Nutrition5kDataset, NutritionModel, LABEL_NAMES,
    SIDE_CAMERAS, load_metadata,
)


# =====================================================================
# DEFAULTS — override via command-line arguments
# =====================================================================

DATA_DIR = "./data"
CHECKPOINT = "./checkpoints/best_model.pth"
BATCH_SIZE = 32
NUM_WORKERS = 4
MAX_DISHES = None
OUTPUT_CSV = None
IMAGE_SOURCES = ["overhead"]

# =====================================================================


@torch.no_grad()
def run_inference(model, loader, device):
    """Run inference on all batches. Returns lists of (dish_id, predictions, ground_truth)."""
    model.eval()
    all_dish_ids = []
    all_preds = []
    all_labels = []

    for images, labels, dish_ids in loader:
        images = images.to(device)
        preds = model(images)
        all_dish_ids.extend(dish_ids)
        all_preds.append(preds.cpu())
        all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_dish_ids, all_preds, all_labels


def compute_metrics(preds, labels):
    """Compute per-nutrient MAE and MAE% (matching the paper)."""
    # preds, labels: (N, 5)
    abs_errors = (preds - labels).abs()
    mae = abs_errors.mean(dim=0)                     # (5,)
    mean_gt = labels.mean(dim=0)                      # (5,)
    mae_pct = 100.0 * mae / mean_gt.clamp(min=1e-6)  # (5,)
    return mae, mae_pct


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Nutrition5k model on the test split.")
    parser.add_argument("--data_dir", default=DATA_DIR, help="Path to data directory (default: %(default)s)")
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="Path to model checkpoint (default: %(default)s)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size (default: %(default)s)")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="DataLoader workers (default: %(default)s)")
    parser.add_argument("--max_dishes", type=int, default=MAX_DISHES, help="Limit number of dishes for quick testing")
    parser.add_argument("--output_csv", default=OUTPUT_CSV, help="Path to save predictions CSV")
    parser.add_argument("--image_sources", nargs="+", default=IMAGE_SOURCES,
                        choices=["overhead", "side_angles"],
                        help="Image sources to use (default: %(default)s)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Transforms ---
    test_transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Side-angle test transform (images are pre-flipped at download time)
    side_angle_test_transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # --- Dataset ---
    print("=== Loading test dataset ===")
    test_dataset = Nutrition5kDataset(
        args.data_dir, "dish_ids/splits/rgb_test_ids.txt",
        transform=test_transform,
        side_angle_transform=side_angle_test_transform,
        max_dishes=args.max_dishes,
        image_sources=args.image_sources,
        side_cameras=SIDE_CAMERAS,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print()

    # --- Load model ---
    print("=== Loading model ===")
    model = NutritionModel().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Checkpoint val MAE: {checkpoint.get('val_loss', '?')}")

    # --- Label mean for denormalization ---
    # Model outputs normalized predictions (pred / label_mean during training).
    # We need label_mean to convert back to real units.
    if "label_mean" in checkpoint:
        label_mean = checkpoint["label_mean"]
        print(f"  Label mean (from checkpoint): {', '.join(f'{LABEL_NAMES[i]}: {label_mean[i]:.1f}' for i in range(5))}")
    else:
        # Fallback for old checkpoints: compute from training split
        print("  Warning: checkpoint missing label_mean, computing from training split")
        import random
        split_path = os.path.join(args.data_dir, "dish_ids/splits/rgb_train_ids.txt")
        with open(split_path, "r") as f:
            train_ids = [line.strip() for line in f if line.strip()]
        metadata = load_metadata(args.data_dir)
        train_labels = torch.tensor(
            [metadata[did] for did in train_ids if did in metadata],
            dtype=torch.float32,
        )
        label_mean = train_labels.mean(dim=0).clamp(min=1.0)
        print(f"  Label mean (computed): {', '.join(f'{LABEL_NAMES[i]}: {label_mean[i]:.1f}' for i in range(5))}")
    print()

    # --- Inference ---
    print("=== Running inference ===")
    dish_ids, preds, labels = run_inference(model, test_loader, device)
    # Denormalize predictions back to real units
    preds = preds * label_mean
    print(f"  Evaluated {len(dish_ids)} samples")
    print()

    # --- Metrics ---
    mae, mae_pct = compute_metrics(preds, labels)

    print("=== Results ===")
    print(f"  {'Nutrient':<12} {'MAE':>10} {'MAE%':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:<12} {mae[i]:>10.2f} {mae_pct[i]:>9.1f}%")

    overall_mae = mae.mean().item()
    overall_pct = mae_pct.mean().item()
    print(f"  {'-'*12} {'-'*10} {'-'*10}")
    print(f"  {'Overall':<12} {overall_mae:>10.2f} {overall_pct:>9.1f}%")
    print()

    # --- Save predictions CSV ---
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            for i, dish_id in enumerate(dish_ids):
                row = [dish_id] + [f"{preds[i, j].item():.6f}" for j in range(5)]
                writer.writerow(row)
        print(f"  Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()
