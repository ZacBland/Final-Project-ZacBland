#!/usr/bin/env python3
"""
Evaluate a trained Nutrition5k model on the test split.

Computes per-nutrient MAE and MAE% (matching the paper's evaluation protocol)
and optionally saves a predictions CSV compatible with compute_statistics.py.

Configure all settings via the global variables below, then run:
    python Code/test.py
"""

import csv
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Import from train.py
from train import (
    Nutrition5kDataset, NutritionModel, LABEL_NAMES,
    SIDE_CAMERAS, SIDE_ANGLE_VFLIP_PROB,
)


# =====================================================================
# CONFIGURATION — edit these variables to control evaluation
# =====================================================================

DATA_DIR = "./data"
CHECKPOINT = "./checkpoints/best_model.pth"
BATCH_SIZE = 32
NUM_WORKERS = 4
MAX_DISHES = None          # set to an int for quick testing
OUTPUT_CSV = None          # set to a path (e.g. "predictions.csv") to save predictions

# Image sources to include: "overhead", "side_angles", or both
IMAGE_SOURCES = ["overhead"]  # should match what the model was trained on

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

    # Side-angle test transform: flip with same probability as training
    side_angle_test_transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.RandomVerticalFlip(p=SIDE_ANGLE_VFLIP_PROB),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # --- Dataset ---
    print("=== Loading test dataset ===")
    test_dataset = Nutrition5kDataset(
        DATA_DIR, "dish_ids/splits/rgb_test_ids.txt",
        transform=test_transform,
        side_angle_transform=side_angle_test_transform,
        max_dishes=MAX_DISHES,
        image_sources=IMAGE_SOURCES,
        side_cameras=SIDE_CAMERAS,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    print()

    # --- Load model ---
    print("=== Loading model ===")
    model = NutritionModel().to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Checkpoint val MAE: {checkpoint.get('val_loss', '?')}")
    print()

    # --- Inference ---
    print("=== Running inference ===")
    dish_ids, preds, labels = run_inference(model, test_loader, device)
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
    if OUTPUT_CSV:
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            for i, dish_id in enumerate(dish_ids):
                row = [dish_id] + [f"{preds[i, j].item():.6f}" for j in range(5)]
                writer.writerow(row)
        print(f"  Predictions saved to {OUTPUT_CSV}")
        print(f"  Evaluate with: python scripts/compute_statistics.py "
              f"data/metadata/dish_metadata_cafe1.csv {OUTPUT_CSV} results.json")


if __name__ == "__main__":
    main()
