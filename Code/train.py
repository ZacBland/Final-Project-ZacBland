#!/usr/bin/env python3
"""
Train a nutrition prediction model on the Nutrition5k dataset.

Architecture closely follows the CVPR 2021 paper (arXiv 2103.03375):
  InceptionV3 backbone → shared 2x4096 FC → per-task FC → 5 regression outputs
  (calories, mass, fat, carbs, protein)

Configure all settings via the global variables below, then run:
    python Code/train.py
"""

import csv
import glob
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


# =====================================================================
# CONFIGURATION — edit these variables to control training
# =====================================================================

DATA_DIR = "./data"
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VAL_SPLIT = 0.1           # fraction of training data used for validation
CHECKPOINT_DIR = "./checkpoints"
PLOTS_DIR = "./plots"       # directory for training plots
MAX_DISHES = None          # set to an int (e.g. 50) for smoke testing
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-4           # L2 regularization for optimizer
EARLY_STOP_PATIENCE = 15      # stop if val MAE doesn't improve for N epochs

# Image sources to include: "overhead", "side_angles", or both
IMAGE_SOURCES = ["overhead", "side_angles"]  # e.g. ["overhead", "side_angles"]

# Side-angle cameras to use (only relevant if "side_angles" in IMAGE_SOURCES)
SIDE_CAMERAS = ["camera_B", "camera_C"]

# Freeze backbone for the first N epochs (train only FC heads to stabilize)
FREEZE_BACKBONE_EPOCHS = 5

# =====================================================================

METADATA_CSVS = [
    "metadata/dish_metadata_cafe1.csv",
    "metadata/dish_metadata_cafe2.csv",
]

LABEL_NAMES = ["calories", "mass", "fat", "carbs", "protein"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_metadata(data_dir):
    """Parse both cafe metadata CSVs into {dish_id: [cal, mass, fat, carb, protein]}."""
    metadata = {}
    for csv_rel in METADATA_CSVS:
        csv_path = os.path.join(data_dir, csv_rel)
        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} not found, skipping")
            continue
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or not row[0].startswith("dish_"):
                    continue
                dish_id = row[0].strip()
                try:
                    labels = [float(row[i]) for i in range(1, 6)]
                    metadata[dish_id] = labels
                except (ValueError, IndexError):
                    continue
    return metadata


class Nutrition5kDataset(Dataset):
    """Dataset for Nutrition5k images (overhead + optional side-angle frames)."""

    def __init__(self, data_dir, split_file=None, dish_ids=None, transform=None,
                 side_angle_transform=None, max_dishes=None,
                 image_sources=None, side_cameras=None,
                 max_side_frames=None):
        self.data_dir = data_dir
        self.transform = transform
        self.side_angle_transform = side_angle_transform or transform

        if image_sources is None:
            image_sources = ["overhead"]
        if side_cameras is None:
            side_cameras = SIDE_CAMERAS

        # Load split IDs — either from pre-filtered list or split file
        if dish_ids is not None:
            split_ids = list(dish_ids)
        elif split_file is not None:
            split_path = os.path.join(data_dir, split_file)
            with open(split_path, "r") as f:
                split_ids = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError("Must provide either split_file or dish_ids")

        if max_dishes:
            split_ids = split_ids[:max_dishes]

        # Load metadata
        metadata = load_metadata(data_dir)

        # Build sample list: (image_path, labels, dish_id, source_type)
        self.samples = []
        n_overhead = 0
        n_side = 0

        for dish_id in split_ids:
            if dish_id not in metadata:
                continue
            labels = metadata[dish_id]

            # Overhead RGB image
            if "overhead" in image_sources:
                img_path = os.path.join(
                    data_dir, "imagery", "realsense_overhead", dish_id, "rgb.png"
                )
                if os.path.exists(img_path):
                    self.samples.append((img_path, labels, dish_id, "overhead"))
                    n_overhead += 1

            # Side-angle extracted frames
            if "side_angles" in image_sources:
                for camera in side_cameras:
                    frames_dir = os.path.join(
                        data_dir, "imagery", "side_angles", dish_id, camera
                    )
                    if not os.path.isdir(frames_dir):
                        continue
                    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
                    if not frame_files:
                        continue
                    if max_side_frames and len(frame_files) > max_side_frames:
                        # Store all paths; randomly pick in __getitem__ for per-epoch variance
                        self.samples.append((frame_files, labels, dish_id, "side"))
                        n_side += 1
                    else:
                        for frame_path in frame_files:
                            self.samples.append((frame_path, labels, dish_id, "side"))
                            n_side += 1

        source = split_file if split_file else f"{len(split_ids)} dish IDs"
        print(f"  Loaded {len(self.samples)} samples from {source} "
              f"({n_overhead} overhead, {n_side} side-angle frames, "
              f"{len(split_ids)} dishes, {len(metadata)} in metadata)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels, dish_id, source_type = self.samples[idx]
        if isinstance(img_path, list):
            img_path = random.choice(img_path)
        image = Image.open(img_path).convert("RGB")

        if source_type == "side" and self.side_angle_transform:
            image = self.side_angle_transform(image)
        elif self.transform:
            image = self.transform(image)

        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels, dish_id


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NutritionModel(nn.Module):
    """
    InceptionV3 backbone with multi-task regression head.

    Matches the paper's architecture:
      backbone features → shared 2x FC(4096) → per-task FC(4096) → FC(1) x 5
    """

    def __init__(self, num_tasks=5, dropout=0.6):
        super().__init__()

        # InceptionV3 backbone (pretrained on ImageNet)
        # Pretrained weights require aux_logits=True; we disable after loading
        self.backbone = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True,
        )
        self.backbone.aux_logits = False
        self.backbone.AuxLogits = None
        # Replace the classification head with identity to get 2048-dim features
        self.backbone.fc = nn.Identity()

        # Shared FC layers (paper: 2x 4096-dim)
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Per-task heads (paper: FC(4096) → FC(1) per task)
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, 1),
            )
            for _ in range(num_tasks)
        ])

    def forward(self, x):
        features = self.backbone(x)
        shared_out = self.shared(features)
        outputs = [head(shared_out) for head in self.task_heads]
        return torch.cat(outputs, dim=1)  # (batch, num_tasks)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs, label_mean=None):
    """Returns (norm_loss, denorm_mae)."""
    model.train()
    total_norm_loss = 0.0
    total_denorm_loss = 0.0
    n_batches = len(loader)
    lm = label_mean.to(device) if label_mean is not None else None
    for i, (images, labels, _) in enumerate(loader, 1):
        images, labels = images.to(device), labels.to(device)
        if lm is not None:
            norm_labels = labels / lm
        else:
            norm_labels = labels
        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, norm_labels)
        loss.backward()
        optimizer.step()
        total_norm_loss += loss.item()
        # Denormalized MAE for reporting
        with torch.no_grad():
            if lm is not None:
                denorm_mae = (preds * lm - labels).abs().mean().item()
            else:
                denorm_mae = loss.item()
            total_denorm_loss += denorm_mae
        pct = 100 * i / n_batches
        print(f"\r  Epoch {epoch:3d}/{total_epochs} | "
              f"Batch {i}/{n_batches} ({pct:5.1f}%) | "
              f"Running MAE: {total_denorm_loss / i:.2f}", end="", flush=True)
    print()  # newline after epoch finishes
    n = max(n_batches, 1)
    return total_norm_loss / n, total_denorm_loss / n


@torch.no_grad()
def validate(model, loader, criterion, device, label_mean=None):
    """Returns (avg_loss, per_nutrient_mae, per_nutrient_mae_pct).

    If label_mean is provided, labels are normalized for loss computation
    and predictions are denormalized for MAE reporting.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []
    lm = label_mean.to(device) if label_mean is not None else None
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        if lm is not None:
            norm_labels = labels / lm
        else:
            norm_labels = labels
        preds = model(images)
        loss = criterion(preds, norm_labels)
        total_loss += loss.item()
        n_batches += 1
        # Denormalize predictions for metric computation
        if lm is not None:
            all_preds.append((preds * lm).cpu())
        else:
            all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    avg_loss = total_loss / max(n_batches, 1)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    per_mae = (all_preds - all_labels).abs().mean(dim=0)
    mean_gt = all_labels.mean(dim=0).clamp(min=1e-6)
    per_mae_pct = 100.0 * per_mae / mean_gt
    return avg_loss, per_mae, per_mae_pct


def save_plots(history, plots_dir):
    """Generate and save training plots for the final presentation."""
    os.makedirs(plots_dir, exist_ok=True)
    epochs = history["epoch"]

    # --- 1. Train & Val Loss (log-log scale) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(epochs, history["train_loss"], "b-o", markersize=3, label="Train MAE")
    ax.loglog(epochs, history["val_loss"], "r-o", markersize=3, label="Val MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (log scale)")
    ax.set_title("Training & Validation Loss (Log-Log Scale)")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "loss_loglog.png"), dpi=150)
    plt.close(fig)

    # --- 2. Train & Val Loss (linear scale) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train MAE")
    ax.plot(epochs, history["val_loss"], "r-o", markersize=3, label="Val MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "loss_linear.png"), dpi=150)
    plt.close(fig)

    # --- 3. Per-nutrient MAE% over epochs ---
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(LABEL_NAMES):
        key = f"val_mae_pct_{name}"
        ax.plot(epochs, history[key], "-o", markersize=3, color=colors[i], label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE %")
    ax.set_title("Validation MAE% Per Nutrient Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "val_mae_pct_per_nutrient.png"), dpi=150)
    plt.close(fig)

    # --- 4. Per-nutrient absolute MAE over epochs ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for i, name in enumerate(LABEL_NAMES):
        key = f"val_mae_{name}"
        ax = axes[i]
        ax.plot(epochs, history[key], "-o", markersize=3, color=colors[i])
        ax.set_title(f"{name.capitalize()} MAE")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE (absolute)")
        ax.grid(True, alpha=0.3)
    axes[-1].axis("off")  # hide the 6th subplot
    fig.suptitle("Per-Nutrient Absolute MAE Over Training", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "val_mae_per_nutrient.png"), dpi=150)
    plt.close(fig)

    # --- 5. Learning rate schedule ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history["lr"], "g-o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "learning_rate.png"), dpi=150)
    plt.close(fig)

    # --- Save raw history as JSON for later use ---
    json_path = os.path.join(plots_dir, "training_history.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"  Plots saved to {plots_dir}/")
    print(f"  Training history saved to {json_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- Transforms ---
    # InceptionV3 expects 299x299. We resize to 320 then center crop to 299.
    train_transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

    # Side-angle training transform (images are pre-flipped at download time)
    side_angle_train_transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Side-angle val transform (images are pre-flipped at download time)
    side_angle_val_transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # --- Dataset (dish-level train/val split to prevent data leakage) ---
    print("=== Loading dataset ===")

    # Read all dish IDs from split file
    split_path = os.path.join(DATA_DIR, "dish_ids/splits/rgb_train_ids.txt")
    with open(split_path, "r") as f:
        all_dish_ids = [line.strip() for line in f if line.strip()]
    if MAX_DISHES:
        all_dish_ids = all_dish_ids[:MAX_DISHES]

    # Dish-level split (fixed seed for reproducibility)
    rng = random.Random(42)
    shuffled_ids = list(all_dish_ids)
    rng.shuffle(shuffled_ids)
    n_val_dishes = int(len(shuffled_ids) * VAL_SPLIT)
    val_dish_ids = shuffled_ids[:n_val_dishes]
    train_dish_ids = shuffled_ids[n_val_dishes:]
    print(f"  Dish-level split: {len(train_dish_ids)} train dishes, {len(val_dish_ids)} val dishes")

    train_dataset = Nutrition5kDataset(
        DATA_DIR, dish_ids=train_dish_ids,
        transform=train_transform,
        side_angle_transform=side_angle_train_transform,
        image_sources=IMAGE_SOURCES,
        side_cameras=SIDE_CAMERAS,
        max_side_frames=1,
    )
    val_dataset = Nutrition5kDataset(
        DATA_DIR, dish_ids=val_dish_ids,
        transform=val_transform,
        side_angle_transform=side_angle_val_transform,
        image_sources=IMAGE_SOURCES,
        side_cameras=SIDE_CAMERAS,
    )

    # Compute label normalization from training set
    all_labels = torch.tensor([s[1] for s in train_dataset.samples], dtype=torch.float32)
    label_mean = all_labels.mean(dim=0)  # (5,)
    label_mean = label_mean.clamp(min=1.0)  # avoid division by zero
    print(f"  Label means (normalization): {', '.join(f'{LABEL_NAMES[i]}: {label_mean[i]:.1f}' for i in range(5))}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    print(f"  Train: {n_train} samples | Val: {n_val} samples")
    print()

    # --- Model ---
    print("=== Building model ===")
    model = NutritionModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())

    # Freeze backbone for the first N epochs
    if FREEZE_BACKBONE_EPOCHS > 0:
        for param in model.backbone.parameters():
            param.requires_grad = False
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {total_params:,} ({trainable_params:,} trainable, backbone frozen for {FREEZE_BACKBONE_EPOCHS} epochs)")
    else:
        trainable_params = total_params
        print(f"  Parameters: {total_params:,} ({trainable_params:,} trainable)")
    print()

    # --- Optimizer, loss, scheduler ---
    # Only pass trainable params to optimizer initially
    optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- Save model summary ---
    summary_path = os.path.join(CHECKPOINT_DIR, "model_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Nutrition5k Model Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Device:              {device}\n")
        f.write(f"Backbone:            InceptionV3 (ImageNet pretrained)\n")
        f.write(f"Input size:          299x299 RGB\n")
        f.write(f"Outputs:             {', '.join(LABEL_NAMES)}\n")
        f.write(f"Loss function:       L1 (MAE)\n")
        f.write(f"Optimizer:           RMSProp (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})\n")
        f.write(f"LR scheduler:        CosineAnnealingLR (T_max={EPOCHS})\n")
        f.write(f"Batch size:          {BATCH_SIZE}\n")
        f.write(f"Epochs:              {EPOCHS}\n")
        f.write(f"Val split:           {VAL_SPLIT}\n")
        f.write(f"Image sources:       {IMAGE_SOURCES}\n")
        f.write(f"Side cameras:        {SIDE_CAMERAS}\n")
        f.write(f"Freeze backbone:     {FREEZE_BACKBONE_EPOCHS} epochs\n")
        f.write(f"Early stop patience: {EARLY_STOP_PATIENCE}\n")
        f.write(f"Dropout:             0.6\n")
        f.write(f"Train samples:       {n_train}\n")
        f.write(f"Val samples:         {n_val}\n")
        f.write(f"Total parameters:    {total_params:,}\n")
        f.write(f"Trainable params:    {trainable_params:,}\n\n")
        f.write("Architecture\n")
        f.write("-" * 60 + "\n")
        f.write(str(model))
        f.write("\n\n")
        f.write("Per-layer Parameter Counts\n")
        f.write("-" * 60 + "\n")
        for name, param in model.named_parameters():
            f.write(f"  {name:<55} {param.numel():>12,}  {'train' if param.requires_grad else 'frozen'}\n")
    print(f"  Model summary saved to {summary_path}")

    # --- Training loop ---
    print("=== Training ===")
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # History tracking for plots
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }
    for name in LABEL_NAMES:
        history[f"val_mae_{name}"] = []
        history[f"val_mae_pct_{name}"] = []

    for epoch in range(1, EPOCHS + 1):
        # Unfreeze backbone after FREEZE_BACKBONE_EPOCHS
        if epoch == FREEZE_BACKBONE_EPOCHS + 1 and FREEZE_BACKBONE_EPOCHS > 0:
            print(f"  >>> Unfreezing backbone at epoch {epoch}")
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Rebuild optimizer with all parameters and a lower LR for backbone
            optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE * 0.01, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - FREEZE_BACKBONE_EPOCHS)

        t0 = time.time()
        train_norm_loss, train_mae = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, EPOCHS, label_mean=label_mean)
        val_norm_loss, val_mae, val_mae_pct = validate(model, val_loader, criterion, device, label_mean=label_mean)
        scheduler.step()
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]

        # Use denormalized val MAE for display and best-model tracking
        val_mae_avg = val_mae.mean().item()

        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_mae)
        history["val_loss"].append(val_mae_avg)
        history["lr"].append(lr)
        for i, name in enumerate(LABEL_NAMES):
            history[f"val_mae_{name}"].append(val_mae[i].item())
            history[f"val_mae_pct_{name}"].append(val_mae_pct[i].item())

        print(f"  ─── Epoch {epoch}/{EPOCHS} ────────────────────────────────────")
        print(f"    Train  ─  MAE: {train_mae:7.2f}  |  norm loss: {train_norm_loss:.3f}")
        print(f"    Val    ─  MAE: {val_mae_avg:7.2f}  |  norm loss: {val_norm_loss:.3f}")
        print(f"    Val MAE%:  "
              + "  ".join(f"{name}: {val_mae_pct[i]:.1f}%"
                         for i, name in enumerate(LABEL_NAMES)))
        print(f"    LR: {lr:.6f}  |  Time: {elapsed:.1f}s")

        # Save best model
        if val_mae_avg < best_val_loss:
            best_val_loss = val_mae_avg
            epochs_without_improvement = 0
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_mae_avg,
                "train_loss": train_mae,
                "label_mean": label_mean,
            }, ckpt_path)
            print(f"    > Saved best model (val MAE: {val_mae_avg:.2f})")
        else:
            epochs_without_improvement += 1
            if EARLY_STOP_PATIENCE and epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                save_plots(history, PLOTS_DIR)
                break

        # Update plots every 5 epochs (and on the last epoch)
        if epoch % 5 == 0 or epoch == EPOCHS:
            save_plots(history, PLOTS_DIR)

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "final_model.pth")
    torch.save({
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_mae_avg,
        "train_loss": train_mae,
        "label_mean": label_mean,
    }, final_path)

    # Final plots
    save_plots(history, PLOTS_DIR)

    print()
    print("=== Training Complete ===")
    print(f"  Best val MAE: {best_val_loss:.2f}")
    print(f"  Best model:   {os.path.join(CHECKPOINT_DIR, 'best_model.pth')}")
    print(f"  Final model:  {final_path}")
    print(f"  Plots:        {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
