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
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image


# =====================================================================
# CONFIGURATION — edit these variables to control training
# =====================================================================

DATA_DIR = "./data"
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VAL_SPLIT = 0.1           # fraction of training data used for validation
CHECKPOINT_DIR = "./checkpoints"
MAX_DISHES = None          # set to an int (e.g. 50) for smoke testing
NUM_WORKERS = 4

# Image sources to include: "overhead", "side_angles", or both
IMAGE_SOURCES = ["overhead", "side_angles"]  # e.g. ["overhead", "side_angles"]

# Side-angle cameras to use (only relevant if "side_angles" in IMAGE_SOURCES)
SIDE_CAMERAS = ["camera_A", "camera_B"]

# Side-angle images are typically upside down — flip most of the time
SIDE_ANGLE_VFLIP_PROB = 0.8

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

    def __init__(self, data_dir, split_file, transform=None,
                 side_angle_transform=None, max_dishes=None,
                 image_sources=None, side_cameras=None):
        self.data_dir = data_dir
        self.transform = transform
        self.side_angle_transform = side_angle_transform or transform

        if image_sources is None:
            image_sources = ["overhead"]
        if side_cameras is None:
            side_cameras = SIDE_CAMERAS

        # Load split IDs
        split_path = os.path.join(data_dir, split_file)
        with open(split_path, "r") as f:
            split_ids = [line.strip() for line in f if line.strip()]

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
                    for frame_path in frame_files:
                        self.samples.append((frame_path, labels, dish_id, "side"))
                        n_side += 1

        print(f"  Loaded {len(self.samples)} samples from {split_file} "
              f"({n_overhead} overhead, {n_side} side-angle frames, "
              f"{len(split_ids)} in split, {len(metadata)} in metadata)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels, dish_id, source_type = self.samples[idx]
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

    def __init__(self, num_tasks=5, dropout=0.5):
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

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    for i, (images, labels, _) in enumerate(loader, 1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pct = 100 * i / n_batches
        avg_loss = total_loss / i
        print(f"\r  Epoch {epoch:3d}/{total_epochs} | "
              f"Batch {i}/{n_batches} ({pct:5.1f}%) | "
              f"Running MAE: {avg_loss:.2f}", end="", flush=True)
    print()  # newline after epoch finishes
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        loss = criterion(preds, labels)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


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
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Side-angle training transform: adds a high-probability vertical flip
    # because side-angle frames are typically captured upside down
    side_angle_train_transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.RandomVerticalFlip(p=SIDE_ANGLE_VFLIP_PROB),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Side-angle val transform: deterministic vertical flip (always flip)
    side_angle_val_transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.RandomVerticalFlip(p=SIDE_ANGLE_VFLIP_PROB),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # --- Dataset ---
    print("=== Loading dataset ===")
    train_dataset = Nutrition5kDataset(
        DATA_DIR, "dish_ids/splits/rgb_train_ids.txt",
        transform=train_transform,
        side_angle_transform=side_angle_train_transform,
        max_dishes=MAX_DISHES,
        image_sources=IMAGE_SOURCES,
        side_cameras=SIDE_CAMERAS,
    )
    val_dataset = Nutrition5kDataset(
        DATA_DIR, "dish_ids/splits/rgb_train_ids.txt",
        transform=val_transform,
        side_angle_transform=side_angle_val_transform,
        max_dishes=MAX_DISHES,
        image_sources=IMAGE_SOURCES,
        side_cameras=SIDE_CAMERAS,
    )

    # Split into train / val using a fixed seed for reproducibility
    n_val = int(len(train_dataset) * VAL_SPLIT)
    n_train = len(train_dataset) - n_val
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(len(train_dataset)), [n_train, n_val], generator=generator
    )
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    print(f"  Train: {n_train} | Val: {n_val}")
    print()

    # --- Model ---
    print("=== Building model ===")
    model = NutritionModel().to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # --- Optimizer, loss, scheduler ---
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # --- Training loop ---
    print("=== Training ===")
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, EPOCHS)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Train MAE: {train_loss:8.2f} | Val MAE: {val_loss:8.2f} | "
              f"LR: {lr:.6f} | Time: {elapsed:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
            }, ckpt_path)
            print(f"           > Saved best model (val MAE: {val_loss:.2f})")

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "final_model.pth")
    torch.save({
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "train_loss": train_loss,
    }, final_path)

    print()
    print("=== Training Complete ===")
    print(f"  Best val MAE: {best_val_loss:.2f}")
    print(f"  Best model:   {os.path.join(CHECKPOINT_DIR, 'best_model.pth')}")
    print(f"  Final model:  {final_path}")


if __name__ == "__main__":
    main()
