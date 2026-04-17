#!/usr/bin/env python3
"""
Run inference on a single image to predict nutritional content.

Usage:
    python Code/predict.py path/to/image.png
    python Code/predict.py path/to/image.png --checkpoint checkpoints/best_model.pth
"""

import argparse
import sys
import os

import torch
from torchvision import transforms
from PIL import Image

# Import model and label names from train.py
from train import NutritionModel, LABEL_NAMES


def predict(image_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same preprocessing as test.py
    preprocess = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)  # (1, 3, 299, 299)

    # Load model
    model = NutritionModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Predict
    with torch.no_grad():
        preds, _ = model(input_tensor)
        preds = preds.squeeze(0)  # (5,)

    # Display results
    print(f"\n{'=' * 40}")
    print(f"  Image: {image_path}")
    print(f"  Model: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    print(f"{'=' * 40}")
    for i, name in enumerate(LABEL_NAMES):
        unit = "kcal" if name == "calories" else "g"
        print(f"  {name:<12} {preds[i].item():>10.2f} {unit}")
    print(f"{'=' * 40}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict nutrition from a food image.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--checkpoint", default="./checkpoints/best_model.pth",
                        help="Path to model checkpoint (default: ./checkpoints/best_model.pth)")
    args = parser.parse_args()
    predict(args.image, args.checkpoint)
