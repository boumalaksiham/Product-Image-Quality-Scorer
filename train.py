"""
Training Script — Image Quality Scorer
========================================
Fine-tunes EfficientNet-B0 on a synthetic labeled dataset
of product images with quality scores 1-5.

In production you'd use:
    - AVA dataset (255k images with human quality ratings)
    - Labeled eBay product images
    - Amazon product image quality dataset

Here we use our synthetic generator to demonstrate the full pipeline.

Usage:
    python train.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.image_generator import generate_all
from models.quality_scorer import EfficientNetScorer, get_transform

CONFIG = {
    "epochs": 10,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "test_size": 0.2,
    "random_state": 42,
}

# Synthetic quality labels for each generated image
# In production these would come from human annotators
IMAGE_LABELS = {
    "01_high_quality.jpg":         4.8,
    "02_good_quality.jpg":         3.8,
    "03_blurry.jpg":               1.8,
    "04_dark.jpg":                 1.5,
    "05_overexposed.jpg":          2.0,
    "06_cluttered_background.jpg": 2.5,
    "07_low_resolution.jpg":       2.2,
}


class ImageQualityDataset(Dataset):
    """Dataset of product images with quality score labels."""

    def __init__(self, image_paths: list, labels: list, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        tensor = self.transform(img)
        # Normalize label to 0-1 for Sigmoid output
        label = torch.tensor((self.labels[idx] - 1) / 4, dtype=torch.float32)
        return tensor, label


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            preds = model(imgs).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # Convert back to 1-5 scale
    preds_1_5 = [1 + p * 4 for p in all_preds]
    labels_1_5 = [1 + l * 4 for l in all_labels]

    mae = np.mean(np.abs(np.array(preds_1_5) - np.array(labels_1_5)))
    return mae, preds_1_5, labels_1_5


def main():
    print("=" * 60)
    print("IMAGE QUALITY SCORER — TRAINING")
    print("=" * 60)

    device = torch.device("cpu")

    # 1. Generate synthetic images
    print("\nStep 1: Generating synthetic product images...")
    image_paths = generate_all("sample_images")

    # 2. Build dataset with labels
    all_paths, all_labels = [], []
    for path in image_paths:
        filename = os.path.basename(path)
        if filename in IMAGE_LABELS:
            all_paths.append(path)
            all_labels.append(IMAGE_LABELS[filename])

    print(f"\nDataset: {len(all_paths)} images")
    for p, l in zip(all_paths, all_labels):
        print(f"  {os.path.basename(p):<40} score: {l}")

    # 3. Train/test split
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"]
    )

    transform = get_transform()
    train_dataset = ImageQualityDataset(train_paths, train_labels, transform)
    test_dataset = ImageQualityDataset(test_paths, test_labels, transform)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"])

    # 4. Model
    print("\nLoading EfficientNet-B0 (pretrained on ImageNet)...")
    model = EfficientNetScorer().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.MSELoss()  # regression loss

    # 5. Train
    print(f"\nTraining for {CONFIG['epochs']} epochs...")
    print("-" * 60)

    best_mae = float("inf")
    for epoch in range(1, CONFIG["epochs"] + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        mae, _, _ = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:>2}/{CONFIG['epochs']} | Loss: {loss:.4f} | MAE: {mae:.4f}")

        if mae < best_mae:
            best_mae = mae
            os.makedirs("models/saved", exist_ok=True)
            torch.save(model.state_dict(), "models/saved/best_model.pt")

    print(f"\nBest MAE: {best_mae:.4f} (on 1-5 scale)")

    # 6. Save config
    with open("models/saved/config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    print("\n✅ Done! Run: python demo.py")


if __name__ == "__main__":
    main()
