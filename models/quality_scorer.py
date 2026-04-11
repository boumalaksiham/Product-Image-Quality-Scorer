"""
Image Quality Scorer — EfficientNet + Computer Vision Signals
==============================================================
Two-stage pipeline:
    Stage 1: Classical CV signals (sharpness, brightness, contrast, etc.)
    Stage 2: EfficientNet deep features
    Final:   Weighted combination → score 1-5
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os


class CVSignalExtractor:
    """Extracts measurable quality signals from product images."""

    def extract(self, image_path: str) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        signals = {}

        # ── 1. Sharpness (Laplacian variance) ─────────────────
        # Recalibrated: synthetic clean images score ~50-100
        # Real blurry images score near 0, sharp images score 200+
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        signals["sharpness"] = min(1.0, laplacian_var / 150.0)

        # ── 2. Brightness ──────────────────────────────────────
        mean_brightness = np.mean(gray) / 255.0
        if mean_brightness < 0.15:
            signals["brightness"] = mean_brightness / 0.15
        elif mean_brightness > 0.92:
            # Overexposed: score drops sharply above 0.92
            signals["brightness"] = max(0.0, 1.0 - (mean_brightness - 0.92) / 0.08)
        else:
            signals["brightness"] = 1.0

        # ── 3. Contrast ────────────────────────────────────────
        std_dev = np.std(gray)
        # Recalibrated: synthetic images have std ~30-80
        # Good contrast needs std > 40
        signals["contrast"] = min(1.0, std_dev / 60.0)

        # ── 4. Colorfulness ────────────────────────────────────
        r, g, b = img_rgb[:,:,0].astype(float), img_rgb[:,:,1].astype(float), img_rgb[:,:,2].astype(float)
        rg = r - g
        yb = 0.5 * (r + g) - b
        colorfulness = np.sqrt(rg.std()**2 + yb.std()**2) + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2)
        signals["colorfulness"] = min(1.0, colorfulness / 80.0)

        # ── 5. Background simplicity ───────────────────────────
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / edges.size
        signals["background_simplicity"] = max(0.0, 1.0 - edge_density * 8)

        # ── 6. Resolution ──────────────────────────────────────
        h, w = gray.shape
        min_dim = min(h, w)
        signals["resolution"] = min(1.0, min_dim / 600.0)

        # ── 7. Noise level ─────────────────────────────────────
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = np.std(gray.astype(float) - blurred.astype(float))
        signals["noise_level"] = max(0.0, 1.0 - noise / 15.0)

        return signals

    def compute_cv_score(self, signals: dict) -> float:
        weights = {
            "sharpness":             0.30,
            "brightness":            0.20,
            "contrast":              0.20,
            "colorfulness":          0.10,
            "background_simplicity": 0.10,
            "resolution":            0.05,
            "noise_level":           0.05,
        }
        score = sum(signals[k] * weights[k] for k in weights if k in signals)
        return round(score, 4)


class EfficientNetScorer(nn.Module):
    """EfficientNet-B0 with regression head for quality scoring."""

    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class ProductImageQualityScorer:
    """Combined CV + EfficientNet quality scorer."""

    def __init__(self, model_path: str = None):
        self.cv_extractor = CVSignalExtractor()
        self.transform = get_transform()
        self.device = torch.device("cpu")

        self.dl_model = EfficientNetScorer().to(self.device)
        if model_path and os.path.exists(model_path):
            self.dl_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.use_dl = True
        else:
            self.use_dl = False
        self.dl_model.eval()

    def score_image(self, image_path: str) -> dict:
        # CV signals
        signals = self.cv_extractor.extract(image_path)
        cv_score = self.cv_extractor.compute_cv_score(signals)

        # Deep learning score
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dl_score = self.dl_model(tensor).item()

        # Combined — weight CV more heavily when model is not well trained
        if self.use_dl:
            combined = 0.35 * dl_score + 0.65 * cv_score
        else:
            combined = cv_score

        final_score = round(1 + combined * 4, 2)

        if final_score >= 4.5:   grade = "Excellent"
        elif final_score >= 3.5: grade = "Good"
        elif final_score >= 2.5: grade = "Acceptable"
        elif final_score >= 1.5: grade = "Poor"
        else:                    grade = "Very Poor"

        issues, recommendations = self._diagnose(signals)

        return {
            "image_path": image_path,
            "final_score": final_score,
            "grade": grade,
            "dl_score": round(dl_score, 4),
            "cv_score": round(cv_score, 4),
            "signals": {k: round(v, 4) for k, v in signals.items()},
            "issues": issues,
            "recommendations": recommendations
        }

    def _diagnose(self, signals):
        issues, recommendations = [], []

        if signals.get("sharpness", 1) < 0.3:
            issues.append("Image is blurry")
            recommendations.append("Use a tripod or stabilize your camera")
        if signals.get("brightness", 1) < 0.4:
            issues.append("Image is too dark")
            recommendations.append("Add more lighting or shoot near a window")
        if signals.get("brightness", 1) < 0.3 and np.mean([signals.get("contrast",1)]) < 0.2:
            issues.append("Image is overexposed")
            recommendations.append("Reduce lighting or use diffused light")
        if signals.get("contrast", 1) < 0.3:
            issues.append("Low contrast — product blends into background")
            recommendations.append("Use a contrasting background color")
        if signals.get("background_simplicity", 1) < 0.3:
            issues.append("Background is cluttered or busy")
            recommendations.append("Use a plain white or neutral background")
        if signals.get("resolution", 1) < 0.5:
            issues.append("Image resolution is too low")
            recommendations.append("Use at least 600x600px for product images")
        if signals.get("noise_level", 1) < 0.5:
            issues.append("Image is grainy/noisy")
            recommendations.append("Shoot in better lighting to reduce camera noise")

        if not issues:
            issues.append("No major issues detected")
            recommendations.append("Image meets quality standards")

        return issues, recommendations

    def score_batch(self, image_paths):
        results = []
        for path in image_paths:
            try:
                results.append(self.score_image(path))
            except Exception as e:
                results.append({"image_path": path, "error": str(e)})
        results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return results