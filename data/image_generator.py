"""
Synthetic Image Generator
==========================
Since we don't have real product photos, we generate synthetic test images
that simulate different quality levels a seller might upload.

This lets us demonstrate and test the scoring system without needing
a real labeled dataset of product images.

In a real production system, you'd use:
    - AVA dataset (255,000 rated images)
    - Amazon product image dataset
    - Internally labeled eBay listing images
"""

import numpy as np
import cv2
import os


def generate_high_quality(path: str):
    """
    Simulates a professional product photo:
    - White background
    - Sharp subject in center
    - Good lighting
    - High resolution
    """
    img = np.ones((800, 800, 3), dtype=np.uint8) * 250  # near-white background

    # Draw a colorful, sharp product shape (simulating a shoe box or phone)
    cv2.rectangle(img, (200, 250), (600, 550), (45, 85, 255), -1)   # blue product
    cv2.rectangle(img, (220, 270), (580, 530), (80, 120, 255), -1)   # lighter face
    cv2.rectangle(img, (200, 250), (600, 550), (20, 60, 200), 3)     # sharp edge

    # Add some realistic detail lines
    for y in range(280, 520, 30):
        cv2.line(img, (220, y), (580, y), (60, 100, 235), 1)

    # No blur — sharp image
    cv2.imwrite(path, img)
    print(f"Generated: {path} (HIGH QUALITY)")


def generate_blurry(path: str):
    """
    Simulates a blurry product photo (camera shake or out of focus).
    """
    img = np.ones((800, 800, 3), dtype=np.uint8) * 230

    cv2.rectangle(img, (200, 250), (600, 550), (45, 85, 255), -1)
    cv2.rectangle(img, (220, 270), (580, 530), (80, 120, 255), -1)

    # Apply heavy Gaussian blur to simulate camera shake
    img = cv2.GaussianBlur(img, (51, 51), 15)

    cv2.imwrite(path, img)
    print(f"Generated: {path} (BLURRY)")


def generate_dark(path: str):
    """
    Simulates a dark product photo (poor lighting).
    """
    img = np.ones((800, 800, 3), dtype=np.uint8) * 40  # dark background

    # Draw product with dark colors
    cv2.rectangle(img, (200, 250), (600, 550), (20, 40, 100), -1)
    cv2.rectangle(img, (220, 270), (580, 530), (30, 55, 120), -1)

    cv2.imwrite(path, img)
    print(f"Generated: {path} (DARK)")


def generate_cluttered_background(path: str):
    """
    Simulates a product photo with a busy, cluttered background.
    (e.g. photo taken on a messy table or in a room)
    """
    # Busy patterned background
    img = np.random.randint(50, 200, (800, 800, 3), dtype=np.uint8)

    # Add random lines and shapes to simulate clutter
    for _ in range(30):
        x1, y1 = np.random.randint(0, 800, 2)
        x2, y2 = np.random.randint(0, 800, 2)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(img, (x1, y1), (x2, y2), color, 2)

    # Draw product on top
    cv2.rectangle(img, (200, 250), (600, 550), (45, 85, 255), -1)
    cv2.rectangle(img, (220, 270), (580, 530), (80, 120, 255), -1)
    cv2.rectangle(img, (200, 250), (600, 550), (20, 60, 200), 2)

    cv2.imwrite(path, img)
    print(f"Generated: {path} (CLUTTERED BACKGROUND)")


def generate_low_resolution(path: str):
    """
    Simulates a low-resolution product photo (old phone camera or thumbnail).
    """
    # Create small image then upscale (introduces pixelation)
    small = np.ones((100, 100, 3), dtype=np.uint8) * 240

    cv2.rectangle(small, (25, 30), (75, 70), (45, 85, 255), -1)
    cv2.rectangle(small, (25, 30), (75, 70), (20, 60, 200), 2)

    # Upscale with nearest neighbor (pixelated look)
    img = cv2.resize(small, (800, 800), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(path, img)
    print(f"Generated: {path} (LOW RESOLUTION)")


def generate_overexposed(path: str):
    """
    Simulates an overexposed product photo (too much light/flash).
    """
    img = np.ones((800, 800, 3), dtype=np.uint8) * 255  # blown out white

    # Product barely visible
    cv2.rectangle(img, (200, 250), (600, 550), (230, 235, 255), -1)
    cv2.rectangle(img, (200, 250), (600, 550), (210, 215, 250), 2)

    cv2.imwrite(path, img)
    print(f"Generated: {path} (OVEREXPOSED)")


def generate_good_quality(path: str):
    """
    Simulates a good (but not perfect) product photo.
    Clean background, decent sharpness, slight imperfections.
    """
    img = np.ones((800, 800, 3), dtype=np.uint8) * 240  # slightly grey bg

    cv2.rectangle(img, (200, 250), (600, 550), (200, 100, 50), -1)   # warm product color
    cv2.rectangle(img, (220, 270), (580, 530), (220, 130, 80), -1)
    cv2.rectangle(img, (200, 250), (600, 550), (160, 80, 30), 2)

    # Very slight blur — not perfect
    img = cv2.GaussianBlur(img, (3, 3), 1)

    cv2.imwrite(path, img)
    print(f"Generated: {path} (GOOD QUALITY)")


def generate_all(output_dir: str = "sample_images"):
    """Generates all test images."""
    os.makedirs(output_dir, exist_ok=True)

    images = {
        "01_high_quality.jpg":        generate_high_quality,
        "02_good_quality.jpg":        generate_good_quality,
        "03_blurry.jpg":              generate_blurry,
        "04_dark.jpg":                generate_dark,
        "05_overexposed.jpg":         generate_overexposed,
        "06_cluttered_background.jpg": generate_cluttered_background,
        "07_low_resolution.jpg":      generate_low_resolution,
    }

    print("Generating synthetic test images...")
    for filename, generator in images.items():
        generator(os.path.join(output_dir, filename))

    print(f"\n✅ Generated {len(images)} test images in '{output_dir}/'")
    return [os.path.join(output_dir, f) for f in images.keys()]


if __name__ == "__main__":
    generate_all()
