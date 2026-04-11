"""
Demo — Image Quality Scorer
=============================
Scores all generated sample images and shows detailed breakdowns.
Can also score any image you provide.

Usage:
    python demo.py
    python demo.py --image /path/to/your/image.jpg
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.quality_scorer import ProductImageQualityScorer
from data.image_generator import generate_all


def print_result(result: dict):
    """Pretty prints a single image scoring result."""
    if "error" in result:
        print(f"\n❌ Error scoring {result['image_path']}: {result['error']}")
        return

    score = result["final_score"]
    grade = result["grade"]

    # Score bar visualization
    bar_filled = int((score - 1) / 4 * 40)
    bar = "█" * bar_filled + "░" * (40 - bar_filled)

    # Grade emoji
    grade_emoji = {"Excellent": "🌟", "Good": "✅", "Acceptable": "⚠️",
                   "Poor": "❌", "Very Poor": "💀"}.get(grade, "")

    print("\n" + "─" * 62)
    print(f"  Image   : {os.path.basename(result['image_path'])}")
    print(f"  Score   : [{bar}] {score:.1f}/5.0")
    print(f"  Grade   : {grade_emoji} {grade}")
    print()
    print(f"  Score Breakdown:")
    print(f"    Deep Learning (EfficientNet) : {result['dl_score']:.3f}")
    print(f"    CV Signals (combined)        : {result['cv_score']:.3f}")
    print()
    print(f"  Signal Details:")
    signals = result["signals"]
    signal_labels = {
        "sharpness":             "Sharpness      ",
        "brightness":            "Brightness     ",
        "contrast":              "Contrast       ",
        "colorfulness":          "Colorfulness   ",
        "background_simplicity": "Background     ",
        "resolution":            "Resolution     ",
        "noise_level":           "Noise Level    ",
    }
    for key, label in signal_labels.items():
        val = signals.get(key, 0)
        signal_bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
        status = "✅" if val >= 0.6 else "⚠️" if val >= 0.3 else "❌"
        print(f"    {label} [{signal_bar}] {val:.2f} {status}")

    print()
    if result["issues"] != ["No major issues detected"]:
        print(f"  Issues:")
        for issue in result["issues"]:
            print(f"    ⚠️  {issue}")
        print(f"  Recommendations:")
        for rec in result["recommendations"]:
            print(f"    💡 {rec}")
    else:
        print(f"  ✅ Image meets quality standards")
    print("─" * 62)


def score_all_samples(scorer):
    """Scores all generated sample images."""
    sample_dir = "sample_images"

    # Generate samples if they don't exist
    if not os.path.exists(sample_dir) or not os.listdir(sample_dir):
        print("Generating sample images first...")
        generate_all(sample_dir)

    image_files = sorted([
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if f.endswith((".jpg", ".jpeg", ".png"))
    ])

    if not image_files:
        print("No images found in sample_images/")
        return

    print(f"\nScoring {len(image_files)} sample images...")
    results = scorer.score_batch(image_files)

    print("\n" + "=" * 62)
    print("  IMAGE QUALITY SCORING RESULTS")
    print("  (sorted best → worst)")
    print("=" * 62)

    for result in results:
        print_result(result)

    # Summary table
    print("\n" + "=" * 62)
    print("  SUMMARY TABLE")
    print("=" * 62)
    print(f"  {'Image':<40} {'Score':>6}  {'Grade'}")
    print(f"  {'─'*40} {'─'*6}  {'─'*12}")
    for r in results:
        if "error" not in r:
            name = os.path.basename(r["image_path"])
            print(f"  {name:<40} {r['final_score']:>5.1f}/5  {r['grade']}")


def score_custom_image(scorer, image_path):
    """Scores a user-provided image."""
    print(f"\nScoring: {image_path}")
    result = scorer.score_image(image_path)
    print_result(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to image to score")
    parser.add_argument("--model", type=str, default="models/saved/best_model.pt",
                        help="Path to trained model weights")
    args = parser.parse_args()

    print("=" * 62)
    print("  PRODUCT IMAGE QUALITY SCORER")
    print("  EfficientNet-B0 + CV Signal Analysis")
    print("=" * 62)

    # Load scorer
    model_path = args.model if os.path.exists(args.model) else None
    scorer = ProductImageQualityScorer(model_path=model_path)

    if args.image:
        score_custom_image(scorer, args.image)
    else:
        score_all_samples(scorer)

        # Interactive mode
        print("\n" + "=" * 62)
        print("  INTERACTIVE MODE")
        print("  Enter a path to score your own image")
        print("  (Press Ctrl+C to quit)")
        print("=" * 62)

        while True:
            try:
                path = input("\nImage path (or Enter to skip): ").strip()
                if not path:
                    continue
                if not os.path.exists(path):
                    print(f"File not found: {path}")
                    continue
                result = scorer.score_image(path)
                print_result(result)
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break


if __name__ == "__main__":
    main()
