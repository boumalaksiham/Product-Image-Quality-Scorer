# Product Image Quality Scoring

An explainable image-quality demonstration combining classical computer-vision signals and an EfficientNet-B0 regression head. It scores product photos on a heuristic 1–5 scale and identifies measurable issues.

## What the score is intended to explain

A photo may be sharp but underexposed, or bright but blurred. The demo reports individual signals alongside a combined score so a reviewer can inspect which measurable issue affected the result.

**Design choice:** combine fixed image measurements with a neural regression head. The classical branch remains available when a trained checkpoint is absent; the two scoring modes should therefore be evaluated separately. Neither mode determines whether the pictured product is correct or whether a marketplace will accept the listing.

**Start here:** [models/quality_scorer.py](models/quality_scorer.py) contains the measurements and combination; [data/image_generator.py](data/image_generator.py) makes the controlled examples; [demo.py](demo.py) exposes the signal breakdown.

## Method

The classical extractor measures sharpness, brightness, contrast, colorfulness, background simplicity, resolution, and noise. When a trained checkpoint is available, the combined normalized score uses **35% neural output and 65% classical score**. Without a checkpoint, the final score falls back to the classical score. The model is still initialized, so pretrained ImageNet weights may be downloaded.

Grades and recommendations are threshold-based heuristics. They are not guarantees of marketplace acceptance or calibrated human judgments.

## Synthetic demonstration data

[data/image_generator.py](data/image_generator.py) creates **seven synthetic images** illustrating blur, darkness, overexposure, clutter and resolution differences. Labels in `train.py` are assigned demonstration scores, not collected human ratings.

The trainer uses five images for training and two for repeated validation/checkpoint selection (seed 42). These two images are explicitly named validation data in the trainer. Python, NumPy, and PyTorch are seeded. The selected checkpoint is reloaded and its predictions, targets, image paths, and MAE are saved to `models/saved/validation_report.json`. The demo scores all seven examples, including training images. This tiny synthetic workflow is not a real-world generalization benchmark.

## Setup

Use Python 3.10 or 3.11 for the repository's pinned PyTorch 2.2.2 environment. Run commands from the repository root. The first model load downloads pretrained weights; an internet connection is needed unless the model cache is populated. A GPU is not required by the current scripts. Runtime depends on hardware and is not benchmarked here.

```bash
git clone https://github.com/boumalaksiham/Product-Image-Quality-Scorer.git
cd Product-Image-Quality-Scorer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. If installation reports an unsupported wheel, check Python version and architecture before changing the pinned environment.

## Run

```bash
python train.py
python demo.py
python demo.py --image /absolute/path/to/product_photo.jpg
```

Training runs for 10 epochs and writes `models/saved/best_model.pt` and `models/saved/config.json`. Generated examples are stored in `sample_images/`. The demo defaults to that checkpoint path and also accepts `--model` for a compatible checkpoint. If the file is absent, scoring falls back to the classical signals; this is not equivalent to the trained combined scorer.

## Repository map

| File | Purpose |
|---|---|
| [data/image_generator.py](data/image_generator.py) | Synthetic examples |
| [models/quality_scorer.py](models/quality_scorer.py) | Signals, EfficientNet head, combination and recommendations |
| [train.py](train.py) | Synthetic labels, training and checkpoint selection |
| [demo.py](demo.py) | Example/custom-image scoring and signal breakdown |

## Interpretation and limitations

Brightness, sharpness and background thresholds encode preferences that may not suit every product category. Complex but useful backgrounds can be penalized. Synthetic labels and the two-image validation set are too small to establish ranking quality. The demo's correct ordering should not be presented as held-out accuracy.

Evaluate a larger independent human-rated dataset using regression error and rank correlation, report scorer variants separately, analyze category-specific failures, and calibrate thresholds before relying on the scores for decisions.

The modified training script passes Python syntax compilation. Training has not been rerun; existing artifacts predate this repair. Rerun training to generate the new selected-checkpoint validation report.
