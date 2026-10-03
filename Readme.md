# Product Image Quality Scoring

An explainable image-quality demonstration combining classical computer-vision signals and an EfficientNet-B0 regression head. It scores product photos on a heuristic 1–5 scale and identifies measurable issues.

## Method

The classical extractor measures sharpness, brightness, contrast, colorfulness, background simplicity, resolution, and noise. When a trained checkpoint is available, the combined normalized score uses **35% neural output and 65% classical score**. Without a checkpoint, the final score falls back to the classical score. The model is still initialized, so pretrained ImageNet weights may be downloaded.

Grades and recommendations are threshold-based heuristics. They are not guarantees of marketplace acceptance or calibrated human judgments.

## Synthetic demonstration data

[data/image_generator.py](data/image_generator.py) creates **seven synthetic images** illustrating blur, darkness, overexposure, clutter and resolution differences. Labels in `train.py` are assigned demonstration scores, not collected human ratings.

The trainer uses five images for training and two for repeated validation/checkpoint selection (seed 42). Although named `test` in the code, those two images are validation data. The demo scores all seven examples, including training images. This tiny synthetic workflow is not a real-world generalization benchmark.

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
