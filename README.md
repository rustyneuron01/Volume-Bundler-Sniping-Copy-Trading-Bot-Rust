# AI-Generated Content Detection

Multi-modal detection of real vs. AI-generated content across **image**, **video**, and **audio**. This repository contains benchmark tooling, model training code, and evaluation pipelines for binary classifiers that distinguish genuine media from synthetic or AI-manipulated content.

---

## Purpose

The project trains and evaluates detection models that answer: *Is this image, video, or audio real (human-captured) or AI-generated?* Use cases include content authentication, media safety, and research on generalization across diverse generators and datasets.

- **Three modalities**: Separate models for image, video, and audio.
- **Unified evaluation**: A single benchmark suite (GASBench) runs all models on the same datasets and metrics.
- **Focus on generalization**: Training and evaluation are designed to perform well on both public datasets and unseen (holdout) data, avoiding dataset memorization.

---

## How Evaluation Works

### Data split

- **Public datasets**: Large set of curated datasets (real, synthetic, semisynthetic) from multiple sources (Hugging Face, parquet/zip/tar, multiple generators). Used for training and for public benchmark reporting.
- **Holdout datasets**: Additional datasets not visible in the public config. Evaluators use them to measure generalization and prevent overfitting to the public set. Holdout names are obfuscated in the benchmark output.

### Metrics

Models are scored with a combined metric that rewards both **discrimination** (correct real vs. fake) and **calibration** (reliable probabilities):

- **MCC** (Matthews Correlation Coefficient): Discrimination in [-1, 1], normalized to [0, 1].
- **Brier score**: Calibration (mean squared error between predicted probabilities and labels); lower is better, random baseline ≈ 0.25.
- **Combined score** = geometric mean of normalized MCC and normalized Brier, with exponents (e.g. α=1.2, β=1.8) so calibration is weighted strongly. This penalizes overconfident wrong predictions and rewards well-calibrated probabilities.

Additional reported metrics: accuracy, cross-entropy, per-dataset breakdowns, inference time (average and p95).

### Inference and constraints

- Evaluation runs on fixed hardware; models must complete inference within time limits per sample.
- Inputs: images/videos/audio in standard shapes (e.g. 224×224 or 518×518 for vision); pipeline supplies 0–255 float tensors; models apply normalization internally.
- Output: logits `[batch, 2]` (real, fake). The pipeline applies softmax for probabilities and computes all metrics.

---

## Tech Stack

### Models

| Modality | Model / approach | Notes |
|----------|------------------|--------|
| **Image** | DINOv3 (ViT backbone) | Patch-level or CLS features + classification head; optional LoRA for better generalization. |
| **Video** | DINOv2 + temporal head (GenD-style) or ReStraV-style | **GenD-style**: DINOv2 ViT-L, LayerNorm-only tuning, temporal transformer + head. **ReStraV-style**: Frozen DINOv2, trajectory geometry (stepwise distances, curvature) → 21-D vector, small MLP only trained. Both output binary logits. |
| **Audio** | BreathNet | Audio-specific model for real vs. synthetic speech/audio. |

All models are exported as **safetensors** with a small wrapper (`model_config.yaml`, `model.py`, `*.safetensors`) so the same benchmark loader runs image, video, and audio.

### Data preprocessing

- **Image**: Decode to RGB, resize (shortest edge or fixed size), optional center crop; DeeperForensics-style augmentations (JPEG compression, blur, noise, color) at multiple levels during evaluation; no normalization in pipeline (model does /255 and ImageNet mean/std in `forward`).
- **Video**: Frame extraction (Decord), fixed number of frames and FPS; same resize/crop and augmentation strategy as images; stored as NumPy (e.g. `.npy`) or compressed (`.npy.zst`) with optional local decode cache for fast training.
- **Audio**: Resampling, fixed-duration windows, optional preprocessed waveforms (e.g. `.pt`); pipeline supports raw bytes or precomputed tensors.

Datasets are defined in YAML (path, modality, media_type, format, sampling). The benchmark downloads and caches data, and supports dataset-balanced and weighted sampling for training.

### Core challenges addressed

1. **Generalization vs. memorization**  
   Head-only or heavy fine-tuning led to strong train/val accuracy but poor performance on unseen holdout datasets (especially unseen real domains). Mitigations: frozen or lightly tuned backbones (LayerNorm-only or LoRA), geometry-based temporal features (ReStraV), small classifiers, and training across many datasets and generators.

2. **Calibration**  
   The combined score heavily weights Brier, so overconfident wrong predictions are costly. We use Brier-aware loss (e.g. Focal CE + Brier), learnable temperature/bias (e.g. LogitCalibrator), and optional temperature sweep on a calibration set.

3. **Balance on public benchmarks**  
   Good performance on both public and (simulated or actual) holdout data is required. We use dataset-balanced sampling, focus on worst-performing datasets, and strict train/val and cross-dataset evaluation to avoid overfitting to a few sources.

---

## Repository layout

| Path | Description |
|------|-------------|
| **`external/gasbench`** | Benchmark package: dataset loading, preprocessing, metrics, CLI (`gasbench run`, `gasbench download`, etc.). |
| **`alternative_models/`** | Training and export code: **dinov2** (video, GenD-style), **restrav** (video, ReStraV-style), and references for image (DINOv3) and audio (BreathNet). |
| **`docs/`** | Additional documentation (model spec, installation, dataset analysis). |

---

## Quick start

### Install

```bash
git clone <this-repo>
cd detection-tool
./install.sh
```

Optional: `./install.sh --no-system-deps` to skip system dependencies if you only need the benchmark and Python env.

### Run benchmarks

Activate the environment and run the benchmark with a model directory (must contain `model_config.yaml`, `model.py`, and `*.safetensors`):

```bash
source .venv/bin/activate

# Image
gasbench run --image-model ./path/to/image_model/ --debug

# Video
gasbench run --video-model ./path/to/video_model/ --debug

# Audio
gasbench run --audio-model ./path/to/audio_model/ --debug

# Full run (all datasets)
gasbench run --image-model ./path/to/image_model/ --full
```

Results are written to the configured results directory (e.g. JSON, parquet, summary).

### Train (example: ReStraV-style video)

From the ReStraV folder, using CSVs and a data root (e.g. mounted cloud storage):

```bash
cd alternative_models/restrav
python train_restrav.py \
  --train-csv ../dinov2/train.csv \
  --val-csv ../dinov2/val_eval.csv \
  --data-root /mnt/your_data \
  --output-dir checkpoints/restrav \
  --epochs 10 --batch-size 32
```

See `alternative_models/restrav/README.md` and `alternative_models/dinov2/README.md` for full training and export instructions.

---

## Model format (submission)

Each modality’s model is a directory (or zip of that directory) with:

- **`model_config.yaml`** — Preprocessing (e.g. `resize: [224, 224]`), `num_classes: 2`, optional `weights_file`.
- **`model.py`** — Defines `load_model(weights_path, num_classes=2)` returning a `torch.nn.Module`; `forward(x)` returns logits `[B, 2]`.
- **`*.safetensors`** — Weights loaded by `load_model`.

Inputs from the pipeline are float32 in 0–255; the model must normalize inside `forward`. See `external/gasbench/docs/Safetensors.md` for the full specification and allowed imports.

---

## References

- **ReStraV-style video**: Frozen backbone + trajectory geometry (stepwise distances, curvature) + MLP — see `alternative_models/restrav/`.
- **GenD-style video**: LayerNorm-only tuning of DINOv2 + temporal transformer — see `alternative_models/dinov2/`.
- **Evaluation and datasets**: `external/gasbench/`, `external/gasbench/docs/IMAGE_DATASETS_V11_V12_ANALYSIS.md`.
- **Multi-modal summary**: `alternative_models/RESUME_TECH_SUMMARY.md`.
