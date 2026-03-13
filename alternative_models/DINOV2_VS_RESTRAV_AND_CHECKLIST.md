# DINOv2 (Yours) vs ReStraV — What Went Wrong & Pre-Training Checklist

## 1. How ReStraV Actually Does It (from their code)

Source: [ReStraV GitHub](https://github.com/ChristianInterno/ReStraV) — `dinov2_features.py`, `train.py`.

| Step | ReStraV implementation |
|------|-------------------------|
| **Backbone** | `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` — **ViT-S/14** (small), **always `.eval()` and `torch.no_grad()`**. No training of backbone. |
| **Input** | 24 frames, **224×224**, `frames.float()/255` (no ImageNet mean/std in their preprocess; DINOv2 hub may normalize internally). |
| **Per-frame embedding** | `forward_features()` → **CLS + all patch tokens** concatenated and flattened: `x_norm_clstoken` (1×384) + `x_norm_patchtokens` (196×384) → one vector per frame. So trajectory is in **75,648-D** space (ViT-S). |
| **Temporal signal** | **Fixed formula**, no learned temporal module: |
| | • Stepwise distance: `d_i = ‖z_{i+1} - z_i‖` for i=1..T-1 |
| | • Curvature: `θ_i = arccos(cos_sim(Δz_i, Δz_{i+1}))` in degrees |
| **21-D feature vector** | First 7 distances `d₁..d₇`, first 6 curvatures `θ₁..θ₆`, plus 8 stats: mean, min, max, var for both `d` and `θ`. Total **21-D** per video. |
| **Classifier** | MLP **21 → 64 → 32 → 1** (binary). Only ~3K trainable parameters. Trained with BCE; threshold τ optimized on training F1. |
| **Normalization** | Features (21-D) normalized with train set mean/std before classifier. |

So: **frozen backbone → fixed geometry (21-D) → tiny MLP**. No learned temporal aggregation, no backbone tuning.

---

## 2. Your DINOv2 Setup (what you ran)

| Step | Your implementation |
|------|----------------------|
| **Backbone** | **ViT-L/14** (large), 518×518. Default `--backbone-tune-mode ln`: **LayerNorm params are trainable** (GenD-style). Backbone LR is 0 for first 10% of steps, then ramped to full. So backbone **is** updated for most of training. |
| **Input** | 24 frames, **518×518**, ImageNet normalize. `EXPECTED_SHAPE = (24, 518, 518, 3)` in training. |
| **Per-frame embedding** | **CLS token only** (1024-D) from backbone. |
| **Temporal signal** | **Learned**: `TemporalTransformer` (2 layers, 512 hidden, 8 heads) on [B, T, 1024] → [B, 512]. So the model **learns** how to combine frames; this can (and does) learn dataset/generator-specific patterns. |
| **Classifier** | 512 → 256 → 2 with dropout, then `LogitCalibrator`. Many more parameters than ReStraV’s head. |
| **Trainable** | Backbone LN + full temporal transformer + classifier. Millions of trainable parameters. |

So: **tuned backbone + learned temporal aggregation on high-dim features** → much higher capacity and strong risk of memorizing datasets/generators.

---

## 3. Side-by-Side Comparison

| Aspect | ReStraV | Your DINOv2 |
|--------|---------|-------------|
| **Backbone** | ViT-S/14, **frozen** | ViT-L/14, **LN tuned** (default) |
| **Resolution** | 224×224 | 518×518 |
| **Per-frame rep** | CLS + patch tokens (75,648-D) | CLS only (1024-D) |
| **Temporal** | **Fixed** curvature + distance → 21-D | **Learned** transformer 1024→512 |
| **Classifier input** | 21-D (geometry) | 512-D (learned) |
| **Trainable params** | ~3K (MLP only) | Millions (backbone LN + temporal + head) |
| **Why it generalizes** | Signal is intrinsic (trajectory geometry); no capacity to memorize generators | High capacity; can memorize generator-specific and dataset-specific patterns |

---

## 4. What Went Wrong (Why Low Score + Memorization)

1. **Backbone was not frozen**  
   With `--backbone-tune-mode ln` (default), LayerNorm in the backbone is trained. That lets the representation adapt to your training generators and hurts cross-generator generalization. ReStraV does **not** train the backbone at all.

2. **Learned temporal aggregation**  
   Your `TemporalTransformer` learns how to combine frame features. It has enough capacity to fit generator- or dataset-specific temporal patterns (e.g. “this flicker = dataset A”). ReStraV uses a **fixed** geometric recipe (curvature + distance), so the temporal signal is “natural vs synthetic” trajectory shape, not learned shortcuts.

3. **No bottleneck on the temporal signal**  
   You feed 24×1024-D into a transformer then 512-D into the classifier. ReStraV compresses the whole video into **21-D** by a fixed formula before any learning. That 21-D bottleneck makes it almost impossible to memorize high-dimensional generator-specific structure.

4. **Higher resolution + larger backbone**  
   518×518 and ViT-L give more capacity. When the backbone is tuned (even only LN), that extra capacity can overfit to training data and hurt generalization.

5. **Possible train/val overlap**  
   If your val/gasbench sets share generators or datasets with the training set, val can look good while gasbench (unseen generators) stays low. ReStraV is designed for **cross-generator** settings; your setup is more prone to train-generator overfitting.

---

## 5. Pre-Training Checklist — Confirm Before Training Again

Use this to avoid repeating the same mistakes.

### 5.1 Strategy choice

- [ ] **A. ReStraV-style (recommended for generalization)**  
  - Frozen DINOv2 (ViT-S or ViT-L), 224×224 (or 518→224 at inference if you keep 518 in data).  
  - Compute trajectory geometry (stepwise distance + curvature) → 21-D.  
  - Train **only** a small classifier (e.g. 21→64→32→2) on 21-D.  
  - No temporal transformer, no backbone tuning.

- [ ] **B. Keep current architecture but reduce memorization**  
  - Use **`--backbone-tune-mode frozen`** (full backbone frozen).  
  - Keep TemporalTransformer + classifier.  
  - Accept that learned temporal aggregation can still overfit; consider stronger augmentation and/or less head capacity.

### 5.2 If you stay with current DINOv2 training script

- [ ] **Backbone**: Confirm you use **`--backbone-tune-mode frozen`** if you want “linear probe” style (no backbone updates). For ReStraV-style you’d need a different pipeline (geometry features + small MLP), not just this flag.
- [ ] **Resolution**: If you move toward ReStraV, data should be **224×224** for the backbone that computes trajectory (or resize 518→224 before backbone in a new model). Your current `EXPECTED_SHAPE` and `model_config.yaml` use 518; that’s fine for current model, wrong for ReStraV-style.
- [ ] **Data**: Confirm train/val/gasbench **don’t leak** (no same video/generator in train and eval). Check that val has **unseen** generators/datasets if you care about generalization.
- [ ] **Evaluation**: Track **per-dataset** (or per-generator) accuracy on val; if some datasets are near 100% and others near 50%, that suggests memorization of easy datasets.

### 5.3 If you implement ReStraV-style in this repo

- [ ] Backbone: DINOv2 ViT-S/14 or ViT-L/14, **frozen**, 224×224 input.
- [ ] Embedding: Use **final-block** CLS + patch tokens (or CLS only if you match paper’s ablations) and compute trajectory.
- [ ] Features: Implement stepwise distance `d_i` and curvature `θ_i`, then 21-D vector (7 d + 6 θ + 8 stats).
- [ ] Classifier: Only MLP on 21-D → 2 logits (for gasbench). Optional: learnable temperature/bias for Brier.
- [ ] Data pipeline: Either precompute 21-D features to disk and train the MLP only, or implement a small wrapper that runs backbone + geometry + MLP in one `forward()` for gasbench.
- [ ] Normalization: Apply the same normalization (e.g. train mean/std on 21-D) at inference as in training.

---

## 6. Summary

- **ReStraV**: Frozen DINOv2, **224×224**, **fixed** trajectory geometry (21-D), **tiny** MLP. No backbone or temporal learning → generalization comes from the geometry signal.
- **Your run**: Tuned backbone (LN) + **learned** temporal transformer + large head on 518×518 → high capacity and memorization, low cross-generator score.

Before training again: pick either a **ReStraV-style** pipeline (frozen backbone + 21-D geometry + small MLP) or at least **freeze the backbone** and confirm data/eval and checklist above.
