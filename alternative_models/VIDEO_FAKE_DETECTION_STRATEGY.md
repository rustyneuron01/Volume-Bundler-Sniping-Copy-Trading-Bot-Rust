# Video-Only Fake Detection: Research-Backed Strategy

**Constraint:** Input is **video only** (no text, no prompts). Goal: **correct** real/fake detection with **good generalization** to unseen datasets/generators.

---

## 1. Why Generalization Fails (Evidence)

Recent work (2024–2026) agrees on the causes:

- **Shortcut learning:** Models learn generator-specific cues (watermarks, texture, style) instead of universal “fake” cues.
- **Artifact memorization:** Full fine-tuning on your data overwrites the backbone’s “natural world” prior with training-generator artifacts, so performance collapses on new generators.
- **Wrong signal:** Detectors that rely on “high-level semantic flaws of a specific model” do not transfer. What transfers is **“intrinsic low-level artifacts introduced by the generative architecture”** (frequency, temporal geometry, representation-space dynamics), which are shared across many generators.

So: **any strategy that keeps the backbone frozen and uses an intrinsic, generator-agnostic signal will generalize better.**

---

## 2. What Actually Works (Video-Only, No Text)

| Approach | Principle | Why it generalizes |
|----------|-----------|--------------------|
| **ReStraV** (NeurIPS 2025) | **Temporal geometry in representation space.** Real videos → “straighter” trajectories in DINOv2 space; AI videos → more curved/jittery (they violate natural-world statistics). | Backbone **frozen**. Signal is curvature/stepwise distance of the trajectory, not generator-specific. No end-to-end tuning of the encoder. |
| **D3** (ICCV 2025) | **Second-order temporal features** (e.g. acceleration in feature space). Real = more volatile; AI = flatter. | **Training-free** or minimal head. Same idea: fixed backbone, intrinsic temporal statistic. |
| **Forensic-oriented augmentation** (“Seeing What Matters”) | Train on **low-level forensic cues** (e.g. wavelet/frequency), with augmentation that preserves those cues but varies style/content. | Classifier is forced to use generator-agnostic, low-level cues instead of memorizing one generator. |
| **Per-frame forgery modules** (e.g. ForgeLens on frames) | Frozen CLIP + small “forgery-aware” modules per frame; no text. | Frozen backbone; only lightweight modules trained; generalizes across 19 generators in image setting; applied to video by aggregating frame-level scores. |

Common factor: **frozen (or lightly adapted) backbone + a signal that is intrinsic to “natural vs synthetic” (temporal geometry, low-level forensic, or forgery-aware features), not to a specific generator.**

---

## 3. Recommended Strategy You Can Follow

**Core idea:** Do **not** full fine-tune the backbone. Use a **frozen** pretrained vision encoder and build the detector from an **intrinsic** signal (temporal geometry in representation space). Train only a **small classifier** on top of that signal so the model cannot memorize generator-specific artifacts.

### Step 1: Backbone (frozen)

- Use **DINOv2** (e.g. ViT-S/14 or ViT-L/14). ReStraV’s analysis shows DINOv2 gives the **largest curvature gap** between real and AI videos in representation space; CLIP and others are worse for this.
- **Do not update backbone weights.** Freeze it. This keeps “natural world” statistics intact so that “straight” vs “curved” trajectories remain meaningful.

### Step 2: Input (video only)

- Sample **N frames** per video (e.g. 24 over ~2 s, or match your current pipeline).
- Resize to **224×224** (DINOv2 standard); normalize with ImageNet mean/std.
- No text, no prompts.

### Step 3: Intrinsic signal — trajectory geometry (ReStraV-style)

- For each video, get **per-frame embeddings** (e.g. CLS token, or CLS + patch tokens concatenated) from the **last block** of DINOv2.
- Build the **trajectory** \( (z_1, z_2, \ldots, z_T) \) in representation space.
- Compute:
  - **Stepwise distance:** \( d_i = \|z_{i+1} - z_i\|_2 \) for \( i = 1, \ldots, T-1 \).
  - **Curvature:** angle \( \theta_i \) between consecutive displacement vectors \( \Delta z_i \) and \( \Delta z_{i+1} \) (Eq. 1 in ReStraV).
- Aggregate into a **small feature vector** per video, e.g.:
  - First 7 distances \( [d_1, \ldots, d_7] \), first 6 curvatures \( [\theta_1, \ldots, \theta_6] \),
  - Plus mean, min, max, variance for both \( \{d_i\} \) and \( \{\theta_i\} \) → **21-D** (as in ReStraV).

### Step 4: Classifier (only trainable part)

- Train **only** a small classifier on the 21-D (or similar) feature vector:
  - Options: logistic regression, 2-layer MLP (e.g. 64→32→2), or random forest + calibrator.
- Output **logits [B, 2]** (real, fake) for gasbench; use softmax for probabilities so Brier/calibration are well-defined.
- You can add a **learnable temperature/bias** (e.g. LogitCalibrator) on top of the classifier logits to improve calibration on your val set.

### Step 5: Why this generalizes

1. **Frozen backbone** → no generator-specific tuning; the representation space stays “natural-world aligned.”
2. **Curvature/distance** is a property of **temporal evolution in that space**: natural dynamics vs synthetic dynamics, not Sora vs Pika.
3. **Very few trainable parameters** (classifier on 21-D) → limited capacity to overfit to specific generators or datasets.
4. **Video-only, no text** throughout.

---

## 4. Alternative (Same Philosophy, Different Signal)

- **D3-style:** Same frozen backbone; use **second-order** (acceleration) statistics of the same trajectory; real = more volatile, AI = flatter. Then either a fixed threshold or a small calibrator for Brier.
- **Forensic augmentation:** If you prefer to keep training a CNN/ViT head, use **forensic-oriented augmentation** (e.g. wavelet-based) so the model is forced to rely on low-level, generator-agnostic cues instead of high-level style.

---

## 5. What to Avoid

- **Full fine-tuning** of the backbone on your 313K samples without the above principles → high risk of shortcut learning and poor cross-generator performance.
- **Relying on a single “top” model name** without the strategy (frozen backbone + intrinsic signal + minimal head) → same generalization issues.
- **Text or multi-modal input** → out of scope for your video-only setting.

---

## 6. Summary: The Strategy in One Sentence

**Use a frozen DINOv2 backbone, compute per-video trajectory geometry (curvature + stepwise distance) in its representation space, and train only a small classifier (and optional calibrator) on that 21-D feature vector so the detector relies on an intrinsic “natural vs synthetic” temporal signal instead of generator-specific artifacts.**

This is the strategy that current research supports for **video-only**, **correct** real/fake detection with **good generalization**.

---

**See also:** `DINOV2_VS_RESTRAV_AND_CHECKLIST.md` for a direct comparison of your DINOv2 run vs ReStraV (why you got low score and memorization) and a **pre-training checklist** to confirm before training again.
