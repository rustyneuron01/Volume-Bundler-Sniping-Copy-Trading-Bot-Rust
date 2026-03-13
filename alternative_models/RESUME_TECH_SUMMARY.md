# Multi-Modal Fake/Real Detection — Video, Image & Audio (Resume Summary)

---

## Project purpose

This project builds a **multi-modal fake vs real detector** for **video, image, and audio**. The goal is to tell whether a given sample was created by a human (real) or by an AI or synthetic tool (fake), for content authentication and media safety. Each modality is handled by a dedicated model: video with a video-specific detector, image with an image model, and audio with an audio model.

---

## What I did — models per modality

- **Video detection:** I used the **GenD**-style model (DINOv2 backbone with LayerNorm-only tuning for generalization) and also implemented a **ReStraV-style** detector (frozen backbone + trajectory geometry + small MLP) as an alternative. Both take only video as input (no text or other modalities).
- **Image detection:** I used the **DINOv3** model for image-level real vs fake classification.
- **Audio detection:** I used the **BreathNet** model for audio-level real vs fake classification.

---

## Data preprocessing

- **Video:** Training and evaluation use preprocessed videos stored as **NumPy arrays** (`.npy`), with optional compression (`.npy.zst`). Each sample is a fixed-shape clip (e.g. 24 frames × height × width × 3 channels). CSVs list `npy_path`, `label` (real/fake), and `dataset` (source). Data is read from **mounted cloud storage** (e.g. Backblaze B2 via rclone); for compressed files, a **decoded cache** on local SSD is used so decoding does not block training. Frames are normalized (e.g. 0–255 or 0–1) and resized per model (e.g. 224×224 or 518×518); ImageNet normalization is applied inside the model.
- **Image / Audio:** Preprocessing follows each model’s requirements (e.g. resize and normalization for image; feature extraction or waveform handling for BreathNet). Data is organized so the evaluation pipeline can load and batch samples consistently across modalities.

---

## ReStraV training style — why I used it and its advantages

For video, I adopted a **ReStraV-style** training pipeline: a **frozen** pretrained vision backbone (e.g. DINOv2 ViT-S), **no** fine-tuning of the backbone. Instead of learning a heavy temporal module, the model uses a **fixed geometric signal**: for each video, we compute per-frame embeddings, then **stepwise distances** and **curvature** along the trajectory in representation space. These are summarized into a small **21-D feature vector** per video. Only a **lightweight MLP** (e.g. 21 → 64 → 32 → 2) is trained on top of this vector; the rest of the network stays frozen.

**Advantages of this style:**  
(1) **Generalization** — The backbone keeps its “natural world” prior, so the detector relies on **intrinsic** temporal geometry (real videos tend to have straighter trajectories in representation space, AI videos more curved) instead of generator-specific artifacts.  
(2) **Less memorization** — Very few trainable parameters (only the MLP), so the model has limited capacity to overfit to particular datasets or generators.  
(3) **Stable and interpretable** — The signal is based on trajectory geometry rather than a black-box temporal network, which aligns with research (e.g. ReStraV, NeurIPS 2025) on video fake detection.

**Why I selected it:**  
Initial video models that fine-tuned the backbone or used a large temporal head achieved good train/val accuracy but **poor cross-dataset and cross-generator performance** (memorization). I chose the ReStraV-style approach so that the detector would **generalize** to unseen datasets and generators while still being trainable on our data (only the small classifier is trained).

---

## How inference works and what the output is

The evaluation pipeline loads each modality’s model via a standard interface: **`load_model(weights_path, **kwargs)`** returns a PyTorch model, and **`forward(x)`** runs inference. For video, the input `x` is a batch of videos in shape **[batch, num_frames, channels, height, width]** as float in the 0–255 range; the model handles normalization and any resizing internally. The pipeline calls `forward` on batched samples and collects the raw **logits** (e.g. **[batch, 2]** for real vs fake). It then applies **softmax** to get probabilities and uses these for **Brier score** and other metrics (e.g. a combined score using MCC and Brier). So the **output** of inference is **logits** from the model; the pipeline turns them into **probabilities** and aggregates into evaluation scores for reporting and comparison.

---

## Main challenge: keeping generalization and avoiding memorization — how I solved it

The biggest challenge was **training models that generalize** to new datasets and generators instead of **memorizing** training sources. When the backbone or a large temporal head was trained end-to-end, validation on held-out datasets showed **overfitting**: good train/val accuracy but poor performance on unseen generators and low scores on the evaluation metrics.

**How I addressed it:**

1. **Training strategy**  
   - **Frozen or minimal backbone:** For video, I used either a fully frozen backbone (ReStraV-style) or only **LayerNorm-only tuning** (GenD-style) so the representation stays close to the pretrained “natural world” prior and does not adapt to generator-specific cues.  
   - **Intrinsic signal:** I relied on **trajectory geometry** (ReStraV-style) or light temporal modeling on frozen features so the discriminative signal is **intrinsic** to real vs synthetic content rather than generator-specific.  
   - **Small head:** Only a small classifier (e.g. MLP on 21-D features) is trained, limiting capacity to memorize.

2. **Calibration and loss**  
   - **Brier-aware training** (e.g. Focal CE + Brier loss) and **post-training temperature sweep** on a calibration set so predicted probabilities are well calibrated and evaluation scores that use Brier are improved.  
   - **Learnable temperature and bias** (e.g. LogitCalibrator) in the model for better probability outputs.

3. **Data and evaluation**  
   - **Dataset-balanced sampling** and, where applicable, **focus on worst-performing datasets** so the model is trained across many sources and does not overfit to a few.  
   - **Strict train/val split** and evaluation on **unseen datasets and generators** to measure generalization, not just in-distribution accuracy.

4. **ReStraV-style pipeline**  
   - Using a **frozen backbone + fixed 21-D geometry + MLP** for video gave a clear path to generalization: the model cannot memorize high-dimensional generator-specific patterns because the temporal signal is fixed and the only learned part is a small classifier on 21-D inputs.

---

## Technologies used

- **Frameworks:** PyTorch, timm (for DINOv2/DINOv3 backbones), safetensors (model weights).
- **Training:** Single- and multi-GPU (DDP), mixed precision (bf16), gradient checkpointing where needed; cosine and cosine-restart learning rate schedules; EMA, optional SWA; dataset-balanced and weighted sampling.
- **Data:** NumPy (`.npy`), zstd (`.npy.zst`), Pandas (CSVs); **rclone** for mounting cloud storage (e.g. Backblaze B2); DataLoader with multiple workers, prefetch, and persistent workers so **data loading and preprocessing run in parallel with GPU training**.
- **Evaluation and inference:** Custom model loader (`load_model` + `forward`), Brier score, MCC; calibration (temperature sweep, learnable temperature/bias).
- **Reproducibility:** Fixed seeds, stratified splits, saving normalization/calibration parameters with the model (e.g. 21-D mean/std for ReStraV-style).

Use or shorten this for your resume or project description as needed.
