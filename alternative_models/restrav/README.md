# ReStraV-style video detector

Frozen DINOv2 + trajectory geometry (21-D) + MLP. Only the small classifier is trained on your data.

- **Input**: Video only, `[B, T, C, H, W]` float 0–255 (e.g. 24×518×518×3; model resizes to 224×224).
- **Output**: Logits `[B, 2]` (real, fake). gasbench applies softmax for Brier.

**Parallelism (like dinov2):** Load, decode (e.g. .npy.zst), and augmentations run on CPU in DataLoader workers; prefetch overlaps with GPU training so the GPU is not waiting. Use `--workers 0` for auto (4–16 workers) and `--prefetch-factor 4`.

## Train (single GPU)

Use the same CSVs and data root as dinov2. Videos can be 518×518; the model resizes to 224 internally.

### Using mount (rclone / B2)

From the `restrav` folder, run the setup script, then the training command it prints:

```bash
cd alternative_models/restrav
./gpu_setup.sh
```

Then run the command printed at the end (it uses `--data-root` at the mount point). Example:

```bash
python train_restrav.py \
  --train-csv ../dinov2/train.csv \
  --val-csv ../dinov2/val_eval.csv \
  --data-root /mnt/b2data \
  --output-dir checkpoints/restrav \
  --epochs 10 --batch-size 32 --workers 0 --prefetch-factor 4
```

Override mount: `MOUNT_POINT=/mnt/mydata ./gpu_setup.sh`. Override CSVs: `TRAIN_CSV=/path/train.csv VAL_CSV=/path/val.csv ./gpu_setup.sh`.

### With compressed `.npy.zst`

Use a **local** decoded cache (SSD) so decode doesn’t hit the mount:

```bash
python train_restrav.py \
  --train-csv ../dinov2/train.csv \
  --val-csv ../dinov2/val_eval.csv \
  --data-root /mnt/b2data \
  --compressed --decoded-root /path/to/local/ssd/_decoded_restrav \
  --output-dir checkpoints/restrav
```

Omit `--decoded-root` to use `<data-root>/_decoded_step_cache` (writes to mount; slower).

## Export for gasbench

```bash
python export_safetensors.py \
  --input checkpoints/restrav/best.pt \
  --output checkpoints/restrav/model.safetensors
```

Point gasbench at the `restrav` folder; `model_config.yaml` and `load_model` in `model.py` are set up for it.

## What gets trained

- **Frozen**: DINOv2 backbone (ViT-S/14 by default).
- **Calibrated once**: 21-D feature mean/std from a sample of training videos.
- **Trained**: MLP classifier (21 → 64 → 32 → 2) only.

## Rough training time

- **Calibration**: ~20k samples at ~same throughput as training.
- **Training**: Bottleneck is frozen backbone forward (ViT-S 224, 32×24 images/batch). On a V100/A100, ~0.2–0.5 s/batch is typical.
- **Example**: 300k samples, batch 32, 10 epochs → ~94k batches → ~9–13 hours. Script prints an estimate at start and elapsed time at end.
