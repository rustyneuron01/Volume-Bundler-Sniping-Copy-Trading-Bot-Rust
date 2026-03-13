#!/bin/bash
#
# GPU setup for ReStraV training: mount B2 (or your data) and run training.
# Data is read through the mount; first epoch streams, then VFS cache speeds up.
#
# Usage:
#   1. Set MOUNT_POINT if needed (default: /mnt/b2data).
#   2. Run: ./gpu_setup.sh
#   3. After mount is ready, run the printed training command from the restrav folder.
#
# For .npy.zst data: use --compressed and --decoded-root (local SSD recommended).
# For plain .npy on mount: use only --data-root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Mount point (override with MOUNT_POINT=/path ./gpu_setup.sh)
MOUNT_POINT="${MOUNT_POINT:-/mnt/b2data}"
B2_REMOTE="${B2_REMOTE:-b3:sn34-bucket}"
VFS_CACHE_DIR="${VFS_CACHE_DIR:-$HOME/bittensor/vfs_cache}"
CACHE_SIZE="${CACHE_SIZE:-4700G}"
VFS_CACHE_MAX_AGE="${VFS_CACHE_MAX_AGE:-720h}"

# CSV paths (relative to restrav folder; dinov2 CSVs live in ../dinov2)
TRAIN_CSV="${TRAIN_CSV:-../dinov2/train.csv}"
VAL_CSV="${VAL_CSV:-../dinov2/val_eval.csv}"

if ! mkdir -p "$MOUNT_POINT" 2>/dev/null; then
    fallback_mount="${HOME}/b2data"
    echo "WARN: Cannot create mount point: $MOUNT_POINT, using $fallback_mount"
    MOUNT_POINT="$fallback_mount"
fi

# ---------- Step 1: Install rclone if needed ----------
if ! command -v rclone &>/dev/null; then
    echo "Installing rclone..."
    curl -sSf https://rclone.org/install.sh | bash
fi

# ---------- Step 2: Mount ----------
mkdir -p "$MOUNT_POINT" "$VFS_CACHE_DIR"
fusermount -u "$MOUNT_POINT" 2>/dev/null || true

echo "Mounting $B2_REMOTE → $MOUNT_POINT"
rclone mount "$B2_REMOTE" "$MOUNT_POINT" \
    --vfs-cache-mode full \
    --vfs-cache-max-age "$VFS_CACHE_MAX_AGE" \
    --vfs-cache-max-size "$CACHE_SIZE" \
    --cache-dir "$VFS_CACHE_DIR" \
    --transfers 64 \
    --checkers 32 \
    --buffer-size 256M \
    --vfs-read-ahead 512M \
    --vfs-cache-poll-interval 1m \
    --dir-cache-time 1h \
    --attr-timeout 1h \
    --daemon

echo "Waiting for mount to become ready..."
ready=0
for _ in $(seq 1 30); do
    if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
        if timeout 5 ls -d "$MOUNT_POINT"/prepared-data-* &>/dev/null; then
            dataset_count=$(ls -d "$MOUNT_POINT"/prepared-data-* 2>/dev/null | wc -l)
            echo "Mount OK: $dataset_count datasets visible"
            ready=1
            break
        fi
    fi
    sleep 2
done

if [[ "$ready" != "1" ]]; then
    echo "ERROR: Mount not ready or no prepared-data-* found under $MOUNT_POINT"
    echo "  Check: ls -la $MOUNT_POINT | head"
    exit 1
fi

# ---------- Step 3: Print training commands ----------
echo ""
echo "============================================================"
echo "ReStraV training (mount at $MOUNT_POINT)"
echo "  CSVs: train=$TRAIN_CSV  val=$VAL_CSV"
echo "  Run from: $SCRIPT_DIR"
echo "============================================================"
echo ""
echo "--- Uncompressed .npy on mount (typical) ---"
echo "  python train_restrav.py \\"
echo "    --train-csv $TRAIN_CSV --val-csv $VAL_CSV \\"
echo "    --data-root $MOUNT_POINT \\"
echo "    --output-dir checkpoints/restrav \\"
echo "    --epochs 10 --batch-size 32 --workers 0 --prefetch-factor 4"
echo ""
echo "--- With .npy.zst (decoded cache on local disk; faster after first pass) ---"
echo "  python train_restrav.py \\"
echo "    --train-csv $TRAIN_CSV --val-csv $VAL_CSV \\"
echo "    --data-root $MOUNT_POINT \\"
echo "    --compressed --decoded-root /path/to/local/ssd/_decoded_restrav \\"
echo "    --output-dir checkpoints/restrav \\"
echo "    --epochs 10 --batch-size 32 --workers 0 --prefetch-factor 4"
echo ""
echo "--- Optional: --mmap (use if you have enough RAM; less copy from mount) ---"
echo "  add: --mmap"
echo ""
echo "--- Quick test (few epochs, small calib) ---"
echo "  python train_restrav.py \\"
echo "    --train-csv $TRAIN_CSV --val-csv $VAL_CSV \\"
echo "    --data-root $MOUNT_POINT \\"
echo "    --epochs 2 --calib-samples 5000 --batch-size 32 \\"
echo "    --output-dir checkpoints/restrav_test"
echo ""
echo "First epoch reads through mount (VFS cache fills). Later epochs use cache → higher GPU util."
echo "============================================================"
