#!/usr/bin/env python3
"""
Download a GasBench video repo (parquet) and convert videos to BGR .npy.
Outputs a CSV manifest with columns: bgr_path,label,dataset.
"""

import argparse
import hashlib
import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from decord import VideoReader, cpu
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


LABEL_MAP = {
    "real": 0,
    "synthetic": 1,
    "semisynthetic": 1,
}


def resize_shortest_edge_center_crop(frame: np.ndarray, target: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h < w:
        new_h = target
        new_w = int(w * (target / h))
    else:
        new_w = target
        new_h = int(h * (target / w))
    new_h = max(new_h, target)
    new_w = max(new_w, target)
    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    i = (new_h - target) // 2
    j = (new_w - target) // 2
    return frame[i : i + target, j : j + target, :]


def extract_frames_rgb(video_bytes: bytes, max_frames: int = 16) -> Optional[np.ndarray]:
    if not video_bytes:
        return None
    # Decord expects a file-like source; use a temporary file on disk
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tf.write(video_bytes)
        temp_path = tf.name
    try:
        vr = VideoReader(temp_path, ctx=cpu(0), num_threads=1)
        total = len(vr)
        if total == 0:
            return None
        frames = []
        for i in range(min(max_frames, total)):
            frame = vr[i].asnumpy()  # RGB HWC
            if frame is None or frame.size == 0:
                continue
            frames.append(frame)
        if not frames:
            return None
        if len(frames) < max_frames:
            last = frames[-1]
            while len(frames) < max_frames:
                frames.append(last)
        return np.asarray(frames, dtype=np.uint8)
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass


def process_video_bytes_to_bgr(video_bytes: bytes, target_size: int) -> Optional[np.ndarray]:
    rgb = extract_frames_rgb(video_bytes, max_frames=16)
    if rgb is None:
        return None
    frames = [resize_shortest_edge_center_crop(f, target_size) for f in rgb]
    rgb = np.asarray(frames, dtype=np.uint8)
    bgr = rgb[:, :, :, ::-1]  # RGB -> BGR
    return bgr


def find_parquet_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.parquet"))


def detect_video_column(schema) -> Optional[str]:
    candidates = ["video_bytes", "video", "bytes", "data"]
    # ParquetSchema: use names only
    if hasattr(schema, "names"):
        fields = {name: None for name in schema.names}
    else:
        fields = {getattr(f, "name", str(f)): None for f in schema}
    for name in candidates:
        if name in fields:
            return name
    for name, ftype in fields.items():
        if str(ftype).startswith("binary") or str(ftype).startswith("large_binary"):
            return name
    return None


def build_archive_index(root: Path) -> dict:
    index = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith((".tar", ".tar.gz", ".tgz", ".zip")):
                index[name] = str(Path(dirpath) / name)
    return index


def read_bytes_from_archive(archive_path: str, inner_path: str) -> Optional[bytes]:
    if archive_path.endswith((".tar", ".tar.gz", ".tgz")):
        import tarfile
        with tarfile.open(archive_path, "r:*") as tf:
            try:
                member = tf.getmember(inner_path)
            except KeyError:
                return None
            f = tf.extractfile(member)
            return f.read() if f else None
    if archive_path.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(archive_path, "r") as zf:
            try:
                with zf.open(inner_path, "r") as f:
                    return f.read()
            except KeyError:
                return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True, help="HF repo id, e.g., gasstation/gs-videos-v3")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to write .npy files")
    parser.add_argument("--dataset-name", type=str, default="gasstation-generated-videos")
    parser.add_argument("--media-type", type=str, default="synthetic", choices=["real", "synthetic", "semisynthetic"])
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples for testing (0 = all)")
    parser.add_argument("--hf-token", type=str, default=None, help="Optional HF token (or use HF_TOKEN env)")
    parser.add_argument("--csv-path", type=str, default="prepared_data_gasbench.csv")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8, help="CPU workers for conversion")
    parser.add_argument("--log-every", type=int, default=200, help="Log progress every N saved samples")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = out_dir / args.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    token = args.hf_token or os.environ.get("HF_TOKEN")
    local_repo = snapshot_download(repo_id=args.repo_id, token=token, repo_type="dataset")
    parquet_files = find_parquet_files(Path(local_repo))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in {local_repo}")

    label = LABEL_MAP[args.media_type]
    csv_lines = ["bgr_path,label,dataset\n"]
    saved = 0

    archive_index = build_archive_index(Path(local_repo))
    for pq_path in parquet_files:
        pq_file = pq.ParquetFile(pq_path)
        video_col = detect_video_column(pq_file.schema)
        has_archive_cols = False
        archive_col = None
        inner_col = None
        if video_col is None:
            names = pq_file.schema.names
            if "archive_filename" in names and "file_path_in_archive" in names:
                has_archive_cols = True
                archive_col = "archive_filename"
                inner_col = "file_path_in_archive"
            else:
                raise ValueError(
                    f"No video bytes column found in schema fields: {list(names)[:20]}"
                )
        for batch in pq_file.iter_batches():
            col = batch.column(video_col) if video_col else None
            archive_batch = batch.column(archive_col) if has_archive_cols else None
            inner_batch = batch.column(inner_col) if has_archive_cols else None
            batch_len = len(col) if col is not None else len(archive_batch)
            tasks = []
            for i in range(batch_len):
                if args.max_samples and saved + len(tasks) >= args.max_samples:
                    break
                if video_col:
                    video_bytes = col[i].as_py()
                else:
                    archive_name = archive_batch[i].as_py()
                    inner_path = inner_batch[i].as_py()
                    archive_path = archive_index.get(archive_name)
                    if not archive_path:
                        candidate = Path(local_repo) / archive_name
                        archive_path = str(candidate) if candidate.exists() else None
                    if archive_path:
                        video_bytes = read_bytes_from_archive(archive_path, inner_path)
                    else:
                        video_bytes = None
                tasks.append(video_bytes)

            if not tasks:
                continue

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(process_video_bytes_to_bgr, vb, args.target_size): vb for vb in tasks}
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Converting {pq_path.name}", leave=False):
                    if args.max_samples and saved >= args.max_samples:
                        break
                    bgr = fut.result()
                    if bgr is None:
                        continue
                    stem = f"{pq_path.stem}_{saved}"
                    name = hashlib.sha1(stem.encode()).hexdigest()[:16] + "_bgr.npy"
                    rel_path = f"{args.dataset_name}/{name}"
                    np.save(dataset_dir / name, bgr)
                    csv_lines.append(f"{rel_path},{label},{args.dataset_name}\n")
                    saved += 1
                    if args.log_every > 0 and (saved % args.log_every == 0):
                        print(f"[progress] saved={saved}")

            if args.max_samples and saved >= args.max_samples:
                break
        if args.max_samples and saved >= args.max_samples:
            break

    csv_path = Path(args.csv_path)
    csv_path.write_text("".join(csv_lines))
    print(f"Saved {saved} samples to {dataset_dir}")
    print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
