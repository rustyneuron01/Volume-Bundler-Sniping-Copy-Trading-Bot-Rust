#!/usr/bin/env python3
"""
Create Master CSV from Backblaze - No Train/Val Split

This script:
1. Scans ALL datasets in Backblaze (b2:sn33-bucket/prepared-data-*)
2. Lists all BGR files from each dataset
3. Creates ONE master CSV with all files
4. Labels as real (0) or fake (1)
5. NO splitting - you create custom train/val later

Output: master_all.csv (all 12M+ samples)

Usage:
    python create_master_csv.py --output master_all.csv
"""

import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import re
import argparse

# Dataset labels (based on dataset name)
REAL_DATASETS = [
    "eidon-video", "imagenet-vidvrd", "pe-video", "deepaction-pexels",
    "dfd-real", "physics-101", "moments-in-time", "ucf101-fullvideo",
    "first-impressions-v2", "soccernet-10s-5class", "sports-qa",
    "egocentric-100k", "lovora-real", "FreeTacMan", "humanoid-everyday",
    "workout-vids",
]

def get_dataset_label(dataset_name):
    """Get label for dataset: 0=real, 1=fake/synthetic."""
    return 0 if dataset_name in REAL_DATASETS else 1


def scan_dataset(dataset):
    """Scan one dataset from Backblaze."""
    remote_path = f"b2:sn33-bucket/prepared-data-{dataset}"
    
    try:
        # Use rclone lsf for fast listing
        cmd = [
            "rclone", "lsf", remote_path,
            "--max-depth", "1",
            "--files-only",
            "--fast-list",
        ]
        
        # No timeout for massive datasets
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
        
        if result.returncode != 0:
            return (dataset, None, 0)
        
        # Process files
        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        bgr_files = [f for f in files if f.endswith('_bgr.npy') or f.endswith('_BGR.npy')]
        
        if not bgr_files:
            return (dataset, None, 0)
        
        label = get_dataset_label(dataset)
        
        # Create entries
        entries = []
        for bgr_file in bgr_files:
            sample_id = bgr_file.replace('_bgr.npy', '').replace('_BGR.npy', '')
            rgb_file = bgr_file.replace('_bgr.npy', '_rgb.npy').replace('_BGR.npy', '_RGB.npy')
            
            entry = {
                'sample_id': sample_id,
                'dataset': dataset,
                'label': label,
                'bgr_path': f"prepared-data-{dataset}/{bgr_file}",
                'rgb_path': f"prepared-data-{dataset}/{rgb_file}",
            }
            entries.append(entry)
        
        return (dataset, entries, len(bgr_files))
        
    except Exception as e:
        return (dataset, None, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='master_all.csv',
                        help='Output CSV file (all samples, no split)')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle the output CSV (recommended for training!)')
    
    args = parser.parse_args()
    
    cpu_count = mp.cpu_count()
    num_workers = cpu_count * 4  # Max I/O parallelization
    
    print("=" * 70)
    print("CREATE MASTER CSV FROM BACKBLAZE - NO SPLIT")
    print("=" * 70)
    print(f"\nSystem:")
    print(f"  CPUs: {cpu_count}")
    print(f"  Parallel workers: {num_workers}")
    print(f"\nOutput: {args.output}")
    print(f"Shuffle: {args.shuffle}")
    print("=" * 70)
    
    # Get datasets from Backblaze
    print("\n[1/4] Scanning Backblaze folders...")
    cmd = ["rclone", "lsd", "b2:sn33-bucket", "--max-depth", "1", "--fast-list"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    
    datasets = []
    for line in result.stdout.strip().split('\n'):
        match = re.search(r'prepared-data-(\S+)', line)
        if match:
            dataset_name = match.group(1)
            datasets.append(dataset_name)  # Include ALL datasets
    
    print(f"  ✓ Found {len(datasets)} datasets")
    
    # Scan datasets in parallel
    print(f"\n[2/4] Scanning {len(datasets)} datasets ({num_workers} workers)...")
    
    all_entries = []
    dataset_stats = {}
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(scan_dataset, ds): ds for ds in datasets}
        
        with tqdm(total=len(datasets), desc="  Scanning") as pbar:
            for future in as_completed(futures):
                dataset, entries, count = future.result()
                
                if entries is None:
                    tqdm.write(f"  ⚠️  {dataset}: No files found")
                elif isinstance(entries, list):
                    all_entries.extend(entries)
                    label = entries[0]['label'] if entries else None
                    label_str = 'REAL' if label == 0 else 'FAKE'
                    dataset_stats[dataset] = {'count': count, 'label': label_str}
                    tqdm.write(f"  ✓ {dataset:30s}: {count:7,} files ({label_str})")
                
                pbar.update(1)
    
    print(f"\n  Total entries: {len(all_entries):,}")
    
    if not all_entries:
        print("\n❌ No entries found!")
        return 1
    
    # Create DataFrame
    print(f"\n[3/4] Creating DataFrame...")
    df = pd.DataFrame(all_entries)
    
    # Remove duplicates
    df_before = len(df)
    df = df.drop_duplicates(subset=['bgr_path'], keep='first')
    df_after = len(df)
    
    if df_before > df_after:
        print(f"  Removed {df_before - df_after:,} duplicates")
    
    print(f"  Total unique samples: {df_after:,}")
    print(f"  Real: {(df['label'] == 0).sum():,}")
    print(f"  Fake: {(df['label'] == 1).sum():,}")
    print(f"  Datasets: {df['dataset'].nunique()}")
    
    # Shuffle if requested
    if args.shuffle:
        print(f"\n  Shuffling dataset...")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"  ✓ Shuffled (seed=42)")
    
    # Save
    print(f"\n[4/4] Saving master CSV...")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Backup if exists
    if output_path.exists():
        backup = output_path.parent / f"{output_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path.rename(backup)
        print(f"  Backed up existing file")
    
    df.to_csv(output_path, index=False)
    
    print(f"  ✓ Saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / (1024**2):.1f} MB")
    
    # Show dataset distribution
    print(f"\n{'='*70}")
    print("DATASET DISTRIBUTION")
    print(f"{'='*70}")
    
    print(f"\n{'Dataset':<30s} {'Count':>10s} {'Label':>8s}")
    print("-" * 50)
    
    for dataset, stats in sorted(dataset_stats.items(), key=lambda x: x[1]['count'], reverse=True):
        print(f"{dataset:<30s} {stats['count']:>10,} {stats['label']:>8s}")
    
    print("=" * 70)
    print("✅ MASTER CSV CREATED")
    print("=" * 70)
    print(f"\nTotal samples: {len(df):,}")
    print(f"Real samples: {(df['label'] == 0).sum():,} ({100*(df['label']==0).sum()/len(df):.1f}%)")
    print(f"Fake samples: {(df['label'] == 1).sum():,} ({100*(df['label']==1).sum()/len(df):.1f}%)")
    print(f"Datasets: {df['dataset'].nunique()}")
    
    print(f"\nNext steps:")
    print(f"  1. Create balanced training CSV (3.34M, 1:1 ratio):")
    print(f"     python create_balanced_csv.py --input {args.output} --output balanced_3.3M.csv")
    print(f"\n  2. Create validation CSV (50K hard datasets):")
    print(f"     python create_val_csv.py --input {args.output} --output val_50k.csv")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
