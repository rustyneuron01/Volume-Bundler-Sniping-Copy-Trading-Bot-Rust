#!/usr/bin/env python3
"""
Create balanced validation CSV(s) for model selection and calibration.

This creates validation sets that:
- Sample by (dataset, label) to reduce skew
- Globally balance real/fake (1:1)
- Optionally split into val_select and val_calib (non-overlapping)

Usage:
    python create_val_csv.py --input master_all.csv \
      --output-select val_select.csv \
      --output-calib val_calib.csv \
      --total-size 40000 \
      --calib-size 10000
"""

import pandas as pd
import argparse
from pathlib import Path

# Samples per dataset for validation (total per dataset, split across labels)
SAMPLES_PER_DATASET = 2000  # Target per dataset, split across labels
# Small dataset threshold
SMALL_DATASET_THRESHOLD = 5000


def create_validation_csv(
    input_csv: str,
    output_select: str,
    output_calib: str,
    total_size: int,
    calib_size: int,
    seed: int,
):
    """
    Create balanced validation CSV(s) from ALL datasets.
    """
    print(f"Reading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    df["dataset"] = df["dataset"].astype(str).str.strip()
    # Stable unique id to enforce true non-overlapping splits later.
    df["_row_id"] = df.index.astype("int64")

    print(f"Total samples in input: {len(df):,}")
    print(f"Datasets in input: {df['dataset'].nunique()}")

    # Check dataset label purity
    label_counts = df.groupby("dataset")["label"].nunique()
    mixed = label_counts[label_counts > 1]
    if not mixed.empty:
        print("\n⚠️  Datasets with mixed labels (need special handling):")
        for dataset in mixed.index.tolist():
            counts = df[df["dataset"] == dataset]["label"].value_counts().to_dict()
            print(f"  - {dataset}: {counts}")

    per_label = max(1, SAMPLES_PER_DATASET // 2)
    val_samples = []

    print("\nSampling per dataset/label:")
    for dataset in sorted(df["dataset"].unique()):
        dataset_df = df[df["dataset"] == dataset]
        dataset_size = len(dataset_df)
        is_small = dataset_size < SMALL_DATASET_THRESHOLD

        for label in [0, 1]:
            label_df = dataset_df[dataset_df["label"] == label]
            if label_df.empty:
                continue
            if is_small:
                n_samples = max(1, int(0.1 * len(label_df)))
            else:
                n_samples = min(per_label, len(label_df))
            sampled = label_df.sample(n=n_samples, random_state=seed)
            val_samples.append(sampled)

        print(
            f"✓ {dataset:30s}: real={len(dataset_df[dataset_df['label']==0]):5d} "
            f"fake={len(dataset_df[dataset_df['label']==1]):5d} "
            f"({'10%' if is_small else 'cap'})"
        )

    val_df = pd.concat(val_samples, ignore_index=True)

    # Balance globally
    real_df = val_df[val_df["label"] == 0]
    fake_df = val_df[val_df["label"] == 1]
    n_each = min(len(real_df), len(fake_df))
    real_df = real_df.sample(n=n_each, random_state=seed)
    fake_df = fake_df.sample(n=n_each, random_state=seed)
    val_df = pd.concat([real_df, fake_df], ignore_index=True)

    # Cap to total_size (if provided)
    if total_size > 0:
        n_each = total_size // 2
        real_df = real_df.sample(n=min(n_each, len(real_df)), random_state=seed)
        fake_df = fake_df.sample(n=min(n_each, len(fake_df)), random_state=seed)
        val_df = pd.concat([real_df, fake_df], ignore_index=True)

    val_df = val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Split into selection/calibration sets with balanced labels
    if calib_size > 0:
        calib_size = min(calib_size, len(val_df) // 2 * 2)
        per_label = calib_size // 2
        real_df = val_df[val_df["label"] == 0]
        fake_df = val_df[val_df["label"] == 1]
        calib_real = real_df.sample(n=min(per_label, len(real_df)), random_state=seed)
        calib_fake = fake_df.sample(n=min(per_label, len(fake_df)), random_state=seed)
        calib_df = pd.concat([calib_real, calib_fake], ignore_index=True).sample(
            frac=1.0, random_state=seed
        )
        # IMPORTANT: split by persistent row ids, not dataframe positional index.
        # Using calib_df.index here causes overlap after concat/reset_index operations.
        calib_ids = set(calib_df["_row_id"].tolist())
        select_df = val_df[~val_df["_row_id"].isin(calib_ids)]
    else:
        calib_df = None
        select_df = val_df

    print(f"\n{'='*70}")
    print("Validation Set Summary:")
    print(f"  Total samples: {len(val_df):,}")
    print(f"  Real: {(val_df['label'] == 0).sum():,}")
    print(f"  Fake: {(val_df['label'] == 1).sum():,}")
    print(f"  Datasets: {val_df['dataset'].nunique()}")
    print(f"{'='*70}")

    if select_df is not None:
        select_out = select_df.drop(columns=["_row_id"], errors="ignore")
        select_out.to_csv(output_select, index=False)
        print(f"\n✓ Saved val_select: {output_select} ({len(select_out):,} samples)")
    if calib_df is not None:
        calib_out = calib_df.drop(columns=["_row_id"], errors="ignore")
        calib_out.to_csv(output_calib, index=False)
        print(f"✓ Saved val_calib:  {output_calib} ({len(calib_out):,} samples)")


def main():
    parser = argparse.ArgumentParser(description="Create balanced validation CSV(s)")
    parser.add_argument('--input', type=str, required=True,
                        help='Input master CSV with all data')
    parser.add_argument('--output-select', type=str, default='val_select.csv',
                        help='Output validation CSV for model selection')
    parser.add_argument('--output-calib', type=str, default='val_calib.csv',
                        help='Output validation CSV for calibration')
    parser.add_argument('--total-size', type=int, default=40000,
                        help='Total size for combined validation set (0 for no cap)')
    parser.add_argument('--calib-size', type=int, default=10000,
                        help='Size for calibration set (0 to disable split)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    
    args = parser.parse_args()
    
    create_validation_csv(
        args.input,
        args.output_select,
        args.output_calib,
        args.total_size,
        args.calib_size,
        args.seed,
    )


if __name__ == '__main__':
    main()

