#!/usr/bin/env python3
"""
Create a training CSV for GAsBench performance.

Now that the training pipeline handles dataset balance at runtime via:
  --dataset-balance   (WeightedRandomSampler: each dataset gets equal gradient)
  --macro-avg-eval    (checkpoint selection treats all datasets equally)

...the CSV only needs to:
1. Include ALL data (no artificial downsampling of paired domains)
2. Cap extremely large datasets to keep total size manageable
3. Maintain reasonable global real:synthetic ratio (light touch only)
4. NO oversampling — runtime weighted sampling handles it without memorization
"""

import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Create GAsBench training CSV (all data, light caps only)"
    )
    parser.add_argument("--input-csv", type=str,
                        default="prepared-data/master_all.csv")
    parser.add_argument("--output-csv", type=str,
                        default="prepared-data/gasbench_balanced.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-cap", type=int, default=30000,
                        help="Max samples per real dataset (prevent single source domination)")
    parser.add_argument("--synth-cap", type=int, default=25000,
                        help="Max samples per synthetic dataset")
    parser.add_argument("--senorita-cap", type=int, default=10000,
                        help="Max samples per senorita variant (there are many)")
    parser.add_argument("--max-total", type=int, default=0,
                        help="If >0, cap total dataset size (proportional downsample)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    print(f"Loading {args.input_csv}...")
    df = pd.read_csv(args.input_csv, low_memory=False)
    print(f"Loaded {len(df):,} samples ({df['dataset'].nunique()} datasets)")

    # ---- Step 1: Apply per-dataset caps ----
    print("\n=== Step 1: Apply per-dataset caps (keep ALL small datasets) ===")
    parts = []
    for ds_name, group in df.groupby("dataset"):
        original_count = len(group)
        label = group["label"].iloc[0]

        is_senorita = "senorita" in ds_name.lower()

        if label == 0:
            cap = args.real_cap
        elif is_senorita:
            cap = args.senorita_cap
        else:
            cap = args.synth_cap

        if original_count > cap:
            group = group.sample(n=cap, random_state=args.seed)
            print(f"  {ds_name}: {original_count:,} -> {len(group):,} (capped)")

        parts.append(group)

    final = pd.concat(parts, ignore_index=True)

    # ---- Step 2: Optional total cap ----
    if args.max_total > 0 and len(final) > args.max_total:
        print(f"\n=== Step 2: Total cap {len(final):,} -> {args.max_total:,} ===")
        ratio = args.max_total / len(final)
        capped_parts = []
        for ds_name, group in final.groupby("dataset"):
            n = max(1, int(len(group) * ratio))
            # Never remove small datasets entirely
            n = max(n, min(len(group), 50))
            if n < len(group):
                group = group.sample(n=n, random_state=args.seed)
            capped_parts.append(group)
        final = pd.concat(capped_parts, ignore_index=True)

    # ---- Shuffle ----
    final = final.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # ---- Summary ----
    real_total = (final["label"] == 0).sum()
    synth_total = (final["label"] == 1).sum()

    print(f"\n=== Final summary ===")
    print(f"{'Dataset':42s} {'Count':>8s}  {'Label':>6s}")
    print("-" * 60)
    for ds_name in sorted(final["dataset"].unique()):
        sub = final[final["dataset"] == ds_name]
        tag = "REAL" if sub["label"].iloc[0] == 0 else "SYNTH"
        print(f"  {ds_name:40s} {len(sub):8,}  {tag}")

    print("-" * 60)
    print(f"  {'TOTAL':40s} {len(final):8,}")
    print(f"  Real:  {real_total:,} ({100*real_total/len(final):.1f}%)")
    print(f"  Synth: {synth_total:,} ({100*synth_total/len(final):.1f}%)")
    print(f"\n  Note: Dataset balance is handled at RUNTIME by --dataset-balance")
    print(f"  and --macro-avg-eval flags. No oversampling needed in the CSV.")

    final.to_csv(args.output_csv, index=False)
    print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
