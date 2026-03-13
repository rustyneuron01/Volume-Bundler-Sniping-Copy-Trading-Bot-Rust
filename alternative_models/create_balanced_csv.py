#!/usr/bin/env python3
"""
Create a rebalanced training CSV.
Caps dominant datasets, optionally oversamples weak ones.
"""

import pandas as pd
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, default="prepared-data/balanced_3.3M_seed789.csv")
    parser.add_argument("--output-csv", type=str, default="prepared-data/rebalanced_full.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-cap", type=int, default=100000,
                        help="Max samples per dominant real dataset")
    parser.add_argument("--synthetic-cap", type=int, default=50000,
                        help="Max samples per synthetic dataset")
    parser.add_argument("--no-oversample", action="store_true",
                        help="Only cap, never oversample (preserves natural distribution)")
    parser.add_argument("--real-target", type=int, default=3000,
                        help="Target for weak real datasets (ignored if --no-oversample)")
    parser.add_argument("--target-real-fake-ratio", type=float, default=1.0,
                        help="Final global ratio real/synthetic (default 1.0 for 1:1)")
    parser.add_argument("--no-label-balance", action="store_true",
                        help="Disable final global label balancing step")
    args = parser.parse_args()

    np.random.seed(args.seed)
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df):,} samples from {args.input_csv}")

    real = df[df["label"] == 0]
    synth = df[df["label"] == 1]

    # Real datasets: cap dominant ones
    print("\n=== REAL DATASETS ===")
    real_parts = []
    for ds, subset in real.groupby("dataset"):
        count = len(subset)
        if count > args.real_cap:
            sampled = subset.sample(n=args.real_cap, random_state=args.seed)
            print(f"  {ds}: {count:,} -> {len(sampled):,} (capped)")
            real_parts.append(sampled)
        elif not args.no_oversample and count < args.real_target:
            target = args.real_target
            repeats = target // count
            remainder = target % count
            parts = [subset] * repeats
            if remainder > 0:
                parts.append(subset.sample(n=remainder, random_state=args.seed))
            oversampled = pd.concat(parts, ignore_index=True)
            print(f"  {ds}: {count:,} -> {len(oversampled):,} (oversampled)")
            real_parts.append(oversampled)
        else:
            print(f"  {ds}: {count:,} (kept)")
            real_parts.append(subset)

    rebalanced_real = pd.concat(real_parts, ignore_index=True)

    # Synthetic datasets: cap dominant ones
    print("\n=== SYNTHETIC DATASETS ===")
    synth_parts = []
    for ds, subset in synth.groupby("dataset"):
        count = len(subset)
        if count > args.synthetic_cap:
            sampled = subset.sample(n=args.synthetic_cap, random_state=args.seed)
            print(f"  {ds}: {count:,} -> {len(sampled):,} (capped)")
            synth_parts.append(sampled)
        else:
            print(f"  {ds}: {count:,} (kept)")
            synth_parts.append(subset)

    rebalanced_synth = pd.concat(synth_parts, ignore_index=True)

    # Combine and (optionally) enforce final global label ratio
    combined = pd.concat([rebalanced_real, rebalanced_synth], ignore_index=True)
    if not args.no_label_balance:
        ratio = float(args.target_real_fake_ratio)
        if ratio <= 0:
            raise ValueError("--target-real-fake-ratio must be > 0")

        real_pool = combined[combined["label"] == 0]
        synth_pool = combined[combined["label"] == 1]
        real_n = len(real_pool)
        synth_n = len(synth_pool)

        # Maximize kept samples while satisfying real/synth ~= ratio.
        if synth_n == 0 or real_n == 0:
            raise RuntimeError("Cannot balance labels: one class is empty after rebalancing.")

        if (real_n / synth_n) >= ratio:
            target_synth = synth_n
            target_real = int(target_synth * ratio)
        else:
            target_real = real_n
            target_synth = int(target_real / ratio)

        target_real = max(1, min(target_real, real_n))
        target_synth = max(1, min(target_synth, synth_n))

        print("\n=== GLOBAL LABEL BALANCE ===")
        print(f"Before: real={real_n:,}, synthetic={synth_n:,}, ratio={real_n/max(synth_n,1):.4f}")
        print(f"Target: real={target_real:,}, synthetic={target_synth:,}, ratio={target_real/max(target_synth,1):.4f}")

        real_part = real_pool.sample(n=target_real, random_state=args.seed)
        synth_part = synth_pool.sample(n=target_synth, random_state=args.seed)
        combined = pd.concat([real_part, synth_part], ignore_index=True)

    combined = combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Stats
    label_counts = combined["label"].value_counts().to_dict()
    print(f"\n=== FINAL ===")
    print(f"Total: {len(combined):,}")
    print(f"  Real (0): {label_counts.get(0, 0):,}")
    print(f"  Synthetic (1): {label_counts.get(1, 0):,}")

    print(f"\nReal distribution:")
    for ds, cnt in combined[combined["label"] == 0].groupby("dataset").size().sort_values(ascending=False).items():
        print(f"  {ds}: {cnt:,}")

    combined.to_csv(args.output_csv, index=False)
    print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
