# scripts/create_stratified_dev.py
"""
Create a stratified-ish dev dataset by sampling from source CSV files.

Writes a single merged CSV: dst/ocs_lab_dev_merged.csv

Usage:
  python scripts/create_stratified_dev.py --src datasets/ocs-lab-chunks --dst datasets/ocs-lab-dev --samples 20000
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def sample_from_files(src_dir: Path, dst_dir: Path, total_samples: int = 20000, label_col_names=None, seed=42):
    dst_dir.mkdir(parents=True, exist_ok=True)
    label_col_names = label_col_names or ["attack", "label", "is_attack", "y"]
    rng = np.random.default_rng(seed)

    files = sorted([p for p in src_dir.glob("*.csv")] + [p for p in src_dir.glob("*.csv.gz")])
    if not files:
        raise SystemExit(f"No CSV files found in {src_dir}")

    # Read files, collect small metadata, but do not keep everything in memory unnecessarily
    dfs = []
    total_rows = 0
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            # find label column (case-insensitive)
            label_col = None
            for cand in label_col_names:
                if cand in df.columns:
                    label_col = cand
                    break
            if label_col is None:
                for c in df.columns:
                    if c.lower() in [x.lower() for x in label_col_names]:
                        label_col = c
                        break
            if label_col is None:
                # no label column found; create placeholder
                df["_dev_label"] = -1
                label_col = "_dev_label"
            df["_dev_label_col"] = label_col  # store which label column is used for this df
            dfs.append((f.name, df, label_col))
            total_rows += len(df)
            print(f"Loaded {f.name}: rows={len(df)} label_col={label_col}")
        except Exception as e:
            print("Failed to read", f, e)

    if total_rows == 0:
        raise SystemExit("No rows available in source files")

    # Build combined stratified sample across files
    # Approach: concatenate but sample per-label proportionally across all data
    all_concat = pd.concat([df for (_, df, _) in dfs], ignore_index=True, sort=False)
    # ensure label column unified under 'dev_label'
    def unify_label_row(r):
        col = r["_dev_label_col"]
        return r[col] if pd.notna(r[col]) else -1
    all_concat["dev_label"] = all_concat.apply(lambda r: unify_label_row(r), axis=1)

    # Now perform stratified sampling by dev_label (proportional to class frequency)
    # If some classes are rare and cause issues, we'll fall back to random sample
    try:
        grouped = all_concat.groupby("dev_label", group_keys=False)
        # compute per-group quotas proportional to class size
        quotas = (grouped.size() / len(all_concat) * total_samples).round().astype(int).to_dict()
        # ensure sum equals total_samples (adjust largest class)
        current_sum = sum(quotas.values())
        if current_sum != total_samples:
            diff = total_samples - current_sum
            # add diff to the largest class
            largest = max(quotas.items(), key=lambda x: x[1])[0]
            quotas[largest] += diff

        sampled_parts = []
        for label_val, q in quotas.items():
            grp = all_concat[all_concat["dev_label"] == label_val]
            if len(grp) <= q:
                sampled_parts.append(grp)
            else:
                sampled_parts.append(grp.sample(n=q, random_state=seed))
        merged = pd.concat(sampled_parts, ignore_index=True, sort=False)
    except Exception as e:
        # fallback random sample if stratified approach fails
        print("Stratified sampling failed, falling back to random sample:", e)
        merged = all_concat.sample(n=min(total_samples, len(all_concat)), random_state=seed).reset_index(drop=True)

    # final adjust to exact size
    if len(merged) > total_samples:
        merged = merged.sample(n=total_samples, random_state=seed).reset_index(drop=True)
    elif len(merged) < total_samples:
        deficit = total_samples - len(merged)
        pool = all_concat.drop(merged.index, errors='ignore')
        if len(pool) > 0:
            extra = pool.sample(n=min(deficit, len(pool)), random_state=seed)
            merged = pd.concat([merged, extra], ignore_index=True, sort=False)

    # remove helper columns before writing
    for c in ["_dev_label_col", "dev_label"]:
        if c in merged.columns:
            merged = merged.drop(columns=[c])

    merged_path = dst_dir / "ocs_lab_dev_merged.csv"
    merged.to_csv(merged_path, index=False)
    print("Wrote merged dev CSV:", merged_path, "rows=", len(merged))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    sample_from_files(Path(args.src), Path(args.dst), total_samples=args.samples, seed=args.seed)
