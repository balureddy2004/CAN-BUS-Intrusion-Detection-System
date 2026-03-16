# scripts/split_large_csvs.py
"""
Split large CSV files into smaller chunk files.

Usage:
  python scripts/split_large_csvs.py --src datasets/ocs-lab --dst datasets/ocs-lab-chunks --rows 100000

This will create files like:
  datasets/ocs-lab-chunks/Attack_free_dataset.csv.part000.csv
  datasets/ocs-lab-chunks/Attack_free_dataset.csv.part001.csv
...
"""

import os
import argparse
import csv
from pathlib import Path

def split_file(src: Path, dst_dir: Path, max_rows: int = 100000, compress: bool = False):
    dst_dir.mkdir(parents=True, exist_ok=True)
    base = src.name
    part_idx = 0
    fh_out = None
    writer = None
    header = None
    out_path = None

    open_fn = open
    if compress:
        import gzip
        open_fn = gzip.open

    with open(src, "r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        rows_written = 0
        for row in reader:
            if rows_written % max_rows == 0:
                if fh_out:
                    fh_out.close()
                suffix = f".part{part_idx:03d}.csv"
                out_name = base + suffix
                out_path = dst_dir / out_name
                if compress:
                    out_path = out_path.with_suffix(out_path.suffix + ".gz")
                    fh_out = open_fn(out_path, "wt", encoding="utf-8")
                else:
                    fh_out = open_fn(out_path, "w", encoding="utf-8", newline="")
                writer = csv.writer(fh_out)
                if header:
                    writer.writerow(header)
                part_idx += 1
            writer.writerow(row)
            rows_written += 1

    if fh_out:
        fh_out.close()
    print(f"Split {src} -> {part_idx} parts (last: {out_path})")

def split_folder(src_dir: Path, dst_dir: Path, max_rows: int, compress: bool):
    for p in sorted(src_dir.glob("*.csv")):
        print("Splitting:", p)
        split_file(p, dst_dir, max_rows=max_rows, compress=compress)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source folder with big CSVs")
    ap.add_argument("--dst", required=True, help="Destination folder to place chunks")
    ap.add_argument("--rows", type=int, default=100000, help="Max rows per chunk (default 100k)")
    ap.add_argument("--compress", action="store_true", help="Compress output .csv.gz")
    args = ap.parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    split_folder(src, dst, max_rows=args.rows, compress=args.compress)

if __name__ == "__main__":
    main()
