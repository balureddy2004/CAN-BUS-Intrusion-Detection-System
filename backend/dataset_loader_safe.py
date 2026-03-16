"""
backend/dataset_loader_safe.py

Super-strict safe loader for UI/dev mode with tolerant encoding fallback.
- Tries UTF-8 first, then falls back to latin1 when reading CSV samples.
"""

import os
from pathlib import Path
import pandas as pd
import logging

log = logging.getLogger("dataset_loader_safe")
log.addHandler(logging.NullHandler())

ALLOWED_EXT = {".csv", ".txt", ".log"}
DEFAULT_MAX_FILE_MB = 50
DEFAULT_SAMPLE_ROWS_PER_FILE = 5000
DEFAULT_CHUNKSIZE = 20000

def list_data_files(dataset_path):
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"dataset path not found: {dataset_path}")
    if p.is_file():
        # single file provided
        if p.suffix.lower() not in ALLOWED_EXT:
            raise RuntimeError(f"File extension not allowed for safe loader: {p.suffix}")
        return [p]
    # directory -> non-recursive listing of allowed files
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXT]
    return sorted(files, key=lambda x: x.name)

def read_csv_sample(path, nrows_per_file=DEFAULT_SAMPLE_ROWS_PER_FILE, chunksize=DEFAULT_CHUNKSIZE):
    """
    Try reading CSV in chunks. Try 'utf-8' first; on UnicodeDecodeError, retry with 'latin1'.
    """
    total = 0
    pieces = []
    try:
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=True, encoding="utf-8"):
            pieces.append(chunk)
            total += len(chunk)
            if total >= nrows_per_file:
                break
    except UnicodeDecodeError:
        # Retry with latin1
        try:
            pieces = []
            total = 0
            for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=True, encoding="latin1"):
                pieces.append(chunk)
                total += len(chunk)
                if total >= nrows_per_file:
                    break
        except Exception as e:
            log.exception("Error reading %s with latin1: %s", path, e)
            raise
    except Exception as e:
        log.exception("Error reading %s: %s", path, e)
        raise
    if pieces:
        df = pd.concat(pieces, ignore_index=True)
        if len(df) > nrows_per_file:
            df = df.iloc[:nrows_per_file]
        return df
    return pd.DataFrame()

def safe_load_dataset(dataset_name_or_path, verbose=False, max_file_size_mb=DEFAULT_MAX_FILE_MB, sample_rows_per_file=DEFAULT_SAMPLE_ROWS_PER_FILE):
    """
    Safe loader entrypoint.

    Accepts:
     - explicit path to a file or directory
     - a short dataset name that resolves to ./datasets/<name> (no other searching)
    """
    p = Path(dataset_name_or_path)
    if not p.exists():
        # Resolve only under ./datasets/<name>
        alt = Path(os.getcwd()) / "datasets" / dataset_name_or_path
        p = alt
    if not p.exists():
        raise FileNotFoundError(f"Could not find dataset path: {dataset_name_or_path} or ./datasets/{dataset_name_or_path}")

    files = list_data_files(p)
    processed_files = []
    skipped = []
    samples = []
    for f in files:
        size_mb = f.stat().st_size / 1024.0 / 1024.0
        if size_mb > max_file_size_mb:
            skipped.append({"file": str(f), "reason": f"size {size_mb:.1f}MB > {max_file_size_mb}MB"})
            if verbose:
                log.warning("Skipping large file %s (%.1f MB)", f.name, size_mb)
            continue
        try:
            if verbose:
                log.info("Reading sample from %s (%.1fMB)", f.name, size_mb)
            # read_csv_sample now tries utf-8 then latin1
            df_sample = read_csv_sample(f, nrows_per_file=sample_rows_per_file)
            if not df_sample.empty:
                samples.append(df_sample)
                processed_files.append(str(f))
        except Exception as e:
            skipped.append({"file": str(f), "reason": f"read_error: {e}"})
    if samples:
        try:
            df_concat = pd.concat(samples, ignore_index=True)
        except Exception:
            # fallback: use first successful sample
            df_concat = samples[0]
    else:
        df_concat = pd.DataFrame()

    meta = {
        "df": df_concat,
        "files": processed_files,
        "skipped": skipped,
        "rows": int(len(df_concat)),
        "resolved_path": str(p),
    }
    if verbose:
        log.info("safe_load_dataset done: processed=%d skipped=%d rows=%d", len(processed_files), len(skipped), len(df_concat))
    return meta

def load_dataset_safe(name_or_path, **kwargs):
    return safe_load_dataset(name_or_path, **kwargs)
