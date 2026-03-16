"""
backend/dataset_loader.py

Safe, single-responsibility dataset loader for CAN IDS project.

Behavior summary:
 - Loads ONLY the file or directory explicitly passed as `path`.
 - If `path` is a directory, will only scan for .csv/.parquet files inside that directory (no recursion).
 - Enforces MAX_ROWS and MAX_FILE_MB caps to avoid multi-GB loads.
 - Maps "ocs-lab" preset to a small dev sample (ocs-lab-dev) only.
 - Returns (X, y, meta) where X is numpy array (N, feature_dim), y is int array (0/1/-1), meta contains df and resolved_path.
 - No automatic fallback to "." and no recursive fallback attempts.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List, Iterable
import numpy as np
import pandas as pd
import re

# Safety constants
MAX_ROWS_GLOBAL = 250_000         # absolute cap for any read
MAX_FILE_MB = 250                 # skip files > this size unless explicitly allowed
PAD_TO = 8                        # payload bytes padded/truncated to this
ALLOWED_DIR_EXTS = (".csv", ".parquet", ".parq", ".tsv", ".txt", ".log")

HEX_PAIR_RE = re.compile(r"[0-9A-Fa-f]{2}")

def _parse_data_field(s: Optional[str]) -> List[int]:
    """Parse '05 20 ea 0a' -> list of ints (len variable). Robust but conservative."""
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        out = []
        for x in s:
            try:
                out.append(int(x))
            except Exception:
                try:
                    out.append(int(str(x), 0))
                except Exception:
                    mm = re.findall(r"[0-9a-fA-F]{2}", str(x))
                    out.extend([int(m, 16) for m in mm])
        return out
    s = str(s).strip().replace(",", " ").replace("0x", "")
    parts = re.findall(r"[0-9a-fA-F]{2}", s)
    return [int(p, 16) for p in parts]

def _gather_files_from_dir(path: str, exts: Iterable[str] = ALLOWED_DIR_EXTS) -> List[str]:
    """Return sorted list of files in directory matching allowed extensions (non-recursive)."""
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return []
    out = []
    for e in exts:
        out.extend(sorted([str(x) for x in p.iterdir() if x.is_file() and x.suffix.lower() == e]))
    return out

def _read_csv_safe(path: str, nrows: Optional[int] = None, chunksize: int = 20000) -> pd.DataFrame:
    """
    Conservative CSV read: read in chunks up to nrows and return concatenated DataFrame.
    Will raise on unrecoverable read errors instead of attempting exotic fallback paths.
    """
    pieces = []
    read_rows = 0
    try:
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=True):
            pieces.append(chunk)
            read_rows += len(chunk)
            if nrows is not None and read_rows >= nrows:
                break
    except Exception as e:
        # Reraise to let caller decide fallback (we avoid cascading attempts here)
        raise
    if not pieces:
        return pd.DataFrame()
    df = pd.concat(pieces, ignore_index=True)
    if nrows is not None and len(df) > nrows:
        df = df.iloc[:nrows]
    return df

def _file_size_mb(path: str) -> float:
    try:
        return Path(path).stat().st_size / 1024.0 / 1024.0
    except Exception:
        return float("inf")

def load_dataset(path: str,
                 max_rows: Optional[int] = None,
                 pad_to: int = PAD_TO,
                 normalize_can_id: bool = True,
                 allow_large_files: bool = False,
                 verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Main loader that returns (X, y, meta).

    Parameters:
      - path: explicit file path or directory path or preset name (see PRESETS).
      - max_rows: optional cap (min with MAX_ROWS_GLOBAL).
      - allow_large_files: if True, some file-size skips are disabled (default False).
    """
    meta: Dict[str, Any] = {"requested_path": path}
    if not path:
        raise ValueError("dataset path is required and must be explicit (auto-detection disabled).")

    # PRESETS -> map dangerous names to small dev samples only
    PRESETS = {
        "ocs-lab": "datasets/ocs-lab-dev",
        "ocs": "datasets/ocs-lab-dev",
    }
    lp = str(path).lower()
    if lp in PRESETS:
        path = PRESETS[lp]
        meta["preset_mapped"] = True

    # enforce global max rows
    if max_rows is None:
        max_rows = MAX_ROWS_GLOBAL
    else:
        max_rows = min(max_rows, MAX_ROWS_GLOBAL)

    p = Path(path)
    df: Optional[pd.DataFrame] = None
    resolved_path = None

    # If directory: gather allowed files (non-recursive) and concat safely
    if p.exists() and p.is_dir():
        files = _gather_files_from_dir(str(p))
        if not files:
            raise FileNotFoundError(f"No allowed data files found in directory: {path}")
        pieces = []
        rows_acc = 0
        for f in files:
            size_mb = _file_size_mb(f)
            if not allow_large_files and size_mb > MAX_FILE_MB:
                if verbose:
                    print(f"[dataset_loader] SKIP {f} ({size_mb:.1f} MB) > {MAX_FILE_MB}MB")
                continue
            try:
                df_chunk = _read_csv_safe(f, nrows=min(max_rows - rows_acc, 20000) if max_rows else None)
            except Exception:
                if verbose:
                    print(f"[dataset_loader] Failed reading file {f}; skipping it.")
                continue
            pieces.append(df_chunk)
            rows_acc += len(df_chunk)
            if max_rows and rows_acc >= max_rows:
                break
        if not pieces:
            raise FileNotFoundError(f"No readable CSV/Parquet files in directory: {path}")
        df = pd.concat(pieces, ignore_index=True)
        if max_rows and len(df) > max_rows:
            df = df.iloc[:max_rows]
        resolved_path = files if len(files) > 1 else files[0]

    else:
        # path should point to a file
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        ext = p.suffix.lower()
        if (not allow_large_files) and _file_size_mb(str(p)) > MAX_FILE_MB:
            raise RuntimeError(f"File too large: {_file_size_mb(str(p)):.1f}MB (exceeds {MAX_FILE_MB}MB).")
        if ext == ".parquet":
            df = pd.read_parquet(str(p))
            if max_rows and len(df) > max_rows:
                df = df.iloc[:max_rows]
            resolved_path = str(p)
        elif ext in (".csv", ".tsv", ".txt", ".log"):
            # conservative read for CSV/TSV
            sep = "," if ext == ".csv" else None
            try:
                df = _read_csv_safe(str(p), nrows=max_rows)
            except Exception:
                # last-resort: try parse as raw lines (line-by-line) using hex extraction
                rows = []
                with open(str(p), "r", encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh):
                        if max_rows and i >= max_rows:
                            break
                        # extract hex bytes
                        bytes_found = re.findall(HEX_PAIR_RE, line)
                        data = [int(b, 16) for b in bytes_found] if bytes_found else []
                        # attempt to find timestamp and id tokens conservatively
                        ts = None
                        can_id = None
                        m_ts = re.search(r'\b(?:timestamp|time)[:\s]*([0-9]+\.[0-9]+|[0-9]+)', line, re.IGNORECASE)
                        if m_ts:
                            try:
                                ts = float(m_ts.group(1))
                            except Exception:
                                ts = None
                        m_id = re.search(r'\b(?:id|can_id)\s*[=:]\s*([0-9A-Fa-fxX]+)\b', line, re.IGNORECASE)
                        if m_id:
                            tok = m_id.group(1)
                            try:
                                can_id = int(tok, 0)
                            except Exception:
                                try:
                                    can_id = int(tok, 16)
                                except Exception:
                                    can_id = None
                        rows.append({"ts": ts, "can_id": can_id, "dlc": len(data), "data": data, "attack": -1})
                if not rows:
                    raise RuntimeError(f"Unable to parse file as CSV or raw log: {p}")
                df = pd.DataFrame(rows)
            resolved_path = str(p)
        else:
            raise RuntimeError(f"Unsupported file extension: {ext}")

    # Basic validation
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError("Loaded DataFrame is empty or invalid.")

    meta["resolved_path"] = resolved_path
    meta["rows_raw"] = int(len(df))

    # Normalize column names and extract table fields conservatively
    col_lower = {c.lower(): c for c in df.columns}
    def get_col(*cands):
        for c in cands:
            if c in col_lower:
                return df[col_lower[c]]
        return None

    ts_col = get_col("ts", "timestamp", "time")
    can_col = get_col("can_id", "id", "canid", "can")
    dlc_col = get_col("dlc", "length")
    data_col = get_col("data", "payload", "data_hex")
    attack_col = get_col("attack", "label", "is_attack", "y", "target", "class")

    # Build row-wise normalized DataFrame (df2)
    rows = []
    total = len(df)
    Nrows = total if max_rows is None else min(total, max_rows)
    for i in range(Nrows):
        r = {}
        # ts
        try:
            if ts_col is not None and not pd.isna(ts_col.iloc[i]):
                r["ts"] = float(ts_col.iloc[i])
            else:
                r["ts"] = None
        except Exception:
            r["ts"] = None
        # can_id
        try:
            if can_col is not None and not pd.isna(can_col.iloc[i]):
                try:
                    r["can_id"] = int(str(can_col.iloc[i]), 0)
                except Exception:
                    try:
                        r["can_id"] = int(str(can_col.iloc[i]), 16)
                    except Exception:
                        r["can_id"] = 0
            else:
                r["can_id"] = 0
        except Exception:
            r["can_id"] = 0
        # dlc
        try:
            if dlc_col is not None and not pd.isna(dlc_col.iloc[i]):
                r["dlc"] = int(dlc_col.iloc[i])
            else:
                r["dlc"] = None
        except Exception:
            r["dlc"] = None
        # data
        try:
            if data_col is not None and not pd.isna(data_col.iloc[i]):
                r["data"] = _parse_data_field(data_col.iloc[i])
            else:
                # search for textual columns containing hex bytes (conservative)
                found = []
                for c in df.columns:
                    val = df.iloc[i][c]
                    if isinstance(val, str) and re.search(r"[0-9A-Fa-f]{2}", val):
                        found = _parse_data_field(val)
                        if found:
                            break
                r["data"] = found
        except Exception:
            r["data"] = []
        # attack/label
        try:
            if attack_col is not None and not pd.isna(attack_col.iloc[i]):
                v = attack_col.iloc[i]
                try:
                    r["attack"] = int(v)
                except Exception:
                    sval = str(v).strip().lower()
                    r["attack"] = 1 if sval in ("1", "true", "yes", "attack") else 0
            else:
                r["attack"] = -1
        except Exception:
            r["attack"] = -1
        rows.append(r)

    df2 = pd.DataFrame(rows)

    # compute ts_delta
    if "ts" in df2.columns and df2["ts"].notna().any():
        try:
            df2 = df2.sort_values("ts").reset_index(drop=True)
            ts_vals = df2["ts"].ffill().fillna(0).astype(float)
            df2["ts_delta"] = ts_vals.diff().fillna(0.0).astype(float)
        except Exception:
            df2["ts_delta"] = 0.0
    else:
        df2["ts_delta"] = 0.0

    # canonicalize can_id
    df2["can_id"] = df2.get("can_id", pd.Series([0]*len(df2))).fillna(0).astype(int)

    # Build feature arrays
    N = len(df2)
    byte_arr = np.zeros((N, pad_to), dtype=np.int32)
    dlc_arr = np.zeros((N,), dtype=np.int32)
    for i, d in enumerate(df2["data"].to_list()):
        b = d if isinstance(d, (list, tuple)) else []
        try:
            b = [int(x) for x in b]
        except Exception:
            b = _parse_data_field(str(b))
        b = b[:pad_to] + [0] * max(0, pad_to - len(b))
        byte_arr[i, :] = np.array(b, dtype=np.int32)
        # dlc fallback
        try:
            if "dlc" in df2.columns and not pd.isna(df2.iloc[i].get("dlc", None)):
                dlc_arr[i] = int(df2.iloc[i].get("dlc", 0))
            else:
                dlc_arr[i] = min(len([x for x in b if x != 0]), pad_to)
        except Exception:
            dlc_arr[i] = min(len([x for x in b if x != 0]), pad_to)

    bytes_mean = byte_arr.mean(axis=1) if N > 0 else np.zeros((N,))
    bytes_std = byte_arr.std(axis=1) if N > 0 else np.zeros((N,))

    can_ids = df2["can_id"].astype(int).to_numpy()
    if normalize_can_id and len(can_ids) > 0:
        min_id, max_id = int(can_ids.min()), int(can_ids.max())
        if max_id == min_id:
            can_norm = np.zeros_like(can_ids, dtype=float)
        else:
            can_norm = (can_ids - min_id) / max(1, (max_id - min_id))
    else:
        can_norm = can_ids.astype(float)

    # assemble features: can_norm, dlc, bytes..., bytes_mean, bytes_std, ts_delta
    if N > 0:
        features = np.concatenate([
            can_norm.reshape(-1, 1).astype(float),
            dlc_arr.reshape(-1, 1).astype(float),
            byte_arr.astype(float),
            bytes_mean.reshape(-1, 1),
            bytes_std.reshape(-1, 1),
            df2["ts_delta"].to_numpy().reshape(-1, 1).astype(float),
        ], axis=1)
    else:
        features = np.zeros((0, 2 + pad_to + 3), dtype=float)

    # labels
    if "attack" in df2.columns:
        try:
            y = df2["attack"].astype(int).to_numpy()
        except Exception:
            y = np.array([int(bool(x)) if x != -1 else -1 for x in df2["attack"].fillna(-1).to_list()], dtype=int)
    else:
        y = np.full((N,), -1, dtype=int)

    meta_out = {
        "df": df2,
        "pad_to": pad_to,
        "feature_names": ["can_norm", "dlc"] + [f"byte{i}" for i in range(pad_to)] + ["bytes_mean", "bytes_std", "ts_delta"],
        "resolved_path": meta.get("resolved_path", resolved_path),
        "rows": int(N),
    }
    return features, y, meta_out
