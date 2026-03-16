# backend/dataset_loader.py
"""
Dataset loader for CAN IDS project.

Supports:
 - CSV with columns: ts,can_id,dlc,data,attack (data = "aa bb cc ...")
 - Raw log text lines like: "Timestamp: 0.000224 ID: 0329 DLC: 8  d7 a7 7f ..."
 - Auto-detects file kind, returns (X, y, meta) where:
    X: numpy array (N, feature_dim)
    y: numpy array (N,) 0/1 or -1 if unknown
 - If `path` is a directory, all CSV / Parquet files inside are loaded and concatenated.
"""

from __future__ import annotations
import re
import os
import glob
from typing import Tuple, Dict, Any, Optional, List, Iterable
import numpy as np
import pandas as pd
from pathlib import Path

HEX_BYTES_RE = re.compile(r"\b([0-9a-fA-F]{2})(?:\s+|$)")

def _parse_data_field(s: Optional[str]) -> List[int]:
    """Parse '05 20 ea 0a' -> list of ints (len variable)."""
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        # assume already ints or hex strings
        out = []
        for x in s:
            try:
                out.append(int(x))
            except Exception:
                try:
                    out.append(int(str(x), 0))
                except Exception:
                    # try hex pairs
                    mm = re.findall(r"[0-9a-fA-F]{2}", str(x))
                    out.extend([int(m, 16) for m in mm])
        return out
    s = str(s).strip().replace(",", " ").replace("0x", "")
    parts = re.findall(r"[0-9a-fA-F]{2}", s)
    return [int(p, 16) for p in parts]

# --- BEGIN REPLACED _parse_raw_line (fast, safe version) ---
# Precompiled regexes (fast)
_HEX_PAIR_RE = re.compile(r'[0-9A-Fa-f]{2}')
_ID_TOKEN_RE = re.compile(r'\b(?:id|can_id)\s*[=:]\s*([0-9A-Fa-fxX]+)\b', re.IGNORECASE)

def _parse_raw_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Fast, robust parser for raw log lines -> {'ts', 'can_id', 'dlc', 'data'} or None.
    - Uses simple heuristics for common formats (0xNNN, "ID: NNN", space-separated hex bytes)
    - Avoids expensive regex constructs that blow up on long lines.
    """
    if not line:
        return None
    # ensure string and trim
    try:
        s = str(line).strip()
    except Exception:
        return None
    if not s:
        return None

    ts = None
    can_id = None
    dlc = None
    data_bytes: List[int] = []

    # quick timestamp extraction (float-like)
    # matches "Timestamp: 0.000224" or "time: 0.5" etc.
    try:
        m_ts = re.search(r'\b(?:timestamp|time)[:\s]*([0-9]+\.\d+|[0-9]+)', s, re.IGNORECASE)
        if m_ts:
            try:
                ts = float(m_ts.group(1))
            except Exception:
                ts = None
    except Exception:
        ts = None

    # Attempt to find explicit ID token first (ID: 0329, can_id=0x080 etc.)
    try:
        m_id = _ID_TOKEN_RE.search(s)
        if m_id:
            id_token = m_id.group(1)
            try:
                # prefer automatic base detection (0x prefix or decimal)
                can_id = int(id_token, 0)
            except Exception:
                try:
                    can_id = int(id_token, 16)
                except Exception:
                    # keep None if unparsable
                    can_id = None
    except Exception:
        can_id = None

    # Fast-path: space/comma/colon separated hex bytes (most common)
    # Split on common separators and pick tokens that are exactly two hex chars
    try:
        # common separators
        sep_tokens = re.split(r'[\s,;:]+', s)
        for t in sep_tokens:
            if len(t) == 2 and all(c in "0123456789abcdefABCDEF" for c in t):
                try:
                    data_bytes.append(int(t, 16))
                except Exception:
                    pass
        # If we found bytes, attempt to infer dlc and possibly a leading hex ID
        if data_bytes:
            if dlc is None:
                dlc = len(data_bytes)
            # try to infer a leading hex id like "080" or "0x080" at start
            m_lead = re.match(r'^\s*(0x)?([0-9A-Fa-f]{3,8})\b', s)
            if can_id is None and m_lead:
                try:
                    can_id = int(m_lead.group(0), 0)
                except Exception:
                    # try parsing the second group as hex
                    try:
                        can_id = int(m_lead.group(2), 16)
                    except Exception:
                        can_id = can_id
            return {"ts": ts, "can_id": can_id, "dlc": dlc, "data": data_bytes}
    except Exception:
        # fall through to general extractor
        pass

    # Generic robust extractor using precompiled finditer
    try:
        it = _HEX_PAIR_RE.finditer(s)
        data_bytes = [int(m.group(0), 16) for m in it]
    except Exception:
        data_bytes = []

    if data_bytes:
        if dlc is None:
            dlc = len(data_bytes)
        # If can_id still unknown, attempt other heuristics
        if can_id is None:
            # look for 'ID=123' or id token again (safer)
            try:
                m_id2 = re.search(r'\b(?:id|can_id)\s*[=:]\s*([0-9A-Fa-fxX]+)\b', s, re.IGNORECASE)
                if m_id2:
                    tok = m_id2.group(1)
                    try:
                        can_id = int(tok, 0)
                    except Exception:
                        try:
                            can_id = int(tok, 16)
                        except Exception:
                            can_id = None
            except Exception:
                can_id = None

            # final fallback: leading hex-ish token
            if can_id is None:
                m_start = re.match(r'^\s*(0x)?([0-9A-Fa-f]{3,8})\b', s)
                if m_start:
                    try:
                        tok = m_start.group(0)
                        can_id = int(tok, 0)
                    except Exception:
                        try:
                            can_id = int(m_start.group(2), 16)
                        except Exception:
                            can_id = None

        return {"ts": ts, "can_id": can_id, "dlc": dlc, "data": data_bytes}

    # No bytes parsed and no ID/timestamp found -> return minimal (so caller can skip)
    return {"ts": ts, "can_id": can_id, "dlc": dlc if dlc is not None else 0, "data": []}
# --- END REPLACED _parse_raw_line ---

def _read_csv_with_fallback(path: str, nrows: Optional[int] = None, sep: Optional[str] = None) -> pd.DataFrame:
    """
    Try reading CSV with utf-8, fallback to latin1.
    Additional fallback attempts:
      - try pandas with engine='python' and sep=None/regex to handle whitespace-delimited logs
      - try delim_whitespace=True
      - finally return a one-column DataFrame containing raw lines (caller can parse raw lines)
    """
    kwargs = {"nrows": nrows} if nrows is not None else {}
    if sep is not None:
        kwargs["sep"] = sep

    # 1) normal read
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="latin1", **kwargs)
        except Exception:
            pass
    except Exception:
        pass

    # 2) python engine with regex sep (handle variable whitespace logs)
    try:
        kw = dict(kwargs)
        # if explicit sep not provided, try splitting on whitespace
        if "sep" not in kw or kw.get("sep") is None:
            kw["engine"] = "python"
            kw["sep"] = r"\s+"
        return pd.read_csv(path, **kw)
    except Exception:
        pass

    # 3) delim_whitespace (pandas convenience)
    try:
        kw = dict(kwargs)
        kw["delim_whitespace"] = True
        return pd.read_csv(path, **kw)
    except Exception:
        pass

    # 4) last resort: read file as raw lines and return single-column DF
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            lines = [L.rstrip("\n") for L in fh]
            # optionally limit rows if nrows present
            if nrows is not None:
                lines = lines[:nrows]
        return pd.DataFrame({"raw_line": lines})
    except Exception:
        # final fallback: try latin1 read of raw lines
        with open(path, "r", encoding="latin1", errors="ignore") as fh:
            lines = [L.rstrip("\n") for L in fh]
            if nrows is not None:
                lines = lines[:nrows]
        return pd.DataFrame({"raw_line": lines})

def _gather_files_from_dir(path: str, exts: Iterable[str] = (".csv", ".parquet")) -> List[str]:
    p = Path(path)
    out = []
    if not p.exists():
        return []
    for e in exts:
        out.extend(sorted([str(x) for x in p.glob(f"*{e}") if x.is_file()]))
    return out

def _load_and_concat_csvs(files: List[str], max_rows: Optional[int] = None,
                          MAX_FILE_MB: Optional[int] = 200, CHUNK_SIZE: int = 20000,
                          verbose: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """
    Safe load CSV files one-by-one using chunked reads and skipping very large files.
    - MAX_FILE_MB: skip files > this size (MB). Set to None to disable skipping.
    - CHUNK_SIZE: pandas chunksize for streaming.
    - max_rows: global cap across all files.
    - verbose: prints lightweight progress (or use logging)
    Returns (df_concat, used_files)
    """
    dfs = []
    used = []
    rows_acc = 0

    for fpath in files:
        try:
            # Skip oversized files
            if MAX_FILE_MB is not None:
                try:
                    size_mb = Path(fpath).stat().st_size / 1024.0 / 1024.0
                except Exception:
                    size_mb = None
                if size_mb is not None and size_mb > MAX_FILE_MB:
                    if verbose:
                        print(f"[dataset_loader] SKIP {fpath} ({size_mb:.1f} MB) > {MAX_FILE_MB} MB")
                    continue

            if verbose:
                print(f"[dataset_loader] STREAM {fpath} (chunksize={CHUNK_SIZE})")

            # Stream file in chunks
            pieces = []
            try:
                reader = pd.read_csv(fpath, chunksize=CHUNK_SIZE, low_memory=True)
            except Exception:
                # If pandas cannot stream (weird format), fallback to normal read (nrows trimmed later)
                try:
                    df_file = pd.read_csv(fpath, nrows=min(CHUNK_SIZE, max_rows or CHUNK_SIZE))
                    pieces = [df_file]
                except Exception:
                    # Give up on this file
                    if verbose:
                        print(f"[dataset_loader] Error opening {fpath}; skipping")
                    continue
            else:
                for chunk in reader:
                    if chunk is None or len(chunk) == 0:
                        continue
                    # Trim chunk if global max_rows applies
                    if max_rows is not None:
                        remaining = max_rows - rows_acc
                        if remaining <= 0:
                            break
                        if len(chunk) > remaining:
                            chunk = chunk.iloc[:remaining]
                    pieces.append(chunk)
                    rows_acc += len(chunk)
                    if max_rows is not None and rows_acc >= max_rows:
                        break

            if not pieces:
                # Fallback light-read to capture header/first rows
                try:
                    df_file = pd.read_csv(fpath, nrows=min(100, max_rows or 100), low_memory=True)
                except Exception:
                    if verbose:
                        print(f"[dataset_loader] No usable data in {fpath}; skipping")
                    continue
                else:
                    pieces = [df_file]

            df_file = pd.concat(pieces, ignore_index=True)
            dfs.append(df_file)
            used.append(fpath)

            if max_rows is not None and rows_acc >= max_rows:
                break

        except pd.errors.EmptyDataError:
            if verbose:
                print(f"[dataset_loader] Empty file: {fpath}")
            continue
        except Exception as e:
            if verbose:
                print(f"[dataset_loader] Error reading {fpath}: {e}")
            continue

    if not dfs:
        return pd.DataFrame(), []

    out = pd.concat(dfs, ignore_index=True)
    if max_rows is not None and len(out) > max_rows:
        out = out.iloc[:max_rows]
    return out, used

def load_dataset(path: str, max_rows: Optional[int] = None, pad_to: int = 8,
                 normalize_can_id: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load dataset file (CSV, Parquet) or directory and return X, y, meta.

    X will contain features:
      [can_id_norm, dlc, byte0..byte7, bytes_mean, bytes_std, ts_delta]
    If can_id not numeric, it will be mapped.
    y is 0/1 for attack; if attack column missing returns -1.

    pad_to: number of payload bytes to pad/truncate to (default 8).
    """
    meta: Dict[str, Any] = {"requested_path": path}

    # ----- PRESET FOLDER RESOLUTION -----
    # Maps short names (as UI sends) to real directories/files
    PRESETS = {
    # point UI logical name to the *small* dev sample to avoid freezes
    "ocs-lab": "datasets/ocs-lab-dev",
    "ocs": "datasets/ocs-lab-dev",
    # keep other fallbacks
    "datasets": "backend/datasets",
    "tmp": "backend/datasets",
}
    try:
        low = str(path).lower()
        if low in PRESETS:
            path = PRESETS[low]
    except Exception:
        # If anything odd happens, continue with original path
        pass
    # ------------------------------------

    df = None

    # If path is a directory -> discover CSVs/parquets and concat
    if os.path.isdir(path):
        files = _gather_files_from_dir(path)
        path_to_report = path
        if not files:
            # try nested data/ocs-lab form (e.g. 'ocs-lab' -> backend/data/ocs-lab)
            alt = os.path.join("backend", "data", path)
            if os.path.isdir(alt):
                files = _gather_files_from_dir(alt)
                path_to_report = alt
        if not files:
            raise FileNotFoundError(f"No CSV/Parquet files found in directory: {path}")
        df, used = _load_and_concat_csvs(files, max_rows=max_rows)
        if df is None or df.empty:
            raise FileNotFoundError(f"No usable CSV files in directory: {path_to_report}")
        meta["resolved_path"] = files if len(files) > 1 else (files[0] if files else path_to_report)
    else:
        # path is likely a file or a preset name
        p = Path(path)
        if not p.exists():
            # try common fallbacks: backend/data/<path> or backend/datasets/<path>.csv
            candidates = [
                Path("backend") / "data" / path,
                Path("backend") / "datasets" / f"{path}.csv",
                Path("backend") / "datasets" / f"{path}",
                Path(path),
            ]
            resolved = None
            for c in candidates:
                if c.exists():
                    resolved = c
                    break
            if resolved:
                p = resolved
            else:
                # if path looks like a directory name under backend/data, try that too:
                alt_dir = Path("backend") / "data" / path
                if alt_dir.exists() and alt_dir.is_dir():
                    files = _gather_files_from_dir(str(alt_dir))
                    if files:
                        df, used = _load_and_concat_csvs(files, max_rows=max_rows)
                        if df is None or df.empty:
                            raise FileNotFoundError(f"No usable CSV files in directory: {alt_dir}")
                        meta["resolved_path"] = files if len(files) > 1 else files[0]
                    else:
                        raise FileNotFoundError(f"No CSV/Parquet files found in directory: {alt_dir}")
                else:
                    raise FileNotFoundError(path)
        # Now p should point to an existing file (or we already loaded df)
        if "resolved_path" not in meta:
            p = Path(p)
            ext = p.suffix.lower()
            if ext == ".parquet":
                df = pd.read_parquet(str(p))
                meta["resolved_path"] = str(p)
            elif ext in (".csv", ".tsv", ".txt"):
                # read csv with fallback
                try:
                    sep = "," if ext == ".csv" else None
                    df = _read_csv_with_fallback(str(p), nrows=None, sep=sep)
                except Exception as e:
                    # if reading as CSV fails, attempt raw-line parsing
                    rows = []
                    with open(str(p), "r", encoding="utf-8", errors="ignore") as fh:
                        for i, line in enumerate(fh):
                            if max_rows is not None and i >= max_rows:
                                break
                            parsed = _parse_raw_line(line)
                            if parsed:
                                parsed["attack"] = parsed.get("attack", 0)
                                rows.append(parsed)
                    if rows:
                        df = pd.DataFrame(rows).fillna(value=np.nan)
                        meta["resolved_path"] = str(p)
                    else:
                        raise
                else:
                    meta["resolved_path"] = str(p)
            else:
                # attempt to read as CSV anyway
                try:
                    df = _read_csv_with_fallback(str(p), nrows=None)
                    meta["resolved_path"] = str(p)
                except Exception:
                    # try reading as raw text lines
                    rows = []
                    with open(str(p), "r", encoding="utf-8", errors="ignore") as fh:
                        for i, line in enumerate(fh):
                            if max_rows is not None and i >= max_rows:
                                break
                            parsed = _parse_raw_line(line)
                            if parsed:
                                parsed["attack"] = parsed.get("attack", 0)
                                rows.append(parsed)
                    if rows:
                        df = pd.DataFrame(rows).fillna(value=np.nan)
                        meta["resolved_path"] = str(p)
                    else:
                        raise FileNotFoundError(f"Unable to interpret file: {p}")

    # At this point `df` should be a pandas DataFrame
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Unable to load dataset as a DataFrame.")

    # ------------------------------------------------------------
    # NEW: handle case where CSV was read as a single text column
    # (common for OCS-Lab files that are raw logs placed into a .csv)
    # If DataFrame has exactly one column and the column name or first
    # values appear to be raw log lines, parse each row with _parse_raw_line
    # and convert into a DataFrame with ts, can_id, dlc, data columns.
    try:
        if df.shape[1] == 1:
            single_col_name = str(df.columns[0])
            # sample a few values to detect raw-line format
            sample_vals = df.iloc[:5, 0].astype(str).tolist() if len(df) >= 1 else []
            looks_like_raw = False
            if "timestamp" in single_col_name.lower() or "id:" in single_col_name.lower():
                looks_like_raw = True
            else:
                for v in sample_vals:
                    if v and ("Timestamp" in v or "ID:" in v or re.search(r"\b[0-9a-fA-F]{2}\b", v)):
                        looks_like_raw = True
                        break
            if looks_like_raw:
                parsed_rows = []
                for i, val in enumerate(df.iloc[:, 0].astype(str)):
                    parsed = _parse_raw_line(val)
                    if parsed:
                        parsed_rows.append(parsed)
                    else:
                        # keep an empty / placeholder row so lengths align
                        parsed_rows.append({"ts": None, "can_id": None, "dlc": None, "data": []})
                df = pd.DataFrame(parsed_rows).fillna(value=np.nan)
                # update resolved_path to indicate we parsed a single-column source
                if "resolved_path" in meta and isinstance(meta["resolved_path"], str):
                    meta["resolved_path"] = meta["resolved_path"]
                else:
                    meta["resolved_path"] = meta.get("resolved_path", "in-memory (parsed-raw-column)")
    except Exception:
        # non-fatal — continue with original df if parsing fails
        pass
    # ------------------------------------------------------------

    # For safety, ensure resolved_path present in meta
    if "resolved_path" not in meta:
        meta["resolved_path"] = "in-memory"

    # try flexible column names
    col_map = {c.lower(): c for c in df.columns}

    def get_col(*cands):
        for c in cands:
            if c in col_map:
                return df[col_map[c]]
        return None

    ts_col = get_col("ts", "timestamp", "time", "timestamp_s")
    can_col = get_col("can_id", "id", "canid", "can")
    dlc_col = get_col("dlc", "length")
    data_col = get_col("data", "payload", "data_hex")
    attack_col = get_col("attack", "label", "is_attack", "y")

    rows = []
    # compute how many rows to iterate
    total_df_rows = len(df)
    Nrows = total_df_rows if max_rows is None else min(total_df_rows, max_rows)

    for i in range(Nrows):
        row = {}
        if ts_col is not None:
            try:
                row["ts"] = float(ts_col.iloc[i]) if not pd.isna(ts_col.iloc[i]) else None
            except Exception:
                row["ts"] = None
        if can_col is not None:
            try:
                row_can = can_col.iloc[i]
                if pd.isna(row_can):
                    row["can_id"] = None
                else:
                    try:
                        row["can_id"] = int(str(row_can), 0)
                    except Exception:
                        try:
                            row["can_id"] = int(str(row_can), 16)
                        except Exception:
                            row["can_id"] = None
            except Exception:
                row["can_id"] = None
        else:
            row["can_id"] = None

        if dlc_col is not None:
            try:
                dlc_val = dlc_col.iloc[i]
                row["dlc"] = int(dlc_val) if not pd.isna(dlc_val) else None
            except Exception:
                row["dlc"] = None
        else:
            row["dlc"] = None

        if data_col is not None:
            try:
                row["data"] = _parse_data_field(data_col.iloc[i])
            except Exception:
                row["data"] = []
        else:
            # attempt to parse raw textual fields in DataFrame row (some CSVs embed raw line)
            # check common textual columns if present
            parsed_any = None
            for c in df.columns:
                val = df.iloc[i][c]
                if isinstance(val, str) and ("ID:" in val or "Timestamp" in val or re.search(r"\b[0-9a-fA-F]{2}\b", val)):
                    parsed_any = _parse_raw_line(str(val))
                    break
            if parsed_any:
                row["data"] = parsed_any.get("data", [])
                if "ts" in parsed_any and parsed_any["ts"] is not None:
                    row["ts"] = parsed_any["ts"]
                if "can_id" in parsed_any and parsed_any["can_id"] is not None:
                    row["can_id"] = parsed_any["can_id"]
            else:
                row["data"] = []

        if attack_col is not None:
            try:
                val = attack_col.iloc[i]
                # handle booleans/strings/numeric
                if pd.isna(val):
                    row["attack"] = -1
                else:
                    try:
                        row["attack"] = 1 if int(val) else 0
                    except Exception:
                        sval = str(val).strip().lower()
                        row["attack"] = 1 if sval in ("1", "true", "yes", "attack") else 0
            except Exception:
                row["attack"] = -1
        else:
            row["attack"] = -1

        rows.append(row)

    df2 = pd.DataFrame(rows).fillna(value=np.nan)

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

    # canonicalize can_id: fill missing with 0
    if "can_id" not in df2.columns:
        df2["can_id"] = 0
    df2["can_id"] = df2["can_id"].fillna(0).astype(float).astype(int)

    # feature arrays
    N = len(df2)
    byte_arr = np.zeros((N, pad_to), dtype=np.int32)
    dlc_arr = np.zeros((N,), dtype=np.int32)
    data_series = df2.get("data", pd.Series([[] for _ in range(N)]))

    for i, d in enumerate(data_series):
        b = d if isinstance(d, (list, tuple)) else []
        try:
            b = [int(x) for x in b]
        except Exception:
            # try parse fallback
            b = _parse_data_field(str(b))
        b = b[:pad_to] + [0] * max(0, pad_to - len(b))
        byte_arr[i, :] = np.array(b, dtype=np.int32)
        # compute dlc fallback
        try:
            if "dlc" in df2.columns and not pd.isna(df2.iloc[i].get("dlc", None)):
                dlc_arr[i] = int(df2.iloc[i].get("dlc", 0))
            else:
                dlc_arr[i] = min(len([x for x in b if x != 0]), pad_to)
        except Exception:
            dlc_arr[i] = min(len([x for x in b if x != 0]), pad_to)

    # bytes stats
    bytes_mean = byte_arr.mean(axis=1) if N > 0 else np.zeros((N,))
    bytes_std = byte_arr.std(axis=1) if N > 0 else np.zeros((N,))

    # normalize can_id mapping to [0,1] if requested
    can_ids = df2["can_id"].astype(int).to_numpy()
    if normalize_can_id:
        if len(can_ids) > 0:
            min_id, max_id = int(can_ids.min()), int(can_ids.max())
            if max_id == min_id:
                can_norm = np.zeros_like(can_ids, dtype=float)
            else:
                can_norm = (can_ids - min_id) / max(1, (max_id - min_id))
        else:
            can_norm = np.zeros_like(can_ids, dtype=float)
    else:
        can_norm = can_ids.astype(float)

    # assemble feature matrix
    features = np.concatenate([
        can_norm.reshape(-1, 1).astype(float),
        dlc_arr.reshape(-1, 1).astype(float),
        byte_arr.astype(float),
        bytes_mean.reshape(-1, 1),
        bytes_std.reshape(-1, 1),
        df2["ts_delta"].to_numpy().reshape(-1, 1).astype(float),
    ], axis=1) if N > 0 else np.zeros((0, 2 + pad_to + 3), dtype=float)

    # y
    if "attack" in df2.columns:
        try:
            y = df2["attack"].astype(int).to_numpy()
        except Exception:
            y = np.array([int(bool(x)) for x in df2["attack"].fillna(-1).to_list()], dtype=int)
    else:
        y = np.full((N,), -1, dtype=int)  # unknown

    meta_out = {
        "df": df2,
        "pad_to": pad_to,
        "feature_names": ["can_norm", "dlc"] + [f"byte{i}" for i in range(pad_to)] + ["bytes_mean", "bytes_std", "ts_delta"],
        "resolved_path": meta.get("resolved_path", None),
        "rows": int(N),
    }
    return features, y, meta_out

# convenience small loader when run directly
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()
    X, y, meta = load_dataset(args.path, max_rows=args.max_rows)
    print("X.shape", X.shape, "y.shape", y.shape)
    print("feature names:", meta["feature_names"])
