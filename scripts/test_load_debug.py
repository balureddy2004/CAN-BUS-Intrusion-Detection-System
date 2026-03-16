# scripts/test_load_debug.py
"""
Run a short diagnostic of load_dataset and the CSV/raw fallback.
Run from project root:
(.venv) python scripts/test_load_debug.py
"""
import sys
from pathlib import Path
import pprint

# ensure project root is on sys.path
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from backend.dataset_loader import load_dataset, _read_csv_with_fallback
except Exception as e:
    print("ERROR: cannot import backend.dataset_loader:", e)
    sys.exit(2)

# Candidate path (explicit file) to inspect via fallback reader
candidate = ROOT / "backend" / "data" / "ocs-lab" / "DoS_attack_dataset.csv"
print("project root:", ROOT)
print("candidate file:", candidate)
print("exists:", candidate.exists())

print("\n=== trying _read_csv_with_fallback on resolved file (first 20 rows) ===")
try:
    df = _read_csv_with_fallback(str(candidate), nrows=20)
    print("read_csv_with_fallback -> shape:", getattr(df, "shape", None))
    print("columns:", list(df.columns)[:12])
    print("first 8 rows preview:")
    # to_string avoids ellipsis
    print(df.head(8).to_string())
except Exception as e:
    print("read_csv_with_fallback failed:", e)

print("\n=== trying load_dataset with max_rows=200 ===")
try:
    X, y, meta = load_dataset("ocs-lab", max_rows=200)
    print("load_dataset -> X type:", type(X), "shape:", getattr(X, "shape", None))
    print("load_dataset -> y type:", type(y), "shape:", getattr(y, "shape", None))
    print("meta keys:", list(meta.keys()))
    pprint.pprint({k: meta.get(k) for k in ("resolved_path", "rows")})
    if isinstance(meta.get("df"), type(None)):
        print("meta['df'] is not present")
    else:
        print("meta['df'] head:")
        print(meta.get("df").head(6).to_string())
except Exception as e:
    print("load_dataset raised:", e)
