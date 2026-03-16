# scripts/parse_test.py
"""
Quick smoke-test for raw-line parser in backend.dataset_loader.
Run from project root with virtualenv active:
(.venv) python scripts/parse_test.py
"""
import sys
from pathlib import Path

# ensure project root is on sys.path so `backend` package is importable
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]  # project root (one level above scripts/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from backend.dataset_loader import _parse_raw_line
except Exception as e:
    print("ERROR: cannot import backend.dataset_loader:", e)
    sys.exit(2)

candidate = ROOT / "backend" / "data" / "ocs-lab" / "DoS_attack_dataset.csv"
if not candidate.exists():
    print("ERROR: expected file not found:", candidate)
    sys.exit(3)

print("=== reading first 20 lines and parsing with _parse_raw_line ===")
with open(candidate, "r", encoding="utf-8", errors="ignore") as fh:
    for i, L in enumerate(fh):
        if i >= 20:
            break
        parsed = _parse_raw_line(L.rstrip("\n"))
        print(f"{i+1:02d} -> parsed:", parsed)
