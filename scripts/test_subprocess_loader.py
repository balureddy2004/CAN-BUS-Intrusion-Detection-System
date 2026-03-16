# scripts/test_subprocess_loader.py
from backend.loader_helper import load_dataset_with_timeout
import pprint

def main():
    path = "backend/test_small.csv"
    try:
        X, y, meta = load_dataset_with_timeout(path, max_rows=1000, timeout=15)
    except Exception as e:
        print("SUBPROCESS LOADER: FAILED:", e)
        raise
    print("SUBPROCESS LOADER:")
    print("  Rows (meta rows or len(X)):", meta.get("rows") or (len(X) if hasattr(X, "__len__") else None))
    print("  Resolved path:", meta.get("resolved_path"))
    print("  X.shape:", getattr(X, "shape", None))
    print("  y:", y)
    pprint.pprint(meta)

if __name__ == "__main__":
    main()
