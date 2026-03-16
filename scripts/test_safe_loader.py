# scripts/test_safe_loader.py
from backend.dataset_loader_safe import safe_load_dataset
import pprint

def main():
    # safe_load_dataset accepts a file or directory or short name -> resolves under ./datasets if needed
    target = "backend/test_small.csv"
    meta = safe_load_dataset(target, verbose=True, max_file_size_mb=1, sample_rows_per_file=5000)
    print("SAFE LOADER:")
    print("  resolved_path:", meta.get("resolved_path"))
    print("  processed files:", meta.get("files"))
    print("  skipped:", meta.get("skipped"))
    print("  rows returned:", meta.get("rows"))
    print("  df columns:", list(meta.get("df").columns) if meta.get("df") is not None else None)
    pprint.pprint({k: meta.get(k) for k in ("files","rows","skipped")})

if __name__ == "__main__":
    main()
