# scripts/test_direct_loader.py
from backend.dataset_loader import load_dataset
import pprint

def main():
    path = "backend/test_small.csv"
    X, y, meta = load_dataset(path, max_rows=1000)
    print("DIRECT LOADER:")
    print("  Loaded rows (meta['rows']):", meta.get("rows"))
    print("  Resolved path:", meta.get("resolved_path"))
    print("  Feature shape:", getattr(X, "shape", None))
    print("  Labels:", y)
    pprint.pprint({"feature_names": meta.get("feature_names"), "pad_to": meta.get("pad_to")})

if __name__ == "__main__":
    main()
