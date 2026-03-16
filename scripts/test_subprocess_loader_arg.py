# scripts/test_subprocess_loader_arg.py

import sys
from backend.loader_helper import load_dataset_with_timeout

if len(sys.argv) < 2:
    print("Usage: python test_subprocess_loader_arg.py <path>")
    sys.exit(2)

path = sys.argv[1]
X, y, meta = load_dataset_with_timeout(path, max_rows=10000, timeout=60)

print("X.shape:", getattr(X, "shape", None))
print("resolved_path:", meta.get("resolved_path"))
print("y sample:", y[:20] if hasattr(y, "__len__") else y)
print("meta keys:", list(meta.keys()))
