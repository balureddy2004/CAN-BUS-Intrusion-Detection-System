# scripts/test_load.py
import sys, pprint
# ensure repo root in path
sys.path.insert(0, r"C:\Code\CAN_IDS_PROJECT")
from backend.train_job import _fallback_load_dataset

X, y, meta = _fallback_load_dataset("ocs-lab")
print("== fallback_load_dataset('ocs-lab') result ==")
print("X type:", type(X))
print("y type:", type(y))
print("X shape / len:", getattr(X, "shape", None) or (len(X) if X is not None else None))
print("y shape / len:", getattr(y, "shape", None) or (len(y) if y is not None else None))
print("meta:")
pprint.pprint(meta)
