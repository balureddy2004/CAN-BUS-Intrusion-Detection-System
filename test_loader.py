# scripts/test_load.py
"""
Simple test runner for backend.dataset_loader.load_dataset

Run from project root while your virtualenv is active:
(.venv) python scripts/test_load.py ocs-lab
"""

from __future__ import annotations
import sys
import os
import traceback

# ---- Robust path setup ----
# Determine project root (folder that contains "backend", "config", "scripts", etc).
# This should work whether the script is executed from project root or another cwd.
SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
# If this file is in scripts/, project root is parent of scripts/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "scripts" else SCRIPT_DIR

# Ensure both project root and backend folder are on sys.path (helps runtime imports).
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ---- Import the loader with fallbacks ----
load_dataset = None
try:
    # Primary preferred import (when backend is a package or in sys.path)
    from backend.dataset_loader import load_dataset  # type: ignore
    print("[test_load] imported backend.dataset_loader via package import")
except Exception:
    try:
        # Secondary fallback: try importing plain module name if present ("dataset_loader.py" on sys.path)
        from dataset_loader import load_dataset  # type: ignore
        print("[test_load] imported dataset_loader as top-level module")
    except Exception:
        # Final fallback: load module directly from backend/dataset_loader.py via importlib
        import importlib.util
        module_path = os.path.join(BACKEND_DIR, "dataset_loader.py")
        if os.path.exists(module_path):
            try:
                spec = importlib.util.spec_from_file_location("backend.dataset_loader", module_path)
                module = importlib.util.module_from_spec(spec)  # type: ignore
                loader = spec.loader  # type: ignore
                if loader is None:
                    raise ImportError("spec.loader is None")
                loader.exec_module(module)  # type: ignore
                # expose load_dataset from loaded module
                load_dataset = getattr(module, "load_dataset", None)
                if callable(load_dataset):
                    print(f"[test_load] dynamically loaded dataset_loader from {module_path}")
                else:
                    raise ImportError("load_dataset not found in loaded module")
            except Exception:
                print("[test_load] dynamic import from backend/dataset_loader.py failed")
                traceback.print_exc()
        else:
            print(f"[test_load] expected module not found at: {module_path}")

# if load_dataset still unresolved, print helpful diagnostics and exit
if "load_dataset" not in globals() or not callable(load_dataset):
    print("ERROR: cannot import load_dataset from backend.dataset_loader or dataset_loader")
    print("Paths tried (project root, backend):")
    print("  PROJECT_ROOT:", PROJECT_ROOT)
    print("  BACKEND_DIR :", BACKEND_DIR)
    print("sys.path (first 10 items):")
    for p in sys.path[:10]:
        print(" ", p)
    traceback.print_stack()
    sys.exit(2)

# ---- test runner logic ----
def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "ocs-lab"
    print(f"== running load_dataset('{arg}') ==")
    try:
        X, y, meta = load_dataset(arg, max_rows=None)
        print("== load_dataset result ==")
        print("X type:", type(X))
        try:
            print("X shape / len:", X.shape if hasattr(X, "shape") else (len(X) if X is not None else None))
        except Exception:
            print("X length: (error reading shape)")

        print("y type:", type(y))
        try:
            print("y shape / len:", y.shape if hasattr(y, "shape") else (len(y) if y is not None else None))
        except Exception:
            print("y length: (error reading shape)")

        print("meta:")
        for k, v in (meta or {}).items():
            print(" ", k, ":", v)

        # if X or y look empty, dump the first 20 raw lines of the resolved source for inspection
        if (X is None or (hasattr(X, "shape") and getattr(X, "shape")[0] == 0)) and meta and meta.get("resolved_path"):
            rp = meta["resolved_path"]
            # resolved_path may be list or single path
            if isinstance(rp, list) and rp:
                rp = rp[0]
            if isinstance(rp, str) and os.path.exists(rp):
                print("\n--- dumping first 20 lines of resolved_path for inspection:", rp, "---\n")
                with open(rp, "r", encoding="utf-8", errors="ignore") as fh:
                    for i, L in enumerate(fh):
                        if i >= 20:
                            break
                        print(i + 1, L.rstrip())
                # also write a debug file for easy copy-paste
                outfn = os.path.join(os.getcwd(), "debug_head.txt")
                with open(outfn, "w", encoding="utf-8") as out:
                    with open(rp, "r", encoding="utf-8", errors="ignore") as fh:
                        for i, L in enumerate(fh):
                            if i >= 200:
                                break
                            out.write(L)
                print(f"\nWrote first 200 lines to {outfn} for inspection.")
        return 0
    except Exception:
        print("load_dataset raised an exception:")
        traceback.print_exc()
        return 3

if __name__ == "__main__":
    sys.exit(main())
