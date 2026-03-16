# backend/loader_helper.py
import subprocess
import sys
import tempfile
import pickle
import os
from pathlib import Path
from typing import Optional, Tuple, Any
import time

def load_dataset_with_timeout(dataset_arg: str, max_rows: Optional[int] = None, timeout: int = 60) -> Tuple[Any, Any, dict]:
    """
    Spawn a subprocess to run backend/loader_worker.py which calls dataset_loader.load_dataset.
    Returns (X, y, meta) on success.
    Raises TimeoutError on timeout or RuntimeError with loader traceback on failure.
    """
    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))  # backend dir
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    tmp.close()
    out_path = tmp.name

    # Build command: python backend/loader_worker.py <module_path> <dataset_arg> <max_rows> <out_path>
    cmd = [sys.executable, os.path.join(module_path, "loader_worker.py"), module_path, dataset_arg, str(max_rows or ""), out_path]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        raise TimeoutError(f"Dataset loading timed out after {timeout} seconds (dataset={dataset_arg})")

    # If worker wrote .err file, raise with its contents
    err_path = out_path + ".err"
    if os.path.exists(err_path):
        with open(err_path, "r", encoding="utf-8") as fh:
            tb = fh.read()
        # cleanup
        try:
            os.remove(err_path)
        except Exception:
            pass
        raise RuntimeError(f"Loader subprocess failed:\n{tb}")

    # Read the result pickle
    try:
        with open(out_path, "rb") as fh:
            res = pickle.load(fh)
    except Exception as e:
        raise RuntimeError(f"Failed to read loader output: {e}")

    # cleanup
    try:
        os.remove(out_path)
    except Exception:
        pass

    return res["X"], res["y"], res["meta"]
