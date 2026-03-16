"""
backend/loader_helper.py

Improved subprocess loader helper.

- Spawns loader_worker.py with the provided module path.
- Waits for worker to finish.
- If output pickle cannot be read, attempts to include worker stdout/stderr
  and any .err file contents in the raised RuntimeError for clearer diagnostics.
- Waits briefly for the output file to reach a minimal size before reading to
  reduce EOFErrors on Windows due to quick process termination / filesystem lags.
"""

import subprocess
import sys
import tempfile
import pickle
import os
from pathlib import Path
from typing import Optional, Tuple, Any
import time

def _read_file_if_exists(path: str) -> Optional[str]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
    except Exception:
        try:
            with open(path, "rb") as fh:
                return fh.read().decode("utf-8", errors="replace")
        except Exception:
            return None
    return None

def load_dataset_with_timeout(dataset_arg: str, max_rows: Optional[int] = None, timeout: int = 60) -> Tuple[Any, Any, dict]:
    """
    Spawn a subprocess to run backend/loader_worker.py which calls dataset_loader.load_dataset.
    Returns (X, y, meta) on success.
    Raises TimeoutError on timeout or RuntimeError with loader traceback on failure.
    """

    backend_dir = Path(__file__).resolve().parent
    worker = backend_dir / "loader_worker.py"
    if not worker.exists():
        raise FileNotFoundError(f"loader_worker.py not found in backend directory ({backend_dir}). Subprocess loader unavailable.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    tmp.close()
    out_path = tmp.name

    # IMPORTANT: include module_path (backend_dir) as first arg to loader_worker.py
    module_path = str(backend_dir)

    cmd = [sys.executable, str(worker), module_path, dataset_arg, str(max_rows or ""), out_path]

    # Launch subprocess and capture stdout/stderr
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        # attempt to capture any output
        try:
            so, se = proc.communicate(timeout=5)
        except Exception:
            so, se = "", ""
        raise TimeoutError(f"Dataset loading timed out after {timeout} seconds (dataset={dataset_arg}).\nSTDOUT:\n{so}\nSTDERR:\n{se}")

    # at this point process finished; capture stdout/stderr
    try:
        so, se = proc.communicate(timeout=1)
    except Exception:
        so, se = "", ""

    # If the worker wrote an .err file, include it
    err_path = out_path + ".err"
    if os.path.exists(err_path):
        err_txt = _read_file_if_exists(err_path) or ""
        # cleanup err file
        try:
            os.remove(err_path)
        except Exception:
            pass
        raise RuntimeError(f"Loader subprocess reported an error. .err contents:\n{err_txt}\n\nWorker STDOUT:\n{so}\n\nWorker STDERR:\n{se}")

    # Wait briefly until the pickle file exists and is non-empty (avoid EOFError race)
    max_wait = 3.0  # seconds
    waited = 0.0
    sleep_step = 0.05
    min_size = 64  # bytes; pick a sensible minimal size for a valid pickle
    while waited < max_wait:
        if os.path.exists(out_path):
            try:
                size = os.path.getsize(out_path)
                if size >= min_size:
                    break
            except Exception:
                pass
        time.sleep(sleep_step)
        waited += sleep_step

    # If file doesn't exist or too small, include worker stdout/stderr in error
    if not os.path.exists(out_path) or (os.path.exists(out_path) and os.path.getsize(out_path) < 1):
        raise RuntimeError(f"Loader subprocess finished but output file is missing or empty. Worker STDOUT:\n{so}\n\nWorker STDERR:\n{se}")

    # Read the result pickle
    try:
        with open(out_path, "rb") as fh:
            res = pickle.load(fh)
    except Exception as e:
        # include stdout/stderr for diagnostics
        raise RuntimeError(f"Failed to read loader output: {e}\n\nWorker STDOUT:\n{so}\n\nWorker STDERR:\n{se}")

    # cleanup
    try:
        os.remove(out_path)
    except Exception:
        pass

    return res["X"], res["y"], res["meta"]
