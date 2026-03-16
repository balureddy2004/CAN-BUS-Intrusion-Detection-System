# backend/loader_worker.py
"""
Run from a subprocess to safely call dataset_loader.load_dataset()
and write results to a pickle file. This process can be killed by the parent
if it exceeds the timeout.

Usage:
    python loader_worker.py <module_path> <dataset_arg> <max_rows_or_empty> <out_path>

Behavior / safety notes:
 - module_path must be an existing directory (where backend/ lives). We only
   insert that directory at the front of sys.path (no other sys.path surgery).
 - We attempt to import `dataset_loader` from that path; if not importable we fail fast.
 - The call to dataset_loader.load_dataset is wrapped in try/except and any
   traceback is written to <out_path>.err so the parent process can surface it.
 - Results are written atomically: we pickle to a temp file and then os.replace()
   to the requested out_path so the caller sees either full result or nothing.
 - We explicitly call with allow_large_files=False to enforce file-size guards
   implemented in the loader.
"""

import sys
import os
import pickle
import traceback
import tempfile
from pathlib import Path
from importlib import import_module, reload

def _write_err(out_path: str, tb: str):
    try:
        err_path = out_path + ".err"
        with open(err_path, "w", encoding="utf-8") as fh:
            fh.write(tb)
    except Exception:
        # best-effort; nothing more we can do here
        pass

def main():
    if len(sys.argv) < 5:
        print("USAGE: python loader_worker.py <module_path> <dataset_arg> <max_rows_or_empty> <out_path>")
        sys.exit(2)

    module_path_raw, dataset_arg, max_rows_raw, out_path = sys.argv[1:5]
    max_rows = int(max_rows_raw) if max_rows_raw else None

    # Normalize and validate module_path
    module_path = os.path.abspath(module_path_raw)
    if not os.path.exists(module_path) or not os.path.isdir(module_path):
        tb = f"Provided module_path does not exist or is not a directory: {module_path}\n"
        _write_err(out_path, tb)
        print("ERROR")
        sys.exit(3)

    # Insert module_path at front of sys.path (temporary for this process)
    sys.path.insert(0, module_path)

    try:
        # Import dataset_loader from the provided module_path
        # Attempt reload if already present to ensure latest code from that path
        try:
            dataset_loader = import_module("dataset_loader")
            reload(dataset_loader)
        except Exception as imp_e:
            # If import fails, write traceback and exit
            tb = "Failed importing 'dataset_loader' from module_path.\n" + traceback.format_exc()
            _write_err(out_path, tb)
            print("ERROR")
            sys.exit(4)

        # Ensure the loader has the expected function
        if not hasattr(dataset_loader, "load_dataset"):
            tb = "module 'dataset_loader' does not expose load_dataset(path, ...)\n"
            _write_err(out_path, tb)
            print("ERROR")
            sys.exit(5)

        # Call the loader with conservative safety flags
        try:
            X, y, meta = dataset_loader.load_dataset(
                dataset_arg,
                max_rows=max_rows,
                allow_large_files=False
            )
        except TypeError:
            # Some variants of loaders may not accept allow_large_files kwarg.
            # Fallback to calling with only (dataset_arg, max_rows)
            try:
                X, y, meta = dataset_loader.load_dataset(dataset_arg, max_rows=max_rows)
            except Exception:
                tb = "dataset_loader.load_dataset raised an exception (fallback call).\n" + traceback.format_exc()
                _write_err(out_path, tb)
                print("ERROR")
                sys.exit(6)
        except Exception:
            tb = "dataset_loader.load_dataset raised an exception.\n" + traceback.format_exc()
            _write_err(out_path, tb)
            print("ERROR")
            sys.exit(7)

        # Prepare result dict (ensure it is pickle-friendly)
        result = {"X": X, "y": y, "meta": meta}

        # Write atomically: pickle to temp file then replace
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pkl", prefix="loader_worker_", dir=str(Path(out_path).parent))
            os.close(tmp_fd)
            with open(tmp_path, "wb") as fh:
                pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
            # Atomic replace (works on POSIX and Windows)
            os.replace(tmp_path, out_path)
        except Exception:
            tb = "Failed writing result pickle atomically.\n" + traceback.format_exc()
            _write_err(out_path, tb)
            # cleanup tmp if exists
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            print("ERROR")
            sys.exit(8)

        # Success
        print("OK")
        sys.exit(0)

    except SystemExit:
        raise
    except Exception:
        tb = "Unhandled exception in loader_worker:\n" + traceback.format_exc()
        _write_err(out_path, tb)
        print("ERROR")
        sys.exit(9)
    finally:
        # Remove the module_path we inserted to avoid leaking modifications (not strictly necessary in subprocess)
        try:
            if sys.path and sys.path[0] == module_path:
                del sys.path[0]
        except Exception:
            pass

if __name__ == "__main__":
    main()
