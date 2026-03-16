# backend/train_job.py
"""
run_training(params, broadcast_train, record_fn, stop_event, pause_event, state, model_out_dir="backend/models")

- params: dict of hyperparams (model, epochs, batchSize, dataset, dataset_path, maxRows, ...)
- broadcast_train: callable(event_dict) -> used to send WS events to clients
- record_fn: callable(event_dict) -> used to persist events (optional)
- stop_event, pause_event: threading.Event-like objects
- state: shared state dict (mutate with state['state']=..., state['msg']=...)
- model_out_dir: where to save model files
"""

from __future__ import annotations
import os
import time
import logging
import glob
from typing import Any, Callable, Dict, Optional, Tuple, List

log = logging.getLogger("can-ids.train_job")

# ML lib flags
SKLEARN_OK = False
XGBOOST_OK = False
LGBM_OK = False
PANDAS_OK = False
NUMPY_OK = False

# Attempt light imports at module load to set capability flags.
# We keep the actual heavy aliases (e.g., pd, np) out of top-level namespace when possible.
try:
    import numpy as _np  # type: ignore
    NUMPY_OK = True
except Exception:
    NUMPY_OK = False

try:
    import sklearn  # noqa: F401
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import xgboost as _xgb  # type: ignore
    XGBOOST_OK = True
except Exception:
    XGBOOST_OK = False

try:
    import lightgbm as _lgb  # type: ignore
    LGBM_OK = True
except Exception:
    LGBM_OK = False

# pandas availability flag (we lazy-import pd only in fallback loader)
try:
    import pandas as _pd  # type: ignore
    PANDAS_OK = True
except Exception:
    PANDAS_OK = False

# If a subprocess-based loader helper exists, prefer it for isolation
try:
    from backend.loader_helper import load_dataset_with_timeout  # type: ignore
    _USE_SUBPROCESS_LOADER = True
except Exception:
    load_dataset_with_timeout = None  # type: ignore
    _USE_SUBPROCESS_LOADER = False

# Direct dataset_loader import fallback (may be heavy)
try:
    from backend.dataset_loader import load_dataset as _direct_load_dataset  # type: ignore
except Exception:
    try:
        from dataset_loader import load_dataset as _direct_load_dataset  # fallback if in PATH
    except Exception:
        _direct_load_dataset = None

# Default safety cap for training rows
DEFAULT_MAX_TRAIN_ROWS = 50000

def _safe_broadcast(bcast: Optional[Callable[[Dict[str, Any]], None]], payload: Dict[str, Any]):
    try:
        if callable(bcast):
            bcast(payload)
    except Exception:
        log.exception("broadcast failed")

def _safe_record(record_fn: Optional[Callable[[Dict[str, Any]], None]], payload: Dict[str, Any]):
    try:
        if callable(record_fn):
            record_fn(payload)
    except Exception:
        log.exception("record_fn failed")

def _resolve_dataset_param(dataset_param: Optional[str]) -> Optional[str]:
    """
    Heuristic resolution for dataset identifiers to file or directory paths.
    Only checks a small set of known backend locations; does NOT attempt wide scanning.
    """
    if not dataset_param:
        return None

    ds = str(dataset_param).strip()
    # absolute or relative path exist?
    if os.path.isabs(ds) and os.path.exists(ds):
        return ds
    cand = os.path.normpath(os.path.join(os.getcwd(), ds))
    if os.path.exists(cand):
        return cand

    base_backend = os.path.join(os.getcwd(), "backend")
    candidates = [
        os.path.join(base_backend, "data"),
        os.path.join(base_backend, "datasets"),
        os.path.join(base_backend, "data", "ocs-lab"),
    ]

    lowered = ds.lower()
    if lowered in ("ocs-lab", "ocs_lab", "ocslab"):
        folder = os.path.join(base_backend, "data", "ocs-lab")
        if os.path.isdir(folder):
            # prefer known small sample files if present
            prefs = [
                "DoS_attack_dataset.csv", "Attack_free_dataset.csv",
                "Fuzzy_attack_dataset.csv", "Impersonation_attack_dataset.csv"
            ]
            for p in prefs:
                pth = os.path.join(folder, p)
                if os.path.exists(pth):
                    return pth
            # otherwise pick first csv in folder
            files = sorted(glob.glob(os.path.join(folder, "*.csv"))) + sorted(glob.glob(os.path.join(folder, "*.parquet")))
            if files:
                return files[0]

    # search candidate dirs for matching name
    basename = os.path.basename(ds)
    for base in candidates:
        pth = os.path.join(base, basename)
        if os.path.exists(pth):
            return pth
    for base in candidates:
        if not os.path.isdir(base):
            continue
        matches = sorted(glob.glob(os.path.join(base, f"*{ds}*.csv"))) + sorted(glob.glob(os.path.join(base, f"*{ds}*.parquet")))
        if matches:
            return matches[0]
    return None

# --------- pandas fallback loader --------------------------------------------
def _infer_label_column(df: "object") -> Optional[str]:
    """
    Infer a candidate label column name from a pandas DataFrame-like object.
    This function does a best-effort check; it expects a pandas.DataFrame when PANDAS_OK is True.
    """
    try:
        # lazy import inside function scope if needed
        import pandas as pd  # type: ignore
    except Exception:
        return None

    candidates = ["label", "attack", "target", "class", "is_attack", "y"]
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_cols:
            return lower_cols[cand]
    # fallback: column with 2-3 unique values
    for c in df.columns:
        try:
            nunique = df[c].nunique(dropna=True)
            if 2 <= nunique <= 3:
                return c
        except Exception:
            continue
    return None

def _fallback_load_dataset(dataset_param: Optional[str], max_rows: Optional[int] = None) -> Tuple[Optional[Any], Optional[Any], Dict[str, Any]]:
    """
    Use pandas to read CSV/Parquet files from resolved path and return X,y,meta.
    Conservative: reads full files only when needed and returns None if unable.
    Supports max_rows to limit memory usage.
    """
    meta: Dict[str, Any] = {}
    if not PANDAS_OK:
        log.debug("[train_job] pandas not available — cannot fallback load")
        return None, None, meta

    # Lazy import pandas (pd) and numpy (np) inside this function
    try:
        import pandas as pd  # type: ignore
    except Exception:
        log.debug("[train_job] pandas import failed in fallback loader")
        return None, None, meta
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore

    try:
        resolved = _resolve_dataset_param(dataset_param) if dataset_param else None
        if not resolved:
            log.debug("[train_job] fallback loader: no resolved path for dataset_param=%r", dataset_param)
            return None, None, meta

        # find files
        files = []
        if os.path.isfile(resolved):
            files = [resolved]
        elif os.path.isdir(resolved):
            files = sorted(glob.glob(os.path.join(resolved, "*.csv"))) + sorted(glob.glob(os.path.join(resolved, "*.parquet")))

        if not files:
            log.debug("[train_job] fallback loader: no csv/parquet found at %s", resolved)
            return None, None, meta

        dfs = []
        rows_accum = 0
        for fn in files:
            try:
                if fn.lower().endswith(".parquet"):
                    df = pd.read_parquet(fn)
                else:
                    # Use pandas read_csv with nrows when max_rows provided to avoid reading whole huge files unnecessarily.
                    if max_rows is not None and max_rows > 0:
                        # read only up to remaining rows needed
                        remaining = max_rows - rows_accum
                        if remaining <= 0:
                            break
                        df = pd.read_csv(fn, nrows=remaining)
                    else:
                        df = pd.read_csv(fn)
                dfs.append(df)
                rows_accum += len(df)
                # if we've reached max_rows across files, stop reading further files
                if max_rows is not None and max_rows > 0 and rows_accum >= max_rows:
                    break
            except Exception:
                log.exception("[train_job] failed to read dataset file: %s", fn)
                continue

        if not dfs:
            log.debug("[train_job] fallback loader: no readable files in %s", files)
            return None, None, meta

        df_all = pd.concat(dfs, ignore_index=True, sort=False)

        # apply head truncation just in case (defensive)
        if max_rows is not None and max_rows > 0 and len(df_all) > max_rows:
            df_all = df_all.head(max_rows)

        meta["source_files"] = files
        meta["rows"] = int(len(df_all))
        label_col = _infer_label_column(df_all)
        if not label_col:
            log.warning("[train_job] fallback loader: couldn't infer label column; columns=%s", list(df_all.columns)[:10])
            return None, None, meta

        y_ser = df_all[label_col].copy()
        X_df = df_all.drop(columns=[label_col])

        # encode non-numeric labels
        if y_ser.dtype == object or not getattr(y_ser.dtype, "kind", None) in ("i", "f"):
            try:
                y_ser, uniques = pd.factorize(y_ser, sort=True)
                meta["label_map"] = {str(v): i for i, v in enumerate(uniques)}
            except Exception:
                try:
                    y_ser = y_ser.astype(int)
                except Exception:
                    log.exception("[train_job] fallback loader: cannot coerce label to numeric")
                    return None, None, meta

        try:
            X = X_df.values.astype(float)
        except Exception:
            X = X_df.values
        y = np.asarray(y_ser) if np is not None else y_ser.to_numpy()
        return X, y, meta
    except Exception:
        log.exception("[train_job] fallback loader exception")
        return None, None, meta

# -------------------- model save util ---------------------------------------
def _save_model(clf: Any, path: str) -> bool:
    """
    Attempt to save model. Prefer joblib.dump. If fails, attempt library-specific saves.
    Returns True if any save succeeded.
    """
    try:
        import joblib  # type: ignore
        joblib.dump(clf, path)
        return True
    except Exception:
        pass
    try:
        # lazy check for xgboost/lightgbm methods
        if XGBOOST_OK and hasattr(clf, "save_model"):
            clf.save_model(path + ".xgb")
            return True
    except Exception:
        pass
    try:
        if LGBM_OK and hasattr(clf, "save_model"):
            clf.save_model(path + ".lgb")
            return True
    except Exception:
        pass
    return False

# -------------------- run_training ------------------------------------------
def run_training(params: Dict[str, Any],
                 broadcast_train: Optional[Callable[[Dict[str, Any]], None]] = None,
                 record_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
                 stop_event: Optional[Any] = None,
                 pause_event: Optional[Any] = None,
                 state: Optional[Dict[str, Any]] = None,
                 model_out_dir: str = "backend/models"):
    """
    Start a training job. Strategy:
      - Try subprocess loader (isolated) if available
      - Else try direct loader import
      - Else use pandas fallback
      - If ML libs available and data suitable -> run training (RF / XGBoost / LightGBM)
      - Else run a simulated training loop (progress + metrics) for UI
    """
    try:
        max_rows = int(params.get("maxRows", DEFAULT_MAX_TRAIN_ROWS)) if isinstance(params, dict) else DEFAULT_MAX_TRAIN_ROWS
        max_rows = min(max_rows, DEFAULT_MAX_TRAIN_ROWS)

        state = state or {}
        stop_event = stop_event or DummyEvent()
        pause_event = pause_event or DummyEvent()

        model_name = str(params.get("model", "RandomForest"))
        epochs = int(params.get("epochs", 10) or 10)
        batch_size = int(params.get("batchSize", 256) or 256)
        train_split = float(params.get("trainSplit", 0.8) or 0.8)
        dataset = params.get("dataset")
        seed = int(params.get("seed", 42) or 42)

        log.info("[train_job] start: model=%s epochs=%s batch=%s dataset=%s", model_name, epochs, batch_size, dataset)
        state.update({"state": "training", "msg": "starting"})
        _safe_broadcast(broadcast_train, {"type": "status", "state": "training", "msg": "starting"})

        # ------------------ NEW: ensure model output dir exists & writable ------------------
        try:
            os.makedirs(model_out_dir, exist_ok=True)
            try:
                os.chmod(model_out_dir, 0o755)
            except Exception:
                # best-effort; ignore on Windows or restricted systems
                pass
            # quick write test
            testf = os.path.join(model_out_dir, f".write_test_{int(time.time())}")
            with open(testf, "w") as fh:
                fh.write("ok")
            os.remove(testf)
        except PermissionError as pe:
            _safe_broadcast(broadcast_train, {"type": "error", "msg": f"PermissionError creating model_out_dir {model_out_dir}: {pe}"})
            state.update({"state": "error", "msg": f"PermissionError: {pe}"})
            log.exception("Permission error creating model_out_dir")
            return
        except Exception as e:
            _safe_broadcast(broadcast_train, {"type": "error", "msg": f"Error creating model_out_dir {model_out_dir}: {e}"})
            state.update({"state": "error", "msg": str(e)})
            log.exception("Error creating model_out_dir")
            return
        # -----------------------------------------------------------------------------------

        X, y, meta = None, None, {}

        # 1) Subprocess loader (preferred)
        if _USE_SUBPROCESS_LOADER and load_dataset_with_timeout is not None:
            try:
                resolved = None
                if isinstance(params.get("dataset_path"), str) and params.get("dataset_path"):
                    resolved = params.get("dataset_path")
                else:
                    resolved = _resolve_dataset_param(dataset) if dataset else None

                if resolved:
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Subprocess loader: resolved -> {resolved}"})
                    try:
                        X, y, meta = load_dataset_with_timeout(resolved, max_rows=max_rows, timeout=120)
                    except Exception:
                        log.exception("[train_job] subprocess loader failed")
                        _safe_broadcast(broadcast_train, {"type": "log", "msg": "Subprocess loader failed — falling back."})
                        X, y, meta = None, None, {}
                else:
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Subprocess loader: attempting with dataset param '{dataset}'"})
                    try:
                        X, y, meta = load_dataset_with_timeout(dataset, max_rows=max_rows, timeout=120)
                    except Exception:
                        log.exception("[train_job] subprocess loader failed for dataset param")
                        _safe_broadcast(broadcast_train, {"type": "log", "msg": "Subprocess loader failed — falling back."})
                        X, y, meta = None, None, {}
            except Exception:
                log.exception("[train_job] unexpected error in subprocess loader path")
                X, y, meta = None, None, {}

        # 2) Direct loader fallback
        if (X is None or y is None) and _direct_load_dataset is not None:
            try:
                resolved = None
                if isinstance(params.get("dataset_path"), str) and params.get("dataset_path"):
                    resolved = params.get("dataset_path")
                else:
                    resolved = _resolve_dataset_param(dataset) if dataset else None

                if resolved:
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Direct loader: resolved -> {resolved}"})
                    try:
                        X, y, meta = _direct_load_dataset(resolved, max_rows=max_rows)
                    except TypeError:
                        X, y, meta = _direct_load_dataset(resolved)
                else:
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Direct loader: trying dataset param '{dataset}'"})
                    try:
                        X, y, meta = _direct_load_dataset(dataset, max_rows=max_rows) if dataset else (None, None, {})
                    except TypeError:
                        X, y, meta = _direct_load_dataset(dataset) if dataset else (None, None, {})
            except Exception:
                log.exception("[train_job] direct loader failed")
                X, y, meta = None, None, {}

        # 3) pandas fallback
        if (X is None or y is None):
            try:
                X, y, meta = _fallback_load_dataset(dataset, max_rows=max_rows)
                if X is not None and y is not None:
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Pandas fallback loaded rows={len(X)}"})
            except Exception:
                log.exception("[train_job] pandas fallback failed")
                X, y, meta = None, None, {}

        # Announce load results
        try:
            rows_loaded = len(X) if hasattr(X, "__len__") else (meta.get("rows") if isinstance(meta, dict) else None)
        except Exception:
            rows_loaded = None
        _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Load result rows={rows_loaded}"})
        log.info("[train_job] load result rows=%s", rows_loaded)

        # Decide if we can run real training
        try_real = (SKLEARN_OK or XGBOOST_OK or LGBM_OK) and (X is not None) and (y is not None) and (len(y) >= 10) and (len(set(y)) > 1)

        if try_real:
            _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Preparing real training (model={model_name})."})
            log.info("[train_job] attempting real training with model=%s", model_name)

            # lazy imports for ML libs we need at training time
            try:
                import numpy as np  # type: ignore
            except Exception:
                np = None  # type: ignore

            try:
                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import confusion_matrix, accuracy_score
            except Exception:
                # If sklearn unexpectedly unavailable, fallback to simulated loop below
                log.exception("[train_job] sklearn import failed at training time")
                train_test_split = None
                RandomForestClassifier = None
                confusion_matrix = None
                accuracy_score = None

            # train/test split (attempt stratify when possible)
            stratify_obj = (y if (y is not None and len(set(y)) > 1) else None)
            try:
                if train_test_split is not None:
                    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_split, random_state=seed, stratify=stratify_obj)
                else:
                    raise Exception("train_test_split not available")
            except Exception:
                from sklearn.model_selection import train_test_split as _tt  # type: ignore
                X_train, X_val, y_train, y_val = _tt(X, y, train_size=train_split, random_state=seed)

            clf = None
            used_impl = "RandomForest"

            # Choose implementation
            if model_name.lower().startswith("xg"):
                if XGBOOST_OK:
                    used_impl = "XGBoost"
                    try:
                        import xgboost as xgb  # type: ignore
                        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=100, random_state=seed, n_jobs=-1)
                    except Exception:
                        clf = None
                else:
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": "XGBoost requested but not installed — using RandomForest."})
                    log.warning("XGBoost requested but not available; using RandomForest")
            elif model_name.lower().startswith("light") or model_name.lower().startswith("lgb"):
                if LGBM_OK:
                    used_impl = "LightGBM"
                    try:
                        import lightgbm as lgb  # type: ignore
                        clf = lgb.LGBMClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
                    except Exception:
                        clf = None
                else:
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": "LightGBM requested but not installed — using RandomForest."})
                    log.warning("LightGBM requested but not available; using RandomForest")

            if clf is None:
                if SKLEARN_OK:
                    used_impl = "RandomForest"
                    try:
                        from sklearn.ensemble import RandomForestClassifier as RF  # type: ignore
                        clf = RF(n_estimators=100, random_state=seed, n_jobs=-1)
                    except Exception:
                        clf = None
                else:
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": "No ML libs available - falling back to simulated training."})
                    log.warning("No ML libs available; skipping real training")
                    clf = None

            if clf is not None:
                rng = (_np.random.RandomState(seed) if NUMPY_OK else None)
                n_samples = X_train.shape[0] if hasattr(X_train, "shape") else len(X_train)
                cm_list = [[0, 0], [0, 0]]

                for ep in range(1, epochs + 1):
                    if getattr(stop_event, "is_set", lambda: False)():
                        _safe_broadcast(broadcast_train, {"type": "status", "state": "stopped", "msg": "stopped by user"})
                        state.update({"state": "stopped", "msg": "stopped by user"})
                        return
                    while getattr(pause_event, "is_set", lambda: False)():
                        _safe_broadcast(broadcast_train, {"type": "status", "state": "paused", "msg": "paused"})
                        time.sleep(0.5)

                    subset_size = max(1, min(n_samples, batch_size))
                    if rng is not None and n_samples > 0:
                        idx = rng.choice(n_samples, size=subset_size, replace=False)
                    else:
                        # fallback: simple sequential slice
                        idx = list(range(subset_size))

                    try:
                        clf.fit(X_train[idx], y_train[idx])
                    except Exception:
                        try:
                            clf.fit(X_train, y_train)
                        except Exception:
                            log.exception("[train_job] model fit failed at epoch %s", ep)

                    try:
                        y_pred = clf.predict(X_val)
                        acc = float(accuracy_score(y_val, y_pred))
                        cm = confusion_matrix(y_val, y_pred)
                        if cm is not None and getattr(cm, "shape", None) == (2, 2):
                            cm_list = cm.tolist()
                        else:
                            # coerce to 2x2 representation
                            a = [[0, 0], [0, 0]]
                            try:
                                for i in range(min(2, cm.shape[0])):
                                    for j in range(min(2, cm.shape[1])):
                                        a[i][j] = int(cm[i, j])
                                cm_list = a
                            except Exception:
                                cm_list = a
                    except Exception:
                        acc = 0.0
                        cm_list = [[0, 0], [0, 0]]

                    pct = float(100.0 * ep / max(1, epochs))
                    _safe_broadcast(broadcast_train, {"type": "metric", "epoch": ep, "loss": 0.0, "valLoss": 0.0, "accuracy": acc, "valAccuracy": acc})
                    _safe_broadcast(broadcast_train, {"type": "progress", "progress": pct, "epoch": ep})
                    _safe_record(record_fn, {"event": "metric", "epoch": ep, "accuracy": acc, "progress": pct})
                    time.sleep(0.5)

                # Save model
                os.makedirs(model_out_dir, exist_ok=True)
                model_path = os.path.join(model_out_dir, f"{used_impl}_{int(time.time())}.pkl")
                saved = _save_model(clf, model_path)
                if saved:
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Model saved to {model_path}"})
                else:
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": "Model save failed (no supported saver)"})

                _safe_broadcast(broadcast_train, {"type": "confusion", "matrix": cm_list, "labels": ["neg", "pos"]})
                state.update({"state": "completed", "msg": "training completed"})
                _safe_broadcast(broadcast_train, {"type": "status", "state": "completed", "msg": "training completed"})
                return

        # If we reach here, run simulated training loop (fallback)
        _safe_broadcast(broadcast_train, {"type": "log", "msg": "Starting simulated training loop (fallback)."})
        total_epochs = max(1, int(epochs))
        for ep in range(1, total_epochs + 1):
            if getattr(stop_event, "is_set", lambda: False)():
                _safe_broadcast(broadcast_train, {"type": "status", "state": "stopped", "msg": "stopped by user"})
                state.update({"state": "stopped", "msg": "stopped by user"})
                return
            while getattr(pause_event, "is_set", lambda: False)():
                _safe_broadcast(broadcast_train, {"type": "status", "state": "paused", "msg": "paused"})
                time.sleep(0.5)

            loss = max(0.0, 1.0 - (ep / total_epochs) - (0.02 * (0.5 - (time.time() % 1))))
            acc = min(1.0, 0.2 + (ep / total_epochs) * 0.8)
            pct = float(100.0 * ep / max(1, total_epochs))
            _safe_broadcast(broadcast_train, {"type": "metric", "epoch": ep, "loss": round(loss, 4), "valLoss": round(loss + 0.05, 4), "accuracy": round(acc, 4), "valAccuracy": round(acc - 0.02, 4)})
            _safe_broadcast(broadcast_train, {"type": "progress", "progress": pct, "epoch": ep})
            _safe_record(record_fn, {"event": "metric", "epoch": ep, "accuracy": acc, "progress": pct})
            time.sleep(0.8)

        cm_list = [[int(total_epochs * 2), int(total_epochs * 1)], [int(total_epochs * 1), int(total_epochs * 6)]]
        _safe_broadcast(broadcast_train, {"type": "confusion", "matrix": cm_list, "labels": ["neg", "pos"]})
        state.update({"state": "completed", "msg": "simulated training completed"})
        _safe_broadcast(broadcast_train, {"type": "status", "state": "completed", "msg": "simulated training completed"})
        return

    except Exception as e:
        log.exception("[train_job] exception")
        _safe_broadcast(broadcast_train, {"type": "log", "msg": f"[train_job] exception: {e}"})
        _safe_broadcast(broadcast_train, {"type": "status", "state": "error", "msg": str(e)})
        state.update({"state": "error", "msg": str(e)})
        return

# Helper DummyEvent used when no real threading.Event is supplied
class DummyEvent:
    def __init__(self):
        self._flag = False

    def is_set(self):
        return bool(self._flag)

    def set(self):
        self._flag = True

    def clear(self):
        self._flag = False
