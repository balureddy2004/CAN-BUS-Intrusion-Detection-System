# backend/train_job.py
"""
train_job.run_training(params, broadcast_train, record_fn, stop_event, pause_event, state, model_out_dir="backend/models")
- params: dict of hyperparams (model, epochs, batchSize, dataset, ...)
- broadcast_train: callable(event_dict) -> used to send WS events to clients
- record_fn: callable(event_dict) -> used to persist events (optional)
- stop_event, pause_event: threading.Event objects
- state: shared state dict (should be mutated with state['state']=..., state['msg']=...)
- model_out_dir: where to save model (optional)
"""

from __future__ import annotations
import os
import time
import logging
import glob
from typing import Any, Callable, Dict, Optional, Tuple, List

log = logging.getLogger("can-ids.train_job")

# Try importing packages used for real training
SKLEARN_OK = False
XGBOOST_OK = False
LGBM_OK = False
PANDAS_OK = False

try:
    import numpy as np
    import sklearn  # noqa: F401
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import xgboost as xgb  # type: ignore
    XGBOOST_OK = True
except Exception:
    XGBOOST_OK = False

try:
    import lightgbm as lgb  # type: ignore
    LGBM_OK = True
except Exception:
    LGBM_OK = False

try:
    import pandas as pd  # type: ignore
    PANDAS_OK = True
except Exception:
    PANDAS_OK = False

# Prefer the subprocess-safe loader helper if available; otherwise fall back to direct loader import.
# loader_helper provides load_dataset_with_timeout(dataset_arg, max_rows=None, timeout=60)
try:
    from backend.loader_helper import load_dataset_with_timeout  # type: ignore
    _USE_SUBPROCESS_LOADER = True
except Exception:
    load_dataset_with_timeout = None  # type: ignore
    _USE_SUBPROCESS_LOADER = False

# local direct dataset loader fallback (kept for environments that don't want subprocess)
try:
    from backend.dataset_loader import load_dataset as _direct_load_dataset  # type: ignore
except Exception:
    try:
        from dataset_loader import load_dataset as _direct_load_dataset  # fallback
    except Exception:
        _direct_load_dataset = None

# Safety cap to avoid loading multi-million row datasets into memory
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
    Resolve dataset_param to a file path if possible. Simple heuristic: check
    absolute path, backend/data, backend/datasets and preset 'ocs-lab'.
    Returns either a file path or directory path (string) or None.
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
            # prefer known files
            prefs = [
                "DoS_attack_dataset.csv", "Attack_free_dataset.csv",
                "Fuzzy_attack_dataset.csv", "Impersonation_attack_dataset.csv"
            ]
            for p in prefs:
                pth = os.path.join(folder, p)
                if os.path.exists(pth):
                    return pth
            # else pick first csv
            files = sorted(glob.glob(os.path.join(folder, "*.csv"))) + sorted(glob.glob(os.path.join(folder, "*.parquet")))
            if files:
                return files[0]

    # search candidate dirs
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


# --- fallback loader using pandas (if available) -----------------------------
def _find_files_in_path(path: str) -> List[str]:
    """Return list of CSV/Parquet files for a path which may be file or directory."""
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".csv", ".parquet", ".parq", ".ndjson", ".json"):
            return [path]
        return []
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv"))) + sorted(glob.glob(os.path.join(path, "*.parquet")))
        return files
    return []


def _infer_label_column(df: "pd.DataFrame") -> Optional[str]:
    """Heuristic to pick label column from dataframe."""
    candidates = ["label", "attack", "target", "class", "is_attack", "y", "attack_type"]
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_cols:
            return lower_cols[cand]
    # fallback: any column with only 2 or few unique values (likely binary label)
    for c in df.columns:
        try:
            nunique = df[c].nunique(dropna=True)
            if nunique <= 3 and nunique >= 2:
                return c
        except Exception:
            continue
    return None


def _fallback_load_dataset(dataset_param: Optional[str]) -> Tuple[Optional[Any], Optional[Any], Dict[str, Any]]:
    """
    Fallback dataset loader using pandas.
    Accepts dataset_param which can be:
     - absolute/relative file path
     - directory path
     - dataset identifier (e.g. 'ocs-lab') which is resolved via _resolve_dataset_param
    Returns (X (np.ndarray), y (np.ndarray), meta dict) or (None, None, {}).
    """
    meta: Dict[str, Any] = {}
    if not PANDAS_OK:
        log.debug("[train_job] pandas not available — cannot fallback load")
        return None, None, meta

    try:
        resolved = _resolve_dataset_param(dataset_param) if dataset_param else None
        if not resolved:
            log.debug("[train_job] fallback loader: no resolved path for dataset_param=%r", dataset_param)
            return None, None, meta

        files = _find_files_in_path(resolved)
        if not files:
            # maybe resolved is a file-like path (already handled) — bail
            log.debug("[train_job] fallback loader: no csv/parquet found at %s", resolved)
            return None, None, meta

        dfs = []
        for fn in files:
            try:
                if fn.lower().endswith(".parquet"):
                    df = pd.read_parquet(fn)
                else:
                    # CSV: try common separators; pandas can usually infer
                    df = pd.read_csv(fn)
                dfs.append(df)
            except Exception:
                log.exception("[train_job] failed to read dataset file: %s", fn)
                continue

        if not dfs:
            log.debug("[train_job] fallback loader: no readable files in %s", files)
            return None, None, meta

        df_all = pd.concat(dfs, ignore_index=True, sort=False)
        meta["source_files"] = files
        meta["rows"] = int(len(df_all))
        # attempt to find label column
        label_col = _infer_label_column(df_all)
        if not label_col:
            log.warning("[train_job] fallback loader: couldn't infer label column for dataset=%s; columns=%s", dataset_param, list(df_all.columns)[:10])
            return None, None, meta

        # Prepare X, y
        y_ser = df_all[label_col].copy()
        X_df = df_all.drop(columns=[label_col])

        # If label is non-numeric, encode to integers
        if y_ser.dtype == object or not np.issubdtype(getattr(y_ser, "dtype", None), np.number):
            try:
                y_ser, uniques = pd.factorize(y_ser, sort=True)
                meta["label_map"] = {str(v): i for i, v in enumerate(uniques)}
            except Exception:
                # last resort: try to convert booleans
                try:
                    y_ser = y_ser.astype(int)
                except Exception:
                    log.exception("[train_job] fallback loader: cannot coerce label to numeric")
                    return None, None, meta

        # Convert to numpy arrays
        try:
            X = X_df.values.astype(float)
        except Exception:
            # If conversion to float fails (categorical features), let numpy pick object dtype
            X = X_df.values
        y = np.asarray(y_ser)

        return X, y, meta
    except Exception:
        log.exception("[train_job] fallback loader exception")
        return None, None, meta


# -----------------------------------------------------------------------------


def _save_model(clf: Any, path: str, model_name: str):
    """
    Try to save model: prefer joblib.dump, else try library-specific save.
    """
    try:
        import joblib  # type: ignore
        joblib.dump(clf, path)
        return True
    except Exception:
        # xgboost native
        try:
            if XGBOOST_OK and hasattr(clf, "save_model"):
                clf.save_model(path + ".xgb")
                return True
        except Exception:
            pass
        try:
            if LGBM_OK and hasattr(clf, "save_model"):
                # lightgbm: save_model supports filename
                clf.save_model(path + ".lgb")
                return True
        except Exception:
            pass
    return False


def run_training(params: Dict[str, Any],
                 broadcast_train: Optional[Callable[[Dict[str, Any]], None]] = None,
                 record_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
                 stop_event: Optional[Any] = None,
                 pause_event: Optional[Any] = None,
                 state: Optional[Dict[str, Any]] = None,
                 model_out_dir: str = "backend/models"):
    """
    Start a training job. Tries to run a chosen model (RandomForest, XGBoost, LightGBM)
    if dependencies and dataset are suitable. Otherwise falls back to a simulated loop.
    """
    try:
        # safety cap (can be overridden by params['maxRows'])
        DEFAULT_MAX_TRAIN_ROWS = 50000
        max_rows = int(params.get("maxRows", DEFAULT_MAX_TRAIN_ROWS)) if isinstance(params, dict) else DEFAULT_MAX_TRAIN_ROWS

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

        # Load dataset if possible
        X, y, meta = None, None, {}
        # Prefer subprocess-based loader if available, else direct loader, else pandas fallback
        if _USE_SUBPROCESS_LOADER and load_dataset_with_timeout is not None:
            try:
                resolved = None
                # if server provided dataset_path we try that first
                if isinstance(params.get("dataset_path"), str) and params.get("dataset_path"):
                    resolved = params.get("dataset_path")
                else:
                    resolved = _resolve_dataset_param(dataset) if dataset else None

                if resolved:
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Trying loader with resolved dataset -> {resolved}"})
                    log.info("[train_job] attempting load_dataset_with_timeout with resolved path %s", resolved)
                    # Use subprocess-based loader with timeout (seconds). Tune timeout as needed.
                    try:
                        X, y, meta = load_dataset_with_timeout(resolved, max_rows=max_rows, timeout=120)
                    except Exception as e:
                        log.exception("[train_job] subprocess loader failed")
                        _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Subprocess loader failed: {e}. Falling back."})
                        X, y, meta = None, None, {}
                else:
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Trying loader with dataset param '{dataset}'"})
                    log.info("[train_job] attempting load_dataset_with_timeout with dataset param %s", dataset)
                    try:
                        X, y, meta = load_dataset_with_timeout(dataset, max_rows=max_rows, timeout=120)
                    except Exception as e:
                        log.exception("[train_job] subprocess loader failed for dataset param")
                        _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Subprocess loader failed: {e}. Falling back."})
                        X, y, meta = None, None, {}

                if X is not None and y is not None:
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Loaded dataset rows={len(X) if X is not None else 0}, features={getattr(X, 'shape', (None, None))[1] if X is not None else 'unknown'}"})
                else:
                    # attempt direct loader or pandas fallback if subprocess loader returned nothing
                    if _direct_load_dataset is not None:
                        try:
                            _safe_broadcast(broadcast_train, {"type": "log", "msg": "Subprocess loader returned no data — trying direct loader"})
                            # try calling direct loader with max_rows kwarg (if accepted), else without
                            try:
                                X, y, meta = _direct_load_dataset(resolved if resolved else dataset, max_rows=max_rows)
                            except TypeError:
                                # loader may not accept max_rows kw; call without it
                                X, y, meta = _direct_load_dataset(resolved if resolved else dataset)
                            if X is not None and y is not None:
                                _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Direct loader loaded rows={len(X)}"})
                        except Exception:
                            log.exception("[train_job] direct loader also failed after subprocess loader")
                            X, y, meta = None, None, {}
                    if X is None or y is None:
                        try:
                            X, y, meta = _fallback_load_dataset(dataset)
                            if X is not None and y is not None:
                                _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Pandas fallback loaded rows={len(X)}"})
                        except Exception:
                            log.exception("[train_job] fallback loader also failed after subprocess and direct loaders")
                            X, y, meta = None, None, {}
            except Exception as e:
                log.exception("[train_job] dataset load failed (subprocess path)")
                _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Dataset load failed (primary subprocess): {e}"})
                try:
                    X, y, meta = _fallback_load_dataset(dataset)
                    if X is not None and y is not None:
                        _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Pandas fallback loaded rows={len(X)}"})
                except Exception:
                    log.exception("[train_job] fallback loader also failed")
                    X, y, meta = None, None, {}
        else:
            # No subprocess loader available; try direct loader if present
            if _direct_load_dataset is not None:
                try:
                    resolved = None
                    if isinstance(params.get("dataset_path"), str) and params.get("dataset_path"):
                        resolved = params.get("dataset_path")
                    else:
                        resolved = _resolve_dataset_param(dataset) if dataset else None

                    if resolved:
                        _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Trying direct loader with resolved dataset -> {resolved}"})
                        log.info("[train_job] attempting direct load_dataset with resolved path %s", resolved)
                        # try with max_rows if accepted, else without
                        try:
                            X, y, meta = _direct_load_dataset(resolved, max_rows=max_rows)
                        except TypeError:
                            X, y, meta = _direct_load_dataset(resolved)
                    else:
                        _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Trying direct loader with dataset param '{dataset}'"})
                        log.info("[train_job] attempting direct load_dataset with dataset param %s", dataset)
                        try:
                            X, y, meta = _direct_load_dataset(dataset, max_rows=max_rows) if dataset else (None, None, {})
                        except TypeError:
                            X, y, meta = _direct_load_dataset(dataset) if dataset else (None, None, {})

                    if X is not None and y is not None:
                        _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Loaded dataset rows={len(X) if X is not None else 0}, features={getattr(X, 'shape', (None, None))[1] if X is not None else 'unknown'}"})
                    else:
                        _safe_broadcast(broadcast_train, {"type": "log", "msg": "direct loader returned no data — attempting pandas fallback"})
                        X, y, meta = _fallback_load_dataset(dataset)
                        if X is not None and y is not None:
                            _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Pandas fallback loaded rows={len(X)}"})
                except Exception:
                    log.exception("[train_job] dataset load failed (direct loader)")
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": "Dataset load failed (direct). Trying fallback."})
                    try:
                        X, y, meta = _fallback_load_dataset(dataset)
                        if X is not None and y is not None:
                            _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Pandas fallback loaded rows={len(X)}"})
                    except Exception:
                        log.exception("[train_job] fallback loader also failed")
                        X, y, meta = None, None, {}
            else:
                # no supplied loader, use pandas fallback directly
                X, y, meta = _fallback_load_dataset(dataset)
                if X is not None and y is not None:
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Pandas fallback loaded rows={len(X)}"})
                else:
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": "No dataset loader available and fallback failed."})

        # decide whether we can run real training
        try_real = (SKLEARN_OK or XGBOOST_OK or LGBM_OK) and X is not None and y is not None and len(y) >= 10 and len(set(y)) > 1

        if try_real:
            _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Preparing real training (model={model_name})."})
            log.info("[train_job] attempting real training with model=%s", model_name)

            # train/test split (stratify when possible)
            stratify_obj = (y if (y is not None and len(set(y)) > 1) else None)
            try:
                X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_split, random_state=seed, stratify=stratify_obj)
            except Exception:
                # fallback no stratify
                X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_split, random_state=seed)

            clf = None
            used_impl = "RandomForest"

            # choose model implementation by requested model_name
            if model_name.lower().startswith("xg"):
                if XGBOOST_OK:
                    used_impl = "XGBoost"
                    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=100, random_state=seed, n_jobs=-1)
                else:
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": "XGBoost requested but xgboost not installed — falling back to RandomForest."})
                    log.warning("XGBoost requested but not available; using RandomForest")
            elif model_name.lower().startswith("light") or model_name.lower().startswith("lgb"):
                if LGBM_OK:
                    used_impl = "LightGBM"
                    clf = lgb.LGBMClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
                else:
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": "LightGBM requested but lightgbm not installed — falling back to RandomForest."})
                    log.warning("LightGBM requested but not available; using RandomForest")

            # default RandomForest if nothing created yet
            if clf is None:
                if SKLEARN_OK:
                    used_impl = "RandomForest"
                    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
                else:
                    # weird: no sklearn but data present; fallback to simulated loop
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": "No sklearn/xgboost/lightgbm available - using simulated training."})
                    log.warning("No ML libs available; skipping real training")
                    clf = None

            # If clf ready, perform epoch-like loop (fit repeated subsets to simulate epochs)
            if clf is not None:
                rng = np.random.RandomState(seed)
                n_samples = X_train.shape[0]
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
                    idx = rng.choice(n_samples, size=subset_size, replace=False)

                    try:
                        # For XGBoost/LightGBM/RandomForest fit works similarly
                        clf.fit(X_train[idx], y_train[idx])
                    except Exception:
                        # Some libraries dislike extremely small subset fits: fallback to full fit
                        clf.fit(X_train, y_train)

                    try:
                        y_pred = clf.predict(X_val)
                        acc = float(accuracy_score(y_val, y_pred))
                        cm = confusion_matrix(y_val, y_pred)
                        if cm is not None:
                            if cm.shape == (2, 2):
                                cm_list = cm.tolist()
                            else:
                                a = [[0, 0], [0, 0]]
                                for i in range(min(2, cm.shape[0])):
                                    for j in range(min(2, cm.shape[1])):
                                        a[i][j] = int(cm[i, j])
                                cm_list = a
                    except Exception:
                        acc = 0.0
                        cm_list = [[0, 0], [0, 0]]

                    pct = float(100.0 * ep / max(1, epochs))
                    _safe_broadcast(broadcast_train, {"type": "metric", "epoch": ep, "loss": 0.0, "valLoss": 0.0, "accuracy": acc, "valAccuracy": acc})
                    _safe_broadcast(broadcast_train, {"type": "progress", "progress": pct, "epoch": ep})
                    _safe_record(record_fn, {"event": "metric", "epoch": ep, "accuracy": acc, "progress": pct})
                    time.sleep(0.5)

                # save model
                os.makedirs(model_out_dir, exist_ok=True)
                model_path = os.path.join(model_out_dir, f"{used_impl}_{int(time.time())}.pkl")
                saved = _save_model(clf, model_path, used_impl)
                if saved:
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": f"Model saved to {model_path}"})
                else:
                    _safe_broadcast(broadcast_train, {"type": "log", "msg": "Model save failed (joblib/native save not available)"})

                _safe_broadcast(broadcast_train, {"type": "confusion", "matrix": cm_list, "labels": ["neg", "pos"]})
                state.update({"state": "completed", "msg": "training completed"})
                _safe_broadcast(broadcast_train, {"type": "status", "state": "completed", "msg": "training completed"})
                return

        # Fallback: simulated training loop
        _safe_broadcast(broadcast_train, {"type": "log", "msg": "Starting simulated training loop (debug fallback)."})
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


# small helper if events not provided
class DummyEvent:
    def __init__(self):
        self._flag = False

    def is_set(self):
        return bool(self._flag)

    def set(self):
        self._flag = True

    def clear(self):
        self._flag = False
