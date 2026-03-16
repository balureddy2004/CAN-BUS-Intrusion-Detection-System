# server.py
# --- add near the top of Training state section ---
# ---- gevent monkey-patching MUST be first ----
import os
if os.getenv("USE_GEVENT", "1") not in ("0", "false", "False"):
    from gevent import monkey as _monkey
    _monkey.patch_all()
# ----------------------------------------------
# simple-websocket is what Flask-Sock uses; avoid gevent-websocket entirely
try:
    from simple_websocket.errors import ConnectionClosed
except Exception:
    class ConnectionClosed(Exception):
        """Fallback if simple_websocket isn’t available (shouldn’t happen)."""
        pass
from gevent import spawn, sleep
from typing import Any
import threading
train_clients = set()
# NEW: a per-connection send lock map
# --- enable gevent on Windows (must be FIRST) ---
# --- NEW helper: always use this to send on a WS ---
# Track per-client send locks
_train_send_locks: dict[object, threading.Lock] = {}
import json
from flask import jsonify, current_app,request
import joblib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _ws_safe_send(ws, payload: str):
    """
    Thread-safe WS send with useful logging.
    NOTE: We re-raise so callers can mark the socket dead.
    """
    lock = _train_send_locks.get(ws)
    if lock is None:
        lock = threading.Lock()
        _train_send_locks[ws] = lock

    with lock:
        try:
            ws.send(payload)
        except ConnectionClosed as e:
            try:
                peer = getattr(ws, "environ", {}).get("REMOTE_ADDR") or "unknown-peer"
                path = getattr(ws, "environ", {}).get("PATH_INFO") or "?"
            except Exception:
                peer, path = "unknown-peer", "?"
            log.info("[_ws_safe_send] connection closed to %s %s: %s", peer, path, e)
            raise
        except Exception as e:
            log.exception("[_ws_safe_send] unexpected send failure: %s", e)
            raise

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_sock import Sock
import time, threading, io, json, random, os, math, sqlite3, smtplib, ssl, logging, signal, traceback
from collections import deque
from typing import Any, Dict, Optional, List, Tuple
from queue import Queue, Empty
from email.message import EmailMessage
import math as _math
from backend.train_upload import register_upload_routes
from backend.train_job import run_training as run_training_job

# ==============================================================================
# Boot / Logging
# ==============================================================================
START_TIME = time.time()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
)
log = logging.getLogger("can-ids")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
sock = Sock(app)
register_upload_routes(app)
# -----------------------------
# MODEL CACHE (in-memory)
# -----------------------------
# shared dict of name -> loaded model object
# accessible from views via current_app.config["MODEL_CACHE"]
app.config["MODEL_CACHE"] = {}

# models API (model listing, download, predict)
from backend.api_models_and_predict import models_bp as models_api_bp

# detector blueprint (ensure it's the backend module)
from backend.models_detector import detector_bp

# register blueprints under /api
app.register_blueprint(models_api_bp, url_prefix="/api")
app.register_blueprint(detector_bp, url_prefix="/api")
# === Readiness + API-health helpers (place immediately after blueprint registration) ===
import time

@app.get("/api/ready")
def api_ready():
    """
    Readiness probe: best-effort check if models are available.
    Returns JSON { ok: True, models_available: bool }.
    Lazy-imports the model discovery helper to avoid import cycles.
    """
    try:
        # lazy import so server can start even if backend module has problems
        from backend.api_models_and_predict import _gather_models
        models = _gather_models() or []
        models_available = bool(models)
    except Exception:
        # fallback: quick filesystem scan of backend/models
        try:
            mdir = os.path.join(os.path.dirname(__file__), "backend", "models")
            models = [f for f in (os.listdir(mdir) if os.path.isdir(mdir) else []) if f.lower().endswith((".pkl", ".joblib"))]
            models_available = bool(models)
        except Exception:
            models_available = False

    return jsonify({"ok": True, "models_available": models_available}), 200

# Optional: also provide /api/health (so clients relying on /api/health succeed)
@app.get("/api/health")
def api_health():
    return jsonify({"ok": True, "status": "alive", "time": int(time.time())}), 200
# === end readiness helpers ===
# ---------- preload latest model (best-effort) ----------
try:
    from backend.api_models_and_predict import _gather_models
    import joblib
    from pathlib import Path

    def preload_latest_model():
        models = _gather_models()
        if not models:
            app.logger.info("No models found to preload")
            return

        newest = models[0]
        path = Path(newest["path"])
        try:
            clf = joblib.load(str(path))
            # set n_jobs=1 if available to avoid runaway thread pools
            try:
                if hasattr(clf, "set_params"):
                    clf.set_params(**{"n_jobs": 1})
            except Exception:
                pass
            app.config["MODEL_CACHE"][path.name] = clf
            app.logger.info("Preloaded model: %s", path.name)
        except Exception as e:
           app.logger.warning("Failed to preload model %s: %s", path, e)


    # call it (best-effort)
    preload_latest_model()
except Exception:
    # keep server resilient if anything goes wrong here
    pass
# ---------------------------------------------------------

# --- Normalized models list for frontend (drop into server.py after blueprints) ---
import pathlib

def _scan_models_dir_models():
    out = []
    mdir = os.path.join(os.path.dirname(__file__), "backend", "models")
    try:
        if os.path.isdir(mdir):
            for fn in sorted(os.listdir(mdir)):
                if fn.lower().endswith((".pkl", ".joblib", ".xgb", ".lgb")):
                    name = os.path.splitext(fn)[0]
                    out.append({
                        "id": name,
                        "label": name.replace("_", " "),
                        "weights_path": os.path.join("backend", "models", fn).replace("\\", "/"),
                        "enabled": True
                    })
    except Exception:
        log.exception("[models-scan] scanning backend/models failed")
    return out

@app.get("/api/models/list")
def api_models_list_for_frontend():
    """
    Returns: JSON array of model objects [{id,label,weights_path,enabled}, ...]
    This is a normalized format for frontends that expect an array rather than nested {"models": [...] }.
    """
    # 1) Prefer config/models.json if it contains "models" list
    try:
        if os.path.exists(MODELS_JSON_PATH):
            with open(MODELS_JSON_PATH, "r", encoding="utf-8") as f:
                j = json.load(f)
            # support both {"models": [...]} or top-level list
            if isinstance(j, dict) and "models" in j and isinstance(j["models"], list):
                models = j["models"]
            elif isinstance(j, list):
                models = j
            else:
                models = None
        else:
            models = None
    except Exception:
        log.exception("[api_models_list] failed to read config/models.json")
        models = None

    if models:
        out = []
        for m in models:
            try:
                # normalize keys
                mid = str(m.get("id") or m.get("name") or "").strip()
                if not mid:
                    continue
                out.append({
                    "id": mid,
                    "label": m.get("label") or m.get("name") or mid,
                    "weights_path": m.get("weights_path") or m.get("path") or "",
                    "enabled": bool(m.get("enabled", True)),
                })
            except Exception:
                continue
        if out:
            return jsonify(out), 200

    # fallback: scan backend/models dir
    scanned = _scan_models_dir_models()
    return jsonify(scanned), 200
# --- end new route ---

# ----- robust models.json discovery (paste near other PATH constants) -----
# The server may run from project root or backend/; check both locations.
PROJECT_ROOT = os.path.dirname(__file__)    # location of server.py
CANDIDATE_MODELS_CONFIGS = [
    os.path.join(PROJECT_ROOT, "config", "models.json"),           # project-root/config/models.json
    os.path.join(PROJECT_ROOT, "backend", "config", "models.json"),# backend/config/models.json
    os.path.join(PROJECT_ROOT, "..", "config", "models.json"),     # one-up (defensive)
]

# Choose the first path that exists; otherwise default to the first candidate
MODELS_JSON_PATH = next((p for p in CANDIDATE_MODELS_CONFIGS if os.path.exists(p)), CANDIDATE_MODELS_CONFIGS[0])

# Optional: log which path we will attempt to use
log.info("[models] MODELS_JSON_PATH resolved -> %s", MODELS_JSON_PATH)
# -----------------------------------------------------------------------

# ==============================================================================
# Helpers
# ==============================================================================
# ------------------------
# models.json helper
# ------------------------

def _load_models_map():
    """
    Return dict { model_id: model_obj } from config/models.json.
    Safe: returns {} on parse error or missing file.
    """
    try:
        with open(MODELS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            # allow both top-level list or {"models": [...]}
            if isinstance(data, dict) and "models" in data:
                data = data["models"]
            if not isinstance(data, list):
                current_app_logger = getattr(log, "warning", None)
                if current_app_logger:
                    log.warning("models.json has unexpected shape; expected list")
                return {}
            out = {}
            for m in data:
                try:
                    mid = str(m.get("id") or "").strip()
                    if not mid:
                        continue
                    out[mid] = m
                except Exception:
                    continue
            return out
    except FileNotFoundError:
        log.warning("models.json not found at %s", MODELS_JSON_PATH)
        return {}
    except Exception as e:
        log.exception("failed to load models.json: %s", e)
        return {}

def _finite(v: Any, fallback: float | int = 0) -> Any:
    """Coerce NaN/inf to a safe finite number for JSON/websocket clients."""
    try:
        if isinstance(v, (int,)):
            return v
        if isinstance(v, float):
            if _math.isnan(v) or _math.isinf(v):
                return fallback
            return v
        return v
    except Exception:
        return fallback

def _json_dumps_safe(obj: Any) -> str:
    def _fix(o):
        if isinstance(o, dict):
            return {k: _fix(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_fix(v) for v in o]
        if isinstance(o, float):
            return _finite(o, 0.0)
        return o
    return json.dumps(_fix(obj))

# ==============================================================================
# Training state (existing + live monitor fixes)
# ==============================================================================
state = {
    "state": "idle",
    "progress": 0.0,
    "epoch": 0,
    "msg": "ready",
}
_stop = threading.Event()
_pause = threading.Event()

# --- NEW: ring buffers for replay on connect ---
TRAIN_METRICS = deque(maxlen=500)   # {"type":"metric", epoch, loss, valLoss, accuracy, valAccuracy, t}
TRAIN_PROGRESS = deque(maxlen=500)  # {"type":"progress", progress, epoch, t}
TRAIN_LOGS = deque(maxlen=500)      # {"type":"log", msg, t}
TRAIN_STATUS = deque(maxlen=50)     # {"type":"status", state, msg, t}

def _record_train_event(evt: dict):
    t = time.time()
    evt = dict(evt)
    evt["t"] = t
    kind = evt.get("type")
    if kind == "metric":
        TRAIN_METRICS.append(evt)
    elif kind == "progress":
        TRAIN_PROGRESS.append(evt)
    elif kind == "log":
        TRAIN_LOGS.append(evt)
    elif kind == "status":
        TRAIN_STATUS.append(evt)

def _sanitize_training_obj(obj: dict) -> dict:
    """Ensure numbers are finite for JSON encoding & clients."""
    s = {}
    for k, v in obj.items():
        if isinstance(v, (float, int)):
            s[k] = _finite(v, 0.0)
        else:
            s[k] = v
    return s

# --- update broadcast_train to use the lock ---
def broadcast_train(obj: dict):
    obj = _sanitize_training_obj(obj)
    _record_train_event(obj)
    dead = []
    payload = _json_dumps_safe(obj)
    for ws in list(train_clients):
        try:
            _ws_safe_send(ws, payload)   # <-- use safe sender
        except Exception:
            dead.append(ws)
    for ws in dead:
        train_clients.discard(ws)
        _train_send_locks.pop(ws, None)   # cleanup its lock

def _push_status(new_state: str, msg: str):
    state["state"] = new_state
    state["msg"] = msg
    broadcast_train({"type": "status", "state": new_state, "msg": msg})

def training_job(params: dict):
    try:
        _push_status("training", "Training started")
        state["progress"] = 0.0
        state["epoch"] = 0

        epochs = int(params.get("epochs", 10))
        for ep in range(1, epochs + 1):
            if _stop.is_set():
                _push_status("stopped", "Stopped by user")
                _stop.clear()
                return

            while _pause.is_set():
                time.sleep(0.25)

            # Simulated work
            time.sleep(0.6)
            loss = max(0.02, 1.2 / ep)
            val_loss = loss * 1.05
            acc = min(0.99, 0.6 + ep * 0.02)
            val_acc = min(0.99, acc - 0.02)

            state["epoch"] = ep
            state["progress"] = round(ep / epochs * 100, 2)
            state["msg"] = f"Epoch {ep}/{epochs} loss={loss:.3f} acc={acc:.3f}"

            broadcast_train({
                "type": "metric",
                "epoch": ep,
                "loss": loss,
                "valLoss": val_loss,
                "accuracy": acc,
                "valAccuracy": val_acc
            })
            broadcast_train({"type": "progress", "progress": state["progress"], "epoch": ep})
            broadcast_train({"type": "log", "msg": state["msg"]})

        _push_status("completed", "Training completed")
    except Exception as e:
        log.exception("[train] crashed")
        _push_status("error", f"Training crashed: {e}")
# --- dataset helper stub (forward declaration to satisfy linters/editors) ---
PROJECT_ROOT = os.path.dirname(__file__)
CANDIDATE_DATASET_DIRS = [
    os.path.join(PROJECT_ROOT, "datasets"),
    os.path.join(PROJECT_ROOT, "backend", "datasets"),
]
DATASETS_CONFIG = os.path.join(PROJECT_ROOT, "config", "datasets.json")

def _nice_id(name: str) -> str:
    name = os.path.splitext(name)[0]
    return name.strip().lower().replace(" ", "-").replace("_", "-")

def _find_dataset_path(dataset_id: str) -> str | None:
    """
    Given a dataset id (slug) or absolute path, return a candidate path (directory or file)
    that exists, otherwise return None.
    """
    if not dataset_id:
        return None

    # if user passed an absolute (or explicit relative) filesystem path, accept it
    try:
        ds = str(dataset_id).strip()
        # expand user and normalize
        ds_expanded = os.path.expanduser(ds)
        if os.path.isabs(ds_expanded) or ds_expanded.startswith(".") or ds_expanded.startswith(".."):
            if os.path.exists(ds_expanded):
                return os.path.abspath(ds_expanded)
    except Exception:
        pass

    did = str(dataset_id).strip().lower()
    for base in CANDIDATE_DATASET_DIRS:
        cand_dir = os.path.join(base, did)
        if os.path.isdir(cand_dir):
            return cand_dir
        try:
            if os.path.isdir(base):
                for entry in os.listdir(base):
                    if _nice_id(entry) == did:
                        return os.path.join(base, entry)
        except Exception:
            continue

    # try files by slug
    for base in CANDIDATE_DATASET_DIRS:
        if not os.path.isdir(base):
            continue
        try:
            for entry in os.listdir(base):
                ext = os.path.splitext(entry)[1].lower()
                if ext in (".csv", ".parquet", ".json", ".ndjson") and _nice_id(entry) == did:
                    return os.path.join(base, entry)
        except Exception:
            continue

    return None
# --- end stub ---

@app.post("/train/start")
def start_train():
    """
    Robust /train/start that accepts:
      - application/json POST with JSON body
      - multipart/form-data or application/x-www-form-urlencoded (request.form)
      - fallback: try to parse raw body as JSON
    Returns helpful 400 messages when modelId is missing or unknown.
    """
    # clear stop/pause markers up front (safe)
    _stop.clear()
    _pause.clear()

    # 1) try standard Flask JSON parse (respects Content-Type)
    parsed = None
    try:
        parsed = request.get_json(silent=True)
    except Exception:
        parsed = None

    # 2) if no JSON, try to parse raw body as JSON (handles some odd clients)
    if parsed is None:
        try:
            raw = request.get_data(as_text=True)
            if raw and raw.strip():
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = None
            else:
                parsed = None
        except Exception:
            parsed = None

    # 3) fallback to form parameters (multipart/form-data or x-www-form-urlencoded)
    form = request.form or {}
    body = parsed if isinstance(parsed, dict) else {}

    # Build a getval helper that checks form then body
    def _getval(k, default=None):
        # prefer body JSON first (explicit JSON should override form)
        v = body.get(k) if isinstance(body, dict) else None
        if v is None:
            v = form.get(k)
        if v is None:
            v = default
        return v

    # Accept either "modelId" or "model" keys (case-insensitive-ish)
    model_id = (_getval("modelId") or _getval("model") or "").strip() if (_getval("modelId") or _getval("model")) else ""
    # Also check camel/snake variants from some clients
    if not model_id:
        model_id = (_getval("model_id") or _getval("model-id") or "").strip()

    # Validate model exists in models config
    try:
        models_map = _load_models_map()  # your project likely has this helper
    except Exception:
        # fallback: try reading config/models.json directly
        models_map = {}
        try:
            cfgp = os.path.join(os.path.dirname(__file__), "config", "models.json")
            if os.path.exists(cfgp):
                with open(cfgp, "r", encoding="utf-8") as fh:
                    arr = json.load(fh) or []
                for m in arr:
                    models_map[m.get("id")] = m
        except Exception:
            pass

    # If client didn't supply model_id but we only have a single model configured,
    # auto-select that model to reduce UI friction.
    if not model_id and isinstance(models_map, dict) and len(models_map) == 1:
        model_id = next(iter(models_map.keys()))
        log.info("[start_train] no modelId in request; auto-selected single model '%s'", model_id)

    model_obj = models_map.get(model_id)
    if model_obj is None:
        log.warning("[start_train] unknown or missing model id '%s' requested", model_id)
        return jsonify({"ok": False, "error": f"unknown model id '{model_id}'. Run admin scan or upload weights."}), 400

    if not model_obj.get("enabled", True):
        return jsonify({"ok": False, "error": f"model '{model_id}' is currently disabled on the server."}), 400

    # Prepare numeric parameters from either JSON body or form
    def _int(k, default):
        try:
            return int(_getval(k, default))
        except Exception:
            return default

    def _float(k, default):
        try:
            return float(_getval(k, default))
        except Exception:
            return default

    # Resolve dataset (id, filename, or absolute path) -> actual path (if possible)
    dataset_raw = None
    # prefer explicit query param first, then JSON/form body
    if request.args.get("dataset"):
        dataset_raw = request.args.get("dataset")
    elif isinstance(body, dict) and ("dataset" in body):
        dataset_raw = body.get("dataset")
    else:
        dataset_raw = None

    # New: allow client to pass dataset_path directly (absolute or relative)
    dataset_path = None
    try:
        # 1) If explicit dataset_path provided in JSON/form, prefer it (if it exists)
        provided_path = None
        if request.args.get("dataset_path"):
            provided_path = request.args.get("dataset_path")
        elif isinstance(body, dict) and ("dataset_path" in body):
            provided_path = body.get("dataset_path")

        if provided_path:
            # normalize and check existence
            p = str(provided_path)
            # expanduser + normpath
            p_norm = os.path.normpath(os.path.expanduser(p))
            if os.path.isabs(p_norm) and os.path.exists(p_norm):
                dataset_path = p_norm
                log.info("[start_train] using provided absolute dataset_path: %s", dataset_path)
            else:
                # If provided path is relative, try resolving relative to project root and cwd
                cand1 = os.path.normpath(os.path.join(os.path.dirname(__file__), p))
                cand2 = os.path.normpath(os.path.join(os.getcwd(), p))
                if os.path.exists(cand1):
                    dataset_path = cand1
                    log.info("[start_train] using provided relative dataset_path resolved to: %s", dataset_path)
                elif os.path.exists(cand2):
                    dataset_path = cand2
                    log.info("[start_train] using provided relative dataset_path resolved to: %s", dataset_path)
                else:
                    log.warning("[start_train] dataset_path provided but not found: %s (tried %s, %s)", p, cand1, cand2)
                    dataset_path = None

        # 2) If no usable dataset_path provided, resolve dataset_raw via discovery helper
        if dataset_path is None and dataset_raw:
            try:
                # prefer discovery by exact id/slug, then normalized slug
                dataset_path = _find_dataset_path(str(dataset_raw))
                if not dataset_path:
                    dataset_path = _find_dataset_path(_nice_id(str(dataset_raw)))
                if dataset_path:
                    log.info("[start_train] resolved dataset '%s' -> %s", dataset_raw, dataset_path)
            except Exception:
                dataset_path = None
    except Exception:
        dataset_path = None

    # Accept max_rows / maxRows override (row limiter) and expose to training job
    max_rows = _int("maxRows", _int("max_rows", 0) or 0)
    if max_rows <= 0:
        # fallback default if not provided
        max_rows = 10000

    params = {
        "model": model_id,
        "modelId": model_id,
        "weights_path": model_obj.get("weights_path", ""),
        "epochs": _int("epochs", 10),
        "batchSize": _int("batchSize", 256),
        "learningRate": _float("learningRate", 1e-3),
        "optimizer": _getval("optimizer", "adam"),
        "trainSplit": _float("trainSplit", 0.8),
        "seed": _int("seed", 42),
        "augment": str(_getval("augment", "true")).lower() in ("1", "true", "yes", "on"),
        "dataset": dataset_raw,
        "dataset_path": dataset_path,
        "maxRows": max_rows,
    }

    log.info("[train] start called for model '%s' with params: %s", model_id, params)

    # delegate to your existing run_training_job in a thread
    def _trainer_thread_fn(p):
        try:
            run_training_job(p, broadcast_train, _record_train_event, _stop, _pause, state, model_out_dir="backend/models")
        except Exception:
            log.exception("run_training_job crashed for model %s", model_id)
            _push_status("error", "Training job crashed on server")

    threading.Thread(target=_trainer_thread_fn, args=(params,), daemon=True, name=f"trainer-{model_id}").start()

    # Respond with parameters used (hide heavy things like weights_path)
    resp_params = {k: v for k, v in params.items() if k != "weights_path"}
    return jsonify(ok=True, started=True, params=resp_params), 200

@app.post("/train/stop")
def stop_train():
    _stop.set()
    return jsonify(ok=True)

@app.post("/train/pause")
def pause_train():
    if state["state"] == "training":
        _pause.set()
        _push_status("paused", "Paused")
    elif state["state"] == "paused":
        _pause.clear()
        _push_status("training", "Resumed")
    return jsonify(ok=True)

@app.post("/train/reset")
def reset_train():
    """Clear ring buffers and reset state; useful between runs."""
    TRAIN_METRICS.clear()
    TRAIN_PROGRESS.clear()
    TRAIN_LOGS.clear()
    TRAIN_STATUS.clear()
    state.update({"state": "idle", "progress": 0.0, "epoch": 0, "msg": "ready"})
    _push_status("idle", "ready")
    return jsonify(ok=True)

@app.get("/train/status")
def status():
    return jsonify(state)

@app.get("/api/train/history")
def api_train_history():
    return jsonify({
        "status": list(TRAIN_STATUS),
        "metrics": list(TRAIN_METRICS),
        "progress": list(TRAIN_PROGRESS),
        "logs": list(TRAIN_LOGS),
        "state": state,
    })

@app.get("/train/model")
def download_model():
    buf = io.BytesIO(b"pretend this is your model weights")
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="trained_model.bin")

@app.post("/api/train/eval/confusion")
def api_eval_confusion():
    body = request.get_json(silent=True) or {}
    model_path = body.get("model_path", "backend/models/model.pkl")
    dataset = body.get("dataset", "datasets/ocs-lab-dev")
    max_rows = int(body.get("max_rows", 2000) or 2000)

    if not os.path.exists(model_path):
        return jsonify(ok=False, error="model_path not found"), 400

    try:
        try:
            model = joblib.load(model_path)
        except Exception:
            with open(model_path, "rb") as fh:
                import pickle
                model = pickle.load(fh)
    except Exception as e:
        return jsonify(ok=False, error=f"failed to load model: {e}"), 500

    try:
        from backend.dataset_loader import load_dataset
        X, y, meta = load_dataset(dataset, max_rows=max_rows)
    except Exception as e:
        return jsonify(ok=False, error=f"dataset load failed: {e}"), 500

    mask = (y >= 0)
    X_eval = X[mask]
    y_eval = y[mask]
    if len(y_eval) == 0:
        return jsonify(ok=False, error="no labeled samples found"), 400

    try:
        from sklearn.metrics import confusion_matrix, accuracy_score
        y_pred = model.predict(X_eval)
        acc = float(accuracy_score(y_eval, y_pred))
        cm = confusion_matrix(y_eval, y_pred).tolist()
    except Exception as e:
        return jsonify(ok=False, error=f"inference/eval failed: {e}"), 500

    return jsonify(ok=True, accuracy=acc, confusion=cm, meta=meta)

# --- update ws_train to use the lock everywhere it sends ---
# --- update ws_train to use the lock everywhere it sends ---
# --- drop-in replacement for ws_train ---
# --- replace the whole ws_train with this hardened version ---
@sock.route("/ws/train")
def ws_train(ws):
    log.info("WS /ws/train: client connected")
    train_clients.add(ws)
    _train_send_locks[ws] = threading.Lock()  # register per-conn lock

    alive = True

    def keepalive():
        # send ping every 15s regardless of client traffic
        last = time.time()
        while alive:
            sleep(1.0)
            now = time.time()
            if now - last >= 15:
                try:
                    _ws_safe_send(ws, _json_dumps_safe({"type": "ping", "t": now}))
                    last = now
                except Exception as e:
                    # any send failure means the socket is gone
                    log.debug("[ws_train keepalive] ping failed: %s", e)
                    break

    ka = spawn(keepalive)

    try:
        # 1) Snapshot on connect
        _ws_safe_send(ws, _json_dumps_safe({"type": "status", "state": state["state"], "msg": state["msg"]}))
        if TRAIN_METRICS:
            _ws_safe_send(ws, _json_dumps_safe({"type": "replay_metrics", "items": list(TRAIN_METRICS)}))
        if TRAIN_PROGRESS:
            _ws_safe_send(ws, _json_dumps_safe({"type": "replay_progress", "items": list(TRAIN_PROGRESS)}))
        if TRAIN_LOGS:
            _ws_safe_send(ws, _json_dumps_safe({"type": "replay_logs", "items": list(TRAIN_LOGS)}))

        # 2) Command loop (receive only; keepalive runs in separate greenlet)
        while True:
            try:
                msg = ws.receive()
            except ConnectionClosed as e:
                log.info("WS /ws/train: receive closed: %s", e)
                break
            except Exception as e:
                log.info("WS /ws/train: receive error: %r", e)
                break


            if msg is None:
                # client closed
                log.info("WS /ws/train: client closed (None)")
                break

            # parse commands
            try:
                o = json.loads(msg) if isinstance(msg, str) else {}
            except Exception:
                o = {}

            if not isinstance(o, dict):
                continue

            cmd = (o.get("cmd") or "").lower()
            now = time.time()
            if cmd == "ping":
                try:
                    _ws_safe_send(ws, _json_dumps_safe({"type": "pong", "t": now}))
                except Exception:
                    break
            elif cmd == "pause":
                _pause.set(); _push_status("paused", "Paused")
            elif cmd == "resume":
                _pause.clear(); _push_status("training", "Resumed")
            elif cmd == "stop":
                _stop.set()
            elif cmd == "reset":
                TRAIN_METRICS.clear(); TRAIN_PROGRESS.clear(); TRAIN_LOGS.clear(); TRAIN_STATUS.clear()
                state.update({"state": "idle", "progress": 0.0, "epoch": 0, "msg": "ready"})
                _push_status("idle", "ready")
            # else ignore

    except Exception as e:
        log.exception("WS /ws/train: unexpected error: %s", e)
    finally:
        alive = False
        try:
            ka.kill()
        except Exception:
            pass
        train_clients.discard(ws)
        _train_send_locks.pop(ws, None)
        log.info("WS /ws/train: client disconnected")
# ==============================================================================
# SQLite persistence
# ==============================================================================
DB_PATH = os.environ.get("IDS_DB_PATH", "ids.db")

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS alerts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms     INTEGER NOT NULL,
    attack_type TEXT NOT NULL,
    severity  TEXT,
    score     REAL,
    details   TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts_ms);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(attack_type);
CREATE INDEX IF NOT EXISTS idx_alerts_sev ON alerts(severity);

CREATE TABLE IF NOT EXISTS packets (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms     REAL NOT NULL,
    can_id    INTEGER,
    dlc       INTEGER,
    data_hex  TEXT,
    size_bytes INTEGER,
    attack    INTEGER,
    attack_type TEXT,
    severity  TEXT,
    score     REAL,
    signal    REAL,
    bar_height REAL,
    meta      TEXT
);
CREATE INDEX IF NOT EXISTS idx_packets_ts ON packets(ts_ms);
CREATE INDEX IF NOT EXISTS idx_packets_can ON packets(can_id);
"""

def get_conn(readonly=False):
    uri = f"file:{os.path.abspath(DB_PATH)}?mode={'ro' if readonly else 'rwc'}"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn(readonly=False)
    try:
        for stmt in SCHEMA_SQL.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(s)
        conn.commit()
        log.info("SQLite schema ensured at %s", os.path.abspath(DB_PATH))
    finally:
        conn.close()

def parse_since_ms(v: Optional[str]) -> Optional[float]:
    if not v:
        return None
    try:
        return float(v)
    except Exception:
        pass
    try:
        if v.endswith("ms"): return float(v[:-2])
        if v.endswith("s"):  return (time.time() * 1000.0) - float(v[:-1]) * 1000.0
        if v.endswith("m"):  return (time.time() * 1000.0) - float(v[:-1]) * 60_000.0
        if v.endswith("h"):  return (time.time() * 1000.0) - float(v[:-1]) * 3_600_000.0
    except Exception:
        return None
    return None

def parse_window_ms(w: Optional[str]) -> int:
    if not w:
        return 1000
    w = str(w).strip().lower()
    try:
        if w.endswith("ms"): return max(1, int(float(w[:-2])))
        if w.endswith("s"):  return max(1, int(float(w[:-1]) * 1000.0))
        if w.endswith("m"):  return max(1, int(float(w[:-1]) * 60_000.0))
        val = float(w)
        if val <= 1000:
            return max(1, int(val * 1000.0))
        return max(1, int(val))
    except Exception:
        return 1000

_write_q: "Queue[tuple[str, dict]]" = Queue(maxsize=20000)
_stop_writer = threading.Event()

def writer_loop():
    conn = get_conn(readonly=False)
    cur = conn.cursor()
    try:
        while not _stop_writer.is_set():
            try:
                kind, p = _write_q.get(timeout=0.25)
            except Empty:
                continue
            try:
                if kind == "alert":
                    cur.execute(
                        "INSERT INTO alerts(ts_ms,attack_type,severity,score,details) VALUES(?,?,?,?,?)",
                        (int(p["ts"]), p["attack_type"], p.get("severity"), p.get("score"),
                         json.dumps(p.get("details") or {})),
                    )
                elif kind == "packet":
                    cur.execute(
                        """INSERT INTO packets(ts_ms,can_id,dlc,data_hex,size_bytes,attack,attack_type,
                                                severity,score,signal,bar_height,meta)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            float(p["ts_ms"]),
                            p.get("can_id"),
                            p.get("dlc"),
                            p.get("data_hex"),
                            p.get("size_bytes"),
                            1 if p.get("attack") else 0,
                            p.get("attack_type"),
                            p.get("severity"),
                            p.get("score"),
                            p.get("signal"),
                            p.get("bar_height"),
                            json.dumps(p.get("meta") or {}),
                        ),
                    )
                conn.commit()
            except Exception:
                log.exception("[writer] failed to persist %s", kind)
            finally:
                _write_q.task_done()
    finally:
        conn.close()

def enqueue(kind: str, payload: Dict[str, Any]):
    try:
        _write_q.put_nowait((kind, payload))
    except Exception:
        try:
            _write_q.get_nowait()
            _write_q.task_done()
            _write_q.put_nowait((kind, payload))
            log.warning("[writer] queue full, dropped oldest to enqueue %s", kind)
        except Exception:
            log.error("[writer] dropped %s permanently (queue saturated)", kind)

def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = {k: row[k] for k in row.keys()}
    if "details" in d and d["details"]:
        try:
            d["details"] = json.loads(d["details"])
        except Exception:
            pass
    if "meta" in d and d["meta"]:
        try:
            d["meta"] = json.loads(d["meta"])
        except Exception:
            pass
    return d

init_db()
_writer_thread = threading.Thread(target=writer_loop, name="sqlite-writer", daemon=True)
_writer_thread.start()

# ==============================================================================
# Alerts: REST + WebSocket (DB persistence)
# ==============================================================================
ALERTS_MAX = 1000
alerts_buffer = deque(maxlen=ALERTS_MAX)
alerts_clients = set()
alerts_lock = threading.Lock()

def _broadcast_alert(obj: dict):
    dead = []
    payload = _json_dumps_safe(obj)
    with alerts_lock:
        for ws in list(alerts_clients):
            try:
                _ws_safe_send(ws, payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            alerts_clients.discard(ws)
            _train_send_locks.pop(ws, None)


def _make_alert(
    attack_type: str,
    severity: str = "medium",
    score: float | None = None,
    details: dict | None = None,
    ts_ms: int | None = None,
):
    ts = ts_ms if ts_ms is not None else int(time.time() * 1000)
    item = {
        "ts": ts,
        "timestamp": ts,
        "attack_type": attack_type,
        "type": attack_type,
        "severity": severity,
        "score": float(score) if score is not None else round(random.uniform(0.6, 0.99), 3),
        "details": details or {},
    }
    alerts_buffer.append(item)
    _broadcast_alert(item)
    try:
        enqueue("alert", {
            "ts": ts,
            "attack_type": attack_type,
            "severity": severity,
            "score": item["score"],
            "details": details or {},
        })
    except Exception:
        pass
    return item

STREAM_ON = os.getenv("DEMO_STREAM", "1") not in ("0", "false", "False")
try:
    DEMO_RATE_HZ = float(os.getenv("DEMO_RATE_HZ", "5"))
except Exception:
    DEMO_RATE_HZ = 5.0

def _demo_alert_stream():
    types = ["dos", "flood", "suspicious_frame", "impersonation", "replay", "unknown"]
    severities = ["low", "medium", "high"]
    while True:
        if STREAM_ON:
            _make_alert(
                attack_type=random.choice(types),
                severity=random.choice(severities),
                score=round(random.uniform(0.6, 0.99), 3),
                details={"src": hex(random.randint(0, 0x7FF))},
            )
            time.sleep(max(0.01, 1.0 / DEMO_RATE_HZ))
        else:
            time.sleep(0.2)
threading.Thread(target=_demo_alert_stream, daemon=True, name="demo-alerts").start()

if not alerts_buffer:
    now = int(time.time() * 1000)
    types = ["dos", "flood", "suspicious_frame", "impersonation", "replay", "unknown"]
    severities = ["low", "medium", "high"]
    for _ in range(120):
        ts = now - random.randint(0, 59) * 1000 - random.randint(0, 999)
        _make_alert(
            attack_type=random.choice(types),
            severity=random.choice(severities),
            score=round(random.uniform(0.6, 0.99), 3),
            details={"src": hex(random.randint(0, 0x7FF))},
            ts_ms=ts,
        )

@app.post("/api/demo/stream/start")
def api_demo_stream_start():
    global STREAM_ON
    STREAM_ON = True
    return jsonify({"ok": True, "stream": "started"})

@app.post("/api/demo/stream/stop")
def api_demo_stream_stop():
    global STREAM_ON
    STREAM_ON = False
    return jsonify({"ok": True, "stream": "stopped"})

@app.post("/api/demo/trigger")
def api_demo_trigger():
    body = request.get_json(force=True, silent=True) or {}
    attack = str(body.get("attack_type") or body.get("type") or "unknown")
    severity = str(body.get("severity") or "medium")
    try:
        score = float(body["score"]) if "score" in body else None
    except Exception:
        score = None
    details = body.get("details") or {}
    return jsonify(_make_alert(attack, severity=severity, score=score, details=details)), 201

@app.post("/api/demo/burst")
def api_demo_burst():
    try:
        n = int(request.args.get("n", 50))
    except Exception:
        n = 50
    types = ["dos", "flood", "suspicious_frame", "impersonation", "replay", "unknown"]
    severities = ["low", "medium", "high"]
    now = int(time.time() * 1000)
    for _ in range(n):
        ts = now - random.randint(0, 29) * 1000 - random.randint(0, 999)
        _make_alert(
            attack_type=random.choice(types),
            severity=random.choice(severities),
            score=round(random.uniform(0.6, 0.99), 3),
            details={"src": hex(random.randint(0, 0x7FF))},
            ts_ms=ts,
        )
    return jsonify({"ok": True, "added": n}), 200

@app.get("/api/alerts/export.csv")
def alerts_export_csv():
    import csv
    sio = io.StringIO()
    w = csv.writer(sio)
    w.writerow(["ts", "attack_type", "severity", "score", "details"])
    for a in list(alerts_buffer):
        w.writerow([
            a.get("ts", ""),
            a.get("attack_type") or a.get("type") or "unknown",
            a.get("severity", ""),
            a.get("score", ""),
            json.dumps(a.get("details", {})),
        ])
    bio = io.BytesIO(sio.getvalue().encode())
    bio.seek(0)
    return send_file(bio, as_attachment=True, download_name="alerts.csv", mimetype="text/csv")

@app.get("/api/alerts")
def api_alerts():
    try:
        limit = min(max(int(request.args.get("limit", 100)), 1), 5000)
    except Exception:
        limit = 100
    since_arg = request.args.get("since")
    type_arg = request.args.get("type") or request.args.get("attack_type")
    severity_arg = request.args.get("severity")

    since_ms = parse_since_ms(since_arg)

    conds, params = [], []
    if since_ms is not None:
        conds.append("ts_ms > ?")
        params.append(since_ms)
    if type_arg:
        conds.append("attack_type = ?")
        params.append(type_arg)
    if severity_arg:
        conds.append("severity = ?")
        params.append(severity_arg)

    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    sql = f"SELECT ts_ms as ts, attack_type, severity, score, details FROM alerts {where} ORDER BY ts_ms ASC LIMIT ?"
    params.append(limit)

    conn = get_conn(readonly=True)
    try:
        rows = conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            d = row_to_dict(r)
            d["timestamp"] = d["ts"]
            d["type"] = d["attack_type"]
            out.append(d)
        return jsonify(out), 200
    finally:
        conn.close()

@app.post("/api/alerts/clear")
def api_alerts_clear():
    alerts_buffer.clear()
    conn = get_conn(readonly=False)
    try:
        conn.execute("DELETE FROM alerts")
        conn.commit()
    finally:
        conn.close()
    return jsonify({"ok": True, "count": 0})

@app.get("/api/alerts/stats")
def api_alerts_stats():
    try:
        win = int(request.args.get("window", 300))
    except Exception:
        win = 300
    data = list(alerts_buffer)[-win:]
    out = {}
    for a in data:
        k = a.get("attack_type") or a.get("type") or "unknown"
        out[k] = out.get(k, 0) + 1
    return jsonify(out), 200

# --- drop-in replacement for ws_alerts ---
# --- replace the whole ws_alerts with this ---
@sock.route("/ws/alerts")
def ws_alerts(ws):
    log.info("WS /ws/alerts: client connected")
    with alerts_lock:
        alerts_clients.add(ws)

    ping_stop = threading.Event()
    def _pinger():
        while not ping_stop.is_set():
            try:
                _ws_safe_send(ws, _json_dumps_safe({"type": "ping", "t": time.time()}))
            except Exception:
                break
            time.sleep(15)

    ping_thread = threading.Thread(target=_pinger, name="ws-alerts-ping", daemon=True)
    ping_thread.start()

    try:
        recent = list(alerts_buffer)[-50:]
        if recent:
            try:
                _ws_safe_send(ws, _json_dumps_safe({"type": "replay_alerts", "items": recent}))
            except Exception as e:
                log.info("WS /ws/alerts: initial send failed → %r", e)
                return

        while True:
            try:
                msg = ws.receive()
            except Exception as e:
                log.info("WS /ws/alerts: receive error → %r", e)
                break
            if msg is None:
                break
            # push-only stream
            sleep(1.0)  # or time.sleep(1.0)
    finally:
        ping_stop.set()
        try:
            ping_thread.join(timeout=1.0)
        except Exception:
            pass
        with alerts_lock:
            alerts_clients.discard(ws)
        log.info("WS /ws/alerts: client disconnected")

# ==============================================================================
# SQL-optimized time-series
# ==============================================================================
@app.get("/api/alerts/timeseries")
def api_alerts_timeseries():
    window_ms = parse_window_ms(request.args.get("window", "1s"))
    now_ms = time.time() * 1000.0
    since_ms = parse_since_ms(request.args.get("since")) or (now_ms - 5 * 60_000.0)

    types_arg = request.args.get("types")
    params: list[Any] = [window_ms, window_ms, since_ms, now_ms]
    where = "ts_ms >= ? AND ts_ms <= ?"
    if types_arg:
        types_filter = [t.strip() for t in types_arg.split(",") if t.strip()]
        if types_filter:
            placeholders = ",".join(["?"] * len(types_filter))
            where += f" AND attack_type IN ({placeholders})"
            params.extend(types_filter)

    sql = f"""
        SELECT CAST(ts_ms / ? AS INTEGER) * ? AS bucket,
               attack_type,
               COUNT(*) AS n
        FROM alerts
        WHERE {where}
        GROUP BY bucket, attack_type
        ORDER BY bucket ASC
    """

    conn = get_conn(readonly=True)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    per_bucket: dict[int, dict[str, Any]] = {}
    for r in rows:
        b = int(r["bucket"])
        typ = r["attack_type"] or "unknown"
        n = int(r["n"])
        bucket_obj = per_bucket.setdefault(b, {"t": b, "count": 0})
        bucket_obj["count"] += n
        bucket_obj[typ] = bucket_obj.get(typ, 0) + n

    out = []
    start_bucket = int(since_ms // window_ms) * window_ms
    end_bucket = int(now_ms // window_ms) * window_ms
    max_buckets = 10_000
    if ((end_bucket - start_bucket) // window_ms) + 1 > max_buckets:
        start_bucket = end_bucket - (max_buckets - 1) * window_ms

    b = start_bucket
    while b <= end_bucket:
        out.append(per_bucket.get(b, {"t": b, "count": 0}))
        b += window_ms

    return jsonify(out), 200

@app.get("/api/packets/timeseries")
def api_packets_timeseries():
    window_ms = parse_window_ms(request.args.get("window", "1s"))
    now_ms = time.time() * 1000.0
    since_ms = parse_since_ms(request.args.get("since")) or (now_ms - 5 * 60_000.0)

    can_id_param = request.args.get("can_id")
    can_id_val = None
    if can_id_param:
        try:
            can_id_val = int(can_id_param, 0)
        except Exception:
            return jsonify({"error": "invalid can_id"}), 400

    where = "ts_ms >= ? AND ts_ms <= ?"
    params: list[Any] = [window_ms, window_ms, since_ms, now_ms]
    if can_id_val is not None:
        where += " AND can_id = ?"
        params.append(can_id_val)

    sql = f"""
        SELECT CAST(ts_ms / ? AS INTEGER) * ? AS bucket,
               COUNT(*) AS n,
               COALESCE(SUM(size_bytes), 0) AS bytes_total,
               SUM(CASE WHEN attack = 1 THEN 1 ELSE 0 END) AS attacks
        FROM packets
        WHERE {where}
        GROUP BY bucket
        ORDER BY bucket ASC
    """

    conn = get_conn(readonly=True)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    per_bucket = {
        int(r["bucket"]): {
            "t": int(r["bucket"]),
            "count": int(r["n"]),
            "bytes": int(r["bytes_total"] or 0),
            "attacks": int(r["attacks"] or 0),
        } for r in rows
    }

    out = []
    start_bucket = int(since_ms // window_ms) * window_ms
    end_bucket = int(now_ms // window_ms) * window_ms
    max_buckets = 10_000
    if ((end_bucket - start_bucket) // window_ms) + 1 > max_buckets:
        start_bucket = end_bucket - (max_buckets - 1) * window_ms

    b = start_bucket
    while b <= end_bucket:
        out.append(per_bucket.get(b, {"t": b, "count": 0, "bytes": 0, "attacks": 0}))
        b += window_ms

    return jsonify(out), 200

# ==============================================================================
# Traffic generator + WebSocket (/ws/traffic)
# ==============================================================================
TRAFFIC_ON = os.getenv("TRAFFIC_ON", "1") not in ("0", "false", "False")
try:
    TRAFFIC_MEAN_PPS = float(os.getenv("TRAFFIC_MEAN_PPS", "12"))
except Exception:
    TRAFFIC_MEAN_PPS = 12.0
try:
    TRAFFIC_BURST_PROB = float(os.getenv("TRAFFIC_BURST_PROB", "0.03"))
except Exception:
    TRAFFIC_BURST_PROB = 0.03
try:
    TRAFFIC_BURST_DURATION_MEAN = float(os.getenv("TRAFFIC_BURST_DURATION_MEAN", "1.2"))
except Exception:
    TRAFFIC_BURST_DURATION_MEAN = 1.2
try:
    TRAFFIC_BURST_MULT = float(os.getenv("TRAFFIC_BURST_MULT", "8.0"))
except Exception:
    TRAFFIC_BURST_MULT = 8.0
try:
    TRAFFIC_ATTACK_PROB = float(os.getenv("TRAFFIC_ATTACK_PROB", "0.015"))
except Exception:
    TRAFFIC_ATTACK_PROB = 0.015

TRAFFIC_PROFILE = os.getenv("TRAFFIC_PROFILE", "default")

COMMON_ID_WEIGHTS = {0x100: 50, 0x200: 30, 0x300: 20, 0x400: 14, 0x2AA: 10, 0x7FF: 8}
RANDOM_ID_RANGE = (0x000, 0x7FF)

class SimpleNoise1D:
    def __init__(self, seed=None, size=1024):
        self.size = size
        self.seed = seed or random.randint(0, 2**30)
        rnd = random.Random(self.seed)
        self.table = [rnd.random() for _ in range(size+1)]
    @staticmethod
    def _smoothstep(t: float) -> float:
        return t * t * (3 - 2 * t)
    def noise(self, x: float) -> float:
        xi = int(_math.floor(x)) % self.size
        xf = x - _math.floor(x)
        a = self.table[xi]
        b = self.table[xi+1]
        t = self._smoothstep(xf)
        return a * (1-t) + b * t

def fractal_noise(noise_obj: SimpleNoise1D, x: float, octaves=4, lacunarity=2.0, gain=0.5) -> float:
    amp, freq, total, max_amp = 1.0, 1.0, 0.0, 0.0
    for _ in range(octaves):
        total += noise_obj.noise(x * freq) * amp
        max_amp += amp
        amp *= gain
        freq *= lacunarity
    return total / max_amp if max_amp else 0.0

GLOBAL_NOISE = SimpleNoise1D(seed=42, size=2048)

traffic_clients = set()
traffic_lock = threading.Lock()

ATTACK_CLUSTER_ON = False
CURRENT_CLUSTER_UNTIL = 0.0

RANDOMNESS = {
    "bar_base_mean": 20.0,
    "bar_base_std": 8.0,
    "heavy_tail_prob": 0.10,
    "heavy_tail_multiplier": 4.0,
    "attack_base_prob": 0.03,
    "anti_streak_limit": 2,
    "max_jitter_ms": 200,
}
ATTACK_TYPES = ["dos", "flood", "suspicious_frame", "impersonation", "replay", "spoof", "sni", "brute"]
_last_attack_type = None
_last_attack_streak = 0
_last_sent_id = None

def pick_can_id():
    if random.random() < 0.85:
        ids = list(COMMON_ID_WEIGHTS.keys())
        weights = list(COMMON_ID_WEIGHTS.values())
        return random.choices(ids, weights=weights, k=1)[0]
    else:
        return random.randint(*RANDOM_ID_RANGE)

def make_payload(dlc=None, structured=False):
    if dlc is None:
        r = random.random()
        if r < 0.01:
            dlc = 0
        elif r < 0.55:
            dlc = random.randint(1, 3)
        else:
            dlc = random.randint(4, 8)
    if structured and dlc > 0:
        base = random.randint(0, 255)
        return dlc, [(base + i*5 + random.randint(-2,2)) & 0xFF for i in range(dlc)]
    else:
        return dlc, [random.randint(0, 255) for _ in range(dlc)]

def sample_bar_height():
    base = random.gauss(RANDOMNESS["bar_base_mean"], max(1.0, RANDOMNESS["bar_base_std"]))
    base = max(1.0, base)
    if random.random() < RANDOMNESS["heavy_tail_prob"]:
        base *= (1.0 + random.random() * RANDOMNESS["heavy_tail_multiplier"])
    base *= (0.8 + random.random() * 0.8)
    return round(base, 2)

def sample_attack_type():
    global _last_attack_type, _last_attack_streak
    if random.random() > RANDOMNESS["attack_base_prob"]:
        _last_attack_streak = 0
        _last_attack_type = None
        return None
    candidate = random.choice(ATTACK_TYPES)
    if candidate == _last_attack_type:
        _last_attack_streak += 1
    else:
        _last_attack_streak = 1
        _last_attack_type = candidate
    if _last_attack_streak > RANDOMNESS["anti_streak_limit"]:
        other = [a for a in ATTACK_TYPES if a != _last_attack_type]
        if other:
            candidate = random.choice(other)
        _last_attack_streak = 1
        _last_attack_type = candidate
    return candidate

def chaotic_packet_payload(can_id):
    dlc, data = make_payload()
    bar_height = sample_bar_height()
    now_s = time.time()
    signal = _math.sin(now_s * (0.3 + random.random())) * (0.2 + random.random()*0.6) + (random.random()*0.4 - 0.2)
    return dlc, data, bar_height, signal

def broadcast_traffic(obj: dict):
    payload = _json_dumps_safe(obj)
    dead = []
    with traffic_lock:
        for ws in list(traffic_clients):
            try:
                _ws_safe_send(ws, payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            traffic_clients.discard(ws)
            _train_send_locks.pop(ws, None)
    try:
        if obj.get("type") == "packet" and "payload" in obj:
            p = obj["payload"]
            ts_val = p.get("ts", time.time())
            ts_ms = float(ts_val) * 1000.0 if ts_val < 10_000_000_000 else float(ts_val)
            data_hex = "".join(f"{b:02x}" for b in (p.get("data") or []))
            enqueue("packet", {
                "ts_ms": ts_ms,
                "can_id": int(p.get("can_id", 0)),
                "dlc": int(p.get("dlc", 0)),
                "data_hex": data_hex,
                "size_bytes": int(p.get("size_bytes", 0)),
                "attack": bool(p.get("attack")),
                "attack_type": p.get("attack_type"),
                "severity": p.get("severity"),
                "score": p.get("score"),
                "signal": p.get("signal"),
                "bar_height": p.get("bar_height"),
                "meta": {"id_hex": p.get("id")},
            })
    except Exception as e:
        log.exception("[packet persist] error: %s", e)

if os.getenv("TRAFFIC_PROFILE", "default") == "snapshot":
    TRAFFIC_MEAN_PPS = 25.0
    TRAFFIC_BURST_PROB = 0.06
    TRAFFIC_BURST_MULT = 12.0
    TRAFFIC_ATTACK_PROB = 0.10
    RANDOMNESS["attack_base_prob"] = 0.06
    RANDOMNESS["heavy_tail_prob"] = 0.15

def traffic_generator():
    global _last_sent_id, ATTACK_CLUSTER_ON, CURRENT_CLUSTER_UNTIL, TRAFFIC_MEAN_PPS
    burst_until = 0.0
    t0 = time.time()
    while True:
        now = time.time()
        if not TRAFFIC_ON:
            time.sleep(0.2)
            continue

        env = max(0.1, 0.6 + fractal_noise(GLOBAL_NOISE, (now - t0) * (0.05 + random.random()*0.02), octaves=4))
        in_cluster = ATTACK_CLUSTER_ON and (now < CURRENT_CLUSTER_UNTIL)

        if now < burst_until:
            base_rate = TRAFFIC_MEAN_PPS * TRAFFIC_BURST_MULT
        else:
            base_rate = TRAFFIC_MEAN_PPS
            if random.random() < TRAFFIC_BURST_PROB:
                burst_len = random.expovariate(1.0 / max(0.05, TRAFFIC_BURST_DURATION_MEAN))
                burst_until = now + burst_len

        if random.random() < 0.12:
            can_id = random.randint(0x000, 0x7FF)
        else:
            can_id = pick_can_id()

        dlc, payload, bar_height, signal = chaotic_packet_payload(can_id)
        pkt_ts = now + (random.random() * RANDOMNESS["max_jitter_ms"] / 1000.0 * (random.choice([-1,1])))

        attack_type = sample_attack_type()
        if in_cluster and random.random() < 0.5:
            attack_type = attack_type or random.choice(ATTACK_TYPES)

        packet = {
            "ts": pkt_ts,
            "id": hex(can_id),
            "can_id": can_id,
            "dlc": dlc,
            "data": payload,
            "size_bytes": 4 + dlc,
            "signal": signal,
            "bar_height": bar_height,
        }
        if attack_type:
            packet["attack"] = True
            packet["attack_type"] = attack_type
            packet["severity"] = random.choice(["low", "medium", "high"])
            packet["score"] = round(random.uniform(0.6, 0.99), 3)
        else:
            packet["attack"] = False

        if _last_sent_id == can_id and random.random() < 0.3:
            can_id = pick_can_id()
        _last_sent_id = can_id

        try:
            broadcast_traffic({"type": "packet", "payload": packet})
        except Exception:
            pass

        current_rate = max(0.1, base_rate * env * (3.0 if in_cluster else 1.0))
        inter_arrival = random.expovariate(current_rate)
        sleep_t = max(0.0005, inter_arrival * (0.5 + random.random() * 1.8))
        time.sleep(sleep_t)

@sock.route("/ws/traffic")
def ws_traffic(ws):
    log.info("WS /ws/traffic: client connected")
    with traffic_lock:
        traffic_clients.add(ws)
    try:
        _ws_safe_send(ws, _json_dumps_safe({"type": "hello", "msg": "chaotic traffic simulator connected"}))
        last_ping = time.time()
        while True:
            # Keepalive
            now = time.time()
            if (now - last_ping) >= 15:
                try:
                   _ws_safe_send(ws, _json_dumps_safe({"type": "ping", "t": now}))
                except Exception:
                    break
                last_ping = now

            msg = ws.receive()
            if msg is None:
                break
            try:
                o = json.loads(msg)
            except Exception:
                o = {}
            if not isinstance(o, dict):
                continue
            cmd = o.get("cmd")
            if cmd == "set_rate":
                global TRAFFIC_MEAN_PPS
                try:
                    TRAFFIC_MEAN_PPS = max(0.1, float(o.get("rate", TRAFFIC_MEAN_PPS)))
                    ws.send(_json_dumps_safe({"type":"info","msg":f"rate set to {TRAFFIC_MEAN_PPS}"}))
                except Exception:
                    ws.send(_json_dumps_safe({"type":"error","msg":"invalid rate"}))
            elif cmd == "set_profile":
                prof = str(o.get("profile","default"))
                if prof == "snapshot":
                    global TRAFFIC_BURST_PROB, TRAFFIC_BURST_MULT, TRAFFIC_ATTACK_PROB
                    TRAFFIC_MEAN_PPS = 25.0
                    TRAFFIC_BURST_PROB = 0.06
                    TRAFFIC_BURST_MULT = 12.0
                    TRAFFIC_ATTACK_PROB = 0.10
                    RANDOMNESS["attack_base_prob"] = 0.06
                    RANDOMNESS["heavy_tail_prob"] = 0.15
                    ws.send(_json_dumps_safe({"type":"info","msg":"profile snapshot applied"}))
                else:
                    ws.send(_json_dumps_safe({"type":"info","msg":"profile default applied"}))
            elif cmd == "start_cluster":
                dur = float(o.get("duration", 4.0))
                global ATTACK_CLUSTER_ON, CURRENT_CLUSTER_UNTIL
                ATTACK_CLUSTER_ON = True
                CURRENT_CLUSTER_UNTIL = time.time() + max(0.5, dur)
                ws.send(_json_dumps_safe({"type":"info","msg":f"cluster started for {dur}s"}))
            elif cmd == "set_randomness":
                updates = o.get("params", {})
                for k, v in updates.items():
                    if k in RANDOMNESS:
                        try:
                            RANDOMNESS[k] = float(v) if isinstance(RANDOMNESS[k], (int,float)) else v
                        except Exception:
                            RANDOMNESS[k] = v
                ws.send(_json_dumps_safe({"type":"info","msg":"randomness updated","randomness":RANDOMNESS}))
    except Exception:
        pass
    finally:
        with traffic_lock:
            traffic_clients.discard(ws)
        log.info("WS /ws/traffic: client disconnected")

@app.post("/api/traffic/start")
def api_traffic_start():
    global TRAFFIC_ON
    TRAFFIC_ON = True
    return jsonify({"ok": True, "traffic": "started"})

@app.post("/api/traffic/stop")
def api_traffic_stop():
    global TRAFFIC_ON
    TRAFFIC_ON = False
    return jsonify({"ok": True, "traffic": "stopped"})

@app.post("/api/traffic/set_rate")
def api_traffic_set_rate():
    body = request.get_json(force=True, silent=True) or {}
    try:
        r = float(body.get("rate"))
    except Exception:
        return jsonify({"ok": False, "error": "invalid rate"}), 400
    global TRAFFIC_MEAN_PPS
    TRAFFIC_MEAN_PPS = max(0.1, r)
    return jsonify({"ok": True, "rate": TRAFFIC_MEAN_PPS})

@app.get("/api/traffic/randomness")
def api_traffic_randomness_get():
    return jsonify(RANDOMNESS)

@app.post("/api/traffic/randomness")
def api_traffic_randomness_set():
    body = request.get_json(force=True, silent=True) or {}
    for k, v in body.items():
        if k in RANDOMNESS:
            try:
                RANDOMNESS[k] = float(v) if isinstance(RANDOMNESS[k], (int,float)) else v
            except Exception:
                RANDOMNESS[k] = v
    return jsonify({"ok": True, "randomness": RANDOMNESS})

# ------------------ BEGIN: Phase-6 ReplayEngine integration (project-root/replay_engine.py) ------------------
# Place this after your alerts/demo endpoints so _make_alert, broadcast_traffic, alerts_lock exist.

try:
    # project-root/replay_engine.py (you chose project root)
    from replay_engine import ReplayEngine, load_csv_log
except Exception:
    ReplayEngine = None
    load_csv_log = None
    log.warning("[replay-integ] replay_engine import failed; replay endpoints disabled until replay_engine.py is added")

def _replay_ws_emitter(event_name: str, payload: dict):
    """
    Adapter called by ReplayEngine when it *wants* to emit an alert.
    We normalize and forward to server pipeline:
      - call _make_alert(...) so alerts are persisted + broadcast via ws_alerts
      - call broadcast_traffic(...) so UI gets the packet in the packet stream
    """
    try:
        if not isinstance(payload, dict):
            return
        # Accept either {type, attack_type, severity, score, details, frame}
        attack_type = str(payload.get("attack_type") or payload.get("type") or payload.get("label") or "unknown")
        severity = str(payload.get("severity") or "medium")
        score = None
        try:
            if "score" in payload:
                score = float(payload.get("score"))
        except Exception:
            score = None
        details = payload.get("details") or {}

        # If payload includes a frame, attach it and also forward as a packet to traffic stream
        frame = payload.get("frame") or {}
        if frame and isinstance(frame, dict):
            # Make sure ts is present
            ts_val = frame.get("ts", time.time())
            # Persist/broadcast alert (this also appends to alerts_buffer and DB)
            _make_alert(attack_type, severity=severity, score=score, details={**details, "frame": frame, "src": frame.get("can_id")})
            # Also forward the raw packet into traffic stream for UI
            try:
                p = {
                    "ts": ts_val,
                    "id": hex(int(frame.get("can_id", 0))) if frame.get("can_id") is not None else None,
                    "can_id": int(frame.get("can_id", 0) or 0),
                    "dlc": int(frame.get("dlc", 0) or 0),
                    "data": list(frame.get("data") or []),
                    "size_bytes": 4 + int(frame.get("dlc", 0) or 0),
                    "attack": True,
                    "attack_type": attack_type,
                    "severity": severity,
                    "score": score,
                    "signal": frame.get("signal"),
                    "bar_height": frame.get("bar_height"),
                    "meta": {"replay": True},
                }
                broadcast_traffic({"type": "packet", "payload": p})
            except Exception:
                log.exception("[replay-emitter] failed to forward packet to traffic stream")
        else:
            # No frame — still create an alert entry
            _make_alert(attack_type, severity=severity, score=score, details=details)
    except Exception:
        log.exception("[replay-emitter] unexpected error handling replay alert")

# Instantiate engine singleton if available
_replay_engine = None
if ReplayEngine is not None:
    try:
        _replay_engine = ReplayEngine(websocket_emitter=_replay_ws_emitter, vcan_channel=os.getenv("VCAN_CHANNEL", "vcan0"))
        log.info("[replay] ReplayEngine instantiated (model_path=%s)", getattr(_replay_engine, "_model_path", None))
    except Exception:
        log.exception("[replay] ReplayEngine instantiation failed")
        _replay_engine = None

def load_replay_dataset(dataset_name: str):
    """
    Supports:
    - backend/datasets/demo.csv
    - backend/datasets/<folder>/*.csv
    - datasets/<folder>/*.csv
    """

    search_roots = [
        os.path.join(BASE_DIR, "backend", "datasets"),
        os.path.join(BASE_DIR, "datasets"),
    ]

    csv_files = []

    for root in search_roots:
        base = os.path.join(root, dataset_name)

        # Case 1: single CSV
        if os.path.isfile(base + ".csv"):
            csv_files.append(base + ".csv")

        # Case 2: folder of CSV chunks
        elif os.path.isdir(base):
            for f in sorted(os.listdir(base)):
                if f.lower().endswith(".csv"):
                    csv_files.append(os.path.join(base, f))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for dataset '{dataset_name}'")

    rows = []
    for csv_path in csv_files:
        rows.extend(load_csv_log(csv_path))

    return rows

@app.route("/api/replay/start", methods=["POST"])
def api_replay_start():
    if _replay_engine is None:
        return jsonify(ok=False, error="Replay engine not available"), 500

    body = request.get_json(force=True) or {}
    dataset = body.get("dataset", "demo")
    speed = float(body.get("speed", 1.0))

    try:
        rows = load_replay_dataset(dataset)

        log.info(
            "[replay] Loaded %d frames from dataset '%s'",
            len(rows),
            dataset
        )

        _replay_engine.start(
            log_rows=rows,
            speed=speed,
            mode="direct",
            to_vcan=False,
        )

        return jsonify(ok=True)

    except FileNotFoundError as e:
        return jsonify(ok=False, error=str(e)), 400

    except Exception as e:
        log.exception("[replay] start failed")
        return jsonify(ok=False, error=str(e)), 500

@app.post("/api/replay/stop")
def api_replay_stop():
    global _replay_engine
    if _replay_engine is None:
        return jsonify({"ok": False, "error": "replay engine unavailable"}), 500
    try:
        _replay_engine.stop()
        return jsonify({"ok": True, "state": _replay_engine.get_state()}), 200
    except Exception as e:
        log.exception("[api/replay/stop] stop failed")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/api/replay/speed")
def api_replay_speed():
    global _replay_engine
    if _replay_engine is None:
        return jsonify({"ok": False, "error": "replay engine unavailable"}), 500
    body = request.get_json(force=True, silent=True) or {}
    try:
        speed = float(body.get("speed", 1.0))
    except Exception:
        return jsonify({"ok": False, "error": "invalid speed"}), 400
    try:
        _replay_engine.set_speed(speed)
        return jsonify({"ok": True, "speed": speed}), 200
    except Exception as e:
        log.exception("[api/replay/speed] set_speed failed")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/api/replay/inject_attack")
def api_replay_inject_attack():
    global _replay_engine
    if _replay_engine is None:
        return jsonify({"ok": False, "error": "replay engine unavailable"}), 500
    body = request.get_json(force=True, silent=True) or {}
    attack_type = body.get("attack_type") or body.get("type")
    params = body.get("params", {})
    if not attack_type:
        return jsonify({"ok": False, "error": "attack_type required"}), 400
    try:
        injected = _replay_engine.inject_attack(attack_type, params)
        return jsonify({"ok": True, "injected_count": len(injected), "state": _replay_engine.get_state()}), 200
    except Exception as e:
        log.exception("[api/replay/inject_attack] inject failed")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/api/replay/state")
def api_replay_state():
    global _replay_engine
    if _replay_engine is None:
        return jsonify({"ok": True, "state": {"running": False}}), 200
    return jsonify({"ok": True, "state": _replay_engine.get_state()}), 200

@app.route("/api/replay/datasets", methods=["GET"])
def api_replay_datasets():
    backend_ds = os.path.join(BASE_DIR, "backend", "datasets")
    root_ds = os.path.join(BASE_DIR, "datasets")

    datasets = set()

    for ds_dir in (backend_ds, root_ds):
        if not os.path.isdir(ds_dir):
            continue

        for entry in os.listdir(ds_dir):
            full = os.path.join(ds_dir, entry)

            # single CSV
            if entry.lower().endswith(".csv"):
                datasets.add(os.path.splitext(entry)[0])

            # folder with CSV chunks
            elif os.path.isdir(full):
                for f in os.listdir(full):
                    if f.lower().endswith(".csv"):
                        datasets.add(entry)
                        break

    return jsonify({
        "ok": True,
        "datasets": sorted(datasets),
        "debug": {
            "backend_path": backend_ds,
            "root_path": root_ds,
        },
    })


# ------------------ END: Phase-6 ReplayEngine integration ------------------

# ==============================================================================
# Packets REST (DB)
# ==============================================================================
@app.get("/api/packets")
def api_packets():
    since_ms = parse_since_ms(request.args.get("since"))
    try:
        limit = min(max(int(request.args.get("limit", 500)), 1), 10000)
    except Exception:
        limit = 500

    can_id_param = request.args.get("can_id")
    can_id_val = None
    if can_id_param:
        try:
            can_id_val = int(can_id_param, 0)
        except Exception:
            return jsonify({"error": "invalid can_id"}), 400

    conds, params = [], []
    if since_ms is not None:
        conds.append("ts_ms > ?")
        params.append(since_ms)
    if can_id_val is not None:
        conds.append("can_id = ?")
        params.append(can_id_val)

    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    sql = f"""SELECT ts_ms, can_id, dlc, data_hex, size_bytes, attack, attack_type,
                     severity, score, signal, bar_height, meta
              FROM packets {where}
              ORDER BY ts_ms ASC
              LIMIT ?"""
    params.append(limit)

    conn = get_conn(readonly=True)
    try:
        rows = conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            d = row_to_dict(r)
            out.append({
                "ts": d["ts_ms"] / 1000.0,
                "can_id": d["can_id"],
                "id": hex(d["can_id"]) if d["can_id"] is not None else None,
                "dlc": d["dlc"],
                "data_hex": d["data_hex"],
                "size_bytes": d["size_bytes"],
                "attack": bool(d["attack"]),
                "attack_type": d.get("attack_type"),
                "severity": d.get("severity"),
                "score": d.get("score"),
                "signal": d.get("signal"),
                "bar_height": d.get("bar_height"),
                "meta": d.get("meta"),
            })
        return jsonify(out), 200
    finally:
        conn.close()

# Start background traffic thread
threading.Thread(target=traffic_generator, daemon=True, name="traffic-gen").start()

# ==============================================================================
# Email: config, toggle/snooze, send (hardened), branded digest, notifier
# ==============================================================================
SEV_ORDER = {"low": 1, "medium": 2, "high": 3}

def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if (v is not None and str(v).strip() != "") else default

def severity_meets_threshold(sev: Optional[str], min_sev: str) -> bool:
    s = (sev or "").lower()
    return SEV_ORDER.get(s, 0) >= SEV_ORDER.get(min_sev.lower(), 2)

def _mask(s: Optional[str]) -> str:
    if not s:
        return ""
    if len(s) <= 4:
        return "*" * len(s)
    return s[:2] + "*"*(len(s)-4) + s[-2:]

def _parse_recipients(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    return [p for p in parts if p]

# --- EMAIL TOGGLE + SNOOZE ---
EMAIL_ENABLED = threading.Event()
if (_get_env("EMAIL_ENABLED", "1") not in ("0", "false", "False")):
    EMAIL_ENABLED.set()   # default ON
else:
    EMAIL_ENABLED.clear() # default OFF

SNOOZE_UNTIL = 0.0  # epoch seconds; if now < SNOOZE_UNTIL, skip sending

def _email_allowed(now: float | None = None) -> bool:
    n = now if now is not None else time.time()
    return EMAIL_ENABLED.is_set() and n >= SNOOZE_UNTIL

@app.get("/api/alerts/email/status")
def api_email_status():
    now = time.time()
    return jsonify({
        "enabled": bool(EMAIL_ENABLED.is_set()),
        "snoozed_until": SNOOZE_UNTIL if SNOOZE_UNTIL > now else 0,
        "snoozed_for_s": max(0, int(SNOOZE_UNTIL - now)),
    }), 200

@app.post("/api/alerts/email/enabled")
def api_email_enabled():
    body = request.get_json(force=True, silent=True) or {}
    enabled = body.get("enabled")
    if isinstance(enabled, str):
        enabled = enabled.lower() not in ("0", "false", "no")
    enabled = bool(enabled)
    if enabled:
        EMAIL_ENABLED.set()
        msg = "email alerts enabled"
    else:
        EMAIL_ENABLED.clear()
        msg = "email alerts disabled"
    log.info("[email] %s", msg)
    return jsonify({"ok": True, "enabled": bool(EMAIL_ENABLED.is_set())}), 200

@app.post("/api/alerts/email/snooze")
def api_email_snooze():
    global SNOOZE_UNTIL
    minutes = request.args.get("minutes")
    body = request.get_json(force=True, silent=True) or {}
    if minutes is None:
        minutes = body.get("minutes", 15)
    try:
        m = max(1, int(float(minutes)))
    except Exception:
        return jsonify({"ok": False, "error": "invalid minutes"}), 400
    SNOOZE_UNTIL = time.time() + m * 60
    log.info("[email] snoozed for %d minutes", m)
    return jsonify({"ok": True, "snoozed_until": SNOOZE_UNTIL, "minutes": m}), 200

# --- SMTP helpers ---
def _smtp_try_send(host: str, port: int, user: str, pwd: str, msg: EmailMessage) -> bool:
    if port == 587:
        context = ssl.create_default_context()
        with smtplib.SMTP(host, port, timeout=15) as s:
            s.ehlo()
            s.starttls(context=context)
            s.ehlo()
            s.login(user, pwd)
            s.send_message(msg)
        return True
    if port == 465:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context, timeout=15) as s:
            s.login(user, pwd)
            s.send_message(msg)
        return True
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(host, port, timeout=15) as s:
            s.ehlo()
            s.starttls(context=context)
            s.ehlo()
            s.login(user, pwd)
            s.send_message(msg)
        return True
    except Exception:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, 465, context=context, timeout=15) as s:
            s.login(user, pwd)
            s.send_message(msg)
        return True

def send_email(subject: str, html_body: str, text_body: Optional[str] = None) -> bool:
    """
    Env:
      SMTP_HOST, SMTP_PORT(465/587), SMTP_USER, SMTP_PASS
      ALERT_EMAIL_TO = "a@x.com,b@y.com"
      ALERT_EMAIL_FROM (optional, defaults to SMTP_USER)
      EMAIL_RETRIES (default 3)
      EMAIL_BACKOFF_S (default 2)
      EMAIL_SUBJECT_PREFIX (optional, e.g. "[CAN-IDS]")
    """
    host = _get_env("SMTP_HOST")
    try:
        port = int(_get_env("SMTP_PORT", "587") or "587")
    except Exception:
        port = 587
    user = _get_env("SMTP_USER")
    pwd  = _get_env("SMTP_PASS")
    to_raw = _get_env("ALERT_EMAIL_TO")
    to_list = _parse_recipients(to_raw)
    from_addr = _get_env("ALERT_EMAIL_FROM", user) or user
    retries = max(0, int(_get_env("EMAIL_RETRIES", "3") or "3"))
    backoff = max(1, int(_get_env("EMAIL_BACKOFF_S", "2") or "2"))
    subj_prefix = _get_env("EMAIL_SUBJECT_PREFIX", "").strip()
    if subj_prefix:
        subject = f"{subj_prefix} {subject}"

    log.info("[email][cfg] host=%s:%s user=%s to=%s",
             host, port, user, ", ".join(to_list) if to_list else "-")

    if not all([host, port, user, pwd]) or not to_list:
        log.warning("[email] missing SMTP_* or ALERT_EMAIL_TO env; email send skipped")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_list)
    if text_body:
        msg.set_content(text_body)
    else:
        msg.set_content("See HTML part.")
    msg.add_alternative(html_body, subtype="html")

    attempt = 0
    while True:
        try:
            _smtp_try_send(host, port, user, pwd, msg)
            log.info("[email] sent → %s via %s:%s", msg["To"], host, port)
            return True
        except smtplib.SMTPAuthenticationError as e:
            log.error("[email] auth failed: %s", e)
            return False
        except Exception as e:
            attempt += 1
            log.warning("[email] send attempt %d failed: %s", attempt, e)
            if attempt > retries:
                log.error("[email] giving up after %d attempts", retries)
                return False
            time.sleep(backoff * attempt)

def _uptime_str() -> str:
    secs = int(time.time() - START_TIME)
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    parts = []
    if days: parts.append(f"{days}d")
    if hours: parts.append(f"{hours}h")
    if minutes: parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)

def _sev_badge_html(sev: str) -> str:
    s = (sev or "").lower()
    color_bg = {"low": "#16a34a", "medium": "#f59e0b", "high": "#dc2626"}.get(s, "#6b7280")
    return f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;color:white;font-weight:600;background:{color_bg};text-transform:capitalize">{s or "-"}</span>'

def _local_time_str(ms: float) -> str:
    t = time.localtime(ms/1000.0)
    tz = time.tzname[time.daylight] if time.daylight else time.tzname[0]
    return time.strftime("%Y-%m-%d %H:%M:%S", t) + (f" {tz}" if tz else "")

def _render_alert_digest_v2(
    min_sev: str,
    items: List[Dict[str, Any]],
    cooldown_s: float
) -> Tuple[str, str]:
    """
    Consumer-friendly HIGH severity alert email.
    No technical details. Industry-grade language.
    """

    frontend = _get_env("FRONTEND_URL", "http://localhost:5173")

    subject = "⚠️ Vehicle Security Alert – Immediate Attention Recommended"

    html = f"""
<!doctype html>
<html>
  <body style="margin:0;font-family:Arial,Helvetica,sans-serif;background:#f3f4f6;">
    <div style="max-width:640px;margin:0 auto;padding:24px">
      
      <!-- Header -->
      <div style="background:#111827;color:#ffffff;padding:20px;border-radius:12px;">
        <h2 style="margin:0;font-size:20px;">Vehicle Security Alert</h2>
        <p style="margin:6px 0 0;font-size:13px;opacity:0.9;">
          High severity activity detected
        </p>
      </div>

      <!-- Body -->
      <div style="background:#ffffff;border-radius:12px;padding:20px;margin-top:16px;">
        <p style="font-size:14px;color:#111827;">
          Our vehicle security system has detected <b>unusual activity</b> affecting
          your vehicle’s communication network.
        </p>

        <p style="font-size:14px;color:#111827;">
          This alert does <b>not</b> mean your vehicle is unsafe to drive, but it does
          require your attention.
        </p>

        <h3 style="font-size:15px;margin-top:18px;">What you should do now</h3>
        <ul style="font-size:14px;color:#111827;padding-left:18px;">
          <li>Ensure your vehicle is parked safely.</li>
          <li>Avoid connecting unknown devices to the vehicle.</li>
          <li>If the alert persists, contact an authorized service center.</li>
        </ul>

        <p style="font-size:14px;color:#111827;margin-top:14px;">
          Our systems are actively monitoring the situation and have applied
          protective measures automatically.
        </p>

        <!-- CTA -->
        <div style="margin-top:20px;">
          <a href="{frontend}"
             style="display:inline-block;background:#2563eb;color:#ffffff;
                    padding:10px 16px;border-radius:8px;text-decoration:none;
                    font-size:14px;font-weight:600;">
            View Vehicle Status
          </a>
        </div>
      </div>

      <!-- Footer -->
      <p style="font-size:12px;color:#6b7280;text-align:center;margin-top:14px;">
        This is an automated security notification. No action is required unless
        advised above.
      </p>

    </div>
  </body>
</html>
    """

    text = (
        "Vehicle Security Alert (High Severity)\n\n"
        "Unusual activity has been detected in your vehicle’s systems.\n\n"
        "What you should do:\n"
        "- Park your vehicle safely\n"
        "- Avoid connecting unknown devices\n"
        "- Contact an authorized service center if alerts continue\n\n"
        "Our systems are actively monitoring and protecting your vehicle."
    )

    return html, text


_notifier_stop = threading.Event()

def notifier_loop():
    """
    Poll SQLite for new alerts and email when severity crosses threshold.
    Respects EMAIL_ENABLED and SNOOZE_UNTIL.
    """
    global SNOOZE_UNTIL
    min_sev = _get_env("ALERT_MIN_SEVERITY", "high")
    cooldown_s = float(_get_env("ALERT_COOLDOWN_S", "60") or "60")
    last_sent_at = 0.0

    conn = get_conn(readonly=True)
    try:
        row = conn.execute("SELECT COALESCE(MAX(id),0) AS max_id FROM alerts").fetchone()
        last_id = int(row["max_id"] or 0)
    finally:
        conn.close()

    log.info("[notifier] starting: min_sev=%s cooldown=%ss (resume id=%s)", min_sev, cooldown_s, last_id)

    while not _notifier_stop.is_set():
        time.sleep(2.0)

        conn = get_conn(readonly=True)
        try:
            rows = conn.execute(
                "SELECT id, ts_ms, attack_type, severity, score, details FROM alerts WHERE id > ? ORDER BY id ASC",
                (last_id,),
            ).fetchall()
        except Exception:
            log.exception("[notifier] query failed")
            rows = []
        finally:
            conn.close()

        if rows:
            last_id = int(rows[-1]["id"])

        interesting = []
        for r in rows:
            sev = (r["severity"] or "").lower()
            if severity_meets_threshold(sev, min_sev):
                interesting.append(row_to_dict(r))

        if not interesting:
            continue

        now = time.time()
        if not _email_allowed(now):
            continue
        if (now - last_sent_at) < cooldown_s:
            continue

        html, text = _render_alert_digest_v2(min_sev, interesting, cooldown_s)
        subject = f"CAN IDS: Alerts ≥ {min_sev.title()}"
        ok = send_email(subject=subject, html_body=html, text_body=text)
        if ok:
            last_sent_at = now
        else:
            log.error("[notifier] email send failed (%d items)", len(interesting))

@app.post("/api/alerts/test_email")
def api_alerts_test_email():
    sample = [
        {"ts_ms": int(time.time()*1000)-20000, "attack_type": "dos", "severity": "high", "score": 0.97, "details": {"src": "0x2AA"}},
        {"ts_ms": int(time.time()*1000)-15000, "attack_type": "replay", "severity": "medium", "score": 0.83, "details": {"src": "0x1F4"}},
        {"ts_ms": int(time.time()*1000)-10000, "attack_type": "suspicious_frame", "severity": "low", "score": 0.66, "details": {"src": "0x7FF"}},
    ]
    min_sev = _get_env("ALERT_MIN_SEVERITY", "high")
    cooldown_s = float(_get_env("ALERT_COOLDOWN_S", "60") or "60")
    html, text = _render_alert_digest_v2(min_sev, sample, cooldown_s)
    force = request.args.get("force", "0") not in ("0", "false", "False")
    if not force and not _email_allowed():
        return jsonify({"ok": False, "error": "email disabled or snoozed"}), 400
    subject = f"CAN IDS: Test Digest (≥ {min_sev.title()})"
    ok = send_email(subject=subject, html_body=html, text_body=text)
    return jsonify({"ok": bool(ok)}), (200 if ok else 500)

@app.get("/api/alerts/email/config")
def api_email_config():
    cfg = {
        "SMTP_HOST": _get_env("SMTP_HOST"),
        "SMTP_PORT": _get_env("SMTP_PORT", "587"),
        "SMTP_USER": _mask(_get_env("SMTP_USER")),
        "SMTP_PASS": _mask(_get_env("SMTP_PASS")),
        "ALERT_EMAIL_FROM": _get_env("ALERT_EMAIL_FROM", _get_env("SMTP_USER")),
        "ALERT_EMAIL_TO": _get_env("ALERT_EMAIL_TO"),
        "ALERT_MIN_SEVERITY": _get_env("ALERT_MIN_SEVERITY", "high"),
        "ALERT_COOLDOWN_S": _get_env("ALERT_COOLDOWN_S", "60"),
        "EMAIL_RETRIES": _get_env("EMAIL_RETRIES", "3"),
        "EMAIL_BACKOFF_S": _get_env("EMAIL_BACKOFF_S", "2"),
        "EMAIL_SUBJECT_PREFIX": _get_env("EMAIL_SUBJECT_PREFIX", ""),
        "EMAIL_ENABLED": 1 if EMAIL_ENABLED.is_set() else 0,
        "FRONTEND_URL": _get_env("FRONTEND_URL", "http://localhost:5173"),
        "IDS_MODEL_NAME": _get_env("IDS_MODEL_NAME", "RandomForest"),
        "SNOOZED_UNTIL": SNOOZE_UNTIL,
    }
    return jsonify(cfg), 200

# Start notifier thread
_notifier_thread = threading.Thread(target=notifier_loop, name="email-notifier", daemon=True)
_notifier_thread.start()

# ==============================================================================
# Health + graceful shutdown
# ==============================================================================
@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})

def _shutdown():
    log.info("Shutting down...")
    _stop_writer.set()
    _notifier_stop.set()
    try:
        _writer_thread.join(timeout=2)
    except Exception:
        pass
    try:
        _notifier_thread.join(timeout=2)
    except Exception:
        pass

def _install_signal_handlers():
    def handler(signum, frame):
        log.info("Signal %s received", signum)
        raise KeyboardInterrupt()
    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if sig is not None:
            try:
                signal.signal(sig, handler)
            except Exception:
                pass

# --- Datasets API (enhanced & hardened) ---
PROJECT_ROOT = os.path.dirname(__file__)

# Candidate dataset directories (project-root/datasets and backend/datasets)
CANDIDATE_DATASET_DIRS = [
    os.path.join(PROJECT_ROOT, "datasets"),
    os.path.join(PROJECT_ROOT, "backend", "datasets"),
]

# Which absolute roots are allowed to be selected from the dashboard.
# By default we only allow dataset files inside the project dataset dirs.
ALLOWED_DATASET_ROOTS = [
    os.path.join(PROJECT_ROOT, "datasets"),
    os.path.join(PROJECT_ROOT, "backend", "datasets"),
]

# Admin override: if set to "1" allow ANY absolute path (use carefully).
# Set environment variable ALLOW_ABSOLUTE_DATASET_PATH=1 only if you trust clients.
ALLOW_ABSOLUTE_DATASET_PATH = os.getenv("ALLOW_ABSOLUTE_DATASET_PATH", "0") in ("1", "true", "True")

# Server-side default cap for training rows (dashboard can request smaller)
DEFAULT_SERVER_MAX_ROWS = int(os.getenv("TRAINING_MAX_ROWS", "50000"))

# Optional safety: maximum dataset file size (MB)
MAX_DATASET_FILE_MB = int(os.getenv("MAX_DATASET_FILE_MB", "500"))

def _nice_id(name: str) -> str:
    """Make a simple slug id from a name or filename."""
    try:
        name = os.path.splitext(name)[0]
        return name.strip().lower().replace(" ", "-").replace("_", "-")
    except Exception:
        return str(name).strip().lower()

def _nice_label(name: str) -> str:
    """Return a human label for an id/folder name."""
    name = os.path.splitext(name)[0]
    return name.replace("-", " ").replace("_", " ").strip().title()

def _is_dataset_entry_valid(name: str) -> bool:
    """Return True for entries we should expose (skip dotfiles, tmp, pyc, etc)."""
    if not name or not isinstance(name, str):
        return False
    nlow = name.strip().lower()
    if nlow in ("__pycache__", "__init__", "__init__.py", "__init__.pyc"):
        return False
    if nlow.startswith("tmp-") or nlow.startswith("tmp_") or nlow.startswith("tmpsample"):
        return False
    if nlow.startswith("."):
        return False
    if nlow.endswith(".pyc") or nlow.endswith(".pyo"):
        return False
    if len(nlow) < 2:
        return False
    return True

def _resolve_dataset_path_safe(dataset_id_or_path: str) -> str | None:
    """
    Resolve a dataset identifier or relative path to a safe path:
      - If the client provided an absolute path: allow only when ALLOW_ABSOLUTE_DATASET_PATH is True.
      - Otherwise search CANDIDATE_DATASET_DIRS and ALLOWED_DATASET_ROOTS for matching files or directories.
    Returns string path or None if not found/allowed.
    """
    if not dataset_id_or_path:
        return None
    ds = str(dataset_id_or_path).strip()

    # Absolute path handling
    if os.path.isabs(ds):
        if ALLOW_ABSOLUTE_DATASET_PATH:
            return ds if os.path.exists(ds) else None
        log.warning("[datasets] refused absolute dataset path (server policy): %s", ds)
        return None

    did = _nice_id(ds)

    # try each candidate root
    search_bases = list(dict.fromkeys(CANDIDATE_DATASET_DIRS + ALLOWED_DATASET_ROOTS))
    for base in search_bases:
        try:
            if not os.path.isdir(base):
                continue
            # exact directory match
            cand_dir = os.path.join(base, ds)
            if os.path.isdir(cand_dir):
                return cand_dir
            # slug match among entries
            for entry in os.listdir(base):
                if not _is_dataset_entry_valid(entry):
                    continue
                if _nice_id(entry) == did:
                    full = os.path.join(base, entry)
                    if os.path.isdir(full) or os.path.isfile(full):
                        return full
            # try file variants by slug
            for ext in (".csv", ".parquet", ".json", ".ndjson"):
                fn = os.path.join(base, f"{ds}{ext}")
                if os.path.exists(fn):
                    return fn
        except Exception:
            continue

    # fallback check: backend/datasets/<ds>.csv
    fallback = os.path.join(PROJECT_ROOT, "backend", "datasets", f"{ds}.csv")
    if os.path.exists(fallback):
        return fallback

    return None

DATASETS_CONFIG = os.path.join(PROJECT_ROOT, "config", "datasets.json")

@app.route("/api/datasets", methods=["GET"])
def api_datasets():
    """
    Return dataset presets for the frontend as JSON list of {id,label}.
    Priority:
      1. config/datasets.json (if present and non-empty)
      2. otherwise scan candidate dataset directories (filtered)
    """
    try:
        # try config file
        if os.path.exists(DATASETS_CONFIG):
            try:
                with open(DATASETS_CONFIG, "r", encoding="utf-8") as f:
                    j = json.load(f)
                if isinstance(j, list) and j:
                    return jsonify(j), 200
            except Exception:
                log.exception("[datasets] failed to read config file, falling back to scan")

        opts = []
        seen_ids = set()
        for d in CANDIDATE_DATASET_DIRS:
            try:
                if not os.path.isdir(d):
                    continue
                entries = sorted(os.listdir(d))
                for e in entries:
                    if not _is_dataset_entry_valid(e):
                        continue
                    full = os.path.join(d, e)
                    if os.path.isdir(full):
                        _id = _nice_id(e)
                        if _id in seen_ids:
                            continue
                        seen_ids.add(_id)
                        opts.append({"id": _id, "label": _nice_label(e)})
                    else:
                        ext = os.path.splitext(e)[1].lower()
                        if ext in (".csv", ".parquet", ".json", ".ndjson"):
                            _id = _nice_id(e)
                            if _id in seen_ids:
                                continue
                            seen_ids.add(_id)
                            opts.append({"id": _id, "label": _nice_label(e)})
            except Exception:
                log.exception("[datasets] scanning dir failed: %s", d)
                continue

        return jsonify(opts), 200

    except Exception as ex:
        log.exception("[datasets] failed to list datasets")
        return jsonify({"error": "failed to list datasets"}), 500


@app.route("/api/datasets/scan", methods=["POST"])
def api_datasets_scan():
    # optional admin-only endpoint — uses X-Admin-Token header to authenticate
    token = request.headers.get("X-Admin-Token", "")
    if token != os.environ.get("ADMIN_TOKEN", ""):
        return jsonify({"error": "unauthorized"}), 401

    try:
        opts = []
        seen_ids = set()
        for d in CANDIDATE_DATASET_DIRS:
            if os.path.isdir(d):
                for e in sorted(os.listdir(d)):
                    if not _is_dataset_entry_valid(e):
                        continue
                    full = os.path.join(d, e)
                    if os.path.isdir(full):
                        _id = _nice_id(e)
                        if _id in seen_ids:
                            continue
                        seen_ids.add(_id)
                        opts.append({"id": _id, "label": _nice_label(e)})
                    else:
                        ext = os.path.splitext(e)[1].lower()
                        if ext in (".csv", ".parquet", ".json", ".ndjson"):
                            _id = _nice_id(e)
                            if _id in seen_ids:
                                continue
                            seen_ids.add(_id)
                            opts.append({"id": _id, "label": _nice_label(e)})

        # write config
        cfg_dir = os.path.dirname(DATASETS_CONFIG)
        os.makedirs(cfg_dir, exist_ok=True)
        with open(DATASETS_CONFIG, "w", encoding="utf-8") as f:
            json.dump(opts, f, indent=2)
        log.info("[datasets] wrote config to %s (count=%d)", DATASETS_CONFIG, len(opts))
        return jsonify({"ok": True, "count": len(opts), "datasets": opts}), 200
    except Exception:
        log.exception("[datasets] scan failed")
        return jsonify({"error": "scan failed"}), 500

# --- end datasets API ---
if __name__ == "__main__":
    print("✅ Starting Flask (gevent-friendly) …")
    _install_signal_handlers()
    use_gevent = os.getenv("USE_GEVENT", "1") not in ("0", "false", "False")
    port_env = os.getenv("PORT") or os.getenv("FLASK_RUN_PORT") or "5000"
    try:
        port = int(port_env)
    except Exception:
        port = 5000

    try:
        if use_gevent:
            # Run gevent pywsgi server (simple-websocket + flask-sock works well here)
            from gevent import pywsgi
            host = "0.0.0.0"
            print(f"🚀 Starting gevent WSGIServer (simple-websocket) on http://{host}:{port} …")
            server = pywsgi.WSGIServer((host, port), app, log=None)  # ← no handler_class
            server.serve_forever()
        else:
            print(f"👟 Starting Flask dev server on http://0.0.0.0:{port} …")
            app.run(host="0.0.0.0", port=port, threaded=False)
    except KeyboardInterrupt:
        pass
    except Exception:
        log.error("Fatal error:\n%s", traceback.format_exc())
    finally:
        _shutdown()
