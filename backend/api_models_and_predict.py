"""
backend/api_models_and_predict.py

Minimal endpoints:
 - GET  /api/models                -> list models
 - GET  /api/models/<name>/download -> download model file
 - GET  /api/models/<name>/info     -> info about model (cached, supports predict_proba, etc.)
 - POST /api/models/<name>/unload   -> evict model from in-memory cache
 - POST /api/admin/models/reload    -> admin: reload single model or 'all' models into cache
 - POST /api/predict               -> predict single sample with a model

Security: restricts served files to backend/models and tmp_models only.
"""
from __future__ import annotations
import os
import time
import io
import joblib
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from flask import Blueprint, jsonify, send_file, request, current_app, abort

models_bp = Blueprint("models_api", __name__, url_prefix="/api")

# Directories to search for models (only these are served)
MODEL_DIRS = [Path("backend/models"), Path("tmp_models")]

# A small helper to always access the shared cache
def _model_cache() -> Dict[str, Any]:
    cache = current_app.config.get("MODEL_CACHE")
    if cache is None:
        cache = {}
        current_app.config["MODEL_CACHE"] = cache
    return cache

def _gather_models() -> List[Dict[str, Any]]:
    out = []
    for d in MODEL_DIRS:
        try:
            if not d.exists():
                continue
            for p in sorted(d.glob("*.pkl")):
                try:
                    s = p.stat()
                    out.append({
                        "name": p.name,
                        "path": str(p.resolve()),
                        "dir": str(d),
                        "size": s.st_size,
                        "mtime": int(s.st_mtime),
                    })
                except Exception:
                    continue
        except Exception:
            continue
    # newest first
    out = sorted(out, key=lambda x: x["mtime"], reverse=True)
    return out

def _safe_model_path(name: str) -> Optional[Path]:
    # ensure no path traversal; only allow names (no slashes)
    if "/" in name or "\\" in name:
        return None
    for d in MODEL_DIRS:
        p = d / name
        try:
            if p.exists() and p.is_file():
                # ensure it's inside allowed dir
                if str(p.resolve()).startswith(str(d.resolve())):
                    return p
        except Exception:
            continue
    return None

@models_bp.route("/models", methods=["GET"])
def list_models():
    models = _gather_models()
    return jsonify({"ok": True, "models": models})

@models_bp.route("/models/<name>/download", methods=["GET"])
def download_model(name: str):
    p = _safe_model_path(name)
    if p is None:
        return jsonify({"ok": False, "error": "model not found or invalid name"}), 404
    try:
        return send_file(str(p.resolve()), as_attachment=True, download_name=p.name)
    except Exception as e:
        current_app.logger.exception("download_model failed")
        return jsonify({"ok": False, "error": str(e)}), 500

@models_bp.route("/models/<name>/info", methods=["GET"])
def model_info(name: str):
    """
    Returns some metadata about the model file and cached state.
    If model is cached, we attempt to extract features like supports_predict_proba
    and n_jobs (if present on sklearn-like estimators). This is non-blocking
    for disk reads in most cases because we only inspect in-memory cache; if not
    cached, we read only top-level info.
    """
    p = _safe_model_path(name)
    if p is None:
        return jsonify({"ok": False, "error": "model not found"}), 404

    cache = _model_cache()
    cached = False
    info: Dict[str, Any] = {"name": name, "path": str(p.resolve()), "mtime": int(p.stat().st_mtime)}
    if name in cache:
        cached = True
        clf = cache[name]
        try:
            info.update({
                "cached": True,
                "type": type(clf).__name__,
                "supports_predict_proba": hasattr(clf, "predict_proba"),
                "has_predict": hasattr(clf, "predict"),
                "n_jobs": getattr(clf, "n_jobs", None),
            })
            # if sklearn classifier has classes_
            if hasattr(clf, "classes_"):
                try:
                    info["classes"] = [str(x) for x in getattr(clf, "classes_")]
                except Exception:
                    info["classes"] = None
        except Exception:
            current_app.logger.exception("error introspecting cached model")
    else:
        # Not cached: return minimal info (file-based)
        info.update({"cached": False, "type": None, "supports_predict_proba": None, "has_predict": None})
    return jsonify({"ok": True, "info": info})

@models_bp.route("/models/<name>/unload", methods=["POST"])
def unload_model(name: str):
    """
    Evict model from in-memory cache.
    """
    cache = _model_cache()
    removed = False
    if name in cache:
        try:
            del cache[name]
            removed = True
            current_app.logger.info("Unloaded model from cache: %s", name)
        except Exception:
            current_app.logger.exception("Failed to unload model %s", name)
            return jsonify({"ok": False, "error": "failed to unload model"}), 500
    return jsonify({"ok": True, "name": name, "unloaded": removed})

@models_bp.route("/admin/models/reload", methods=["POST"])
def admin_reload_models():
    """
    Admin endpoint to (re)load a model into cache, or load all models.

    JSON body:
      { "name": "RandomForest_xxx.pkl" }    -> reload single
      { "all": true }                       -> reload all models found
    Requires header X-Admin-Token matching env ADMIN_TOKEN or app config check.
    """
    # basic token check
    admin_token_env = os.environ.get("ADMIN_TOKEN")
    provided = request.headers.get("X-Admin-Token", "")
    if admin_token_env and provided != admin_token_env:
        current_app.logger.warning("Unauthorized admin reload attempt from %s", request.remote_addr)
        return jsonify({"ok": False, "error": "forbidden"}), 403

    body = request.get_json(silent=True) or {}
    name = body.get("name")
    all_flag = bool(body.get("all", False))
    cache = _model_cache()
    reloaded = []
    failed = {}

    def _load_to_cache(pth: Path):
        try:
            clf = joblib.load(str(pth))
            cache[pth.name] = clf
            reloaded.append(pth.name)
            current_app.logger.info("Reloaded model into cache: %s", pth.name)
            return True
        except Exception as e:
            current_app.logger.exception("Failed to reload model %s: %s", pth, e)
            failed[str(pth.name)] = str(e)
            return False

    if all_flag:
        models = _gather_models()
        for m in models:
            p = Path(m["path"])
            _load_to_cache(p)
        return jsonify({"ok": True, "reloaded": reloaded, "failed": failed})

    if name:
        p = _safe_model_path(name)
        if p is None:
            return jsonify({"ok": False, "error": "model not found", "name": name}), 404
        ok = _load_to_cache(p)
        if ok:
            return jsonify({"ok": True, "reloaded": [name]})
        return jsonify({"ok": False, "failed": {name: failed.get(name, "load_failed")}}), 500

    return jsonify({"ok": False, "error": "provide 'name' or set 'all': true"}), 400

# ---------------- Prediction endpoint ----------------
def _build_features_from_raw(raw: dict, pad_to: int = 8):
    """
    raw: {'can_id': int, 'dlc': int, 'data': [int,...], 'ts_delta': float (optional)}
    returns numpy array shape (1, feature_dim)
    Feature order matches dataset_loader output:
      [can_norm, dlc, byte0..byte7, bytes_mean, bytes_std, ts_delta]
    NOTE: can_norm is just can_id as float (no global normalization) for single-sample inference.
    """
    can_id = int(raw.get("can_id", 0) or 0)
    dlc = int(raw.get("dlc", 0) or 0)
    data = raw.get("data", []) or []
    # ensure ints
    try:
        data = [int(x) for x in data]
    except Exception:
        data = []
    # pad/truncate to pad_to
    b = data[:pad_to] + [0] * max(0, pad_to - len(data))
    arr_bytes = np.array(b, dtype=float)
    bytes_mean = float(arr_bytes.mean()) if len(arr_bytes) > 0 else 0.0
    bytes_std = float(arr_bytes.std()) if len(arr_bytes) > 0 else 0.0
    ts_delta = float(raw.get("ts_delta", 0.0) or 0.0)
    # can_norm: for single-sample inference we don't have dataset min/max; use can_id as-is (float)
    can_norm = float(can_id)
    feat = np.concatenate([[can_norm, float(dlc)], arr_bytes, [bytes_mean, bytes_std, ts_delta]]).reshape(1, -1)
    return feat

def _choose_model_path(model_name: Optional[str]):
    if model_name:
        p = _safe_model_path(model_name)
        return p
    models = _gather_models()
    if not models:
        return None
    return Path(models[0]["path"])

@models_bp.route("/predict", methods=["POST"])
def predict_one():
    """
    JSON body:
    {
      "model_name": "RandomForest_12345.pkl",   (optional - newest if omitted)
      "features": [can_norm, dlc, byte0, ..., byte7, bytes_mean, bytes_std, ts_delta]  (optional)
      "raw": {"can_id": 817, "dlc": 8, "data": [..], "ts_delta": 0.0}  (optional)
    }
    Returns:
      { ok: True, model: name, pred: int, prob: optional float, raw_pred: optional }
    """
    # read raw request bytes for nicer logging if desired (non-blocking)
    try:
        raw_body = request.get_data(as_text=True)
        current_app.logger.info("========== /api/predict REQUEST ARRIVED ==========")
        current_app.logger.info("Client = %s", request.remote_addr)
        current_app.logger.info("Raw body received: %s", raw_body[:1000])
    except Exception:
        raw_body = None

    body = request.get_json(silent=True) or {}
    model_name = body.get("model_name")
    features = body.get("features")
    raw = body.get("raw")

    model_path = _choose_model_path(model_name)
    if model_path is None:
        return jsonify({"ok": False, "error": "no model available"}), 404

    # build X
    X = None
    try:
        if features is not None:
            X = np.asarray(features, dtype=float).reshape(1, -1)
        elif raw is not None:
            current_app.logger.info("Building features from 'raw'.")
            X = _build_features_from_raw(raw)
        else:
            return jsonify({"ok": False, "error": "provide 'features' or 'raw' in request body"}), 400
    except Exception as e:
        current_app.logger.exception("failed to build features")
        return jsonify({"ok": False, "error": f"failed to build features: {e}"}), 400

    cache = _model_cache()
    clf = cache.get(model_path.name)
    try:
        if clf is None:
            current_app.logger.info("CACHE MISS - loading model from disk: %s", model_path)
            clf = joblib.load(str(model_path))
            # attempt to reduce parallel threads impact: don't change user model unless it's scikit-learn RF-like
            try:
                # set n_jobs=1 to avoid heavy thread pools on server if attribute exists
                if hasattr(clf, "set_params"):
                    clf.set_params(**{"n_jobs": 1})
                    current_app.logger.info("Set model n_jobs=1 via set_params()")
            except Exception:
                # ignore if not applicable
                pass
            cache[model_path.name] = clf
        else:
            current_app.logger.info("CACHE HIT - using cached model: %s", model_path.name)

        # run prediction
        pred = None
        prob = None
        if hasattr(clf, "predict_proba"):
            try:
                prob_arr = clf.predict_proba(X)
                if prob_arr is not None:
                    if prob_arr.shape[1] == 2:
                        prob = float(prob_arr[0, 1])
                    else:
                        prob = float(prob_arr[0].max())
            except Exception:
                current_app.logger.exception("predict_proba failed")
                prob = None
        try:
            predv = clf.predict(X)
            # predv might be array-like
            if hasattr(predv, "__len__"):
                pred = int(predv[0])
            else:
                pred = int(predv)
        except Exception:
            # fallback
            if prob is not None:
                pred = 1 if prob >= 0.5 else 0
            else:
                pred = None

        current_app.logger.info("Prediction success: pred=%s prob=%s", pred, prob)
        return jsonify({"ok": True, "model": model_path.name, "pred": pred, "prob": prob})
    except Exception as e:
        current_app.logger.exception("prediction failed")
        return jsonify({"ok": False, "error": str(e)}), 500
