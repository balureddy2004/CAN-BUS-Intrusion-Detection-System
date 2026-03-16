import os
import json
from flask import Blueprint, request, jsonify, abort, current_app, session
from pathlib import Path
from typing import Optional

detector_bp = Blueprint("detector_bp", __name__)

# Configurable paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODELS_JSON_PATH = os.path.join(os.path.dirname(__file__), "config", "models.json")

# weight file extensions we accept (lowercase)
# weight file extensions we accept (lowercase)
# Extend to common ML formats: scikit-learn joblib/pickle, xgboost boosters, etc.
WEIGHT_EXTS = {".pth", ".pt", ".bin", ".ckpt", ".h5", ".joblib", ".xgb", ".pkl"}

def torch_gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def find_weight_in_dir(dir_path: Path) -> Optional[Path]:
    """
    Search for any weight file inside dir_path (recursively),
    return the Path of the first match or None.
    """
    p = Path(dir_path)
    for f in p.rglob("*"):
        if f.is_file() and f.suffix.lower() in WEIGHT_EXTS:
            return f
    return None

def scan_for_models():
    """
    Scan MODELS_DIR for model artifacts. Heuristics:
      - If entry is a directory: look inside recursively for any weight file and use that.
      - If entry is a file at top level with weight ext: treat it as a model file.
    Returns a list of model dicts suitable for writing to config.
    """
    detected = []
    p = Path(MODELS_DIR)
    current_app.logger.info("Scanning models dir: %s", MODELS_DIR)

    if not p.exists():
        current_app.logger.warning("MODELS_DIR does not exist: %s", MODELS_DIR)
        return detected

    for entry in sorted(p.iterdir()):
        try:
            if entry.is_dir():
                current_app.logger.debug("Inspecting directory: %s", str(entry))
                wf = find_weight_in_dir(entry)
                has_weights = wf is not None
                detected.append({
                    "id": entry.name,
                    "display": entry.name.replace("_", " ").title(),
                    "enabled": bool(has_weights),
                    "requires_gpu": False,
                    "weights_path": str(wf) if wf else "",
                    "notes": "" if has_weights else "no weight files found"
                })

            elif entry.is_file() and entry.suffix.lower() in WEIGHT_EXTS:
                current_app.logger.debug("Found top-level weight file: %s", str(entry))
                detected.append({
                    "id": entry.stem,
                    "display": entry.stem.replace("_", " ").title(),
                    "enabled": True,
                    "requires_gpu": False,
                    "weights_path": str(entry),
                    "notes": "single-weight-file"
                })
            else:
                current_app.logger.debug("Skipping non-model entry: %s", str(entry))
        except Exception as e:
            current_app.logger.warning("Error scanning entry %s: %s", entry, e)

    current_app.logger.info("Scan complete. Models found: %d", len(detected))
    return detected

def write_models_json(data):
    tmp = MODELS_JSON_PATH + ".tmp"
    os.makedirs(os.path.dirname(MODELS_JSON_PATH), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, MODELS_JSON_PATH)
    current_app.logger.info("Wrote models config to %s (count=%d)", MODELS_JSON_PATH, len(data))

#
# Dev helper: read models.json from disk on each request
#
def _load_models_json_from_disk():
    """
    Safely read MODELS_JSON_PATH from disk and return a list.
    Normalizes older schema shapes if necessary.
    """
    try:
        with open(MODELS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "models" in data:
                return data["models"]
            if isinstance(data, list):
                return data
            return []
    except FileNotFoundError:
        current_app.logger.warning("models.json not found at %s", MODELS_JSON_PATH)
        return []
    except Exception as e:
        current_app.logger.exception("failed to load models.json: %s", e)
        return []

@detector_bp.route("/api/models", methods=["GET"])
def list_models():
    """
    Dev-friendly models listing: read config/models.json from disk every time.
    """
    models = _load_models_json_from_disk()
    return jsonify(models)

#
# Admin checks
#
def is_current_user_admin() -> bool:
    """
    Attempt multiple safe checks for admin status:
      1) If flask_login is installed and current_user exists, use current_user.is_authenticated and current_user.is_admin
      2) Fallback to session['is_admin'] if set by your auth system
      3) Otherwise return False (deny)
    Replace or extend this with your real auth logic (JWT, roles, etc.)
    """
    # Try flask_login first (if available)
    try:
        from flask_login import current_user  # type: ignore
        # If your User model exposes `is_admin` or similar, this will work.
        if getattr(current_user, "is_authenticated", False):
            is_admin_attr = getattr(current_user, "is_admin", None)
            if is_admin_attr is not None:
                return bool(is_admin_attr)
            # if no is_admin attribute, you may choose to treat any authenticated user as non-admin:
            return False
    except Exception:
        # flask_login not installed or import failed; fall through to next check
        current_app.logger.debug("flask_login not available or error checking current_user")

    # Fallback: session flag (if you set session['is_admin'] after login)
    try:
        if session and session.get("is_admin"):
            return True
    except Exception:
        current_app.logger.debug("session admin check failed or not available")

    # By default, deny
    return False

#
# Existing token-based admin endpoint (keeps backward compatibility and CI usage)
#
@detector_bp.route("/api/models/scan", methods=["POST"])
def admin_scan_token():
    """
    Token-based admin endpoint. Useful for CLI/CI where you don't have app sessions.
    Requires ADMIN_TOKEN env var to be set on the server and must be supplied in header:
      X-Admin-Token: <ADMIN_TOKEN>
    """
    admin_token = os.environ.get("ADMIN_TOKEN")
    req_token = request.headers.get("X-Admin-Token", "")

    if not admin_token or req_token != admin_token:
        current_app.logger.warning("Unauthorized token-based scan attempt (token present: %s)", bool(admin_token))
        abort(403)

    detected = scan_for_models()
    write_models_json(detected)

    return jsonify({
        "ok": True,
        "count": len(detected),
        "gpu_available": torch_gpu_available(),
        "models": detected
    })

#
# Recommended server-side admin proxy (auth-based)
#
@detector_bp.route("/api/admin/models/scan", methods=["POST"])
def admin_scan_proxy():
    """
    Server-side proxy endpoint for admin-triggered scans.
    Use this from an authenticated admin UI. It performs server-side auth checks
    (see is_current_user_admin) and never exposes secrets to the client.
    """
    if not is_current_user_admin():
        current_app.logger.warning("Unauthorized admin_scan_proxy attempt from %s", request.remote_addr)
        abort(403, "forbidden")

    detected = scan_for_models()
    write_models_json(detected)

    return jsonify({
        "ok": True,
        "count": len(detected),
        "gpu_available": torch_gpu_available(),
        "models": detected
    })

#
# Optional: local-only quick endpoint for dev (safe-ish if only bound to localhost)
# Uncomment and use only for local development. Do NOT enable in production.
#
# @detector_bp.route("/api/admin/models/scan-local", methods=["POST"])
# def admin_scan_local():
#     remote = request.remote_addr
#     allowed = {"127.0.0.1", "::1"}
#     if remote not in allowed:
#         current_app.logger.warning("Blocked non-local admin scan from %s", remote)
#         abort(403)
#
#     ADMIN_SCAN_PASS = os.environ.get("ADMIN_SCAN_PASS")
#     if ADMIN_SCAN_PASS:
#         provided = request.headers.get("X-Admin-Scan-Pass", "")
#         if provided != ADMIN_SCAN_PASS:
#             current_app.logger.warning("Invalid admin-scan-pass")
#             abort(403)
#
#     detected = scan_for_models()
#     write_models_json(detected)
#     return jsonify({"ok": True, "count": len(detected), "gpu_available": torch_gpu_available(), "models": detected})

# End of file
