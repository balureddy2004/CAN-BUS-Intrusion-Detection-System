import os
import json
from flask import Blueprint, jsonify, current_app

models_bp = Blueprint("models_bp", __name__)

# Path to models.json
DEFAULT_MODELS_JSON = os.path.join(os.path.dirname(__file__), "config", "models.json")

def load_models_config():
    try:
        with open(DEFAULT_MODELS_JSON, "r") as f:
            data = json.load(f)

        out = []
        for m in data:
            out.append({
                "id": m.get("id"),
                "display": m.get("display", m.get("id")),
                "available": bool(m.get("enabled", True)),
                "requires_gpu": bool(m.get("requires_gpu", False)),
                "notes": m.get("notes", "")
            })
        return out

    except Exception as e:
        current_app.logger.warning(f"Could not load models config: {e}")
        return []

@models_bp.route("/api/models", methods=["GET"])
def list_models():
    return jsonify(load_models_config())
