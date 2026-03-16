import os
import joblib
import json
from core.logger import get_logger

log = get_logger(__name__)

_model = None
_scaler = None
_meta = None

def reload_model(path="models/model.pkl"):
    """Reload trained model, scaler, and metadata."""
    global _model, _scaler, _meta

    try:
        model_dir = os.path.dirname(path) or "models"
        model_path = path
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        meta_path = os.path.join(model_dir, "meta.json")

        # Load artifacts
        _model = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                _meta = json.load(f)
        else:
            _meta = {"labels": ["benign"], "features": []}

        log.info(f"✅ Model loaded: {model_path}")

    except Exception as e:
        log.exception(f"❌ Failed to load model: {e}")
        _model, _scaler, _meta = None, None, {"labels": ["benign"], "features": []}


def predict_frame(frame):
    """Predict label + score for a CAN frame."""
    global _model, _scaler, _meta

    if _model is None or _scaler is None:
        return "benign", 0.0

    try:
        # Extract features in same order as training
        can_id_num = int(frame["can_id"], 16)
        dlc = int(frame["dlc"])
        bytes_list = [int(b, 16) for b in frame["data"].split()]
        features = [can_id_num, dlc] + bytes_list  # total = 10 features

        # ✅ Feature length validation
        expected_features = _meta.get("features", [])
        if expected_features and len(features) != len(expected_features):
            log.error(f"❌ Feature mismatch: expected {len(expected_features)}, got {len(features)}")
            return "benign", 0.0

        # Apply scaler
        features = _scaler.transform([features])

        # Prediction
        pred = _model.predict(features)[0]
        score = max(_model.predict_proba(features)[0])

        return _meta.get("labels", ["benign"])[pred], score

    except Exception as e:
        log.exception(f"❌ Inference failed: {e}")
        return "benign", 0.0
