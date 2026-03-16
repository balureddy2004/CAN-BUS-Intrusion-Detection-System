import sys
import os

# --- FORCE PROJECT ROOT INTO PYTHON PATH (WINDOWS SAFE) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.model.predict import predict_single_frame


def process_frame(frame):
    """
    Normalizes a CAN frame and runs ML prediction.
    """

    features = [
        frame["can_id"],
        frame["dlc"],
        *frame["data"]
    ]

    prediction, confidence = predict_single_frame(features)

    return {
        **frame,
        "attack": bool(prediction),
        "confidence": float(confidence)
    }
