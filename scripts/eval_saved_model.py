# scripts/eval_saved_model.py
# Usage:
#   python scripts/eval_saved_model.py backend/models/model.pkl

import sys
import os
import pickle

# --- Ensure project root is on sys.path so `backend` package is importable ---
# This makes the script work whether you run it from project root or from elsewhere.
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# Now safe to import
try:
    from backend.dataset_loader import load_dataset
except Exception as e:
    print("Failed to import backend.dataset_loader:", e)
    raise

def load_model(path):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as fh:
            return pickle.load(fh)

def main(model_path):
    if not os.path.exists(model_path):
        print("Model file not found:", model_path)
        sys.exit(2)
    print("Loading model:", model_path)
    model = load_model(model_path)

    print("Loading sample dataset (max_rows=2000)...")
    X, y, meta = load_dataset("datasets/ocs-lab-dev", max_rows=2000)
    print("Sample loaded: X.shape=", getattr(X, "shape", None), "y.shape=", getattr(y, "shape", None))
    if X is None or y is None:
        print("No data returned by loader")
        sys.exit(4)

    # use only labeled samples
    mask = (y >= 0)
    X_eval = X[mask]
    y_eval = y[mask]
    if len(y_eval) == 0:
        print("No labeled samples (y>=0) found in sample")
        sys.exit(5)

    # try sklearn metrics
    try:
        from sklearn.metrics import confusion_matrix, accuracy_score
        y_pred = model.predict(X_eval)
        acc = accuracy_score(y_eval, y_pred)
        cm = confusion_matrix(y_eval, y_pred)
        print("Accuracy:", acc)
        print("Confusion matrix:\n", cm)
    except Exception as e:
        # fallback: simple accuracy count
        print("sklearn unavailable or inference failed:", e)
        try:
            y_pred = model.predict(X_eval)
            acc = float((y_pred == y_eval).sum()) / len(y_eval)
            print("Fallback accuracy:", acc)
        except Exception as e2:
            print("Inference failed totally:", e2)
            sys.exit(6)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/eval_saved_model.py <model_path>")
        sys.exit(1)
    main(sys.argv[1])
