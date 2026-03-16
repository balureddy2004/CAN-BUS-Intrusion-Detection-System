import os, json, joblib

def save_model(model, scaler, out_dir, meta: dict):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "model.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def load_model(out_dir):
    model = joblib.load(os.path.join(out_dir, "model.pkl"))
    scaler = joblib.load(os.path.join(out_dir, "scaler.pkl"))
    return model, scaler
