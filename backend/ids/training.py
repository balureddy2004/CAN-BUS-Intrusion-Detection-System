import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from core.logger import get_logger
from datasets.ocs_lab import load_dataset

log = get_logger(__name__)

def train_and_save(dataset="ocs-lab", model_dir="models"):
    """
    Train a RandomForest model on the OCS-Lab dataset and save artifacts.
    Yields progress % for app.py to update UI.
    """
    log.info("🧠 Loading dataset: %s", dataset)
    df = load_dataset()

    if df.empty:
        raise ValueError("Dataset is empty!")

    log.info("✅ Dataset loaded: %s rows, %s cols", df.shape[0], df.shape[1])

    # Ensure numeric features only
    feature_cols = ["can_id_num", "dlc"] + [f"b{i}" for i in range(8)]
    X = df[feature_cols]
    y = df["label"]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    log.info("📊 Features shape: %s, Labels shape: %s", X.shape, y_encoded.shape)
    log.info("🔍 Labels mapping: %s", dict(zip(le.classes_, le.transform(le.classes_))))
    log.info("🔍 Sample features:\n%s", X.head())
    log.info("🔍 Sample labels:\n%s", y_encoded[:10])

    # Train/val split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    yield 20

    # Train RandomForest
    log.info("🧠 Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    yield 70

    # Validation accuracy
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    log.info("✅ Validation accuracy: %.3f", acc)

    yield 90

    # Save model + scaler + meta
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    meta_path = os.path.join(model_dir, "meta.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    with open(meta_path, "w") as f:
        json.dump(
            {
                "dataset": dataset,
                "val_acc": acc,
                "features": feature_cols,   # ✅ store exact features
                "labels": list(le.classes_),
            },
            f,
            indent=2,
        )

    log.info("💾 Saved model to %s", model_path)
    log.info("💾 Saved scaler to %s", scaler_path)
    log.info("💾 Saved meta to %s", meta_path)

    yield 100
