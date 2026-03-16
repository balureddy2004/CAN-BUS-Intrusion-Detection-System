# backend/training_engine.py
"""
Training engine supporting sklearn (RandomForest), XGBoost/LightGBM if installed,
and a simple PyTorch 1D-CNN model (optional).

Provides:
 - train_sklearn
 - train_xgb / train_lgb (if libs present)
 - train_torch_cnn (lightweight)
 - save_model (joblib or torch.save)
 - export_onnx (for PyTorch model)
"""
from __future__ import annotations
import os
import joblib
import time
from typing import Tuple, Dict, Any, Optional
import numpy as np

# sklearn default
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# try optional packages
_has_xgboost = False
_has_lightgbm = False
try:
    import xgboost as xgb
    _has_xgboost = True
except Exception:
    pass
try:
    import lightgbm as lgb
    _has_lightgbm = True
except Exception:
    pass

# PyTorch optional
_has_torch = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _has_torch = True
except Exception:
    pass

def train_sklearn(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                  params: Optional[dict] = None) -> Dict[str, Any]:
    params = params or {}
    n_estimators = int(params.get("n_estimators", 200))
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=int(params.get("seed", 42)))
    t0 = time.time()
    clf.fit(X_train, y_train)
    t1 = time.time()
    y_pred = clf.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_val, y_pred)
    return {"model": clf, "report": report, "acc": acc, "f1": f1, "cm": cm, "train_time_s": t1 - t0}

def save_model(obj: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if _has_torch and isinstance(obj, torch.nn.Module):
        torch.save(obj.state_dict(), path)
    else:
        joblib.dump(obj, path)

# Optional XGBoost
def train_xgboost(X_train, y_train, X_val, y_val, params=None):
    if not _has_xgboost:
        raise RuntimeError("xgboost not installed")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    p = dict(params or {})
    p.setdefault("objective", "binary:logistic" if len(set(y_train))<=2 else "multi:softprob")
    p.setdefault("verbosity", 0)
    num_round = int(p.pop("num_round", 100))
    watchlist = [(dtrain, "train"), (dval, "eval")]
    bst = xgb.train(p, dtrain, num_round, watchlist, early_stopping_rounds=20, verbose_eval=False)
    preds = bst.predict(dval)
    if preds.ndim > 1:
        y_pred = preds.argmax(axis=1)
    else:
        y_pred = (preds > 0.5).astype(int)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    return {"model": bst, "report": report}

def train_lightgbm(X_train, y_train, X_val, y_val, params=None):
    if not _has_lightgbm:
        raise RuntimeError("lightgbm not installed")
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)
    p = dict(params or {})
    p.setdefault("objective", "binary" if len(set(y_train))<=2 else "multiclass")
    bst = lgb.train(p, dtrain, valid_sets=[dval], num_boost_round=int(p.get("num_round", 200)),
                    early_stopping_rounds=20, verbose_eval=False)
    preds = bst.predict(X_val)
    if preds.ndim > 1:
        y_pred = preds.argmax(axis=1)
    else:
        y_pred = (preds > 0.5).astype(int)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    return {"model": bst, "report": report}

# Tiny PyTorch 1D CNN for sequence-like features (expects shape [N, C=1, L])
if _has_torch:
    class Tiny1DCNN(nn.Module):
        def __init__(self, in_len, n_classes=2, hidden=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(hidden, n_classes)
            )
        def forward(self, x):
            return self.net(x)

    def train_torch_cnn(X_train, y_train, X_val, y_val, params=None):
        params = params or {}
        device = torch.device("cpu")
        X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)  # [N,1,L]
        X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
        in_len = X_train.shape[1]
        n_classes = len(np.unique(np.concatenate([y_train, y_val])))
        model = Tiny1DCNN(in_len, n_classes=n_classes, hidden=int(params.get("hidden", 32))).to(device)
        opt = optim.Adam(model.parameters(), lr=float(params.get("lr", 1e-3)))
        loss_fn = nn.CrossEntropyLoss()
        epochs = int(params.get("epochs", 20))
        for ep in range(epochs):
            model.train()
            opt.zero_grad()
            out = model(X_train_t)
            loss = loss_fn(out, y_train_t)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_val_t).argmax(dim=1).cpu().numpy()
        report = classification_report(y_val, preds, output_dict=True, zero_division=0)
        return {"model": model, "report": report}

    def export_onnx(model: nn.Module, input_shape: Tuple[int, ...], path: str):
        import torch
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        dummy = torch.randn(*input_shape)
        torch.onnx.export(model, dummy, path, opset_version=11)
else:
    # placeholders
    def train_torch_cnn(*a, **k):
        raise RuntimeError("PyTorch not installed")
    def export_onnx(*a, **k):
        raise RuntimeError("PyTorch not installed")

# small helper to evaluate & save
def evaluate_and_save(res: Dict[str, Any], outpath: Optional[str] = None) -> Dict[str, Any]:
    if outpath and "model" in res:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        save_model(res["model"], outpath)
    return res
