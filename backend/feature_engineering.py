# backend/feature_engineering.py
"""
Small helper utilities for feature transforms, scaling, train/test split.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

def train_test_split_scale(X: np.ndarray, y: np.ndarray, test_size=0.2, random_state=42, stratify=None
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Split and fit a StandardScaler on training features.
    Returns X_train, X_test, y_train, y_test, meta
    """
    rs = random_state
    if stratify is None and y is not None and (y.min() >= 0):
        stratify = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs, stratify=stratify)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    meta = {"scaler": scaler}
    return X_train, X_test, y_train, y_test, meta
