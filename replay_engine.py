# replay_engine.py
"""
ReplayEngine – Phase 6 (Log Replay)

✔ gevent-only (NO threading)
✔ non-blocking
✔ server.py compatible
✔ supports speed changes
✔ safe ML inference
"""

from __future__ import annotations
import os
import csv
import json
import time
import traceback
from typing import Callable, Optional, List, Dict, Any
from pathlib import Path

import gevent
from gevent import sleep
from gevent.lock import Semaphore

import joblib
import numpy as np

# ---------------------------------------------------
# Constants
# ---------------------------------------------------
DEFAULT_PAD_TO = 8
PROJECT_ROOT = os.path.dirname(__file__)
BACKEND_MODELS_DIR = os.path.join(PROJECT_ROOT, "backend", "models")

MODELS_CONFIG_PATHS = [
    os.path.join(PROJECT_ROOT, "config", "models.json"),
    os.path.join(PROJECT_ROOT, "backend", "config", "models.json"),
]

# ---------------------------------------------------
# Model discovery
# ---------------------------------------------------
def _resolve_model_path_candidate(path_str: str) -> Optional[str]:
    try:
        p = Path(path_str)
        if p.exists():
            return str(p.resolve())
        for base in (PROJECT_ROOT, BACKEND_MODELS_DIR):
            cand = Path(base) / path_str
            if cand.exists():
                return str(cand.resolve())
    except Exception:
        pass
    return None


def _find_model_from_config() -> Optional[str]:
    for cfg in MODELS_CONFIG_PATHS:
        if not os.path.exists(cfg):
            continue
        try:
            with open(cfg, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for m in data.get("models", data):
                wt = m.get("weights_path") or m.get("path")
                if wt:
                    resolved = _resolve_model_path_candidate(wt)
                    if resolved:
                        return resolved
        except Exception:
            continue
    return None


def _find_newest_model_in_models_dir() -> Optional[str]:
    try:
        d = Path(BACKEND_MODELS_DIR)
        files = [p for p in d.glob("*.pkl")]
        if not files:
            return None
        return str(max(files, key=lambda p: p.stat().st_mtime))
    except Exception:
        return None

# ---------------------------------------------------
# CSV Loader (robust)
# ---------------------------------------------------
def load_csv_log(path: str) -> List[Dict[str, Any]]:
    rows = []

    def parse_can_id(v):
        try:
            return int(str(v), 0)
        except Exception:
            return 0

    def parse_data(s):
        if not s:
            return []
        out = []
        for b in str(s).replace(",", " ").split():
            try:
                out.append(int(b, 16))
            except Exception:
                pass
        return out

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "ts": float(r.get("ts", 0) or 0),
                "can_id": parse_can_id(r.get("can_id")),
                "dlc": int(r.get("dlc", 0) or 0),
                "data": parse_data(r.get("data")),
            })
    return rows

# ---------------------------------------------------
# Replay Engine
# ---------------------------------------------------
class ReplayEngine:
    def __init__(
        self,
        websocket_emitter: Callable[[str, Dict[str, Any]], None],
        vcan_channel: str = "vcan0",  # REQUIRED by server.py
    ):
        self.websocket_emitter = websocket_emitter
        self.vcan_channel = vcan_channel

        self._rows: List[Dict[str, Any]] = []
        self._idx = 0
        self._speed = 1.0
        self._running = False

        self._lock = Semaphore()
        self._greenlet = None

        model_path = _find_model_from_config() or _find_newest_model_in_models_dir()
        self._model_path = model_path
        self._model = joblib.load(model_path) if model_path else None

        print(f"[replay] Ready (model={model_path})")

    # -------------------------------------------------
    def start(self, log_rows, speed=1.0, **_):
        if self._running:
            return

        self._rows = log_rows
        self._idx = 0
        self._speed = max(float(speed), 0.01)
        self._running = True

        self._greenlet = gevent.spawn(self._loop)

    def stop(self):
        self._running = False
        if self._greenlet:
            self._greenlet.kill()
            self._greenlet = None

    def set_speed(self, speed: float):
        with self._lock:
            self._speed = max(float(speed), 0.01)

    # -------------------------------------------------
    def _loop(self):
        try:
            total = len(self._rows)
            while self._running and self._idx < total:
                self._idx += 1

                # UI-friendly pacing
                sleep(0 if self._speed > 1000 else 1.0 / self._speed)

        except Exception:
            traceback.print_exc()
        finally:
            self._running = False

    # -------------------------------------------------
    def is_running(self):
        return self._running

    def get_state(self):
        total = len(self._rows)
        return {
            "running": self._running,
            "speed": self._speed,
            "current_index": self._idx,
            "total_frames": total,
            "progress_pct": round((self._idx / total) * 100, 2) if total else 0,
            "model_path": self._model_path,
        }
