from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO
import io, csv, time, random, threading, logging, os, json, glob

from core.settings import settings
from core.logger import get_logger
from ids.inference import predict_frame, reload_model
from ids.emailer import send_email_alert

# -----------------------------------------------------------------------------
# App & IO
# -----------------------------------------------------------------------------
log = get_logger(__name__)
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

@app.before_request
def _log_request():
    app.logger.info("HTTP %s %s", request.method, request.path)

# Allow CORS for REST and Socket.IO
CORS(app, resources={r"/*": {"origins": "*"}})
sio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
STATE = {
    "totalPackets": 0,
    "benignRate": 0.985,
    "attackRate": 0.015,
    "currentModel": settings.current_model,
    "cpu": 18,
    "memory": 41,
    "timeline": [],
    "alerts": [],
}

# Bootstrap TRAIN state from meta.json if available
TRAIN = {"state": "idle", "progress": 0, "msg": ""}
meta_path = os.path.join("models", "meta.json")
if os.path.exists(meta_path):
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        TRAIN.update({
            "state": "completed",
            "progress": 100,
            "dataset": meta.get("dataset", "unknown"),
            "val_acc": meta.get("val_acc", None),
            "labels": meta.get("labels", []),
            "msg": "Loaded from previous training"
        })
        STATE["currentModel"] = f"{meta.get('dataset','model')} (acc {meta.get('val_acc',0):.3f})"
        log.info("✅ Bootstrapped TRAIN state from meta.json")
    except Exception as e:
        log.warning("⚠️ Failed to bootstrap TRAIN state: %s", e)

SETTINGS = {"emailOn": True, "thresh": settings.alert_threshold}

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/api/metrics", methods=["GET"])
def metrics():
    return jsonify({**STATE, "currentModel": settings.current_model})

@app.route("/api/settings", methods=["POST"])
def api_settings():
    body = request.get_json(force=True)
    SETTINGS.update({
        "emailOn": bool(body.get("emailOn", True)),
        "thresh": float(body.get("thresh", settings.alert_threshold)),
    })
    return {"ok": True}

# ---- training ----
@app.route("/api/training/start", methods=["POST"])
def training_start():
    from ids.training import train_and_save

    if TRAIN["state"] == "running":
        return {"ok": False, "msg": "already running"}, 409

    dataset = request.args.get("dataset", "ocs-lab")
    TRAIN.update({"state": "running", "progress": 1, "msg": ""})

    def job(dataset=dataset):
        try:
            for p in train_and_save(dataset=dataset):
                TRAIN["progress"] = int(p)

            TRAIN.update({"state": "completed", "msg": "Model retrained"})

            meta_path = os.path.join("models", "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                TRAIN["val_acc"] = meta.get("val_acc")
                TRAIN["dataset"] = meta.get("dataset")
                TRAIN["labels"] = meta.get("labels", [])
                STATE["currentModel"] = f"{meta.get('dataset','model')} (acc {meta.get('val_acc',0):.3f})"

            reload_model()
            app.logger.info("Training finished, model reloaded")
        except Exception as e:
            TRAIN.update({"state": "error", "msg": str(e)})
            app.logger.exception("Training job crashed")

    threading.Thread(target=job, daemon=True).start()
    return {"ok": True}

@app.route("/api/training/status", methods=["GET"])
def training_status():
    model_dir = "models"
    meta_path = os.path.join(model_dir, "meta.json")

    response = {
        "state": TRAIN.get("state", "idle"),
        "progress": TRAIN.get("progress", 0),
        "msg": TRAIN.get("msg", ""),
        "dataset": TRAIN.get("dataset"),
        "val_acc": TRAIN.get("val_acc"),
        "labels": TRAIN.get("labels", []),
    }

    if os.path.exists(meta_path) and response["state"] != "running":
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            response.update({
                "dataset": meta.get("dataset", response["dataset"]),
                "val_acc": meta.get("val_acc", response["val_acc"]),
                "labels": meta.get("labels", response["labels"]),
            })
        except Exception as e:
            response.update({"state": "error", "msg": f"Failed to read meta.json: {str(e)}"})

    return jsonify(response)

@app.route("/api/export/csv", methods=["GET"])
def export_csv():
    output = io.StringIO()
    w = csv.writer(output)
    w.writerow(["ts", "can_id", "dlc", "data", "pred", "score"])
    for _ in range(200):
        w.writerow([
            int(time.time() * 1000),
            hex(random.randint(0, 0x7FF)),
            8,
            "00 11 22 33 44 55 66 77",
            random.choice(["benign", "spoofing"]),
            round(random.random(), 3),
        ])
    bio = io.BytesIO(output.getvalue().encode())
    bio.seek(0)
    return send_file(bio, as_attachment=True, download_name="packets.csv", mimetype="text/csv")

@app.route("/ping", methods=["GET"])
def ping():
    return {"ok": True, "msg": "pong"}

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
@app.route("/api/models", methods=["GET"])
def list_models():
    models = []
    for path in glob.glob("models/*.pkl"):
        meta_path = path.replace(".pkl", ".json")
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception as e:
                app.logger.warning("Failed to read meta for %s: %s", path, e)
        models.append({
            "file": os.path.basename(path),
            "dataset": meta.get("dataset", "unknown"),
            "val_acc": meta.get("val_acc", None),
            "timestamp": meta.get("timestamp", None),
            "current": (settings.current_model == os.path.basename(path))
        })
    return jsonify(models)

@app.route("/api/models/load", methods=["POST"])
def load_model():
    body = request.get_json(force=True)
    fname = body.get("file")
    path = os.path.join("models", fname)

    if not fname or not os.path.exists(path):
        return {"ok": False, "msg": "model not found"}, 404

    settings.current_model = fname
    reload_model(path)

    STATE["currentModel"] = fname
    app.logger.info("Loaded model: %s", fname)
    return {"ok": True, "msg": f"Loaded {fname}"}

# -----------------------------------------------------------------------------
# Demo stream
# -----------------------------------------------------------------------------
STREAM_RUNNING = True
DEMO_BUCKET_SPREAD_MINUTES = 6

def demo_stream():
    global STREAM_RUNNING
    while True:
        try:
            time.sleep(0.15)
            if not STREAM_RUNNING:
                continue

            frame = {
                "ts": int(time.time() * 1000),
                "can_id": hex(random.randint(0, 0x7FF)),
                "dlc": 8,
                "data": " ".join(f"{random.randint(0,255):02X}" for _ in range(8)),
            }

            label, score = predict_frame(frame)
            score = float(score)
            pkt = {**frame, "pred": label, "score": score}

            STATE["totalPackets"] += 1
            app.logger.info("Emitting packet: %s", pkt)
            sio.emit("packet", pkt, namespace="/")

            if label != "benign" and score >= SETTINGS["thresh"]:
                jittered_ts = frame["ts"] - random.randint(0, DEMO_BUCKET_SPREAD_MINUTES * 60) * 1000
                alert = {
                    "ts": jittered_ts,
                    "type": "High-severity anomaly",
                    "score": score,
                    "severity": "high",
                }
                STATE["alerts"] = [alert] + STATE["alerts"][:99]
                app.logger.info("Emitting alert: %s", alert)
                sio.emit("alert", alert, namespace="/")

                if SETTINGS.get("emailOn"):
                    try:
                        send_email_alert(pkt)
                    except Exception as e:
                        app.logger.warning("Email send failed (ignored): %s", e)

            if STATE["totalPackets"] % 20 == 0:
                metrics = {"cpu": random.randint(10, 45), "memory": random.randint(35, 60)}
                app.logger.info("Emitting metrics: %s", metrics)
                sio.emit("metrics", metrics, namespace="/")

        except Exception as e:
            app.logger.exception("demo_stream loop error; continuing: %s", e)
            time.sleep(0.5)

threading.Thread(target=demo_stream, daemon=True).start()

# -----------------------------------------------------------------------------
# Socket.IO controls
# -----------------------------------------------------------------------------
@sio.on("stream:stop")
def stop_stream():
    global STREAM_RUNNING
    STREAM_RUNNING = False

@sio.on("stream:start")
def start_stream():
    global STREAM_RUNNING
    STREAM_RUNNING = True

# -----------------------------------------------------------------------------
# Test emit route
# -----------------------------------------------------------------------------
@app.route("/test-emit", methods=["GET"])
def test_emit():
    pkt = {
        "ts": int(time.time() * 1000),
        "can_id": "0x123",
        "dlc": 8,
        "data": "11 22 33 44 55 66 77 88",
        "pred": "spoofing",
        "score": 0.95,
    }
    app.logger.info("Test emit packet: %s", pkt)
    sio.emit("packet", pkt, namespace="/")
    return {"ok": True, "msg": "sent one pkt"}

@app.route("/api/alerts", methods=["GET"])
def api_alerts():
    # optional ?limit=100
    try:
        limit = int(request.args.get("limit", 100))
    except Exception:
        limit = 100
    # STATE["alerts"] is newest-first in your code; slice safely
    return jsonify(STATE.get("alerts", [])[:limit]), 200

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Try load model at startup
    default_model_path = (
        os.path.join("models", settings.current_model)
        if settings.current_model and settings.current_model != "N/A"
        else None
    )

    if default_model_path and os.path.exists(default_model_path):
        print(f"⚙️ Using model: {default_model_path}")
        try:
            reload_model(default_model_path)
            log.info(f"✅ Loaded initial model: {default_model_path}")
        except Exception as e:
            log.warning(f"⚠️ Failed to load model at startup: {e}")
    else:
        print("⚙️ No model found, starting fresh (training required).")
        log.warning("⚠️ No model found, starting without model")

    log.info("✅ Starting Flask-SocketIO on %s:%s", settings.host, settings.port)
    # IMPORTANT: use SocketIO runner so /socket.io works
    sio.run(
        app,
        host=settings.host,
        port=settings.port,
        debug=True,
        allow_unsafe_werkzeug=True,  # suppress warning on Werkzeug with SocketIO
    )
