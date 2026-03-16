# backend/train_upload.py
# Drop this into backend/ and import in server.py (see instructions below)
import os
import io
import time
import uuid
from flask import request, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename

# where datasets are stored
DATASETS_DIR = os.environ.get("IDS_DATASETS_DIR", os.path.join(os.path.dirname(__file__), "datasets"))
os.makedirs(DATASETS_DIR, exist_ok=True)

ALLOWED_EXT = {".csv", ".parquet", ".pq"}

def _allowed(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT

def register_upload_routes(app):
    @app.post("/api/train/upload")
    def api_train_upload():
        """
        Accepts a multipart/form-data file field named 'file'.
        Optional form fields:
          - name: desired dataset name (without extension). If omitted, filename is used.
        Returns: {"ok": True, "dataset": "<name>", "path": "<relpath>"}
        """
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "no file provided"}), 400
        f = request.files["file"]
        if f.filename == "":
            return jsonify({"ok": False, "error": "empty filename"}), 400
        filename = secure_filename(f.filename)
        if not _allowed(filename):
            return jsonify({"ok": False, "error": f"invalid file type; allowed={','.join(ALLOWED_EXT)}"}), 400
        # derive dataset name
        name = request.form.get("name") or os.path.splitext(filename)[0]
        # unique suffix to avoid collisions
        suffix = uuid.uuid4().hex[:8]
        outname = f"{name}_{suffix}{os.path.splitext(filename)[1]}"
        outpath = os.path.join(DATASETS_DIR, outname)
        # save file
        try:
            f.save(outpath)
        except Exception as e:
            return jsonify({"ok": False, "error": f"save failed: {e}"}), 500
        return jsonify({"ok": True, "dataset": name, "filename": outname, "path": outpath}), 201

    @app.get("/api/train/datasets")
    def api_train_datasets():
        """
        List available datasets in backend/datasets (filename, size, mtime).
        """
        out = []
        try:
            for fn in sorted(os.listdir(DATASETS_DIR)):
                p = os.path.join(DATASETS_DIR, fn)
                if not os.path.isfile(p): continue
                stat = os.stat(p)
                out.append({"filename": fn, "size": stat.st_size, "mtime": int(stat.st_mtime)})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "datasets": out}), 200

    @app.get("/api/train/datasets/download/<path:filename>")
    def api_train_datasets_download(filename):
        # safe download helper
        try:
            return send_from_directory(DATASETS_DIR, filename, as_attachment=True)
        except Exception:
            return jsonify({"ok": False, "error": "not found"}), 404
