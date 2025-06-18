#!/usr/bin/env python3
"""
Mask‑R‑CNN inference service ‑ v3.0
====================================

Endpoints
---------

GET  /predict?image=<path_or_url>[&download=filename.png]
    → PNG with detections drawn.

GET  /json?image=<path_or_url>
    →  {"detections":[…]}

POST /api/predict        (Content‑Type: multipart/form‑data)
    • image=<file>  _or_  url=<http/https URL>
    • return_image=[true|false]

POST /api/predict        (Content‑Type: application/json)
    {
      "image": "<base64 PNG/JPEG>" | "url": "<http(s)://…>" | "path": "/opt/in/car.jpg",
      "return_image": true
    }

The JSON “detections” list has:

    {
      "class": "dent",
      "score": 0.94,
      "box":   [y1,x1,y2,x2]
    }

If “return_image” is true, the response also contains:

    "annotated_image": "<base64‑encoded PNG>"

---------------------------------------------------------------------------
Build & Run (example)
---------------------------------------------------------------------------

docker build -t maskrcnn_api .
docker run --rm -p 5000:5000 --network=host maskrcnn_api
"""

import os, sys, io, base64, json
import requests
from flask import Flask, request, jsonify, send_file
import numpy as np
import skimage.io
from skimage.color import gray2rgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Mask‑R‑CNN set‑up ---------------- #
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU mode
ROOT_DIR  = os.path.abspath(".")
MODEL_DIR = os.path.join(ROOT_DIR, "weights")
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib, visualize
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
graph = tf.compat.v1.get_default_graph()

class InferenceConfig(Config):
    NAME                     = "car_damage"
    NUM_CLASSES              = 1 + 6          # BG + 6 classes
    GPU_COUNT                = 1
    IMAGES_PER_GPU           = 1
    DETECTION_MIN_CONFIDENCE = 0.80

config = InferenceConfig()
model  = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
model.load_weights(os.path.join(MODEL_DIR, "mask_rcnn_car_damage.h5"), by_name=True)

CLASS_NAMES = [
    "BG", "broken part", "crack", "dent",
    "lamp broken", "missing part", "scratch"
]

# ---------------- Flask app ---------------- #
app = Flask(__name__)

# ---------------- Helper functions ---------------- #
def _ensure_rgb(img):
    """Convert grayscale to RGB if necessary."""
    if img.ndim != 3:
        img = gray2rgb(img)
    return img

def _decode_base64(b64: "str|bytes"):
    """base‑64 string/bytes → ndarray (RGB)"""
    if isinstance(b64, str):
        b64 = b64.encode()
    return skimage.io.imread(io.BytesIO(base64.b64decode(b64)))

def _load_remote(url: str):
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        raise FileNotFoundError(f"HTTP {r.status_code}: {url}")
    return skimage.io.imread(io.BytesIO(r.content))

def _load_local(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}")
    return skimage.io.imread(path)

def _load_image(src: str):
    """Local path *or* http(s) URL → ndarray"""
    if src.lower().startswith(("http://", "https://")):
        img = _load_remote(src)
    else:
        img = _load_local(src)
    return _ensure_rgb(img)

def _run_detection(image_np):
    """Forward pass & nicely formatted detections."""
    with graph.as_default():
        r = model.detect([image_np], verbose=0)[0]
    dets = [
        {
            "class":  CLASS_NAMES[cid],
            "score":  float(score),
            "box":    [int(v) for v in box]   # [y1,x1,y2,x2]
        }
        for cid, score, box in zip(r["class_ids"], r["scores"], r["rois"])
    ]
    return r, dets

def _annotate_png_bytes(image_np, r):
    """Return BytesIO with PNG overlay (ready for send_file)."""
    fig, ax = plt.subplots(figsize=(12, 10))
    visualize.display_instances(
        image_np, r["rois"], r["masks"], r["class_ids"],
        CLASS_NAMES, r["scores"], ax=ax, title="Prediction"
    )
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def _get_image_from_query():
    src = request.args.get("image") or request.args.get("url")
    if not src:
        raise ValueError("Missing ?image=<path_or_url> or ?url=<url>")
    return _load_image(src)

# ---------------- GET endpoints ---------------- #
@app.route("/predict", methods=["GET"])
def get_predict():
    try:
        img = _get_image_from_query()
    except Exception as e:
        return jsonify(error=str(e)), 400
    r, _ = _run_detection(img)
    filename = request.args.get("download", "prediction.png")
    return send_file(
        _annotate_png_bytes(img, r),
        mimetype="image/png",
        download_name=filename
    )

@app.route("/json", methods=["GET"])
def get_json():
    try:
        img = _get_image_from_query()
    except Exception as e:
        return jsonify(error=str(e)), 400
    _, dets = _run_detection(img)
    return jsonify(detections=dets), 200

# ---------------- legacy /api/predict ---------------- #
@app.route("/api/predict", methods=["POST"])
def api_predict():
    # -------- 1. Acquire image --------
    if request.content_type.startswith("multipart/form-data"):
        file = request.files.get("image")
        if file:
            image_np = _ensure_rgb(skimage.io.imread(file))
        else:
            url = request.form.get("url")
            if not url:
                return jsonify(error="Provide 'image' file or 'url' field"), 400
            image_np = _ensure_rgb(_load_remote(url))
        want_img = request.form.get("return_image", "false").lower() == "true"

    else:  # application/json
        try:
            data: dict = request.get_json(force=True)
        except Exception:
            return jsonify(error="Malformed JSON"), 400

        if "image" in data:
            image_np = _ensure_rgb(_decode_base64(data["image"]))
        elif "url" in data:
            image_np = _ensure_rgb(_load_remote(data["url"]))
        elif "path" in data:
            image_np = _ensure_rgb(_load_local(data["path"]))
        else:
            return jsonify(error="JSON must contain 'image', 'url', or 'path'"), 400
        want_img = bool(data.get("return_image", False))

    # -------- 2. Run inference --------
    r, detections = _run_detection(image_np)

    # -------- 3. Build response --------
    resp = {"detections": detections}
    if want_img:
        resp["annotated_image"] = base64.b64encode(
            _annotate_png_bytes(image_np, r).read()
        ).decode()
    return jsonify(resp), 200

# ---------------- health ---------------- #
@app.route("/")
def health():
    return jsonify(
        service   = "Mask‑R‑CNN inference API",
        version   = "3.0",
        status    = "ready",
        endpoints = ["/predict", "/json", "/api/predict"],
        classes   = CLASS_NAMES[1:]
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
