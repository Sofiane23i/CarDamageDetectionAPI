"""
Mask‑R‑CNN inference service (v2)
---------------------------------
* GET /predict?image=<path>  -> PNG image with detections drawn
* GET /json?image=<path>     -> JSON array of detections
* POST /api/predict          -> legacy multipart / base64 endpoint
"""

import os, sys, io, base64, json
from flask import Flask, request, jsonify, send_file
import numpy as np
import skimage.io
from skimage.color import gray2rgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------‑ Mask‑R‑CNN setup (unchanged) ---------- #
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
    NUM_CLASSES              = 1 + 6        # BG + 6 classes
    GPU_COUNT                = 1
    IMAGES_PER_GPU           = 1
    DETECTION_MIN_CONFIDENCE = 0.80

config = InferenceConfig()
model  = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
model.load_weights(os.path.join(MODEL_DIR, "mask_rcnn_car_damage.h5"), by_name=True)

CLASS_NAMES = ["BG", "broken part", "crack", "dent", "lamp broken", "missing part", "scratch"]

# --------‑ Flask app ---------- #
app = Flask(__name__)

# ---------- utils ------------- #
def _decode_to_ndarray(file_or_b64):
    """Werkzeug FileStorage *or* base64 → RGB ndarray."""
    if isinstance(file_or_b64, (bytes, str)):
        image_bytes = base64.b64decode(file_or_b64)
        img = skimage.io.imread(io.BytesIO(image_bytes))
    else:
        img = skimage.io.imread(file_or_b64)

    if img.ndim != 3:
        img = gray2rgb(img)
    return img

def _load_from_disk(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}")
    img = skimage.io.imread(path)
    if img.ndim != 3:
        img = gray2rgb(img)
    return img

def _run_detection(image_np):
    """Forward pass + formatted detections."""
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
    """Return PNG bytes (ready for send_file)."""
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

# ---------- new GET endpoints ---------- #
@app.route("/predict", methods=["GET"])
def get_predict():
    path = request.args.get("image")
    if not path:
        return jsonify(error="Missing ?image=<path>"), 400
    try:
        img = _load_from_disk(path)
    except Exception as e:
        return jsonify(error=str(e)), 400

    r, _ = _run_detection(img)
    return send_file(_annotate_png_bytes(img, r),
                     mimetype="image/png",
                     download_name="prediction.png")

@app.route("/json", methods=["GET"])
def get_json():
    path = request.args.get("image")
    if not path:
        return jsonify(error="Missing ?image=<path>"), 400
    try:
        img = _load_from_disk(path)
    except Exception as e:
        return jsonify(error=str(e)), 400

    _, dets = _run_detection(img)
    return jsonify(detections=dets), 200

# ---------- legacy POST /api/predict (unchanged) ---------- #
@app.route("/api/predict", methods=["POST"])
def api_predict():
    # … existing multipart / base64 handler …
    ...

@app.route("/")
def health():
    return jsonify(service="Mask‑R‑CNN inference API v2",
                   status="ready",
                   endpoints=["/predict", "/json", "/api/predict"],
                   classes=CLASS_NAMES[1:])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
