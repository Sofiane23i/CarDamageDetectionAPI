"""
Mask‑R‑CNN inference service
----------------------------
• POST /api/predict  – multipart *or* JSON body
      accepted payloads
      ────────────────────────────────────────────────────
      1. multipart/form‑data            field name  image
      2. application/json               {"image": "<base64 PNG/JPEG>",
                                         "return_image": true|false}

      response (HTTP 200)
      ────────────────────────────────────────────────────
      {
        "detections": [
          {"class": "dent",
           "score": 0.923,
           "box":  [y1, x1, y2, x2]},
          ...
        ],
        "annotated_image": "data:image/png;base64,..."   # only if asked
      }
"""

import os, sys, io, base64, json
from flask import Flask, request, jsonify
import numpy as np
import skimage.io
from skimage.color import gray2rgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Mask‑R‑CNN setup ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT_DIR  = os.path.abspath(".")
MODEL_DIR = os.path.join(ROOT_DIR, "weights")
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib, visualize

import tensorflow as tf
tf.get_logger().setLevel("ERROR")        # silence TF spam
graph = tf.compat.v1.get_default_graph() # for TF‑1 style sessions

class InferenceConfig(Config):
    NAME                       = "car_damage"
    NUM_CLASSES                = 1 + 6        # BG + 6 custom classes
    GPU_COUNT                  = 1
    IMAGES_PER_GPU             = 1
    DETECTION_MIN_CONFIDENCE   = 0.80

config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
model.load_weights(os.path.join(MODEL_DIR, "mask_rcnn_balloon_0146.h5"),
                   by_name=True)

CLASS_NAMES = ["BG", "broken part", "crack", "dent",
               "lamp broken", "missing part", "scratch"]

# ---------------- Flask app ----------------
app = Flask(__name__)

def _decode_to_ndarray(file_or_b64: str):
    """Accept a werkzeug FileStorage OR a base64 string and return RGB ndarray."""
    if isinstance(file_or_b64, (bytes, str)):      # base64 path
        image_bytes = base64.b64decode(file_or_b64)
        img = skimage.io.imread(io.BytesIO(image_bytes))
    else:                                          # Werkzeug FileStorage
        img = skimage.io.imread(file_or_b64)

    if img.ndim != 3:                              # ensure RGB
        img = gray2rgb(img)
    return img

def _run_detection(image_np):
    """Run Mask‑R‑CNN and format results."""
    with graph.as_default():
        r = model.detect([image_np], verbose=0)[0]
    dets = []
    for cid, score, box in zip(r["class_ids"], r["scores"], r["rois"]):
        dets.append({
            "class":  CLASS_NAMES[cid],
            "score":  float(score),
            "box":    [int(v) for v in box]        # [y1,x1,y2,x2]
        })
    return r, dets

def _annotate(image_np, r):
    """Draw masks/boxes on an image and return base64 string."""
    fig, ax = plt.subplots(figsize=(12,10))
    visualize.display_instances(
        image_np, r["rois"], r["masks"], r["class_ids"],
        CLASS_NAMES, r["scores"], ax=ax, title="Prediction"
    )
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

@app.route("/api/predict", methods=["POST"])
def api_predict():
    # -------- 1. Get the image --------
    if request.content_type.startswith("multipart/form-data"):
        file = request.files.get("image")
        if file is None:
            return jsonify(error="No image part in multipart request"), 400
        image_np = _decode_to_ndarray(file)
        want_img = request.form.get("return_image", "false").lower() == "true"
    else:  # application/json
        try:
            data = request.get_json(force=True)
            image_b64 = data["image"]
        except (KeyError, TypeError):
            return jsonify(error="JSON must contain 'image' (base64 string)"), 400
        image_np = _decode_to_ndarray(image_b64)
        want_img = bool(data.get("return_image", False))

    # -------- 2. Run inference --------
    r, detections = _run_detection(image_np)

    # -------- 3. Prepare response --------
    resp = {"detections": detections}
    if want_img:
        resp["annotated_image"] = _annotate(image_np, r)
    return jsonify(resp), 200

@app.route("/", methods=["GET"])
def health():
    return jsonify(service="Mask‑R‑CNN inference API",
                   status="ready",
                   endpoint="/api/predict",
                   classes=CLASS_NAMES[1:])

if __name__ == "__main__":
    # Don’t enable Flask debugger in production!
    app.run(host="0.0.0.0", port=5000, threaded=True)
