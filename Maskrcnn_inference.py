from flask import Flask, request, render_template_string
import os
import sys
import io
import base64
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Root paths
ROOT_DIR = os.path.abspath(".")
MODEL_DIR = os.path.join(ROOT_DIR, "weights")
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib, visualize

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
graph = tf.compat.v1.get_default_graph()

# Config for inference
class InferenceConfig(Config):
    NAME = "balloon"
    NUM_CLASSES = 1 + 6  # background + 6 custom classes
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.8

config = InferenceConfig()

# Load model
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_balloon_0146.h5")
model.load_weights(MODEL_PATH, by_name=True)

# Class names
class_names = ["BG", "broken part", "crack", "dent", "lamp broken", "missing part", "scratch"]

# Flask app
app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return '''
            <h1>Upload Image for Inference</h1>
            <form method="POST" action="/predict" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*">
                <input type="submit" value="Upload">
            </form>
        '''

    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    image = skimage.io.imread(file)

    if image.ndim != 3:
        from skimage.color import gray2rgb
        image = gray2rgb(image)

    with graph.as_default():
        results = model.detect([image], verbose=0)

    r = results[0]

    # Plot results and convert to base64 string
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'], ax=ax, title="Prediction")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Render result with new upload form
    html = f'''
        <h1>Prediction Result</h1>
        <img src="data:image/png;base64,{image_base64}" alt="Prediction Result"><br><br>

        <h2>Upload New Image</h2>
        <form method="POST" action="/predict" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*">
            <input type="submit" value="Upload">
        </form>
    '''
    return html

@app.route('/')
def index():
    return '''
        <h1>Upload Image for Inference</h1>
        <form method="POST" action="/predict" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*">
            <input type="submit" value="Upload">
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
