# Mask‑R‑CNN Car‑Damage Inference API

![Python version](https://img.shields.io/badge/python-3.6+-blue)

A lightweight Flask micro‑service that exposes a Mask‑R‑CNN model trained for car‑damage inspection (**broken part, crack, dent, lamp broken, missing part, scratch**).

---

## ✨ Features

| Endpoint       | Method   | Purpose                                                                 |
| -------------- | -------- | ----------------------------------------------------------------------- |
| `/predict`     | **GET**  | Returns a **PNG** image with detections drawn on it.                    |
| `/json`        | **GET**  | Returns raw detection results (class, score, bounding‑box) in **JSON**. |
| `/api/predict` | **POST** | Legacy endpoint that accepts file uploads or base64 payloads.           |

> **Heads‑up:** CUDA is disabled by default, so the service runs fine on a regular CPU‑only server.

---

## 🚀 Quick Start

```bash
# 1. Clone & install deps
$ git clone https://github.com/your‑org/car‑damage‑api.git
$ cd car‑damage‑api
$ python3 -m venv venv && source venv/bin/activate
$ pip install -r requirements.txt

# 2. Download your trained weights ↴
Download weights file from https://drive.google.com/file/d/1RGL8Ojk93fnmbwYHrUkEsHCX1woIYa3J/view?usp=sharing and move it inside weigths folder
$ mkdir -p weights && mv mask_rcnn_car_damage.h5 weights/

# 3. Run the service
$ python api.py  # ⇒ http://localhost:5000
```

Docker fan?

```bash
$ docker build -t car-damage-api .
$ docker run -p 5000:5000 car-damage-api
```

---

## 📑 Endpoints

### `GET /predict`

Returns an **annotated PNG**.

| Query param | Type   | Description                                                    |
| ----------- | ------ | -------------------------------------------------------------- |
| `image`     | string | Absolute or relative path **on the server** to the image file. |

```bash
curl -o out.png \
  "http://localhost:5000/predict?image=/data/car.jpg"
```

---

### `GET /json`

Returns detections in JSON.

```bash
curl "http://localhost:5000/json?image=/data/car.jpg"
```

Response:

```json
{
  "detections": [
    {"class": "dent", "score": 0.92, "box": [105, 48, 302, 249]},
    {"class": "scratch", "score": 0.88, "box": [410, 120, 530, 200]}
  ]
}
```

---

### `POST /api/predict` (legacy)

Accepts either `multipart/form‑data` or `application/json` with a base64 payload.
See [`api.py`](api.py) for the exact request/response schema.

---

## 🐍 Python usage

```python
import requests

img_path = "tests/car.jpg"

# 1) Get visualisation
data = requests.get(
    "http://localhost:5000/predict",
    params={"image": img_path}, timeout=60
).content
open("preview.png", "wb").write(data)

# 2) Get raw detections
resp = requests.get(
    "http://localhost:5000/json",
    params={"image": img_path}, timeout=60
).json()
print(resp)
```

---

## 🛠  Configuration

| Variable                   | Default     | Description                                       |
| -------------------------- | ----------- | ------------------------------------------------- |
| `DETECTION_MIN_CONFIDENCE` | `0.80`      | Minimum score to keep a detection.                |
| `MODEL_DIR`                | `./weights` | Where to find the `mask_rcnn_car_damage.h5` file. |

To tweak thresholds, open **`api.py` → `InferenceConfig`**.

---

## 🤕 Troubleshooting

* **`FileNotFoundError`** – Make sure the `image` path you pass actually exists inside the container or host running the API.
* **Slow predictions?** – Reduce `IMAGES_PER_GPU` or (`GPU_COUNT` + enable CUDA).
* **Matplotlib crashes in headless server** – Already handled by `matplotlib.use("Agg")`.

---

## 📝 License

MIT © 2025 SpidR




