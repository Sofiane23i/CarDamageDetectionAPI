import requests, sys, pathlib

img_path = pathlib.Path(sys.argv[1]).resolve()  # usage: python call_api.py path/to/img.jpg

# 1) Get annotated PNG
png_bytes = requests.get(
    "http://192.168.131.56:5000/predict",
    params={"image": str(img_path)},
    timeout=60
).content
open("prediction.png", "wb").write(png_bytes)
print("Saved → prediction.png")

# 2) Get raw detections
dets = requests.get(
    "http://192.168.131.56:5000/json",
    params={"image": str(img_path)},
    timeout=60
).json()
print("Detections:", dets)
