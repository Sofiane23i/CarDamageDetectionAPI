# 1) Multipart example (curl)
# curl -F image=@car.jpg -F return_image=true http://localhost:5000/api/predict

# 1. Get detections as JSON (remote image URL)
#curl -G "http://SERVER_IP:5000/json" --data-urlencode "image=https://raw.githubusercontent.com/yourrepo/sample_images/car.jpg"

# 2. Get annotated PNG (file on the container)
# curl -o result.png "http://SERVER_IP:5000/predict?image=/data/car.jpg"

# 2) Pure JSON example (Python requests)
import json, base64, requests
with open("../car2.jpg", "rb") as f:
    payload = {
        "image": base64.b64encode(f.read()).decode(),
        "return_image": False
    }
r = requests.post("http://localhost:5000/api/predict",
                  json=payload, timeout=120)
print(json.dumps(r.json(), indent=2))


