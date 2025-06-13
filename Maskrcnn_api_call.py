# 1) Multipart example (curl)
# curl -F image=@car.jpg -F return_image=true http://localhost:5000/api/predict

# 2) Pure JSON example (Python requests)
import json, base64, requests
with open("car.jpg", "rb") as f:
    payload = {
        "image": base64.b64encode(f.read()).decode(),
        "return_image": True
    }
r = requests.post("http://192.168.118.56:5000/api/predict",
                  json=payload, timeout=120)
print(json.dumps(r.json(), indent=2))
