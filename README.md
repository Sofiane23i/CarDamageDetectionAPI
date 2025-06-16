# CarDamageDetectionAPI

Download weights file from https://drive.google.com/file/d/1RGL8Ojk93fnmbwYHrUkEsHCX1woIYa3J/view?usp=sharing and move it inside weigths folder

# CarDamageDetectionAPI Call 
To call the API, start by running the Mask R-CNN Flask server on the first machine.
Then, from a second machine where the test vehicle images are located, call the Flask server using either the curl command or the Python script maskrcnn_api_call.py.
Don’t forget to update the IP address or domain name of the Flask server in the API call.

```python
import json, base64, requests
with open("../car2.jpg", "rb") as f:
    payload = {
        "image": base64.b64encode(f.read()).decode(),
        "return_image": False
    }
r = requests.post("http://ip_adress:5000/api/predict",
                  json=payload, timeout=120)
print(json.dumps(r.json(), indent=2))
