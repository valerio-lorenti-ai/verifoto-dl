import requests

API_URL = "https://verifoto-dl-production.up.railway.app/predict"
IMAGE_PATH = "test_real.jpg"

with open(IMAGE_PATH, "rb") as f:
    files = {
        "file": (IMAGE_PATH, f, "image/jpeg")
    }

    response = requests.post(API_URL, files=files)

print("Status code:", response.status_code)
print("Response JSON:", response.json())