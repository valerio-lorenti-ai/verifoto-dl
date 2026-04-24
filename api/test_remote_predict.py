import os
import requests
from dotenv import load_dotenv

# Carica env
load_dotenv()

API_URL = os.getenv("API_BASE_URL") + "/predict"
API_KEY = os.getenv("INTERNAL_API_KEY")

IMAGE_PATH = "test_real.jpg"

headers = {
    "x-api-key": API_KEY
}

with open(IMAGE_PATH, "rb") as f:
    files = {
        "file": (IMAGE_PATH, f, "image/jpeg")
    }

    response = requests.post(API_URL, files=files, headers=headers)

print("Status code:", response.status_code)
print("Response:", response.text)