import os
import requests
from dotenv import load_dotenv

# Carica env
load_dotenv()

API_URL = os.getenv("API_BASE_URL") + "/model-info"
API_KEY = os.getenv("INTERNAL_API_KEY")

headers = {
    "x-api-key": API_KEY
}

response = requests.get(API_URL, headers=headers)

print("Status code:", response.status_code)
print("Response:", response.text)