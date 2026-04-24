"""
Test HTTP del servizio — richiede server attivo.
Configura API_URL e API_KEY prima di eseguire.
Esegui da api/: python test_api.py
"""
import sys
import os
import requests
from dotenv import load_dotenv

# Carica env
load_dotenv()

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("INTERNAL_API_KEY", "")

headers = {"x-api-key": API_KEY} if API_KEY else {}

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"✅ {label}")
        passed += 1
    else:
        print(f"❌ {label} {detail}")
        failed += 1


# --- GET /health ---
r = requests.get(f"{API_URL}/health")
check("GET /health → 200", r.status_code == 200)
data = r.json()
check("GET /health → model_loaded presente", "model_loaded" in data)
check("GET /health → status ok", data.get("status") == "ok")

# --- GET /model-info ---
r = requests.get(f"{API_URL}/model-info", headers=headers)
check("GET /model-info → 200", r.status_code == 200, f"(got {r.status_code})")
data = r.json()
check("GET /model-info → service_name presente", "service_name" in data)
check("GET /model-info → threshold presente", "threshold" in data)

# --- POST /predict con immagine reale ---
with open("test_real.jpg", "rb") as f:
    r = requests.post(f"{API_URL}/predict", files={"file": ("test_real.jpg", f, "image/jpeg")}, headers=headers)
check("POST /predict (real) → 200", r.status_code == 200, f"(got {r.status_code})")
data = r.json()
check("POST /predict (real) → decision presente", "decision" in data)

# --- POST /predict con immagine manipolata ---
with open("test_manipulated.jpg", "rb") as f:
    r = requests.post(f"{API_URL}/predict", files={"file": ("test_manipulated.jpg", f, "image/jpeg")}, headers=headers)
check("POST /predict (manipulated) → 200", r.status_code == 200, f"(got {r.status_code})")
data = r.json()
check("POST /predict (manipulated) → decision presente", "decision" in data)

print(f"\nRisultato: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
