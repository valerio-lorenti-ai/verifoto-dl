from io import BytesIO
import time
import torch
from PIL import Image
from torchvision import transforms

from app.model_loader import load_model

# carico il modello UNA volta sola
print("🚀 Caricamento modello...")
model = load_model()

# stessa pipeline del training
transform = transforms.Compose([
    transforms.Resize(257),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

MODEL_VERSION = "pico_plus_exp3_aug"
THRESHOLD = 0.2


def predict_image(image_bytes: bytes, timeout_seconds: float = 5.0):

    start_time = time.time()

    if time.time() - start_time > timeout_seconds:
        raise TimeoutError("Inference timeout")

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("Impossibile leggere l'immagine")

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logit = model(x).squeeze(1)
        score = torch.sigmoid(logit).item()
    
    if time.time() - start_time > timeout_seconds:
        raise TimeoutError("Inference timeout")

    predicted_class = "manipulated" if score >= THRESHOLD else "real"

    if score >= 0.8 or score <= 0.1:
        confidence_level = "high"
    elif score >= 0.6 or score <= 0.2:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    if predicted_class == "manipulated" and confidence_level == "high":
        decision = "likely_fraud"
    elif predicted_class == "real" and confidence_level == "high":
        decision = "likely_valid"
    else:
        decision = "uncertain"

    inference_time_ms = round((time.time() - start_time) * 1000, 2)
    return predicted_class, round(score, 4), confidence_level, MODEL_VERSION, THRESHOLD, decision, inference_time_ms