from io import BytesIO

import numpy as np
import torch
from PIL import Image

from app.model_loader import load_model

# carico il modello UNA volta sola
print("🚀 Caricamento modello...")
model = load_model()

THRESHOLD = 0.2


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("Impossibile leggere l'immagine")

    # Resize: lato corto -> 257
    width, height = image.size
    short_side = min(width, height)
    scale = 257 / short_side

    new_width = round(width * scale)
    new_height = round(height * scale)

    image = image.resize((new_width, new_height), Image.BILINEAR)

    # Center crop 224x224
    left = (new_width - 224) // 2
    top = (new_height - 224) // 2
    right = left + 224
    bottom = top + 224

    image = image.crop((left, top, right, bottom))

    # To tensor [C, H, W] in range [0, 1]
    arr = np.array(image).astype("float32") / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)

    # Normalize ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    x = (x - mean) / std

    return x.unsqueeze(0)


def predict_image(image_bytes: bytes):
    x = preprocess_image(image_bytes)

    with torch.no_grad():
        logit = model(x).squeeze(1)
        score = torch.sigmoid(logit).item()

    predicted_class = "manipulated" if score >= THRESHOLD else "real"

    if score >= 0.8 or score <= 0.1:
        confidence_level = "high"
    elif score >= 0.6 or score <= 0.2:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    return predicted_class, round(score, 4), confidence_level