from io import BytesIO

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

THRESHOLD = 0.2


def predict_image(image_bytes: bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logit = model(x).squeeze(1)
        score = torch.sigmoid(logit).item()

    predicted_class = "manipulated" if score >= THRESHOLD else "real"

    return predicted_class, round(score, 4)