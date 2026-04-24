from io import BytesIO
import time
import torch
from PIL import Image
from torchvision import transforms

from app.model_loader import load_model
from app import settings

# ---------------------------------------------------------------------------
# Lazy model loading
#
# The model is NOT loaded at import time. This avoids side effects during
# testing and allows the module to be imported without the weights file.
#
# _model is initialised on the first call to predict_image() via _get_model().
# model_loaded reflects whether the model has been successfully loaded.
# ---------------------------------------------------------------------------
_model = None
model_loaded = False

transform = transforms.Compose([
    transforms.Resize(settings.IMG_RESIZE),
    transforms.CenterCrop(settings.IMG_CROP),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=settings.NORMALIZE_MEAN,
        std=settings.NORMALIZE_STD,
    ),
])


def _get_model():
    """Return the model, loading it on first call (thread-safe via GIL for CPython)."""
    global _model, model_loaded
    if _model is None:
        print("🚀 Caricamento modello...")
        _model = load_model()
        model_loaded = True
        print("✅ Modello caricato")
    return _model


def predict_image(image_bytes: bytes, timeout_seconds: float = 5.0):
    start_time = time.time()

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("Impossibile leggere l'immagine")

    model = _get_model()
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logit = model(x).squeeze(1)
        score = torch.sigmoid(logit).item()

    if time.time() - start_time > timeout_seconds:
        raise TimeoutError("Inference timeout")

    predicted_class = "manipulated" if score >= settings.THRESHOLD else "real"

    if score >= 0.75 or score <= 0.15:
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
    return (
        predicted_class,
        round(score, 4),
        confidence_level,
        settings.MODEL_VERSION,
        settings.THRESHOLD,
        decision,
        inference_time_ms,
    )
