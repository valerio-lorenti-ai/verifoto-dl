from io import BytesIO
import logging
import time
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

from app.model_loader import load_model
from app import settings

# ---------------------------------------------------------------------------
# Download model weights from R2 before anything else.
# This runs once at module import time (which happens at server startup).
# If the file already exists locally the function returns immediately.
# ---------------------------------------------------------------------------
try:
    # When running from inside api/ (Docker: WORKDIR /app, uvicorn app.main:app)
    from download_model import download_model_if_missing
except ModuleNotFoundError:
    # When running from the repo root (e.g. pytest from project root)
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from download_model import download_model_if_missing

try:
    download_model_if_missing()
except Exception as _download_exc:
    logger.critical(
        "Impossibile scaricare il modello all'avvio: %s — il server non può partire.",
        _download_exc,
        exc_info=True,
    )
    raise

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
        logger.info("Inizio caricamento modello in memoria...")
        t0 = time.perf_counter()
        try:
            _model = load_model()
        except Exception as exc:
            logger.critical(
                "Caricamento modello FALLITO: %s — il server non può servire richieste.",
                exc,
                exc_info=True,
            )
            raise
        elapsed = round(time.perf_counter() - t0, 1)
        model_loaded = True
        logger.info("Modello pronto in %.1fs.", elapsed)
    return _model


def predict_image(image_bytes: bytes):
    """
    Esegue preprocessing e inferenza sull'immagine.

    Nessun timeout interno: il tempo di inferenza su CPU può variare tra 5 e 15 secondi.
    Il timeout reale è gestito a livello di client HTTP (Supabase Edge Function)
    e dal proxy Railway (default 300 s).
    """
    t_start = time.perf_counter()

    # --- Preprocessing ---
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("Impossibile leggere l'immagine")

    t_after_open = time.perf_counter()
    x = transform(image).unsqueeze(0)
    t_after_preprocess = time.perf_counter()

    preprocess_ms = round((t_after_preprocess - t_start) * 1000, 1)
    logger.info("predict | preprocessing=%.1fms (open=%.1fms, transform=%.1fms)",
                preprocess_ms,
                (t_after_open - t_start) * 1000,
                (t_after_preprocess - t_after_open) * 1000)

    # --- Inferenza ---
    model = _get_model()
    t_before_infer = time.perf_counter()

    with torch.no_grad():
        logit = model(x).squeeze(1)
        score = torch.sigmoid(logit).item()

    t_after_infer = time.perf_counter()
    inference_ms = round((t_after_infer - t_before_infer) * 1000, 1)
    total_ms = round((t_after_infer - t_start) * 1000, 1)

    logger.info("predict | inference=%.1fms  total=%.1fms  score=%.4f",
                inference_ms, total_ms, score)

    # --- Classificazione ---
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

    return (
        predicted_class,
        round(score, 4),
        confidence_level,
        settings.MODEL_VERSION,
        settings.THRESHOLD,
        decision,
        total_ms,
    )
