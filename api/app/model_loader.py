import logging
import os
from pathlib import Path

import torch
import timm

logger = logging.getLogger(__name__)

DEVICE = torch.device("cpu")


def load_model():
    # MODEL_PATH must point to the same file that download_model_if_missing() wrote.
    # Railway: weights/best.pt  (resolves to /app/weights/best.pt)
    # Local:   api/weights/best.pt  (resolves to <repo-root>/api/weights/best.pt)
    model_path = Path(os.getenv("MODEL_PATH", "weights/best.pt"))

    if not model_path.exists():
        raise FileNotFoundError(
            f"File del modello non trovato: {model_path}. "
            "Assicurati che download_model_if_missing() sia stato chiamato prima."
        )

    model = timm.create_model(
        "convnext_tiny",
        pretrained=False,
        num_classes=1,
        drop_rate=0.3,
    )

    ckpt = torch.load(str(model_path), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    model.to(DEVICE)
    model.eval()

    logger.info("Modello caricato da %s", model_path)

    return model
