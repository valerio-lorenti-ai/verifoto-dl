"""
download_model.py
-----------------
Downloads the ConvNeXt model weights from Cloudflare R2 if they are not
already present on disk.

Required environment variables:
    R2_ACCESS_KEY_ID      – R2 API token access key
    R2_SECRET_ACCESS_KEY  – R2 API token secret key
    R2_ACCOUNT_ID         – Cloudflare account ID (used to build the endpoint URL)
    R2_BUCKET_NAME        – Name of the R2 bucket (e.g. verifoto-models)
    R2_MODEL_KEY          – Object key inside the bucket (default: best.pt)
    MODEL_PATH            – Local destination path (default: weights/best.pt)

Railway:  MODEL_PATH=weights/best.pt   (WORKDIR is /app, weights dir is /app/weights/)
Local:    MODEL_PATH=api/weights/best.pt  (cwd is repo root)

Usage:
    from download_model import download_model_if_missing
    download_model_if_missing()
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def download_model_if_missing() -> None:
    """Download model weights from Cloudflare R2 if the local file is absent."""

    model_path = Path(os.getenv("MODEL_PATH", "weights/best.pt"))

    if model_path.exists():
        logger.info("Modello già presente in %s — skip download.", model_path)
        return

    # --- Read credentials from environment (never log secret values) ---
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    account_id = os.getenv("R2_ACCOUNT_ID")
    bucket_name = os.getenv("R2_BUCKET_NAME")
    model_key = os.getenv("R2_MODEL_KEY", "best.pt")

    missing = [
        name
        for name, val in [
            ("R2_ACCESS_KEY_ID", access_key),
            ("R2_SECRET_ACCESS_KEY", secret_key),
            ("R2_ACCOUNT_ID", account_id),
            ("R2_BUCKET_NAME", bucket_name),
        ]
        if not val
    ]
    if missing:
        raise EnvironmentError(
            f"Variabili d'ambiente mancanti per il download del modello: {', '.join(missing)}"
        )

    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    logger.info(
        "Modello non trovato in %s. Download da R2 bucket '%s', key '%s'...",
        model_path,
        bucket_name,
        model_key,
    )

    # Create destination directory if needed
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import boto3
        from botocore.config import Config

        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )

        s3.download_file(bucket_name, model_key, str(model_path))

    except Exception as exc:
        # Clean up partial download if it exists
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(f"Download del modello fallito: {exc}") from exc

    logger.info("Modello scaricato con successo in %s.", model_path)
