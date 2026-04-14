import logging
import time

from fastapi import FastAPI, UploadFile, File, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from app import settings
from app.inference import predict_image, model_loaded
from app.schemas import HealthResponse, ModelInfoResponse, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="verifoto-dl",
    docs_url=None if settings.ENVIRONMENT == "production" else "/docs",
    redoc_url=None if settings.ENVIRONMENT == "production" else "/redoc",
)

# --- Auth ---

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def require_api_key(key: str = Security(api_key_header)):
    if not settings.INTERNAL_API_KEY:
        return  # API key non configurata: skip (utile in dev locale)
    if key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="API key non valida")


# --- Exception handlers ---

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    raise HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def generic_error_handler(request, exc):
    logger.error(f"Errore non gestito: {exc}")
    raise HTTPException(status_code=500, detail="Errore interno del server")


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model_loaded": model_loaded}


@app.get("/model-info", response_model=ModelInfoResponse, dependencies=[Security(require_api_key)])
def model_info():
    return {
        "service_name": settings.SERVICE_NAME,
        "model_version": settings.MODEL_VERSION,
        "threshold": settings.THRESHOLD,
        "status": "ok",
    }


@app.post("/predict", response_model=PredictionResponse, dependencies=[Security(require_api_key)])
async def predict(file: UploadFile = File(...), request_id: str = None):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File non valido: deve essere un'immagine")

    image_bytes = await file.read()

    if len(image_bytes) > settings.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File troppo grande: massimo {settings.MAX_FILE_SIZE_MB} MB"
        )

    t0 = time.perf_counter()

    try:
        predicted_class, score, confidence_level, model_version, threshold, decision, inference_time_ms = predict_image(image_bytes)
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        f"predict | request_id={request_id} file={file.filename} "
        f"decision={decision} score={score} time={inference_time_ms}ms"
    )

    return {
        "status": "success",
        "service_name": settings.SERVICE_NAME,
        "filename": file.filename,
        "predicted_class": predicted_class,
        "manipulation_probability": score,
        "confidence_level": confidence_level,
        "model_version": model_version,
        "threshold": threshold,
        "decision": decision,
        "inference_time_ms": inference_time_ms,
    }
