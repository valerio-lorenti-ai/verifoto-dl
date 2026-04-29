import logging
import re
import time
from collections import defaultdict
from threading import Lock

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Security
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader

from app import settings
from app.inference import predict_image, model_loaded, _get_model
from app.schemas import HealthResponse, ModelInfoResponse, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="verifoto-dl",
    docs_url=None if settings.ENVIRONMENT == "production" else "/docs",
    redoc_url=None if settings.ENVIRONMENT == "production" else "/redoc",
)

# ---------------------------------------------------------------------------
# Global IP-based rate limiting middleware
# Max 60 requests per minute per IP across all endpoints.
# NOTE: in-memory — resets on process restart. For multi-instance deployments
# replace with a Redis-backed store.
# ---------------------------------------------------------------------------
_ip_rate_store: dict[str, dict] = defaultdict(lambda: {"count": 0, "window_start": 0.0})
_ip_rate_lock = Lock()
_GLOBAL_RATE_LIMIT = 60    # max requests per window
_GLOBAL_RATE_WINDOW = 60.0  # window in seconds


@app.middleware("http")
async def global_rate_limit_middleware(request: Request, call_next):
    # /health is excluded — must always be reachable by load balancers
    if request.url.path == "/health":
        return await call_next(request)

    # Prefer the first IP in X-Forwarded-For (set by reverse proxy / Railway)
    forwarded = request.headers.get("x-forwarded-for")
    client_ip = forwarded.split(",")[0].strip() if forwarded else (
        request.client.host if request.client else "unknown"
    )

    now = time.time()
    with _ip_rate_lock:
        entry = _ip_rate_store[client_ip]
        if now - entry["window_start"] >= _GLOBAL_RATE_WINDOW:
            entry["count"] = 1
            entry["window_start"] = now
        else:
            entry["count"] += 1

        if entry["count"] > _GLOBAL_RATE_LIMIT:
            retry_after = int(_GLOBAL_RATE_WINDOW - (now - entry["window_start"])) + 1
            logger.warning("Rate limit exceeded for IP %s", client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Troppe richieste. Riprova più tardi."},
                headers={"Retry-After": str(retry_after)},
            )

    return await call_next(request)


# ---------------------------------------------------------------------------
# Magic bytes — validate actual file content, not just the MIME type header
# ---------------------------------------------------------------------------
_IMAGE_MAGIC: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),    # WebP: RIFF....WEBP
    (b"II\x2a\x00", "image/tiff"),
    (b"MM\x00\x2a", "image/tiff"),
]

_UUID_RE = re.compile(r"^[a-zA-Z0-9\-]{1,64}$")


def _is_valid_image(data: bytes) -> bool:
    """Return True if `data` starts with a recognised image magic byte sequence."""
    for magic, _ in _IMAGE_MAGIC:
        if data[: len(magic)] == magic:
            if magic == b"RIFF":
                return len(data) >= 12 and data[8:12] == b"WEBP"
            return True
    return False


# ---------------------------------------------------------------------------
# API key auth — fail-closed in production
# ---------------------------------------------------------------------------
_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def require_api_key(key: str = Security(_api_key_header)) -> None:
    if not settings.INTERNAL_API_KEY:
        if settings.ENVIRONMENT == "production":
            # Misconfigured production — refuse all requests rather than open the API
            raise HTTPException(status_code=503, detail="Servizio non disponibile")
        return  # Dev/local: skip if key not set
    if not key or key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Non autorizzato")


# ---------------------------------------------------------------------------
# Exception handlers — never leak internal details to the client
# ---------------------------------------------------------------------------
@app.exception_handler(ValueError)
async def _value_error_handler(request: Request, exc: ValueError):
    raise HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def _generic_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    raise HTTPException(status_code=500, detail="Errore interno del server")


# ---------------------------------------------------------------------------
# Startup — preload model so the first /predict request is never charged
# the cold-start cost inside the 5 s inference timeout window.
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _preload_model():
    logger.info("Precaricamento modello all'avvio...")
    _get_model()
    logger.info("Modello pronto — server operativo")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
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
    # Validate request_id format (UUID-like, alphanumeric + hyphens, max 64 chars)
    if request_id is not None and not _UUID_RE.match(request_id):
        raise HTTPException(status_code=400, detail="request_id non valido")

    # MIME type check (client-supplied, not trusted alone)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File non valido: deve essere un'immagine")

    image_bytes = await file.read()

    # Size check
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="File vuoto")

    if len(image_bytes) > settings.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File troppo grande: massimo {settings.MAX_FILE_SIZE_MB} MB",
        )

    # Magic bytes check — validates actual file content regardless of MIME header
    if not _is_valid_image(image_bytes):
        raise HTTPException(status_code=400, detail="File non valido: formato immagine non riconosciuto")

    try:
        predicted_class, score, confidence_level, model_version, threshold, decision, inference_time_ms = predict_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    logger.info(
        "predict | request_id=%s decision=%s score=%.4f confidence=%s total_ms=%.1f",
        request_id or "n/a", decision, score, confidence_level, inference_time_ms,
    )

    return {
        "status": "success",
        "service_name": settings.SERVICE_NAME,
        "filename": file.filename or "image",
        "predicted_class": predicted_class,
        "manipulation_probability": score,
        "confidence_level": confidence_level,
        "model_version": model_version,
        "threshold": threshold,
        "decision": decision,
        "inference_time_ms": inference_time_ms,
    }
