import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.inference import predict_image
from app.schemas import HealthResponse, PredictionResponse

app = FastAPI()


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    request_id: str = None
):
    t0 = time.perf_counter()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File non valido")

    image_bytes = await file.read()
    t1 = time.perf_counter()

    try:
        predicted_class, score, confidence_level, model_version, threshold, decision, inference_time_ms = predict_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    t2 = time.perf_counter()

    print(f"read_file: {t1 - t0:.3f}s")
    print(f"inference_total: {t2 - t1:.3f}s")
    print(f"request_total: {t2 - t0:.3f}s")

    return {
        "status": "success",
        "service_name": "verifoto-dl",
        "filename": file.filename,
        "predicted_class": predicted_class,
        "manipulation_probability": score,
        "confidence_level": confidence_level,
        "model_version": model_version,
        "threshold": threshold,
        "decision": decision,
        "inference_time_ms": inference_time_ms
    }       