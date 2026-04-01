from fastapi import FastAPI, UploadFile, File, HTTPException
from app.inference import predict_image
from app.schemas import HealthResponse, PredictionResponse

app = FastAPI()


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File non valido")

    image_bytes = await file.read()

    predicted_class, score, confidence_level = predict_image(image_bytes)

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "score": score,
        "confidence_level": confidence_level
    }       