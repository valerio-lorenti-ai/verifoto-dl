from fastapi import FastAPI, UploadFile, File, HTTPException
from app.inference import predict_image

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File non valido")

    image_bytes = await file.read()

    predicted_class, score = predict_image(image_bytes)

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "score": score
    }