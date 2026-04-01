from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    score: float
    confidence_level: str