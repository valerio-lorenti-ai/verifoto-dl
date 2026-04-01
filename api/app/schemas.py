from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class PredictionResponse(BaseModel):
    status: str
    service_name: str
    filename: str
    predicted_class: str
    manipulation_probability: float
    confidence_level: str
    model_version: str
    threshold: float
    decision: str
    inference_time_ms: float
    