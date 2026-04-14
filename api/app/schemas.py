from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    service_name: str
    model_version: str
    threshold: float
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
