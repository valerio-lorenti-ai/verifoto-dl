from typing import Literal
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    service_name: str
    model_version: str
    threshold: float
    status: str


class PredictionResponse(BaseModel):
    status: Literal["success"]
    service_name: str
    filename: str
    predicted_class: Literal["real", "manipulated"]
    manipulation_probability: float = Field(ge=0.0, le=1.0)
    confidence_level: Literal["low", "medium", "high"]
    model_version: str
    threshold: float = Field(ge=0.0, le=1.0)
    decision: Literal["likely_valid", "likely_fraud", "uncertain"]
    inference_time_ms: float = Field(ge=0.0)

    @field_validator("filename")
    @classmethod
    def sanitize_filename(cls, v: str) -> str:
        """Strip path separators and null bytes from filename."""
        import os
        return os.path.basename(v.replace("\x00", ""))
