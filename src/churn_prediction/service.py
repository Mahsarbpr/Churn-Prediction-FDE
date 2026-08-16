from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from churn_prediction.inference.prediction_service import (
    PredictionService,
)
from churn_prediction.runtime import build_prediction_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _prediction_service

    if _prediction_service is None:
        _prediction_service = build_prediction_service()

    yield


app = FastAPI(
    title="Churn Prediction Service",
    version="1.0.0",
    lifespan=lifespan,
)
class HealthResponse(BaseModel):
    status: str

class PredictionRequest(BaseModel):
    customer_id: str

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    model_version: str
    scored_at: str


_prediction_service: PredictionService | None = None

def set_prediction_service(
    service: PredictionService,
) -> None:
    global _prediction_service
    _prediction_service = service


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:
    if _prediction_service is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is not ready.",
        )

    try:
        result = _prediction_service.predict(
            customer_id=request.customer_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return PredictionResponse(**result)