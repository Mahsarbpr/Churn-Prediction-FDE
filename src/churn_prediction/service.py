from __future__ import annotations
import json
import logging
import uuid
import time

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from churn_prediction.inference.prediction_service import (
    PredictionService,
)
from churn_prediction.runtime import build_prediction_service

from churn_prediction.telemetry import (
    configure_metrics,
    prediction_errors,
    request_latency,
    score_distribution,
)

logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_metrics()
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
    request_id = str(uuid.uuid4())
    started = time.perf_counter()

    try:
        result = _prediction_service.predict(
            customer_id=request.customer_id,
        )

        logger.info(
            json.dumps(
            {
                "event": "churn_prediction",
                "request_id": request_id,
                "customer_id": result["customer_id"],
                "churn_probability": result["churn_probability"],
                "model_version": result["model_version"],
                "scored_at": result["scored_at"],
                "status": "success",
            }
            )
        )
        score_distribution.record(
            float(result["churn_probability"])
        )

        return PredictionResponse(**result)

    except ValueError as exc:
        prediction_errors.add(
            1,
            {"error_type": "customer_not_found"},
        )
        logger.warning(
            json.dumps(
            {
                "event": "churn_prediction",
                "request_id": request_id,
                "customer_id": request.customer_id,
                "status": "not_found",
                "error": str(exc),
            }
            )
        )
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception(
            json.dumps(
                {
                    "event": "churn_prediction",
                    "request_id": request_id,
                    "customer_id": request.customer_id,
                    "status": "error",
                }
            )
        )

        prediction_errors.add(
            1,
            {"error_type": "internal"},
        )

        raise

    finally:
        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000

        request_latency.record(
            elapsed_ms,
            {"route": "/predict"},
        )