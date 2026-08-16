from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from churn_prediction.inference.prediction_service import (
    PredictionService,
)
from churn_prediction.inference.xgboost_model import (
    XGBoostChurnModel,
)
from churn_prediction.modeling import train_xgboost
from churn_prediction.repositories.local import (
    LocalEventRepository,
)
from churn_prediction.service import (
    app,
    set_prediction_service,
)


def _training_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_id": f"cust_{i:05d}",
                "recency_days": float(i * 5),
                "has_meaningful_history": 1,
                "sessions_30d": i % 4,
                "sessions_90d": i % 8,
                "purchase_count_90d": i % 3,
                "revenue_90d": float(i % 3) * 10.0,
                "churn": i % 2,
            }
            for i in range(20)
        ]
    )


def _client(
    tmp_path: Path,
) -> TestClient:
    dataset = _training_dataset()

    trained_model = train_xgboost(dataset)

    model_path = tmp_path / "model.json"
    trained_model.save_model(model_path)

    model = XGBoostChurnModel(
        model_path=model_path,
        model_version="test-v1",
    )

    events = pd.DataFrame(
        [
            {
                "event_id": "evt_00000001",
                "customer_id": "cust_00001",
                "event_type": "session",
                "timestamp": pd.Timestamp(
                    "2024-05-20T12:00:00Z"
                ),
                "properties": {
                    "duration_sec": 120,
                },
            },
            {
                "event_id": "evt_00000002",
                "customer_id": "cust_00001",
                "event_type": "purchase",
                "timestamp": pd.Timestamp(
                    "2024-05-21T12:00:00Z"
                ),
                "properties": {
                    "amount_usd": 9.99,
                },
            },
        ]
    )

    repository = LocalEventRepository(events)

    service = PredictionService(
        model=model,
        event_repository=repository,
        clock=lambda: pd.Timestamp(
            "2024-06-01T12:00:00Z"
        ),
    )

    set_prediction_service(service)

    return TestClient(app)


def test_health() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


def test_predict_returns_churn_probability(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/predict",
        json={
            "customer_id": "cust_00001",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["customer_id"] == "cust_00001"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["model_version"] == "test-v1"
    assert (
        body["scored_at"]
        == "2024-06-01T12:00:00+00:00"
    )


def test_predict_returns_404_for_unknown_customer(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/predict",
        json={
            "customer_id": "cust_missing",
        },
    )

    assert response.status_code == 404

    body = response.json()

    assert (
        body["detail"]
        == "No events found for customer: cust_missing"
    )