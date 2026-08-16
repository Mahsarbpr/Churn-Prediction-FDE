from pathlib import Path

import pandas as pd
import pytest

from churn_prediction.inference.xgboost_model import (
    XGBoostChurnModel,
)
from churn_prediction.modeling import (
    MODEL_FEATURES,
    train_xgboost,
)
from churn_prediction.inference.prediction_service import (
    PredictionService,
)
from churn_prediction.repositories.local import (
    LocalEventRepository,
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


def test_xgboost_model_loads_and_predicts(
    tmp_path: Path,
) -> None:
    dataset = _training_dataset()

    trained_model = train_xgboost(dataset)

    model_path = tmp_path / "model.json"
    trained_model.save_model(model_path)

    model = XGBoostChurnModel(
        model_path=model_path,
        model_version="test-v1",
    )

    probabilities = model.predict_probability(dataset)

    assert len(probabilities) == len(dataset)
    assert probabilities.between(0.0, 1.0).all()
    assert model.model_version == "test-v1"
    assert model.feature_names == MODEL_FEATURES


def test_xgboost_model_rejects_missing_features(
    tmp_path: Path,
) -> None:
    dataset = _training_dataset()

    trained_model = train_xgboost(dataset)

    model_path = tmp_path / "model.json"
    trained_model.save_model(model_path)

    model = XGBoostChurnModel(
        model_path=model_path,
        model_version="test-v1",
    )

    incomplete_features = dataset.drop(
        columns=["sessions_30d"]
    )

    with pytest.raises(
        ValueError,
        match="Missing model features",
    ):
        model.predict_probability(incomplete_features)

def test_prediction_service_returns_probability(
    tmp_path: Path,
) -> None:
    dataset = _training_dataset()

    trained_model = train_xgboost(dataset)

    model_path = tmp_path / "model.json"
    trained_model.save_model(model_path)

    model = XGBoostChurnModel(
        model_path=model_path,
        model_version="test-v1",
    )

    service = PredictionService(
        model=model,
        event_repository=repository,
        clock=lambda: pd.Timestamp(
            "2024-06-01T12:00:00Z"
        ),
    )

    customer_features = dataset.iloc[[0]]

    result = service.predict(
        customer_id="cust_00000",
    )

    assert result["customer_id"] == "cust_00000"
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["model_version"] == "test-v1"
    assert (
        result["scored_at"]
        == "2024-06-01T12:00:00+00:00"
    )


def test_prediction_service_returns_probability(
    tmp_path: Path,
) -> None:
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

    result = service.predict(
        customer_id="cust_00001",
    )

    assert result["customer_id"] == "cust_00001"
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["model_version"] == "test-v1"


def test_prediction_service_rejects_unknown_customer(
    tmp_path: Path,
) -> None:
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
            }
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

    with pytest.raises(
        ValueError,
        match="No events found",
    ):
        service.predict(
            customer_id="cust_missing",
        )