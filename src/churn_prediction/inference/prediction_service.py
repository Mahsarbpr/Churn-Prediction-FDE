from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from churn_prediction.features import build_rfm_features
from churn_prediction.inference.base import ChurnModel
from churn_prediction.repositories.base import EventRepository


class PredictionService:
    def __init__(
        self,
        model: ChurnModel,
        event_repository: EventRepository,
        clock: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        self._model = model
        self._event_repository = event_repository
        self._clock = clock or (
            lambda: pd.Timestamp.now(tz="UTC")
        )

    def predict(
        self,
        customer_id: str,
    ) -> dict[str, object]:
        events = self._event_repository.get_customer_events(
            customer_id
        )

        if events.empty:
            raise ValueError(
                f"No events found for customer: {customer_id}"
            )

        scored_at = self._clock()

        features = build_rfm_features(
            events,
            as_of=scored_at,
        )
        if len(features) != 1:
            raise ValueError(
                f"Expected one feature row for customer {customer_id}, "
                f"got {len(features)}."
        )

        if features.iloc[0]["customer_id"] != customer_id:
            raise ValueError(
                f"Feature row does not match customer {customer_id}."
        )
        probabilities = self._model.predict_probability(
            features
        )
        if len(probabilities) != 1:
            raise ValueError(
                f"Expected one prediction for customer {customer_id}, "
                f"got {len(probabilities)}."
            )

        probability = float(probabilities.iloc[0])
        return {
            "customer_id": customer_id,
            "churn_probability": probability,
            "model_version": self._model.model_version,
            "scored_at": scored_at.isoformat(),
        }