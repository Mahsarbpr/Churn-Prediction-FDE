from __future__ import annotations

from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from churn_prediction.modeling import (
    MODEL_FEATURES,
    predict_churn_probability,
)


class XGBoostChurnModel:
    def __init__(
        self,
        model_path: Path,
        model_version: str,
    ) -> None:
        self._model = XGBClassifier()
        self._model.load_model(model_path)
        self._model_version = model_version

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def feature_names(self) -> list[str]:
        return MODEL_FEATURES.copy()

    def predict_probability(
        self,
        features: pd.DataFrame,
    ) -> pd.Series:
        missing_features = (
            set(self.feature_names) - set(features.columns)
        )

        if missing_features:
            raise ValueError(
                "Missing model features: "
                f"{sorted(missing_features)}"
            )

        return predict_churn_probability(
            self._model,
            features,
        )