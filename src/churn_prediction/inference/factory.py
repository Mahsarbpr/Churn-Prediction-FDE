from __future__ import annotations

from pathlib import Path

from churn_prediction.inference.base import ChurnModel
from churn_prediction.inference.xgboost_model import (
    XGBoostChurnModel,
)


def load_churn_model(
    model_type: str,
    model_path: Path,
    model_version: str,
) -> ChurnModel:
    if model_type == "xgboost":
        return XGBoostChurnModel(
            model_path=model_path,
            model_version=model_version,
        )

    raise ValueError(
        f"Unsupported model type: {model_type}"
    )