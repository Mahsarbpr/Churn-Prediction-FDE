from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from churn_prediction.modeling import (
    MODEL_FEATURES,
    train_xgboost,
)


def main() -> None:
    training_dir = Path(
        os.getenv(
            "SM_CHANNEL_TRAINING",
            "/opt/ml/input/data/training",
        )
    )

    model_dir = Path(
        os.getenv(
            "SM_MODEL_DIR",
            "/opt/ml/model",
        )
    )

    dataset_path = (
        training_dir
        / "training_dataset.parquet"
    )

    dataset = pd.read_parquet(
        dataset_path
    )

    model = train_xgboost(
        dataset
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        model_dir
        / "churn_model.json"
    )

    model.save_model(
        model_path
    )

    metadata = {
        "model_type": "xgboost",
        "model_version": "xgb-sagemaker-v1",
        "feature_version": "rfm-v1",
        "features": MODEL_FEATURES,
    }

    metadata_path = (
        model_dir
        / "churn_model_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"Saved model artifacts to {model_dir}"
    )


if __name__ == "__main__":
    main()