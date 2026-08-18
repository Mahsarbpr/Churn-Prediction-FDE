from __future__ import annotations

import pandas as pd
from xgboost import DMatrix, XGBClassifier

from churn_prediction.modeling import MODEL_FEATURES


def global_feature_importance(
    model: XGBClassifier,
) -> pd.DataFrame:
    booster = model.get_booster()
    importance_type = "total_gain"
    gain_scores = booster.get_score(
        importance_type=importance_type,
    )

    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            importance_type: [
                float(gain_scores.get(feature, 0.0))
                for feature in MODEL_FEATURES
            ],
        }
    )

    total = importance[importance_type].sum()

    if total > 0:
        importance["importance_share"] = (
            importance[importance_type] / total
        )
    else:
        importance["importance_share"] = 0.0

    return (
        importance
        .sort_values(
            importance_type,
            ascending=False,
        )
        .reset_index(drop=True)
    )


def explain_prediction(
    model: XGBClassifier,
    features: pd.DataFrame,
) -> pd.DataFrame:
    if len(features) != 1:
        raise ValueError(
            "Expected exactly one feature row."
        )

    feature_values = features[MODEL_FEATURES]

    matrix = DMatrix(
        feature_values,
        feature_names=MODEL_FEATURES,
    )

    contributions = model.get_booster().predict(
        matrix,
        pred_contribs=True,
    )[0]

    rows = []

    for index, feature in enumerate(MODEL_FEATURES):
        contribution = float(contributions[index])

        rows.append(
            {
                "feature": feature,
                "value": float(
                    feature_values.iloc[0][feature]
                ),
                "contribution": contribution,
                "direction": (
                    "increases churn risk"
                    if contribution > 0
                    else "decreases churn risk"
                    if contribution < 0
                    else "neutral"
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .assign(
            absolute_contribution=lambda frame:
                frame["contribution"].abs()
        )
        .sort_values(
            "absolute_contribution",
            ascending=False,
        )
        .reset_index(drop=True)
    )