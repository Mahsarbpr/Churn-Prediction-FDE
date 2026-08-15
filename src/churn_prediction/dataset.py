import pandas as pd

from churn_prediction.features import build_rfm_features
from churn_prediction.labels import build_churn_labels


def build_training_dataset(
    events: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
    horizon_days: int = 60,
) -> pd.DataFrame:
    snapshots: list[pd.DataFrame] = []

    for cutoff in cutoffs:
        cutoff = pd.Timestamp(cutoff)

        features = build_rfm_features(
            events,
            as_of=cutoff,
        )

        labels = build_churn_labels(
            events,
            cutoff=cutoff,
            horizon_days=horizon_days,
        )

        snapshot = features.merge(
            labels,
            on="customer_id",
            how="inner",
        )

        snapshots.append(snapshot)

    if not snapshots:
        return pd.DataFrame()

    return (
        pd.concat(snapshots, ignore_index=True)
        .sort_values(["customer_id", "cutoff"])
        .reset_index(drop=True)
    )