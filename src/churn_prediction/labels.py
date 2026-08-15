import pandas as pd

CHURN_HORIZON_DAYS = 60

MEANINGFUL_ACTIVITY = {
    "session",
    "purchase",
    "push_open",
    "in_app_event",
    "campaign_click",
    "support_ticket",
}


def build_churn_labels(
    events: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon_days: int = CHURN_HORIZON_DAYS,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)

    horizon_end = cutoff + pd.Timedelta(days=horizon_days)

    existing_customers = (
        events.loc[
            events["timestamp"] < cutoff,
            "customer_id",
        ]
        .drop_duplicates()
    )

    future_activity = events[
        (events["timestamp"] >= cutoff)
        & (events["timestamp"] < horizon_end)
        & (events["event_type"].isin(MEANINGFUL_ACTIVITY))
    ]

    active_customers = set(
        future_activity["customer_id"].unique()
    )

    labels = pd.DataFrame(
        {
            "customer_id": existing_customers,
        }
    )

    labels["cutoff"] = cutoff

    labels["churn"] = (
        ~labels["customer_id"].isin(active_customers)
    ).astype(int)

    return labels.reset_index(drop=True)