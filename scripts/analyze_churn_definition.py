from pathlib import Path
import pandas as pd
from churn_prediction.features import load_events


MEANINGFUL_ACTIVITY = {
    "session",
    "purchase",
    "push_open",
    "in_app_event",
    "campaign_click",
    "support_ticket",
}
OBSERVATION_END = pd.Timestamp("2024-06-01T12:00:00Z")
CUTOFFS = [
    "2024-02-01T00:00:00Z",
    "2024-03-01T00:00:00Z",
    "2024-04-01T00:00:00Z",
]

HORIZONS = [30, 45, 60, 90]


def analyze_churn(
    events: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon_days: int,
) -> dict:
    horizon_end = cutoff + pd.Timedelta(days=horizon_days)

    # Only customers who existed before the prediction cutoff
    existing_customers = events.loc[
        events["timestamp"] < cutoff,
        "customer_id",
    ].unique()

    future_activity = events[
        (events["customer_id"].isin(existing_customers))
        & (events["timestamp"] >= cutoff)
        & (events["timestamp"] < horizon_end)
        & (events["event_type"].isin(MEANINGFUL_ACTIVITY))
    ]

    active_customers = future_activity["customer_id"].nunique()
    customer_count = len(existing_customers)
    churned_customers = customer_count - active_customers

    return {
        "cutoff": cutoff.date(),
        "horizon_days": horizon_days,
        "customers": customer_count,
        "churned": churned_customers,
        "active": active_customers,
        "churn_rate": (
            churned_customers / customer_count
            if customer_count
            else 0
        ),
    }


def analyze_dataset(
    name: str,
    events: pd.DataFrame,
) -> None:

    rows = []

    for cutoff_value in CUTOFFS:
        cutoff = pd.Timestamp(cutoff_value)

        for horizon_days in HORIZONS:
            horizon_end = cutoff + pd.Timedelta(days=horizon_days)

            # We cannot assign a label if the dataset does not
            # contain the complete future observation window.
            if horizon_end > OBSERVATION_END:
                continue

            rows.append(
                analyze_churn(
                    events,
                    cutoff,
                    horizon_days,
                )
            )

    results = pd.DataFrame(rows)

    print(f"\n{name}")
    print("=" * len(name))

    if results.empty:
        print("No valid cutoff/horizon combinations.")
        return

    display = results.copy()
    display["churn_rate"] = (
        display["churn_rate"] * 100
    ).round(1)

    display = display.rename(
        columns={"churn_rate": "churn_pct"}
    )

    print(display.to_string(index=False))


def main() -> None:
    real_events = load_events("data/events.json")

    as_of = pd.Timestamp("2024-06-01T12:00:00Z")

    meaningful = real_events[
        real_events["event_type"].isin(MEANINGFUL_ACTIVITY)
    ]

    last_activity = meaningful.groupby("customer_id")[
        "timestamp"
    ].max()

    recency_days = (
        as_of - last_activity
    ).dt.total_seconds() / 86400

    print("\nReal customer recency at June 1")
    print("================================")
    print(recency_days.describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95]
    ))
    
    synthetic_events = pd.read_parquet(
        Path("artifacts/synthetic_events.parquet")
    )

    analyze_dataset(
        "Real customers",
        real_events,
    )

    analyze_dataset(
        "Synthetic customers",
        synthetic_events,
    )


if __name__ == "__main__":
    main()