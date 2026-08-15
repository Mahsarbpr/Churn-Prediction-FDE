from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn_prediction.features import load_events


EVENTS_PATH = ROOT / "data" / "events.json"


def main() -> None:
    events = load_events(str(EVENTS_PATH))

    print(f"Rows: {len(events)}")
    print(f"Customers: {events['customer_id'].nunique()}")

    print("\nDate range:")
    print(f"  First event: {events['timestamp'].min()}")
    print(f"  Last event:  {events['timestamp'].max()}")

    print("\nEvent type counts:")
    print(events["event_type"].value_counts())

    print("\nEvents per customer:")
    print(
        events.groupby("customer_id")
        .size()
        .describe()
    )

    sessions = events[events["event_type"] == "session"].copy()

    sessions["duration_sec"] = sessions["properties"].apply(
        lambda properties: properties.get("duration_sec")
    )

    print("\nSession duration (seconds):")
    print(sessions["duration_sec"].describe())

    purchases = events[events["event_type"] == "purchase"].copy()

    purchases["amount_usd"] = purchases["properties"].apply(
        lambda properties: properties.get("amount_usd")
    )

    print("\nPurchase amount (USD):")
    print(purchases["amount_usd"].describe())

    meaningful = events[
        events["event_type"].isin(
            [
                "session",
                "purchase",
                "push_open",
                "in_app_event",
                "campaign_click",
                "support_ticket",
            ]
        )
    ].sort_values(["customer_id", "timestamp"])

    meaningful["previous_timestamp"] = (
        meaningful.groupby("customer_id")["timestamp"].shift(1)
    )

    meaningful["gap_days"] = (
        meaningful["timestamp"] - meaningful["previous_timestamp"]
    ).dt.total_seconds() / 86400

    print("\nGap between meaningful customer activities (days):")
    print(
        meaningful["gap_days"]
        .dropna()
        .describe(
            percentiles=[0.5, 0.75, 0.9, 0.95]
        )
    )


if __name__ == "__main__":
    main()