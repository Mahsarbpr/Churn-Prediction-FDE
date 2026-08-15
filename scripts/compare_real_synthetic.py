from __future__ import annotations
from pathlib import Path
import pandas as pd
from churn_prediction.features import load_events


def summarize(events: pd.DataFrame) -> dict:
    events = events.copy().sort_values(["customer_id", "timestamp"])

    events_per_customer = events.groupby("customer_id").size()

    gaps = (
        events.groupby("customer_id")["timestamp"]
        .diff()
        .dropna()
        .dt.total_seconds()
        .div(86400)
    )

    session_durations = (
        events.loc[events["event_type"] == "session", "properties"]
        .apply(lambda properties: properties["duration_sec"])
    )

    purchase_amounts = (
        events.loc[events["event_type"] == "purchase", "properties"]
        .apply(lambda properties: properties["amount_usd"])
    )

    return {
        "customers": events["customer_id"].nunique(),
        "events": len(events),
        "events_per_customer_mean": events_per_customer.mean(),
        "gap_median_days": gaps.median(),
        "gap_p90_days": gaps.quantile(0.90),
        "session_duration_mean": session_durations.mean(),
        "purchase_amount_median": purchase_amounts.median(),
        "event_mix": events["event_type"].value_counts(normalize=True),
        "purchase_amount_mean": purchase_amounts.mean(),
        "purchase_amount_p25": purchase_amounts.quantile(0.25),
        "purchase_amount_p75": purchase_amounts.quantile(0.75),
    }


def main() -> None:
    real = load_events("data/events.json")
    synthetic = pd.read_parquet(
        Path("artifacts/synthetic_events.parquet")
    )

    real_summary = summarize(real)
    synthetic_summary = summarize(synthetic)

    print("Real vs synthetic")
    print("-----------------")
    print(
        f"Customers:              "
        f"{real_summary['customers']} vs "
        f"{synthetic_summary['customers']}"
    )
    print(
        f"Events:                 "
        f"{real_summary['events']} vs "
        f"{synthetic_summary['events']}"
    )
    print(
        f"Events/customer mean:   "
        f"{real_summary['events_per_customer_mean']:.2f} vs "
        f"{synthetic_summary['events_per_customer_mean']:.2f}"
    )
    print(
        f"Gap median (days):      "
        f"{real_summary['gap_median_days']:.2f} vs "
        f"{synthetic_summary['gap_median_days']:.2f}"
    )
    print(
        f"Gap p90 (days):         "
        f"{real_summary['gap_p90_days']:.2f} vs "
        f"{synthetic_summary['gap_p90_days']:.2f}"
    )
    print(
        f"Session duration mean:  "
        f"{real_summary['session_duration_mean']:.2f} vs "
        f"{synthetic_summary['session_duration_mean']:.2f}"
    )
    print(
        f"Purchase median:        "
        f"{real_summary['purchase_amount_median']:.2f} vs "
        f"{synthetic_summary['purchase_amount_median']:.2f}"
    )
    print(
        f"Purchase mean:          "
        f"{real_summary['purchase_amount_mean']:.2f} vs "
        f"{synthetic_summary['purchase_amount_mean']:.2f}"
    )
    print(
        f"Purchase p25:           "
        f"{real_summary['purchase_amount_p25']:.2f} vs "
        f"{synthetic_summary['purchase_amount_p25']:.2f}"
    )
    print(
        f"Purchase p75:           "
        f"{real_summary['purchase_amount_p75']:.2f} vs "
        f"{synthetic_summary['purchase_amount_p75']:.2f}"
    )
    event_mix = pd.concat(
        [
            real_summary["event_mix"].rename("real"),
            synthetic_summary["event_mix"].rename("synthetic"),
        ],
        axis=1,
    ).fillna(0)

    print("\nEvent type mix:")
    print(event_mix)


if __name__ == "__main__":
    main()