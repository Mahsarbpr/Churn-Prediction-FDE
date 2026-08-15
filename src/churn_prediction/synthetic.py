import numpy as np
import pandas as pd


def generate_synthetic_events(
    events: pd.DataFrame,
    customer_count: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    events = events.copy().sort_values(["customer_id", "timestamp"])

    event_counts = (
        events.groupby("customer_id")
        .size()
        .to_numpy()
    )

    event_type_probabilities = (
        events["event_type"]
        .value_counts(normalize=True)
        .sort_index()
    )

    event_types = event_type_probabilities.index.to_numpy()
    event_probabilities = event_type_probabilities.to_numpy()

    gaps = (
        events.groupby("customer_id")["timestamp"]
        .diff()
        .dropna()
        .dt.total_seconds()
        .div(86400)
        .to_numpy()
    )

    session_durations = (
        events.loc[events["event_type"] == "session", "properties"]
        .apply(lambda properties: properties["duration_sec"])
        .to_numpy()
    )

    purchase_amounts = (
        events.loc[events["event_type"] == "purchase", "properties"]
        .apply(lambda properties: properties["amount_usd"])
        .to_numpy()
    )

    min_timestamp = events["timestamp"].min()
    max_timestamp = events["timestamp"].max()

    available_span_days = (
        max_timestamp - min_timestamp
    ).total_seconds() / 86400

    rows: list[dict] = []

    for customer_number in range(1, customer_count + 1):
        customer_id = f"syn_cust_{customer_number:05d}"

        event_count = int(rng.choice(event_counts))

        customer_event_types = rng.choice(
            event_types,
            size=event_count,
            p=event_probabilities,
        )

        gap_count = max(event_count - 1, 0)

        customer_gaps = rng.choice(
            gaps,
            size=gap_count,
        ).astype(float)

        total_span_days = float(customer_gaps.sum())

        # Resample unusually long histories until the full customer
        # timeline fits within the source dataset's date range.
        while total_span_days > available_span_days:
            customer_gaps = rng.choice(
                gaps,
                size=gap_count,
            ).astype(float)
            total_span_days = float(customer_gaps.sum())

        latest_start = max_timestamp - pd.Timedelta(
            days=total_span_days
        )

        start_range_seconds = (
            latest_start - min_timestamp
        ).total_seconds()

        if start_range_seconds > 0:
            start_timestamp = min_timestamp + pd.Timedelta(
                seconds=float(
                    rng.uniform(0, start_range_seconds)
                )
            )
        else:
            start_timestamp = min_timestamp

        timestamp = start_timestamp

        for event_index, event_type in enumerate(
            customer_event_types
        ):
            properties: dict = {}

            if event_type == "session":
                properties["duration_sec"] = max(
                    1,
                    int(rng.choice(session_durations)),
                )

            elif event_type == "purchase":
                properties["amount_usd"] = round(
                    float(rng.choice(purchase_amounts)),
                    2,
                )

            elif event_type in {
                "push_sent",
                "push_open",
                "campaign_click",
            }:
                properties["campaign_id"] = (
                    f"camp_{rng.integers(1, 21):02d}"
                )

            elif event_type == "in_app_event":
                properties["event_name"] = (
                    f"event_{rng.integers(1, 11):02d}"
                )

            elif event_type == "support_ticket":
                properties["category"] = rng.choice(
                    ["billing", "bug", "account", "other"]
                )

            rows.append(
                {
                    "event_id": f"syn_evt_{len(rows) + 1:08d}",
                    "customer_id": customer_id,
                    "event_type": event_type,
                    "timestamp": timestamp,
                    "properties": properties,
                }
            )

            if event_index < len(customer_gaps):
                timestamp += pd.Timedelta(
                    days=float(customer_gaps[event_index])
                )

    return (
        pd.DataFrame(rows)
        .sort_values(["customer_id", "timestamp"])
        .reset_index(drop=True)
    )