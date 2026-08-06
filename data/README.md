# Dataset

One sample file of **raw events only** for a churn prediction service. **Pre-built features are not provided.** Feature engineering—transforming raw events into **RFM (Recency, Frequency, Monetary)**—is part of the assignment.

## File

| File | Rows | Description |
| :---- | ---: | :---- |
| [`events.json`](events.json) | 800 | JSON array of raw event objects |

## Observation time

Use **`2024-06-01T12:00:00Z`** as the as-of timestamp when computing recency and lookback windows. See [`dataset_schema.json`](dataset_schema.json).

## Event shape

```json
{
  "event_id": "evt_00000001",
  "customer_id": "cust_00047",
  "event_type": "session",
  "timestamp": "2024-03-03T16:24:29Z",
  "properties": {
    "duration_sec": 180
  }
}
```

| `event_type` | Typical `properties` |
| :---- | :---- |
| `session` | `duration_sec` |
| `purchase` | `amount_usd` |
| `push_sent` / `push_open` | `campaign_id` |
| `in_app_event` | `event_name` |
| `campaign_click` | `campaign_id` |
| `support_ticket` | `category` |

## Feature engineering (required)

Transform raw events → an RFM feature table (one row per `customer_id`), then use that as training / scoring input.

- **Recency** — days since last meaningful activity (e.g. last `session`) relative to as-of  
- **Frequency** — session (or activity) counts in lookback windows (e.g. 30d / 90d)  
- **Monetary** — purchase count and/or revenue in a lookback window (e.g. 90d)

No churn label is provided — you'll need to define one. Think carefully about how your label's time window relates to your feature time windows; defining both from the same underlying activity can make a model look far more accurate than it actually is.

## Scaling the dataset (required)

`events.json` is 800 rows across 80 customers (median ~10 events/customer). That's enough to understand the event shapes and prototype your RFM transform against, but it's too small to train on or evaluate directly — any train/test split or subgroup slice will leave you with single- or low-double-digit sample counts, which won't support a credible baseline comparison or fairness check.

Once you understand the shape of the raw sample, write a script that generates additional **synthetic** customers/events preserving its statistical structure — event-type mix, inter-event timing, session durations, purchase amounts — and merge that with (or use it in place of) the raw sample before training. There's no single required size; pick something large enough to make your evaluation and subgroup analysis meaningful (order of hundreds to a couple thousand customers is a reasonable range), and briefly justify your choice and generative assumptions.

If you want a meaningful bias/fairness check, note that the raw schema has **no profile/demographic fields** — only event-level data. You're welcome to invent synthetic segment attributes (e.g. plan tier, acquisition channel, region) as part of your generation script to give the fairness analysis something real to slice on. If you do, document that they're synthetic and state the assumptions behind them, so it's clear which parts of the analysis rest on invented vs. observed structure.
