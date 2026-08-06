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
