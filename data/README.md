# Dataset

Customer behavioral features for a churn prediction service used in campaign audience selection, plus checks for subgroup performance gaps. The industry context is mobile marketing / app engagement (Localytics-style).

## Schema reference: `dataset_schema.json`

[`dataset_schema.json`](dataset_schema.json) is the source of truth for this dataset. It documents:

| Key | What it describes |
| :---- | :---- |
| `description` / `industry` | High-level purpose and domain |
| `label` | Target column (`churned`): `0` = active/retained, `1` = churned |
| `features` | Every input field and what it measures |
| `datasets` | Which JSON files are included and how many rows each has |
| `notes` | Hints for scoring quality (feature–churn correlation, mild subgroup base-rate differences, class imbalance) |

Use this file when building features, choosing metrics, and deciding which subgroups to check for uneven performance.

## Files

| File | Role |
| :---- | :---- |
| `customers_churn_train.json` | Primary training set (1600 rows) |
| `customers_churn_test.json` | Holdout evaluation set (400 rows) |
| `customers_churn_cold_start.json` | Newly acquired users, tenure ≤ 21 days (200 rows) |
| `customers_churn_premium.json` | Premium / high-value cohort (300 rows) |
| `customers_churn_preview.json` | 10-row preview of the training schema |

## Label

- **`churned = 0`**: active / retained  
- **`churned = 1`**: churned (no meaningful activity in the observation window)

## Feature groups (from the schema)

- **Recency / frequency:** `recency_days`, `frequency_30d`, `frequency_90d`
- **Engagement:** `avg_session_duration_sec`, `push_open_rate`, `push_opt_in`, `in_app_events_30d`, `campaign_clicks_30d`
- **Monetization / tenure:** `purchases_90d`, `revenue_90d`, `days_since_install`, `support_tickets_90d`
- **Demographic / device proxies:** `device_os`, `country_code`, `acquisition_channel`, `age_band`, `plan_tier`
- **Identifier:** `customer_id` (do not use as a predictive feature)

## Notes for scoring

- Behavioral features are correlated with churn (high recency, low frequency/engagement → higher churn).
- Mild subgroup base-rate differences exist for fairness / performance-gap exercises (e.g. `age_band` 55+, `paid_social`, android + emerging markets).
- Class imbalance is intentional; prefer PR-AUC, recall@k, or precision–recall over raw accuracy.
