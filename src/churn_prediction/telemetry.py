from __future__ import annotations

import os

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource


def configure_metrics() -> None:
    endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        "http://localhost:4318/v1/metrics",
    )

    exporter = OTLPMetricExporter(
        endpoint=endpoint,
    )

    reader = PeriodicExportingMetricReader(
        exporter,
        export_interval_millis=10_000,
    )

    provider = MeterProvider(
        resource=Resource.create(
            {
                "service.name": "churn-prediction-service",
                "deployment.environment.name": "take-home",
            }
        ),
        metric_readers=[reader],
    )

    metrics.set_meter_provider(provider)


meter = metrics.get_meter("churn-prediction-service")

request_latency = meter.create_histogram(
    "churn_prediction.request_latency_ms",
    unit="ms",
)

prediction_errors = meter.create_counter(
    "churn_prediction.errors",
)

score_distribution = meter.create_histogram(
    "churn_prediction.score",
)