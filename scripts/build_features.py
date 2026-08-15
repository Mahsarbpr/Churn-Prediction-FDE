from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn_prediction.features import (
    DEFAULT_AS_OF,
    build_rfm_features,
    load_events,
)


EVENTS_PATH = ROOT / "data" / "events.json"
OUTPUT_PATH = ROOT / "artifacts" / "rfm_features.parquet"


def main() -> None:
    events = load_events(str(EVENTS_PATH))

    features = build_rfm_features(
        events,
        as_of=DEFAULT_AS_OF,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    features.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Wrote {len(features)} customer feature rows "
        f"to {OUTPUT_PATH}"
    )

    print("\nSample:")
    print(features.head())


if __name__ == "__main__":
    main()