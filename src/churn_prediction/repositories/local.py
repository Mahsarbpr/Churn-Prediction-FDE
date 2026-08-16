from __future__ import annotations
import pandas as pd


class LocalEventRepository:
    def __init__(self, events: pd.DataFrame) -> None:
        self._events = events

    def get_customer_events(
        self,
        customer_id: str,
    ) -> pd.DataFrame:
        return self._events[
            self._events["customer_id"] == customer_id
        ].copy()