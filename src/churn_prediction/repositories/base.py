from __future__ import annotations
from typing import Protocol
import pandas as pd


class EventRepository(Protocol):
    def get_customer_events(
        self,
        customer_id: str,
    ) -> pd.DataFrame:
        ...