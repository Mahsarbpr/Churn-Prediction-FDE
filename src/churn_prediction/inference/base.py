from __future__ import annotations

from typing import Protocol

import pandas as pd


class ChurnModel(Protocol):
    @property
    def model_version(self) -> str:
        ...

    @property
    def feature_names(self) -> list[str]:
        ...

    def predict_probability(
        self,
        features: pd.DataFrame,
    ) -> pd.Series:
        ...