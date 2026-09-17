from __future__ import annotations

from pathlib import Path
from typing import List, Union

import pandas as pd

TARGET_COLUMN = "target"

QUADRATIC_SOURCE_COLUMN = "6"
ENGINEERED_COLUMN = "6_squared"


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(path)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[ENGINEERED_COLUMN] = df[QUADRATIC_SOURCE_COLUMN] ** 2
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c != TARGET_COLUMN]
