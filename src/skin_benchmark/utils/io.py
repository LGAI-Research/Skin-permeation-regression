"""File I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to CSV with stable defaults."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def round_numeric_columns(df: pd.DataFrame, columns: Iterable[str], digits: int) -> pd.DataFrame:
    """Round the provided numeric columns in a dataframe copy."""

    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].round(digits)
    return out
