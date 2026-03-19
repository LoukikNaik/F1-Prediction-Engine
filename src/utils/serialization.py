"""Shared JSON serialization helpers for API and static export."""

import numpy as np
import pandas as pd


def to_json_safe(val):
    """Convert a single value to a JSON-serialisable Python type."""
    if pd.isna(val):
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return round(float(val), 6)
    if isinstance(val, float):
        return round(val, 6)
    if isinstance(val, int):
        return val
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)


def matrix_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a prediction matrix DataFrame to JSON-friendly records."""
    records = []
    pos_cols = [f"P{i}" for i in range(1, 23) if f"P{i}" in df.columns]
    summary_cols = [
        c
        for c in ("expected_position", "win_prob", "podium_prob", "top5_prob")
        if c in df.columns
    ]
    for driver_name in df.index:
        row = df.loc[driver_name]
        rec: dict = {"driver_name": driver_name}
        for col in pos_cols + summary_cols:
            rec[col] = to_json_safe(row[col])
        records.append(rec)
    return records
