from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "signal_date",
    "next_open_date",
    "source_expiry",
    "flow_vega_true",
    "flow_delta_true",
    "call_iv_diff",
    "put_iv_diff",
    "total_iv_change",
    "close_spot_1520",
    "next_open_spot_0920",
    "overnight_move_points",
]


@dataclass
class FeatureBundle:
    frame: pd.DataFrame
    feature_cols: List[str]
    target_col: str
    regression_target_col: str


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = np.where(np.abs(b) < 1e-9, 0.0, a / b)
    return pd.Series(out, index=a.index)


def engineer_features(df: pd.DataFrame) -> FeatureBundle:
    validate_columns(df)
    x = df.copy()
    x["signal_date"] = pd.to_datetime(x["signal_date"], errors="coerce")
    x["next_open_date"] = pd.to_datetime(x["next_open_date"], errors="coerce")
    x = x.dropna(subset=["signal_date", "next_open_date"]).sort_values("signal_date").reset_index(drop=True)

    # Core structural derivatives.
    x["flow_vega_m"] = x["flow_vega_true"] / 1_000_000.0
    x["flow_delta_m"] = x["flow_delta_true"] / 1_000_000.0
    x["iv_spread"] = x["call_iv_diff"] - x["put_iv_diff"]
    x["iv_sum"] = x["call_iv_diff"] + x["put_iv_diff"]
    x["iv_spread_abs"] = x["iv_spread"].abs()
    x["iv_change_abs"] = x["total_iv_change"].abs()
    x["spot_ret_pct"] = _safe_div(
        x["next_open_spot_0920"] - x["close_spot_1520"],
        x["close_spot_1520"],
    ) * 100.0

    # Interactions (often helpful for directional edge).
    x["delta_x_ivspread"] = x["flow_delta_m"] * x["iv_spread"]
    x["vega_x_ivchange"] = x["flow_vega_m"] * x["total_iv_change"]
    x["delta_x_vega"] = x["flow_delta_m"] * x["flow_vega_m"]
    x["ivsum_x_spread"] = x["iv_sum"] * x["iv_spread"]

    # Regime/context.
    x["dow"] = x["signal_date"].dt.dayofweek
    x["month"] = x["signal_date"].dt.month
    x["week_of_month"] = ((x["signal_date"].dt.day - 1) // 7) + 1
    x["is_month_start"] = (x["signal_date"].dt.day <= 5).astype(int)
    x["is_month_end"] = (x["signal_date"].dt.day >= 25).astype(int)

    # Expiry code context from cached source.
    x["expiry_e1"] = x["source_expiry"].astype(str).str.contains("E1", case=False, na=False).astype(int)
    x["expiry_e2"] = x["source_expiry"].astype(str).str.contains("E2", case=False, na=False).astype(int)
    x["expiry_other"] = ((x["expiry_e1"] == 0) & (x["expiry_e2"] == 0)).astype(int)

    # Binary target: even +1 point in predicted direction counts as win.
    x["target_up"] = (x["overnight_move_points"].astype(float) >= 1.0).astype(int)
    x["target_move_points"] = x["overnight_move_points"].astype(float)

    feature_cols = [
        "flow_vega_m",
        "flow_delta_m",
        "call_iv_diff",
        "put_iv_diff",
        "total_iv_change",
        "iv_spread",
        "iv_sum",
        "iv_spread_abs",
        "iv_change_abs",
        "delta_x_ivspread",
        "vega_x_ivchange",
        "delta_x_vega",
        "ivsum_x_spread",
        "dow",
        "month",
        "week_of_month",
        "is_month_start",
        "is_month_end",
        "expiry_e1",
        "expiry_e2",
        "expiry_other",
    ]

    return FeatureBundle(
        frame=x,
        feature_cols=feature_cols,
        target_col="target_up",
        regression_target_col="target_move_points",
    )


def split_train_test_by_last_year(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Empty dataframe after preprocessing.")
    end_date = df["signal_date"].max()
    test_start = end_date - pd.Timedelta(days=365)
    train_df = df[df["signal_date"] < test_start].copy()
    test_df = df[df["signal_date"] >= test_start].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split failed. Check date span in source data.")
    return train_df, test_df

