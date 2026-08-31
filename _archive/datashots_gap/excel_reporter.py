from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def export_to_excel(results: pd.DataFrame, metrics: Dict[str, Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trade_log = results.copy()
    if not trade_log.empty:
        trade_log = trade_log[
            [
                "signal_date",
                "next_day",
                "index",
                "strength",
                "sentiment",
                "confidence",
                "signal_strength",
                "cmp",
                "next_open",
                "actual_gap",
                "result",
                "points",
                "cumulative_points",
                "conviction",
                "in_sync",
                "action",
                "vega_confirm",
                "theta_confirm",
                "oi_confirm",
                "pos",
            ]
        ].rename(
            columns={
                "signal_date": "Date",
                "index": "Index",
                "sentiment": "Predicted",
                "confidence": "Confidence",
                "actual_gap": "Actual_Gap",
                "result": "Result",
                "points": "Points",
                "cumulative_points": "Cumulative",
            }
        )

    summary_rows = []
    for k, v in metrics.items():
        summary_rows.append(
            {
                "Bucket": k,
                "Trades": v.get("trades", 0),
                "Wins": v.get("wins", 0),
                "Losses": v.get("losses", 0),
                "Win_Rate": v.get("win_rate", 0.0),
                "Avg_Win": v.get("avg_win", 0.0),
                "Avg_Loss": v.get("avg_loss", 0.0),
                "Expectancy": v.get("expectancy", 0.0),
            }
        )
    summary = pd.DataFrame(summary_rows)

    monthly = pd.DataFrame()
    if not trade_log.empty:
        monthly = trade_log.copy()
        monthly["Month"] = pd.to_datetime(monthly["Date"]).dt.to_period("M").astype(str)
        monthly = (
            monthly.groupby(["Month", "Index"], as_index=False)
            .agg(
                Trades=("Result", "size"),
                Wins=("Result", lambda s: int((s == "WIN").sum())),
                Total_Points=("Points", "sum"),
                Avg_Points=("Points", "mean"),
            )
        )
        monthly["Win_Rate"] = monthly["Wins"] / monthly["Trades"]

    strength_dist = pd.DataFrame()
    if not trade_log.empty:
        bins = [-10_000, -10, 10, 10_000]
        labels = ["RED(<-10)", "YELLOW(-10 to 10)", "GREEN(>10)"]
        strength_dist = trade_log.copy()
        strength_dist["Strength_Band"] = pd.cut(strength_dist["strength"], bins=bins, labels=labels).astype(str)
        strength_dist = (
            strength_dist.groupby(["Index", "Strength_Band"], as_index=False, observed=True)
            .agg(
                Trades=("Result", "size"),
                Win_Rate=("Result", lambda s: float((s == "WIN").mean())),
                Avg_Points=("Points", "mean"),
            )
        )

    vega_conf = pd.DataFrame()
    if not trade_log.empty:
        vega_conf = (
            trade_log.groupby(["Index", "vega_confirm"], as_index=False)
            .agg(
                Trades=("Result", "size"),
                Win_Rate=("Result", lambda s: float((s == "WIN").mean())),
                Avg_Points=("Points", "mean"),
            )
        )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        trade_log.to_excel(writer, sheet_name="Trade Log", index=False)
        summary.to_excel(writer, sheet_name="Summary Metrics", index=False)
        monthly.to_excel(writer, sheet_name="Monthly Breakdown", index=False)
        strength_dist.to_excel(writer, sheet_name="Strength Distribution", index=False)
        vega_conf.to_excel(writer, sheet_name="Vega Confirmation", index=False)
