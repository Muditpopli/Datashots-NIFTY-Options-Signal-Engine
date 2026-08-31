from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ThresholdResult:
    min_confidence: float
    trades: int
    win_rate: float
    expectancy: float
    annual_points: float
    avg_win: float
    avg_loss: float


def _metrics_for_threshold(df: pd.DataFrame, min_conf: float) -> ThresholdResult:
    sub = df[df["confidence"] >= min_conf].copy()
    if sub.empty:
        return ThresholdResult(min_conf, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    signed = sub["signed_points"].to_numpy(dtype=float)
    wins = signed[signed >= 1.0]
    losses = signed[signed <= -1.0]
    wr = float(len(wins) / (len(wins) + len(losses))) if (len(wins) + len(losses)) else 0.0
    exp = float(np.mean(signed)) if len(signed) else 0.0
    return ThresholdResult(
        min_confidence=min_conf,
        trades=int(len(sub)),
        win_rate=wr,
        expectancy=exp,
        annual_points=float(np.sum(signed)),
        avg_win=float(np.mean(wins)) if len(wins) else 0.0,
        avg_loss=float(np.mean(losses)) if len(losses) else 0.0,
    )


def evaluate_threshold_grid(
    df: pd.DataFrame,
    thresholds: Iterable[float] = (0, 55, 60, 65, 70, 75, 80),
) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        r = _metrics_for_threshold(df, float(t))
        rows.append(
            {
                "Min_Confidence": float(t),
                "Trades": r.trades,
                "Win_Rate": r.win_rate,
                "Expectancy": r.expectancy,
                "Annual_Points": r.annual_points,
                "Avg_Win": r.avg_win,
                "Avg_Loss": r.avg_loss,
            }
        )
    return pd.DataFrame(rows)


def optimize_global_threshold(
    tune_df: pd.DataFrame,
    min_trades: int = 40,
    target_win_rate: float = 0.55,
    thresholds: Iterable[float] = tuple(range(50, 86, 1)),
) -> ThresholdResult:
    best: Optional[ThresholdResult] = None
    for t in thresholds:
        r = _metrics_for_threshold(tune_df, float(t))
        if r.trades < min_trades:
            continue
        # Primary objective: expectancy; secondary: win-rate.
        if best is None:
            best = r
            continue
        k = (r.expectancy, r.win_rate, -abs(r.min_confidence - 70.0))
        kb = (best.expectancy, best.win_rate, -abs(best.min_confidence - 70.0))
        if k > kb:
            best = r
    if best is None:
        # fallback to all-trade policy
        return _metrics_for_threshold(tune_df, 0.0)
    # Soft guard: if tuned threshold still poor, keep daily trading.
    if best.expectancy <= 0 and best.win_rate < target_win_rate:
        return _metrics_for_threshold(tune_df, 0.0)
    return best


def optimize_regime_thresholds(
    tune_df: pd.DataFrame,
    regimes: Iterable[str],
    min_trades_per_regime: int = 12,
    thresholds: Iterable[float] = tuple(range(50, 86, 1)),
) -> Dict[str, ThresholdResult]:
    out: Dict[str, ThresholdResult] = {}
    for rg in regimes:
        sub = tune_df[tune_df["regime"] == rg].copy()
        if len(sub) < min_trades_per_regime:
            out[rg] = _metrics_for_threshold(sub, 0.0)
            continue
        best: Optional[ThresholdResult] = None
        for t in thresholds:
            r = _metrics_for_threshold(sub, float(t))
            if r.trades < max(6, int(len(sub) * 0.15)):
                continue
            if best is None:
                best = r
                continue
            k = (r.expectancy, r.win_rate, -abs(r.min_confidence - 70.0))
            kb = (best.expectancy, best.win_rate, -abs(best.min_confidence - 70.0))
            if k > kb:
                best = r
        out[rg] = best if best is not None else _metrics_for_threshold(sub, 0.0)
    return out

