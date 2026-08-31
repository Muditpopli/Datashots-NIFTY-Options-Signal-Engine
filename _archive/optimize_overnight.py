"""
Overnight Gap Signal — Parameter Optimizer
===========================================
Run AFTER run_overnight_backtest.py has produced overnight_raw.csv.

    python optimize_overnight.py

Strategy
--------
The raw CSV contains all per-day feature values (day_move, l30_move,
day_oi_skew, l30_oi_skew for both indices) alongside the actual next-day gap.

Instead of re-loading 3 years of options data for every config combination,
we recompute directional checks from the raw numerical columns and evaluate
every (threshold, alignment, conviction) combination analytically.

Grid searched:
  - nifty_min_day_pts  : minimum Nifty full-day move to be directional
  - nifty_min_l30_pts  : minimum Nifty L30 move
  - bn_min_day_pts     : minimum Banknifty full-day move
  - bn_min_l30_pts     : minimum Banknifty L30 move
  - min_oi_skew_chg    : minimum OI skew change (contracts)
  - nifty_min_aligned  : Nifty checks needed (of 4)
  - bn_min_aligned     : Banknifty checks needed (of 4)
  - high_conv_min      : total checks needed for HIGH conviction (of 8)
  - medium_conv_min    : total checks needed for MEDIUM conviction (of 8)

Objective
---------
  Primary : maximise win rate on HIGH conviction trades
  Secondary: maximise expectancy (avg pts/trade), min 20 HIGH trades
  Also reports: MEDIUM conviction metrics and combined metrics

Outputs
-------
  data/backtest/outputs/optimization_results.csv   ← all configs ranked
  data/backtest/outputs/optimization_best.csv      ← top 20 configs
  Console: top 10 configs + recommended config
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

RAW_CSV  = ROOT / "data" / "backtest" / "outputs" / "overnight_raw.csv"
OUT_DIR  = ROOT / "data" / "backtest" / "outputs"

# ── Grid definition ───────────────────────────────────────────────────────────
GRID = {
    "nifty_min_day_pts":  [0, 20, 30, 50, 75],
    "nifty_min_l30_pts":  [0, 10, 15, 20, 30],
    "bn_min_day_pts":     [0, 75, 100, 150, 200],
    "bn_min_l30_pts":     [0, 30, 50, 75, 100],
    "min_oi_skew_chg":    [0, 1_000, 5_000, 10_000, 25_000],
    "nifty_min_aligned":  [2, 3],
    "bn_min_aligned":     [2, 3],
    "high_conv_min":      [6, 7, 8],
    "medium_conv_min":    [4, 5],
}

MIN_TRADES_HIGH   = 20   # Minimum HIGH conviction trades to count a config as valid
MIN_TRADES_MEDIUM = 15   # Minimum MEDIUM conviction trades


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dir_vec(series: pd.Series, threshold: float) -> np.ndarray:
    arr = series.to_numpy(dtype=float)
    out = np.zeros(len(arr), dtype=int)
    out[arr >  threshold] =  1
    out[arr < -threshold] = -1
    return out


def _metrics(pts: np.ndarray) -> Dict:
    if len(pts) == 0:
        return {"trades": 0, "win_rate": 0.0, "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
    wins   = pts[pts > 0]
    losses = pts[pts <= 0]
    return {
        "trades":     int(len(pts)),
        "win_rate":   round(float((pts > 0).mean()), 4),
        "expectancy": round(float(pts.mean()), 3),
        "avg_win":    round(float(wins.mean()),   2) if len(wins)   else 0.0,
        "avg_loss":   round(float(losses.mean()), 2) if len(losses) else 0.0,
    }


def evaluate(df: pd.DataFrame, p: Dict) -> Optional[Dict]:
    """Apply one config to the raw dataframe and return metric dict."""

    # ── Recompute 8 directional checks ───────────────────────────────────
    n_sd = _dir_vec(df["n_day_move"],     p["nifty_min_day_pts"])
    n_sl = _dir_vec(df["n_l30_move"],     p["nifty_min_l30_pts"])
    n_od = _dir_vec(df["n_day_oi_skew"],  p["min_oi_skew_chg"])
    n_ol = _dir_vec(df["n_l30_oi_skew"],  p["min_oi_skew_chg"])

    b_sd = _dir_vec(df["b_day_move"],     p["bn_min_day_pts"])
    b_sl = _dir_vec(df["b_l30_move"],     p["bn_min_l30_pts"])
    b_od = _dir_vec(df["b_day_oi_skew"],  p["min_oi_skew_chg"])
    b_ol = _dir_vec(df["b_l30_oi_skew"],  p["min_oi_skew_chg"])

    n_checks = np.stack([n_sd, n_sl, n_od, n_ol], axis=1)   # (N, 4)
    b_checks = np.stack([b_sd, b_sl, b_od, b_ol], axis=1)   # (N, 4)

    # ── Nifty majority direction ─────────────────────────────────────────
    n_pos = (n_checks == 1).sum(axis=1)
    n_neg = (n_checks == -1).sum(axis=1)
    n_dir = np.where(n_pos > n_neg, 1, np.where(n_neg > n_pos, -1, 0))
    n_aligned = np.where(n_dir == 1, n_pos, np.where(n_dir == -1, n_neg, 0))

    # ── Banknifty majority direction ─────────────────────────────────────
    b_pos = (b_checks == 1).sum(axis=1)
    b_neg = (b_checks == -1).sum(axis=1)
    b_dir = np.where(b_pos > b_neg, 1, np.where(b_neg > b_pos, -1, 0))
    b_aligned = np.where(b_dir == 1, b_pos, np.where(b_dir == -1, b_neg, 0))

    # ── Apply alignment gates ─────────────────────────────────────────────
    nifty_ok = (n_dir != 0) & (n_aligned >= p["nifty_min_aligned"])
    bn_ok    = (b_dir == n_dir) & (b_aligned >= p["bn_min_aligned"])
    both_ok  = nifty_ok & bn_ok

    # ── Total aligned (out of 8) ──────────────────────────────────────────
    all_checks = np.stack([n_sd, n_sl, n_od, n_ol, b_sd, b_sl, b_od, b_ol], axis=1)
    total_aligned = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        if both_ok[i]:
            total_aligned[i] = int((all_checks[i] == n_dir[i]).sum())

    # ── Conviction levels ─────────────────────────────────────────────────
    is_high   = both_ok & (total_aligned >= p["high_conv_min"])
    is_medium = both_ok & ~is_high & (total_aligned >= p["medium_conv_min"])

    # Reversal warning: L30 spot direction opposes day spot direction
    # When active, cap at MEDIUM (don't allow HIGH)
    reversal = (n_sd != 0) & (n_sl != 0) & (n_sd != n_sl)
    is_high   = is_high & ~reversal          # reversal knocks HIGH → MEDIUM
    is_medium = is_medium | (both_ok & (total_aligned >= p["high_conv_min"]) & reversal)

    # ── Signal direction → points ─────────────────────────────────────────
    actual_gap = df["actual_gap"].to_numpy(dtype=float)
    predicted  = np.where(both_ok, n_dir, 0)   # +1 BULL, -1 BEAR, 0 SKIP

    def _pts(pred: int) -> np.ndarray:
        mask = predicted == pred
        gaps = actual_gap[mask]
        if pred == 1:
            return np.where(gaps >= 1.0, gaps, -np.abs(gaps))
        if pred == -1:
            return np.where(gaps <= -1.0, np.abs(gaps), -np.abs(gaps))
        return np.array([])

    # ── Evaluate by conviction ────────────────────────────────────────────
    high_mask   = is_high   & (predicted != 0)
    medium_mask = is_medium & (predicted != 0)

    high_pts   = np.concatenate([_pts(1)[is_high[predicted==1]], _pts(-1)[is_high[predicted==-1]]]) \
        if False else _pts_conv(predicted, actual_gap, is_high)
    medium_pts = _pts_conv(predicted, actual_gap, is_medium)
    overall_pts = _pts_conv(predicted, actual_gap, is_high | is_medium)

    h = _metrics(high_pts)
    m = _metrics(medium_pts)
    o = _metrics(overall_pts)

    if h["trades"] < MIN_TRADES_HIGH and m["trades"] < MIN_TRADES_MEDIUM:
        return None   # not enough trades to be statistically meaningful

    return {
        # ── Config params ───────────────────────────────────────────────
        **{k: v for k, v in p.items()},
        # ── HIGH conviction metrics ─────────────────────────────────────
        "h_trades":     h["trades"],
        "h_win_rate":   h["win_rate"],
        "h_expectancy": h["expectancy"],
        "h_avg_win":    h["avg_win"],
        "h_avg_loss":   h["avg_loss"],
        # ── MEDIUM conviction metrics ───────────────────────────────────
        "m_trades":     m["trades"],
        "m_win_rate":   m["win_rate"],
        "m_expectancy": m["expectancy"],
        # ── Combined metrics ────────────────────────────────────────────
        "total_trades":    o["trades"],
        "total_win_rate":  o["win_rate"],
        "total_expectancy": o["expectancy"],
        # ── Trade rate ──────────────────────────────────────────────────
        "trade_rate": round(float(o["trades"]) / len(df), 3),
        "diverge_rate": round(1.0 - float(o["trades"]) / len(df), 3),
    }


def _pts_conv(predicted: np.ndarray, actual_gap: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute points for all trades matching mask, preserving sign."""
    indices = np.where(mask & (predicted != 0))[0]
    pts = []
    for i in indices:
        p = predicted[i]
        g = actual_gap[i]
        if p == 1:
            pts.append(g if g >= 1.0 else -abs(g))
        elif p == -1:
            pts.append(abs(g) if g <= -1.0 else -abs(g))
    return np.array(pts, dtype=float)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not RAW_CSV.exists():
        print(f"ERROR: {RAW_CSV} not found.")
        print("Run   python run_overnight_backtest.py   first.")
        sys.exit(1)

    print(f"Loading raw backtest data from {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV)

    # Keep only rows where we have all 8 raw feature columns and actual_gap
    required = [
        "n_day_move", "n_l30_move", "n_day_oi_skew", "n_l30_oi_skew",
        "b_day_move", "b_l30_move", "b_day_oi_skew", "b_l30_oi_skew",
        "actual_gap",
    ]
    df = df.dropna(subset=required).reset_index(drop=True)
    print(f"  {len(df)} usable trading days\n")

    # ── Build grid ───────────────────────────────────────────────────────
    keys   = list(GRID.keys())
    values = list(GRID.values())
    combos = list(product(*values))
    total  = len(combos)
    print(f"Grid size: {total:,} combinations -- evaluating ...")

    results: List[Dict] = []
    for i, combo in enumerate(combos):
        if i % 5000 == 0:
            pct = i / total * 100
            print(f"  {i:,}/{total:,}  ({pct:.0f}%)", end="\r", flush=True)

        p = dict(zip(keys, combo))
        # Skip nonsensical combos (medium_conv must be < high_conv)
        if p["medium_conv_min"] >= p["high_conv_min"]:
            continue

        row = evaluate(df, p)
        if row is not None:
            results.append(row)

    print(f"\nValid configs: {len(results):,}")

    if not results:
        print("No valid configs found — try relaxing MIN_TRADES thresholds.")
        return

    opt_df = pd.DataFrame(results)

    # ── Rank by HIGH win rate (primary) then expectancy (secondary) ───────
    opt_df = opt_df.sort_values(
        ["h_win_rate", "h_expectancy", "total_expectancy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # ── Save ─────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_path  = OUT_DIR / "optimization_results.csv"
    best_path = OUT_DIR / "optimization_best.csv"

    opt_df.to_csv(all_path, index=False)
    opt_df.head(20).to_csv(best_path, index=False)
    print(f"All results  : {all_path}")
    print(f"Top 20       : {best_path}")

    # ── Print top 10 ─────────────────────────────────────────────────────
    cols = [
        "nifty_min_day_pts", "nifty_min_l30_pts", "bn_min_day_pts", "bn_min_l30_pts",
        "min_oi_skew_chg", "nifty_min_aligned", "bn_min_aligned",
        "high_conv_min", "medium_conv_min",
        "h_trades", "h_win_rate", "h_expectancy",
        "m_trades", "m_win_rate",
        "total_trades", "total_win_rate", "total_expectancy",
        "trade_rate",
    ]
    print("\n── TOP 10 CONFIGS (ranked by HIGH conviction win rate) ─────────────")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(opt_df[cols].head(10).to_string(index=True))

    # ── Recommended config ────────────────────────────────────────────────
    best = opt_df.iloc[0]
    print("\n── RECOMMENDED CONFIG ──────────────────────────────────────────────")
    print(f"  nifty_min_day_pts  = {best['nifty_min_day_pts']}")
    print(f"  nifty_min_l30_pts  = {best['nifty_min_l30_pts']}")
    print(f"  bn_min_day_pts     = {best['bn_min_day_pts']}")
    print(f"  bn_min_l30_pts     = {best['bn_min_l30_pts']}")
    print(f"  min_oi_skew_chg    = {best['min_oi_skew_chg']:.0f}")
    print(f"  nifty_min_aligned  = {int(best['nifty_min_aligned'])}")
    print(f"  bn_min_aligned     = {int(best['bn_min_aligned'])}")
    print(f"  high_conv_min      = {int(best['high_conv_min'])}")
    print(f"  medium_conv_min    = {int(best['medium_conv_min'])}")
    print(f"\n  HIGH conviction : {int(best['h_trades'])} trades | "
          f"{best['h_win_rate']:.1%} win rate | {best['h_expectancy']:+.2f} pts/trade")
    print(f"  MEDIUM conviction: {int(best['m_trades'])} trades | "
          f"{best['m_win_rate']:.1%} win rate")
    print(f"  Combined         : {int(best['total_trades'])} trades | "
          f"{best['total_win_rate']:.1%} win rate | {best['total_expectancy']:+.2f} pts/trade")
    print(f"  Trade rate       : {best['trade_rate']:.1%} of days\n")

    print("Update OvernightConfig in run_overnight_backtest.py with the above values,")
    print("then re-run to get the optimised backtest report.\n")


if __name__ == "__main__":
    main()
