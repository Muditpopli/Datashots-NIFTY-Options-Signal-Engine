"""
Overnight Gap Signal -- Backtest Runner
========================================
Run from the project root:
    python run_overnight_backtest.py

Loads 3+ years of NIFTY + BANKNIFTY options data (2023-01-01 to 2026-04-07),
runs the 8-factor alignment signal, and saves:
  - data/backtest/outputs/overnight_raw.csv        -- every day, all raw features
  - data/backtest/outputs/overnight_results.csv    -- optimised config results
  - data/backtest/outputs/overnight_results.xlsx   -- formatted Excel (4 sheets)

Raw CSV is used by optimize_overnight.py without re-loading options data.

LOCKED CONFIG (v1) -- derived from grid search over 75,000 combinations on
3 years of NIFTY + BANKNIFTY options data (2023-01-01 to 2026-04-07).
Key finding: Nifty 50pt day-move filter is the strongest single gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from datashots_gap import OvernightConfig, run_backtest, summarize

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_ROOT = ROOT / "data" / "backtest" / "cache" / "rolling_options"
OUT_DIR    = ROOT / "data" / "backtest" / "outputs"
START_DATE = "2023-01-01"
END_DATE   = "2026-04-07"

# ── LOCKED CONFIG v1 ──────────────────────────────────────────────────────────
# Optimised from 75,000-combination grid search. Do not change without re-running
# optimize_overnight.py on fresh data.
#
# Logic:
#   - Nifty must have moved 50+ pts on the full day (filters out drift/noise days)
#   - Nifty L30 must show 15+ pts directional move (closing momentum)
#   - Banknifty day-move not filtered (BN L30 is what matters for confirmation)
#   - Banknifty L30 must show 100+ pts move in same direction
#   - OI skew change >= 5000 contracts to count as directional
#   - Nifty needs 3/4 checks aligned; BN needs 2/4 checks aligned
#   - 6+/8 total checks = HIGH conviction
#   - 5/8 total checks = MEDIUM conviction (trade with smaller size)
LOCKED_CFG = OvernightConfig(
    nifty_min_day_pts=50.0,
    nifty_min_l30_pts=15.0,
    bn_min_day_pts=0.0,
    bn_min_l30_pts=100.0,
    min_oi_skew_chg=5_000.0,
    nifty_min_aligned=3,
    bn_min_aligned=2,
    high_conviction_min=6,
    medium_conviction_min=5,
)

# ── Permissive config for raw feature capture (used by optimizer) ─────────────
RAW_CFG = OvernightConfig(
    nifty_min_day_pts=0.0,
    nifty_min_l30_pts=0.0,
    bn_min_day_pts=0.0,
    bn_min_l30_pts=0.0,
    min_oi_skew_chg=0.0,
    nifty_min_aligned=1,
    bn_min_aligned=1,
    high_conviction_min=8,
    medium_conviction_min=5,
)


def _metrics_row(label: str, p: pd.Series) -> dict:
    if len(p) == 0:
        return {}
    wins   = p[p > 0]
    losses = p[p <= 0]
    return {
        "Conviction":  label,
        "Trades":      len(p),
        "Wins":        int((p > 0).sum()),
        "Losses":      int((p <= 0).sum()),
        "Win Rate":    round(float((p > 0).mean()), 3),
        "Expectancy":  round(float(p.mean()), 2),
        "Avg Win":     round(float(wins.mean()),   2) if len(wins)   else 0.0,
        "Avg Loss":    round(float(losses.mean()), 2) if len(losses) else 0.0,
        "Cum Points":  round(float(p.sum()), 2),
    }


def _print_summary(label: str, metrics: dict) -> None:
    ov = metrics.get("overall", {})
    print(f"\n{'='*58}")
    print(f"  {label}")
    print(f"{'='*58}")
    print(f"  Total days    : {metrics.get('total_days', 0)}")
    print(f"  Traded days   : {metrics.get('traded_days', 0)}  ({1 - metrics.get('diverge_rate', 1):.1%} signal rate)")
    print(f"  Diverge rate  : {metrics.get('diverge_rate', 0):.1%}")
    print(f"\n  -- Overall --")
    print(f"  Trades        : {ov.get('trades', 0)}")
    print(f"  Win rate      : {ov.get('win_rate', 0):.1%}")
    print(f"  Expectancy    : {ov.get('expectancy', 0):+.2f} pts / trade")
    print(f"  Avg win       : +{ov.get('avg_win', 0):.2f} pts")
    print(f"  Avg loss      :  {ov.get('avg_loss', 0):.2f} pts")
    for conv in ("HIGH", "MEDIUM"):
        c = metrics.get(f"conviction_{conv}", {})
        if c.get("trades", 0):
            print(f"\n  -- {conv} conviction --")
            print(f"  Trades        : {c['trades']}")
            print(f"  Win rate      : {c['win_rate']:.1%}")
            print(f"  Expectancy    : {c['expectancy']:+.2f} pts / trade")
    print()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Raw pass -- no filters, captures all feature values ───────────
    print(f"Loading options data: {START_DATE} to {END_DATE} ...")
    print("Pass 1: raw (no filters) -- for optimizer ...")
    df_raw = run_backtest(CACHE_ROOT, START_DATE, END_DATE, cfg=RAW_CFG)
    raw_path = OUT_DIR / "overnight_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    print(f"  Raw saved: {raw_path}  ({len(df_raw)} trading days)")

    # ── 2. Locked config backtest ─────────────────────────────────────────
    print("\nPass 2: locked config backtest ...")
    df = run_backtest(CACHE_ROOT, START_DATE, END_DATE, cfg=LOCKED_CFG)
    df["year"] = pd.to_datetime(df["signal_date"]).dt.year

    csv_path  = OUT_DIR / "overnight_results.csv"
    xlsx_path = OUT_DIR / "overnight_results.xlsx"
    df.to_csv(csv_path, index=False)

    # ── Excel: 4 sheets ───────────────────────────────────────────────────
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:

        # Sheet 1: Full trade log
        df.to_excel(xw, sheet_name="Trade Log", index=False)

        # Sheet 2: Conviction breakdown
        conv_rows = []
        for conv in ("HIGH", "MEDIUM", "overall"):
            sub = df[df["conviction"] == conv] if conv != "overall" else df
            p = sub[sub["trade"] == True]["points"]
            r = _metrics_row(conv.title(), p)
            if r:
                conv_rows.append(r)
        pd.DataFrame(conv_rows).to_excel(xw, sheet_name="Conviction Breakdown", index=False)

        # Sheet 3: Year-by-year breakdown
        yr_rows = []
        for yr in sorted(df["year"].unique()):
            ydf = df[df["year"] == yr]
            for conv in ("HIGH", "MEDIUM", "overall"):
                sub = ydf[ydf["conviction"] == conv] if conv != "overall" else ydf
                p = sub[sub["trade"] == True]["points"]
                r = _metrics_row(conv.title(), p)
                if r:
                    r["Year"] = yr
                    yr_rows.append(r)
        yr_df = pd.DataFrame(yr_rows)
        if not yr_df.empty:
            yr_df = yr_df[["Year", "Conviction"] + [c for c in yr_df.columns if c not in ("Year", "Conviction")]]
        yr_df.to_excel(xw, sheet_name="Year by Year", index=False)

        # Sheet 4: Monthly win rate (equity curve data)
        df["month"] = pd.to_datetime(df["signal_date"]).dt.to_period("M").astype(str)
        monthly = (
            df[df["trade"] == True]
            .groupby("month")
            .apply(lambda g: pd.Series({
                "trades":      len(g),
                "wins":        int((g["points"] > 0).sum()),
                "win_rate":    round(float((g["points"] > 0).mean()), 3),
                "expectancy":  round(float(g["points"].mean()), 2),
                "cum_points":  round(float(g["points"].sum()), 2),
            }))
            .reset_index()
        )
        monthly["running_total"] = monthly["cum_points"].cumsum().round(2)
        monthly.to_excel(xw, sheet_name="Monthly Equity Curve", index=False)

    print(f"  Results CSV  : {csv_path}")
    print(f"  Results Excel: {xlsx_path}")

    metrics = summarize(df)
    _print_summary("LOCKED CONFIG v1 -- FINAL RESULTS", metrics)

    # ── Year-by-year console summary ──────────────────────────────────────
    print("YEAR-BY-YEAR BREAKDOWN")
    print("=" * 58)
    for yr in sorted(df["year"].unique()):
        ydf = df[df["year"] == yr]
        for conv in ("HIGH", "MEDIUM"):
            p = ydf[(ydf["conviction"] == conv) & ydf["trade"]]["points"]
            if len(p):
                print(f"  {yr}  {conv:<7}: {len(p):3d} trades | "
                      f"{(p>0).mean():.1%} WR | exp {p.mean():+.1f} | cum {p.sum():+.0f} pts")
        tot = ydf[ydf["trade"]]["points"]
        if len(tot):
            div = (~ydf["trade"]).sum()
            print(f"  {yr}  TOTAL  : {len(tot):3d} trades | "
                  f"{(tot>0).mean():.1%} WR | exp {tot.mean():+.1f} | "
                  f"cum {tot.sum():+.0f} pts  ({div} diverge days)")
        print()


if __name__ == "__main__":
    main()
