"""
btst_engine/backtest_evaluator.py — 3-Year Backtest Performance Report

Computes win rate and PnL across all 754 labeled dates using saved XGBoost models
applied directly to features_raw.csv (same features used during training).

Four signal categories evaluated:
  1. ML Daily (all)        — ML prediction every single day, no filter
  2. ML Ensemble HIGH      — only days where Model A and B agree
  3. Rule Engine Only      — only the 34 rule-engine BULLISH/BEARISH signals
  4. Tier A                — rule fired + ML HIGH + both directions agree

Train+val period (2023–Jun 2025) is in-sample for the ML model.
Test period (Jul 2025–Apr 2026) is strictly out-of-sample.
Both are reported; the distinction is noted in the report.

Usage:
  python -m btst_engine.backtest_evaluator
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT       = Path(__file__).parent.parent
_ML_DIR     = _ROOT / "data" / "ml"
_REPORT_DIR = _ROOT / "data" / "reports"
_IST        = timezone(timedelta(hours=5, minutes=30))

_TRAIN_END = "2024-09-30"
_VAL_END   = "2025-06-30"
_TEST_START = "2025-07-01"


# ── Step 1: Load data ─────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[1/7] Loading data files...")

    feat  = pd.read_csv(_ML_DIR / "features_raw.csv")
    lbl   = pd.read_csv(_ML_DIR / "labels.csv")
    test  = pd.read_csv(_ML_DIR / "test_predictions.csv")

    print(f"  features_raw.csv  : {len(feat):4d} rows  ({feat['date'].min()} -> {feat['date'].max()})")
    print(f"  labels.csv        : {len(lbl):4d} rows  ({lbl['date'].min()} -> {lbl['date'].max()})")
    print(f"  test_predictions  : {len(test):4d} rows  ({test['date'].min()} -> {test['date'].max()})")

    return feat, lbl, test


# ── Step 2: Apply saved models to all dates ───────────────────────────────────

def apply_models(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Load the saved XGBoost models and apply them to every row in features_raw.
    This reconstructs the ML signal for all 754 dates in a few seconds — far
    faster than calling predict_today() which rebuilds cache snapshots.
    """
    print("\n[2/7] Applying saved XGBoost models to all dates...")

    model_a_path = _ML_DIR / "model_a_winrate.json"
    model_b_path = _ML_DIR / "model_b_pnl.json"
    feat_col_path = _ML_DIR / "feature_columns.json"

    if not model_a_path.exists():
        raise FileNotFoundError(
            "Models not found. Run: python -m btst_engine.ml_predictor --train"
        )

    ma = xgb.XGBClassifier()
    ma.load_model(str(model_a_path))
    mb = xgb.XGBRegressor()
    mb.load_model(str(model_b_path))

    with feat_col_path.open() as f:
        feat_cols = json.load(f)

    X         = feat[feat_cols].values.astype(float)
    prob_ce   = ma.predict_proba(X)[:, 1]
    pred_move = mb.predict(X)

    model_a   = np.where(prob_ce >= 0.5, "CE_BUY", "PE_BUY")
    model_b   = np.where(pred_move > 0,  "CE_BUY", "PE_BUY")
    ensemble  = np.where(model_a == model_b, "HIGH", "LOW")
    ml_conf   = np.where(model_a == "CE_BUY", prob_ce, 1 - prob_ce)

    preds = pd.DataFrame({
        "date":          feat["date"].values,
        "ml_pred":       model_a,
        "ml_conf":       np.round(ml_conf, 4),
        "ensemble":      ensemble,
        "pred_move_pts": np.round(pred_move, 2),
    })

    print(f"  Applied to {len(preds)} dates")
    print(f"  CE_BUY predictions: {(model_a=='CE_BUY').sum()}  "
          f"  PE_BUY: {(model_a=='PE_BUY').sum()}")
    print(f"  Ensemble HIGH: {(ensemble=='HIGH').sum()}  "
          f"  LOW: {(ensemble=='LOW').sum()}")

    return preds


# ── Build full merged DataFrame ───────────────────────────────────────────────

def build_master(feat: pd.DataFrame, lbl: pd.DataFrame,
                 preds: pd.DataFrame) -> pd.DataFrame:
    """
    Merge features + labels + model predictions on date.
    Inner join with labels drops the 3 tail dates (no next-day open available).
    Returns 751 rows with all columns needed for the evaluation.
    """
    print("\n[3/7] Building master dataset...")

    # Decode rule engine signal from btst_direction feature
    dir_map = {1: "CE_BUY", -1: "PE_BUY", 0: "NO_TRADE"}
    feat = feat.copy()
    feat["rule_signal"] = feat["btst_direction"].map(dir_map).fillna("NO_TRADE")

    # Merge: features -> labels -> predictions
    df = (feat[["date", "rule_signal", "btst_signal_fired", "btst_confidence"]]
          .merge(lbl[["date", "label", "move_pts", "move_pct"]], on="date", how="inner")
          .merge(preds,  on="date", how="inner"))

    # Actual direction from label
    df["actual_dir"] = np.where(df["label"] == 1, "CE_BUY", "PE_BUY")

    # Decode ensemble agreement between rule and ML
    df["rule_ml_agree"] = (
        ((df["rule_signal"] == "CE_BUY") & (df["ml_pred"] == "CE_BUY")) |
        ((df["rule_signal"] == "PE_BUY") & (df["ml_pred"] == "PE_BUY"))
    )

    # Period flags
    df["period"] = "train_val"
    df.loc[df["date"] >= _TEST_START, "period"] = "test"

    print(f"  Master: {len(df)} rows  "
          f"  train+val={( df['period']=='train_val').sum()}  "
          f"  test={(df['period']=='test').sum()}")
    print(f"  Rule signals (btst_signal_fired=1): {df['btst_signal_fired'].sum()}")
    print(f"  Tier A candidates (rule + ML HIGH + agree): "
          f"{(df['btst_signal_fired'].astype(bool) & (df['ensemble']=='HIGH') & df['rule_ml_agree']).sum()}")

    return df


# ── PnL and statistics helpers ────────────────────────────────────────────────

def pnl_series(prediction_series: pd.Series, move_pts_series: pd.Series) -> pd.Series:
    """
    CE_BUY: PnL = +move_pts (profit when market up, loss when down)
    PE_BUY: PnL = -move_pts (profit when market down, loss when up)
    """
    return np.where(prediction_series == "CE_BUY",
                    move_pts_series, -move_pts_series)


def compute_stats(pnl: np.ndarray | pd.Series) -> dict:
    """Full statistics from a PnL array."""
    pnl  = np.asarray(pnl, dtype=float)
    n    = len(pnl)
    if n == 0:
        return {k: 0 for k in
                ["n","wins","win_pct","avg_win","avg_loss","total_pnl",
                 "max_dd","sharpe","wl_ratio"]}

    wins  = pnl[pnl > 0]
    loses = pnl[pnl < 0]

    # Max drawdown
    cum   = np.cumsum(pnl)
    peak  = np.maximum.accumulate(cum)
    dd    = peak - cum
    max_dd = float(dd.max())

    std  = float(np.std(pnl))
    sharpe = float(np.mean(pnl) / std) if std > 0 else 0.0

    avg_win  = float(wins.mean())  if len(wins)  > 0 else 0.0
    avg_loss = float(loses.mean()) if len(loses) > 0 else 0.0
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    return {
        "n":         n,
        "wins":      len(wins),
        "win_pct":   len(wins) / n * 100,
        "avg_win":   avg_win,
        "avg_loss":  avg_loss,
        "total_pnl": float(cum[-1]) if n > 0 else 0.0,
        "max_dd":    -max_dd,
        "sharpe":    sharpe,
        "wl_ratio":  wl_ratio,
    }


# ── Step 3-4: Evaluate all categories ────────────────────────────────────────

def evaluate_all(df: pd.DataFrame) -> dict[str, dict]:
    """Compute stats for all four signal categories."""

    # Category 1: ML signal every day
    df["pnl_ml_all"] = pnl_series(df["ml_pred"], df["move_pts"])

    # Category 2: ML ensemble HIGH only
    mask_high = df["ensemble"] == "HIGH"
    df_high   = df[mask_high]
    pnl_high  = pd.Series(pnl_series(df_high["ml_pred"], df_high["move_pts"]))

    # Category 3: Rule engine signals only
    mask_rule = df["btst_signal_fired"].astype(bool)
    df_rule   = df[mask_rule]
    pnl_rule  = pd.Series(pnl_series(df_rule["rule_signal"], df_rule["move_pts"]))

    # Category 4: Tier A — rule fired + ML HIGH + both agree
    mask_a  = mask_rule & (df["ensemble"] == "HIGH") & df["rule_ml_agree"]
    df_a    = df[mask_a]
    pnl_a   = pd.Series(pnl_series(df_a["ml_pred"], df_a["move_pts"]))

    return {
        "ML Daily (all)":    {"df": df,      "pnl": df["pnl_ml_all"].values},
        "ML Ensemble HIGH":  {"df": df_high, "pnl": pnl_high.values},
        "Rule Engine Only":  {"df": df_rule, "pnl": pnl_rule.values},
        "Tier A (best)":     {"df": df_a,    "pnl": pnl_a.values},
    }


# ── Step 4: Performance table ─────────────────────────────────────────────────

def fmt_row(name: str, s: dict) -> str:
    wr     = f"{s['win_pct']:.1f}%"
    aw     = f"+{s['avg_win']:.1f}" if s['avg_win'] != 0 else "—"
    al     = f"{s['avg_loss']:.1f}" if s['avg_loss'] != 0 else "—"
    pnl    = f"{s['total_pnl']:+.1f}"
    dd     = f"{s['max_dd']:+.1f}" if s['max_dd'] != 0 else "0.0"
    sharpe = f"{s['sharpe']:.3f}"
    return (f"  {name:<22} {s['n']:>6}  {wr:>6}   {aw:>9}   {al:>9}  "
            f"{pnl:>10}  {dd:>8}  {sharpe:>7}")


def print_performance_table(categories: dict[str, dict],
                            date_range: tuple[str, str]) -> str:
    print("\n[4/7] Performance summary")

    stats = {name: compute_stats(cat["pnl"]) for name, cat in categories.items()}

    sep = "=" * 80
    lines = [
        "",
        sep,
        "  DATASHOTS — 3-Year Backtest Performance Report",
        f"  Period: {date_range[0]} to {date_range[1]}",
        "  NOTE: Train+val (2023–Jun 2025) is in-sample for ML model.",
        "        Test period (Jul 2025–Mar 2026) is strictly out-of-sample.",
        sep,
        "",
        (f"  {'Category':<22} {'Trades':>6}  {'Win%':>6}   "
         f"{'Avg Win':>9}   {'Avg Loss':>9}  {'Total PnL':>10}  "
         f"{'Max DD':>8}  {'Sharpe':>7}"),
        "  " + "-" * 75,
    ]

    for name, cat in categories.items():
        s = stats[name]
        lines.append(fmt_row(name, s))

    lines += ["", sep, ""]

    report = "\n".join(lines)
    print(report)
    return report


# ── Step 5: Monthly breakdown ─────────────────────────────────────────────────

def print_monthly_breakdown(df: pd.DataFrame) -> str:
    print("[5/7] Monthly breakdown (ML Daily — all dates)")

    df = df.copy()
    df["ym"]  = df["date"].str[:7]
    df["pnl"] = df["pnl_ml_all"]

    monthly = (df.groupby("ym")
               .agg(trades=("pnl", "count"),
                    wins  =("pnl", lambda x: (x > 0).sum()),
                    pnl   =("pnl", "sum"))
               .reset_index())
    monthly["win_pct"] = monthly["wins"] / monthly["trades"] * 100
    monthly["cumulative"] = monthly["pnl"].cumsum()

    lines = [
        "\n  Monthly Breakdown — ML Daily (all)",
        f"  {'Month':<9}  {'Trades':>6}  {'Wins':>5}  {'Win%':>6}  "
        f"{'PnL pts':>10}  {'Cumulative':>12}",
        "  " + "-" * 58,
    ]

    for _, row in monthly.iterrows():
        arrow = "+" if row["pnl"] >= 0 else ""
        cum_arrow = "+" if row["cumulative"] >= 0 else ""
        lines.append(
            f"  {row['ym']:<9}  {int(row['trades']):>6}  {int(row['wins']):>5}  "
            f"{row['win_pct']:>5.1f}%  {arrow}{row['pnl']:>9.1f}  "
            f"{cum_arrow}{row['cumulative']:>11.1f}"
        )

    lines.append("")
    report = "\n".join(lines)
    print(report)
    return report


# ── Step 6: Distribution analysis ────────────────────────────────────────────

def print_distribution(df: pd.DataFrame) -> str:
    print("[6/7] Move distribution analysis")

    buckets = [
        ("< -200",   lambda x: x < -200),
        ("-200->-100", lambda x: (-200 <= x) & (x < -100)),
        ("-100-> -50", lambda x: (-100 <= x) & (x < -50)),
        (" -50->   0", lambda x: (-50  <= x) & (x < 0)),
        ("   0->  50", lambda x: (0    <= x) & (x < 50)),
        ("  50-> 100", lambda x: (50   <= x) & (x < 100)),
        (" 100-> 200", lambda x: (100  <= x) & (x < 200)),
        ("> 200",    lambda x: x >= 200),
    ]

    ce_mask = df["ml_pred"] == "CE_BUY"
    pe_mask = df["ml_pred"] == "PE_BUY"

    lines = [
        "\n  Move Distribution (next-day NIFTY open vs today close, in pts)",
        "  CE_BUY = model predicted market UP | PE_BUY = predicted DOWN",
        "",
        f"  {'Bucket':<12}  {'Total':>6}  {'CE_BUY':>8}  {'PE_BUY':>8}  "
        f"{'CE correct':>11}  {'PE correct':>11}",
        "  " + "-" * 68,
    ]

    for label, cond in buckets:
        mask = cond(df["move_pts"])
        total = mask.sum()
        ce_in = (mask & ce_mask).sum()
        pe_in = (mask & pe_mask).sum()

        # For CE_BUY in this bucket: correct if move_pts > 0 (CE pays when up)
        ce_correct = (mask & ce_mask & (df["move_pts"] > 0)).sum()
        pe_correct = (mask & pe_mask & (df["move_pts"] < 0)).sum()

        lines.append(
            f"  {label:<12}  {total:>6}  {ce_in:>8}  {pe_in:>8}  "
            f"{ce_correct:>11}  {pe_correct:>11}"
        )

    lines += [
        "",
        "  Interpretation:",
        "  When CE_BUY is correct -> actual move is positive (those pts are profit).",
        "  When CE_BUY is wrong   -> actual move is negative (those pts are loss).",
        "",
    ]

    report = "\n".join(lines)
    print(report)
    return report


# ── Additional: test-set breakdown ───────────────────────────────────────────

def print_test_breakdown(df: pd.DataFrame) -> str:
    """Separate stats for train+val vs test (honest out-of-sample)."""
    lines = ["\n  Period Breakdown — ML Daily"]
    lines.append(f"  {'Period':<18}  {'Dates':>6}  {'Win%':>6}  {'PnL':>10}  {'Sharpe':>7}")
    lines.append("  " + "-" * 52)

    for label, mask in [
        ("Train+val (IS)", df["period"] == "train_val"),
        ("Test (OOS)",     df["period"] == "test"),
        ("All combined",   pd.Series([True] * len(df), index=df.index)),
    ]:
        sub  = df[mask]
        pnl_ = sub["pnl_ml_all"].values
        s    = compute_stats(pnl_)
        lines.append(
            f"  {label:<18}  {s['n']:>6}  {s['win_pct']:>5.1f}%  "
            f"{s['total_pnl']:>+10.1f}  {s['sharpe']:>7.3f}"
        )

    lines.append("")
    report = "\n".join(lines)
    print(report)
    return report


# ── Step 7: Save ─────────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame, full_report: str) -> None:
    print("[7/7] Saving outputs...")

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _ML_DIR.mkdir(parents=True, exist_ok=True)

    report_path = _REPORT_DIR / "backtest_performance_3yr.txt"
    report_path.write_text(full_report, encoding="utf-8")
    print(f"  Report  -> {report_path}")

    csv_cols = [
        "date", "period", "rule_signal", "btst_signal_fired", "btst_confidence",
        "ml_pred", "ml_conf", "ensemble", "pred_move_pts",
        "rule_ml_agree", "actual_dir", "label", "move_pts", "move_pct",
        "pnl_ml_all",
    ]
    out = df[csv_cols].copy()
    out["pnl_ml_high"]  = np.where(out["ensemble"] == "HIGH", out["pnl_ml_all"], np.nan)
    out["pnl_rule"]     = np.where(
        out["btst_signal_fired"].astype(bool),
        pnl_series(out["rule_signal"], out["move_pts"]), np.nan
    )
    out["pnl_tier_a"]   = np.where(
        out["btst_signal_fired"].astype(bool) &
        (out["ensemble"] == "HIGH") & out["rule_ml_agree"],
        out["pnl_ml_all"], np.nan
    )

    csv_path = _ML_DIR / "full_predictions_754.csv"
    out.to_csv(csv_path, index=False)
    print(f"  CSV     -> {csv_path}  ({len(out)} rows)")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run() -> None:
    sep = "=" * 80
    now = datetime.now(tz=_IST).strftime("%Y-%m-%d %H:%M IST")
    print(f"\n{sep}")
    print(f"  DATASHOTS — Backtest Evaluator   {now}")
    print(sep)

    feat, lbl, test = load_data()
    preds  = apply_models(feat)
    df     = build_master(feat, lbl, preds)

    date_range = (df["date"].min(), df["date"].max())

    categories = evaluate_all(df)

    # Collect all text for the saved report
    all_text = []

    perf_txt  = print_performance_table(categories, date_range)
    all_text.append(perf_txt)

    period_txt = print_test_breakdown(df)
    all_text.append(period_txt)

    monthly_txt = print_monthly_breakdown(df)
    all_text.append(monthly_txt)

    dist_txt  = print_distribution(df)
    all_text.append(dist_txt)

    # Tier A per-trade details
    mask_a = (df["btst_signal_fired"].astype(bool) &
               (df["ensemble"] == "HIGH") & df["rule_ml_agree"])
    df_a = df[mask_a]
    if len(df_a) > 0:
        tier_lines = [
          "\n  Tier A Signal Details (rule fired + ML HIGH + agree)",
          f"  {'Date':<12}  {'Rule':>8}  {'ML':>8}  {'Move pts':>10}  {'PnL':>8}",
          "  " + "-" * 52,
        ]
        for _, row in df_a.iterrows():
            pnl_v = row["move_pts"] if row["rule_signal"] == "CE_BUY" else -row["move_pts"]
            tier_lines.append(
                f"  {row['date']:<12}  {row['rule_signal']:>8}  "
                f"{row['ml_pred']:>8}  {row['move_pts']:>+10.1f}  {pnl_v:>+8.1f}"
            )
        tier_lines.append("")
        tier_txt = "\n".join(tier_lines)
        print(tier_txt)
        all_text.append(tier_txt)

    full_report = "\n".join(all_text)
    save_outputs(df, full_report)

    # Final summary
    s_all = compute_stats(df["pnl_ml_all"].values)
    print(f"\n{sep}")
    print(f"  Summary (ML Daily, all {s_all['n']} dates)")
    print(f"  Win rate : {s_all['win_pct']:.1f}%  ({s_all['wins']}W / "
          f"{s_all['n']-s_all['wins']}L)")
    print(f"  Total PnL: {s_all['total_pnl']:+.1f} pts over 3+ years")
    print(f"  Sharpe   : {s_all['sharpe']:.3f}")
    print(f"\n  See {_REPORT_DIR}/backtest_performance_3yr.txt for full report")
    print(sep + "\n")


if __name__ == "__main__":
    run()
