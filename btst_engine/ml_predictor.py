"""
btst_engine/ml_predictor.py — BTST ML Direction Predictor

Trains a daily CE/PE direction classifier on 3 years of NIFTY/BANKNIFTY
options data.  Outputs CE_BUY or PE_BUY for every trading day (no NO_TRADE).

Pipeline
--------
  Step 1  Build feature matrix from rolling options cache
  Step 2  Build labels: next-day NIFTY open vs today W2 close
  Step 3  Train Model A (win-rate maximiser) + Model B (PnL maximiser)
  Step 4  Test-set evaluation with per-trade PnL simulation
  Step 5  Ensemble: both-agree = HIGH confidence
  Step 6  Save models, metrics, predictions
  Step 7  predict_today() live inference function

Walk-forward split
  Train : 2023-01-01 to 2024-09-30
  Val   : 2024-10-01 to 2025-06-30
  Test  : 2025-07-01 to 2026-04-08

Usage
  python -m btst_engine.ml_predictor --train
  python -m btst_engine.ml_predictor --predict 2026-04-07
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT       = Path(__file__).parent.parent
_ML_DIR     = _ROOT / "data" / "ml"
_BTST_CSV   = _ROOT / "data" / "tracker" / "btst_signals.csv"
_FEAT_CSV   = _ML_DIR / "features_raw.csv"
_LABEL_CSV  = _ML_DIR / "labels.csv"
_MODEL_A    = _ML_DIR / "model_a_winrate.json"
_MODEL_B    = _ML_DIR / "model_b_pnl.json"
_MODEL_ACE  = _ML_DIR / "model_ace_ce.json"
_MODEL_APE  = _ML_DIR / "model_ape_pe.json"
_FEAT_COLS  = _ML_DIR / "feature_columns.json"
_METRICS    = _ML_DIR / "model_metrics.json"
_TEST_PREDS = _ML_DIR / "test_predictions.csv"
_FULL_PREDS = _ML_DIR / "full_predictions_754.csv"

# Walk-forward split boundaries
_TRAIN_END = date(2024, 9, 30)
_VAL_END   = date(2025, 6, 30)

# Model B: trade only when predicted move exceeds this threshold (points)
_MOVE_THRESHOLD = 15.0


# ── Step 1: Feature extraction ────────────────────────────────────────────────

def _btst_lookup() -> dict[str, dict]:
    """Load btst_signals.csv into a dict keyed by date string."""
    lookup: dict[str, dict] = {}
    if not _BTST_CSV.exists():
        return lookup
    with _BTST_CSV.open() as f:
        for row in csv.DictReader(f):
            lookup[row["date"]] = row
    return lookup


def _index_features(prefix: str, snap) -> dict:
    """Extract scalar features for one index snapshot."""
    w1, w2 = snap.w1, snap.w2

    spot_move_w1 = (
        (w1.spot_close - w1.spot_open) / w1.spot_open * 100
        if w1.spot_open > 0 else 0.0
    )
    spot_move_w2 = (
        (w2.spot_close - w1.spot_close) / w1.spot_close * 100
        if w1.spot_close > 0 else 0.0
    )
    spot_vs_mp = (
        (w2.spot_close - w2.max_pain_strike) / w2.spot_close
        if w2.spot_close > 0 else 0.0
    )

    return {
        f"{prefix}_w1_pcr":             w1.pcr,
        f"{prefix}_w2_pcr":             w2.pcr,
        f"{prefix}_pcr_delta":          w2.pcr - w1.pcr,
        f"{prefix}_w1_net_delta":       w1.net_delta,
        f"{prefix}_w2_net_delta":       w2.net_delta,
        f"{prefix}_delta_delta":        w2.net_delta - w1.net_delta,
        f"{prefix}_w1_iv_skew":         w1.iv_skew,
        f"{prefix}_w2_iv_skew":         w2.iv_skew,
        f"{prefix}_w1_oi_call":         w1.total_call_oi,
        f"{prefix}_w1_oi_put":          w1.total_put_oi,
        f"{prefix}_w2_oi_call":         w2.total_call_oi,
        f"{prefix}_w2_oi_put":          w2.total_put_oi,
        f"{prefix}_w2_oi_change_call":  w2.oi_change_call,
        f"{prefix}_w2_oi_change_put":   w2.oi_change_put,
        f"{prefix}_spot_move_w1":       spot_move_w1,
        f"{prefix}_spot_move_w2":       spot_move_w2,
        f"{prefix}_w2_spot_close":      w2.spot_close,
        f"{prefix}_w2_max_pain":        w2.max_pain_strike,
        f"{prefix}_spot_vs_maxpain":    spot_vs_mp,
    }


def _build_row(d: date, snaps: dict, ctx, btst_row: Optional[dict]) -> dict:
    """Assemble one feature-row for date d."""
    snap_n = snaps["NIFTY"]
    snap_b = snaps["BANKNIFTY"]

    row = {"date": d.strftime("%Y-%m-%d")}
    row.update(_index_features("nifty", snap_n))
    row.update(_index_features("bnf",   snap_b))

    # Cross-index agreement flags
    n_pcr = snap_n.w2.pcr
    b_pcr = snap_b.w2.pcr
    row["pcr_agreement"] = int((n_pcr - 1.0) * (b_pcr - 1.0) > 0)
    row["delta_agreement"] = int(snap_n.w2.net_delta * snap_b.w2.net_delta > 0)

    # Time features
    try:
        dte = (date.fromisoformat(ctx.current_nifty_expiry) - d).days
    except Exception:
        dte = 0
    row["days_to_expiry"]  = max(0, dte)
    row["is_expiry_eve"]   = int(ctx.is_expiry_eve)
    row["day_of_week"]     = d.weekday()

    # BTST signal features
    if btst_row and btst_row.get("signal"):
        sig = btst_row["signal"]
        row["btst_signal_fired"] = int(sig != "NO_TRADE")
        row["btst_direction"] = (
            1 if btst_row.get("direction") == "BULLISH"
            else -1 if btst_row.get("direction") == "BEARISH"
            else 0
        )
        try:
            row["btst_confidence"] = float(btst_row.get("confidence", 0))
        except (ValueError, TypeError):
            row["btst_confidence"] = 0.0
    else:
        row["btst_signal_fired"] = 0
        row["btst_direction"]    = 0
        row["btst_confidence"]   = 0.0

    return row


# ── Step 1b: Derived feature enrichment ──────────────────────────────────────

def _enrich_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add four derived features (Improvement 1) to the feature matrix in-place.
    Idempotent: returns unchanged if all four columns already present.

      prev_day_gap_pts      — yesterday's actual NIFTY next-open minus close
      prev_5day_ml_accuracy — rolling 5-day win rate of past ML predictions
      nifty_5day_momentum   — 5-day spot-close return normalised by today close
      vix_regime            — IV-skew bucket: 0=low(<12), 1=medium(12-18), 2=high(>18)
    """
    NEW = ["prev_day_gap_pts", "prev_5day_ml_accuracy",
           "nifty_5day_momentum", "vix_regime"]
    if all(c in df.columns for c in NEW):
        print("[STEP 1b] Derived features already present — skipping enrichment.")
        return df

    print("[STEP 1b] Adding four derived features...", flush=True)
    df = df.sort_values("date").reset_index(drop=True)

    # 1. vix_regime — no history needed, pure current-row derivation
    abs_iv = df["nifty_w2_iv_skew"].abs()
    df["vix_regime"] = pd.cut(
        abs_iv, bins=[-np.inf, 12.0, 18.0, np.inf], labels=[0, 1, 2]
    ).astype(int)

    # 2. nifty_5day_momentum — 5-day normalised spot return
    spot = df["nifty_w2_spot_close"]
    df["nifty_5day_momentum"] = ((spot - spot.shift(5)) / spot).fillna(0.0)

    # 3. prev_day_gap_pts — shift labels' move_pts by 1 trading day
    if _LABEL_CSV.exists():
        lab = pd.read_csv(_LABEL_CSV)[["date", "move_pts"]]
        lab["date"] = pd.to_datetime(lab["date"])
        df["date"]  = pd.to_datetime(df["date"])
        df = df.merge(lab.rename(columns={"move_pts": "_prev_gap"}),
                      on="date", how="left")
        df["prev_day_gap_pts"] = df["_prev_gap"].shift(1).fillna(0.0)
        df = df.drop(columns=["_prev_gap"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    else:
        df["prev_day_gap_pts"] = 0.0

    # 4. prev_5day_ml_accuracy — rolling 5-day accuracy of past ML predictions
    if _FULL_PREDS.exists():
        pr = pd.read_csv(_FULL_PREDS)[["date", "ml_pred", "actual_dir"]]
        pr["date"] = pd.to_datetime(pr["date"])
        pr = pr.sort_values("date").reset_index(drop=True)
        pr["ml_correct"] = (pr["ml_pred"] == pr["actual_dir"]).astype(int)
        pr["_roll5"] = (
            pr["ml_correct"].rolling(5, min_periods=1).mean()
                           .shift(1).fillna(0.5)
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.merge(pr[["date", "_roll5"]], on="date", how="left")
        df["prev_5day_ml_accuracy"] = df["_roll5"].fillna(0.5)
        df = df.drop(columns=["_roll5"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    else:
        df["prev_5day_ml_accuracy"] = 0.5

    vc = df["vix_regime"].value_counts().sort_index()
    print(f"  vix_regime dist      : low={vc.get(0,0)}  med={vc.get(1,0)}  high={vc.get(2,0)}")
    print(f"  nifty_5day_momentum  : mean={df['nifty_5day_momentum'].mean():.4f}  std={df['nifty_5day_momentum'].std():.4f}")
    print(f"  prev_day_gap_pts     : mean={df['prev_day_gap_pts'].mean():+.2f} pts  std={df['prev_day_gap_pts'].std():.2f}")
    print(f"  prev_5day_ml_acc     : mean={df['prev_5day_ml_accuracy'].mean():.3f}  std={df['prev_5day_ml_accuracy'].std():.3f}")
    nans = df[NEW].isnull().sum().sum()
    print(f"  NaN in new features  : {nans}  (must be 0)")
    print(f"  New feature matrix   : {df.shape}", flush=True)
    return df


def _add_inference_features(row: dict, d: date) -> dict:
    """
    Compute the four derived features for a single live-inference row.
    Reads from cached CSVs; uses safe defaults if data unavailable.
    """
    # 1. vix_regime
    abs_iv = abs(row.get("nifty_w2_iv_skew", 0.0))
    row["vix_regime"] = 0 if abs_iv < 12.0 else (1 if abs_iv < 18.0 else 2)

    # 2. nifty_5day_momentum — needs 5 prior spot closes
    row["nifty_5day_momentum"] = 0.0
    if _FEAT_CSV.exists():
        try:
            cache = pd.read_csv(_FEAT_CSV, usecols=["date", "nifty_w2_spot_close"])
            cache = cache[cache["date"] < d.strftime("%Y-%m-%d")].tail(6)
            if len(cache) >= 5:
                c5 = float(cache.iloc[-5]["nifty_w2_spot_close"])
                cn = row.get("nifty_w2_spot_close", 0.0)
                if cn > 0:
                    row["nifty_5day_momentum"] = (cn - c5) / cn
        except Exception:
            pass

    # 3. prev_day_gap_pts — yesterday's label move_pts
    row["prev_day_gap_pts"] = 0.0
    if _LABEL_CSV.exists():
        try:
            lab = pd.read_csv(_LABEL_CSV)[["date", "move_pts"]].sort_values("date")
            for n in range(1, 8):
                cand = (d - timedelta(days=n)).strftime("%Y-%m-%d")
                hit  = lab[lab["date"] == cand]
                if not hit.empty:
                    row["prev_day_gap_pts"] = float(hit.iloc[0]["move_pts"])
                    break
        except Exception:
            pass

    # 4. prev_5day_ml_accuracy
    row["prev_5day_ml_accuracy"] = 0.5
    if _FULL_PREDS.exists():
        try:
            pr  = pd.read_csv(_FULL_PREDS, usecols=["date", "ml_pred", "actual_dir"])
            pr  = pr.sort_values("date")
            dstr = d.strftime("%Y-%m-%d")
            past = pr[pr["date"] < dstr].tail(5)
            if len(past) > 0:
                row["prev_5day_ml_accuracy"] = float(
                    (past["ml_pred"] == past["actual_dir"]).mean()
                )
        except Exception:
            pass

    return row


def build_feature_matrix(rebuild: bool = False) -> pd.DataFrame:
    """
    Build feature matrix from rolling options cache.
    Caches to data/ml/features_raw.csv; set rebuild=True to force recompute.
    """
    _ML_DIR.mkdir(parents=True, exist_ok=True)

    if _FEAT_CSV.exists() and not rebuild:
        print(f"[STEP 1] Loading cached features from {_FEAT_CSV}")
        df = pd.read_csv(_FEAT_CSV)
        df = _enrich_feature_matrix(df)
        df.to_csv(_FEAT_CSV, index=False)
        print(f"         Shape: {df.shape}  (use --rebuild to recompute)")
        print(df.head().to_string())
        return df

    print("[STEP 1] Building feature matrix from rolling options cache...", flush=True)
    print("         This reads ~84 JSON files per date — expect 15-25 min.\n", flush=True)

    import logging as _log
    _log.getLogger("btst_engine.expiry_manager").setLevel(_log.ERROR)
    _log.getLogger("btst_engine.signal_builder").setLevel(_log.ERROR)
    _log.getLogger("dhanhq").setLevel(_log.CRITICAL)

    from btst_engine.backtest_runner import collect_backtest_dates
    import btst_engine.expiry_manager as _em
    from btst_engine.expiry_manager  import resolve_expiry_context
    from btst_engine.signal_builder  import build_snapshots

    # Skip live Dhan API calls entirely — use cache-derived expiry calendar.
    # Without this, each of 825 resolve_expiry_context() calls attempts a
    # TCP connection that may timeout, turning 25 min into 60+ min.
    _em._fetch_expiries_from_dhan = lambda index: None

    dates    = collect_backtest_dates()
    btst_lup = _btst_lookup()
    rows: List[dict] = []
    skipped  = 0

    for i, d in enumerate(dates):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(dates)} dates processed, {len(rows)} kept, {skipped} skipped...", flush=True)

        try:
            ctx   = resolve_expiry_context(d)
            snaps = build_snapshots(d, ctx)
        except Exception as exc:
            logger.warning("[FEAT] %s  error: %s", d, exc)
            skipped += 1
            continue

        n_w2 = snaps["NIFTY"].w2.data_quality
        b_w2 = snaps["BANKNIFTY"].w2.data_quality
        if n_w2 == "MISSING" or b_w2 == "MISSING":
            skipped += 1
            continue

        btst_row = btst_lup.get(d.strftime("%Y-%m-%d"))
        row = _build_row(d, snaps, ctx, btst_row)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = _enrich_feature_matrix(df)
    df.to_csv(_FEAT_CSV, index=False)

    print(f"\n[STEP 1] Done. Shape: {df.shape}  Saved to {_FEAT_CSV}", flush=True)
    print(f"         Dates kept: {len(rows)}  Skipped (MISSING/error): {skipped}", flush=True)
    print(df.head().to_string(), flush=True)
    return df


# ── Step 2: Label generation ──────────────────────────────────────────────────

def _get_nifty_spot_open(d: date) -> Optional[float]:
    """
    Return NIFTY spot open price at 9:15 AM for date d by reading only the
    ATM CALL cache file — far cheaper than loading the full snapshot.
    """
    from btst_engine.signal_builder import _find_file, _load_leg, _ts_ist
    fp = _find_file("NIFTY", d.strftime("%Y-%m-%d"), "WEEK", "E1", "ATM", "CALL")
    if fp is None:
        return None
    leg = _load_leg(fp, "CALL")
    if leg is None or not leg.get("spot") or not leg.get("timestamp"):
        return None
    ts_start = _ts_ist(d, 9, 15)
    ts_end   = _ts_ist(d, 14, 45)
    spots    = leg["spot"]
    for i, t in enumerate(leg["timestamp"]):
        if ts_start <= t < ts_end and i < len(spots):
            return float(spots[i])
    return None


def _next_trading_date(d: date, available_dates: set[str]) -> Optional[date]:
    """Return the first date after d that is in available_dates."""
    cur = d + timedelta(days=1)
    for _ in range(14):          # max 2 weeks gap
        if cur.strftime("%Y-%m-%d") in available_dates:
            return cur
        cur += timedelta(days=1)
    return None


def build_labels(feat_df: pd.DataFrame, rebuild: bool = False) -> pd.DataFrame:
    """
    For each date, compute:
      label     = 1 (CE_BUY) if next NIFTY open > today W2 close else 0 (PE_BUY)
      move_pts  = next_open - today_close  (positive = market up)
      move_pct  = move_pts / today_close * 100
    """
    if _LABEL_CSV.exists() and not rebuild:
        print(f"\n[STEP 2] Loading cached labels from {_LABEL_CSV}")
        df = pd.read_csv(_LABEL_CSV)
    else:
        print("\n[STEP 2] Building labels (next-day NIFTY open vs today close)...")
        import logging as _log
        _log.getLogger("btst_engine.signal_builder").setLevel(_log.ERROR)

        available = set(feat_df["date"].tolist())
        records = []

        for idx, row in feat_df.iterrows():
            d         = date.fromisoformat(row["date"])
            today_cls = row["nifty_w2_spot_close"]

            if today_cls <= 0:
                continue

            next_d = _next_trading_date(d, available)
            if next_d is None:
                continue

            next_open = _get_nifty_spot_open(next_d)
            if next_open is None or next_open <= 0:
                continue

            move_pts = next_open - today_cls
            records.append({
                "date":      row["date"],
                "label":     1 if move_pts > 0 else 0,
                "move_pts":  round(move_pts, 2),
                "move_pct":  round(move_pts / today_cls * 100, 4),
            })

        df = pd.DataFrame(records)
        df.to_csv(_LABEL_CSV, index=False)
        print(f"         Saved to {_LABEL_CSV}")

    # Summary stats
    ce = df[df["label"] == 1]
    pe = df[df["label"] == 0]
    total = len(df)

    print(f"\n  Label distribution:")
    print(f"    CE_BUY (up)  : {len(ce):4d}  ({len(ce)/total*100:.1f}%)")
    print(f"    PE_BUY (down): {len(pe):4d}  ({len(pe)/total*100:.1f}%)")
    print(f"  Average move on CE days : {ce['move_pts'].mean():+.1f} pts  "
          f"({ce['move_pct'].mean():+.3f}%)")
    print(f"  Average move on PE days : {pe['move_pts'].mean():+.1f} pts  "
          f"({pe['move_pct'].mean():+.3f}%)")

    print(f"\n  Top 10 biggest UP moves:")
    for _, r in ce.nlargest(10, "move_pts").iterrows():
        print(f"    {r['date']}  +{r['move_pts']:.1f} pts  (+{r['move_pct']:.2f}%)")

    print(f"\n  Top 10 biggest DOWN moves:")
    for _, r in pe.nsmallest(10, "move_pts").iterrows():
        print(f"    {r['date']}  {r['move_pts']:.1f} pts  ({r['move_pct']:.2f}%)")

    return df


# ── Step 3: Model training ────────────────────────────────────────────────────

_PARAM_GRID = {
    "max_depth":        [3, 4, 5],
    "n_estimators":     [100, 200, 300],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
}


def _split_data(merged: pd.DataFrame, feature_cols: List[str]):
    """Return train/val/test splits as (X, y, move_pts, dates) tuples."""
    dates = pd.to_datetime(merged["date"])

    mask_tr  = dates <= pd.Timestamp(_TRAIN_END)
    mask_val = (dates > pd.Timestamp(_TRAIN_END)) & (dates <= pd.Timestamp(_VAL_END))
    mask_te  = dates > pd.Timestamp(_VAL_END)

    def _pack(mask):
        sub = merged[mask].reset_index(drop=True)
        return (
            sub[feature_cols].values,
            sub["label"].values.astype(int),
            sub["move_pts"].values.astype(float),
            sub["date"].tolist(),
        )

    return _pack(mask_tr), _pack(mask_val), _pack(mask_te)


def train_model_a(X_tr, y_tr, X_val, y_val) -> Tuple[object, dict]:
    """
    Model A: XGBoost classifier, win-rate maximiser.
    GridSearchCV on training data; evaluate best model on validation set.
    """
    print("\n[STEP 3A] Training Model A (win-rate / F1 classifier)...")
    print(f"  Train: {len(X_tr)} samples  |  Val: {len(X_val)} samples")
    print(f"  Grid size: {3*3*3*2*2} combinations × 3-fold CV "
          f"= {3*3*3*2*2*3} fits (may take a few minutes)")

    base = xgb.XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    cv = TimeSeriesSplit(n_splits=3)
    gs = GridSearchCV(
        base, _PARAM_GRID, scoring="f1", cv=cv,
        n_jobs=-1, refit=True, verbose=1,
    )
    gs.fit(X_tr, y_tr)

    best = gs.best_estimator_
    print(f"\n  Best params (A): {gs.best_params_}")
    print(f"  Best CV F1     : {gs.best_score_:.4f}")

    val_preds = best.predict(X_val)
    val_acc   = accuracy_score(y_val, val_preds)
    val_f1    = f1_score(y_val, val_preds, average="binary")
    print(f"  Val accuracy   : {val_acc:.4f}")
    print(f"  Val F1 (CE=1)  : {val_f1:.4f}")

    metrics = {
        "best_cv_f1":    gs.best_score_,
        "val_accuracy":  val_acc,
        "val_f1":        val_f1,
        "best_params":   gs.best_params_,
    }
    return best, metrics


def train_model_b(X_tr, move_tr, X_val, move_val) -> Tuple[object, dict]:
    """
    Model B: XGBoost regressor predicting move_pts.
    Threshold ±_MOVE_THRESHOLD for CE/PE direction.
    """
    print("\n[STEP 3B] Training Model B (PnL / move_pts regressor)...")

    base = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
    )

    cv = TimeSeriesSplit(n_splits=3)
    gs = GridSearchCV(
        base, _PARAM_GRID, scoring="neg_mean_absolute_error",
        cv=cv, n_jobs=-1, refit=True, verbose=1,
    )
    gs.fit(X_tr, move_tr)

    best = gs.best_estimator_
    val_pred  = best.predict(X_val)
    val_mae   = float(np.mean(np.abs(val_pred - move_val)))
    val_corr  = float(np.corrcoef(val_pred, move_val)[0, 1]) if len(val_pred) > 1 else 0.0

    # Direction accuracy on val
    pred_dir  = np.where(val_pred > 0, 1, 0)
    true_dir  = np.where(move_val > 0, 1, 0)
    dir_acc   = accuracy_score(true_dir, pred_dir)

    print(f"\n  Best params (B): {gs.best_params_}")
    print(f"  Val MAE         : {val_mae:.2f} pts")
    print(f"  Val correlation : {val_corr:.4f}")
    print(f"  Val direction   : {dir_acc:.4f} (raw sign accuracy)")

    metrics = {
        "val_mae":        val_mae,
        "val_correlation": val_corr,
        "val_dir_acc":    dir_acc,
        "best_params":    gs.best_params_,
    }
    return best, metrics


# ── Step 3b: Dual CE/PE models (Improvement 2) ───────────────────────────────

def train_model_ace(X_tr, y_tr, X_val, y_val) -> Tuple[object, dict]:
    """
    Model A-CE: XGBoost classifier optimised for up-gap days (label=1).
    Maximises recall for CE_BUY via recall scoring + scale_pos_weight.
    """
    print("\n[STEP 3-ACE] Training Model A-CE (optimised for CE_BUY / up-gap days)...")
    print(f"  Train: {len(X_tr)} samples  |  Val: {len(X_val)} samples")

    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    spw   = n_neg / max(n_pos, 1)

    base = xgb.XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    cv = TimeSeriesSplit(n_splits=3)
    gs = GridSearchCV(
        base, _PARAM_GRID, scoring="recall", cv=cv,
        n_jobs=-1, refit=True, verbose=1,
    )
    gs.fit(X_tr, y_tr)

    best      = gs.best_estimator_
    val_proba = best.predict_proba(X_val)[:, 1]
    val_preds = (val_proba >= 0.5).astype(int)
    val_acc   = accuracy_score(y_val, val_preds)
    val_rec   = float(np.sum((val_preds == 1) & (y_val == 1)) / max(int((y_val == 1).sum()), 1))

    print(f"\n  Best params (ACE) : {gs.best_params_}")
    print(f"  Best CV recall    : {gs.best_score_:.4f}")
    print(f"  Val accuracy      : {val_acc:.4f}")
    print(f"  Val CE recall     : {val_rec:.4f}")

    return best, {
        "best_cv_recall": gs.best_score_,
        "val_accuracy":   val_acc,
        "val_ce_recall":  val_rec,
        "best_params":    gs.best_params_,
    }


def train_model_ape(X_tr, y_tr, X_val, y_val) -> Tuple[object, dict]:
    """
    Model A-PE: XGBoost classifier optimised for down-gap days (label=0).
    Labels are flipped (PE_BUY=1, CE_BUY=0) so recall scoring favours PE detection.
    At inference predict_proba(X)[:,1] gives P(PE_BUY).
    """
    print("\n[STEP 3-APE] Training Model A-PE (optimised for PE_BUY / down-gap days)...")
    print(f"  Train: {len(X_tr)} samples  |  Val: {len(X_val)} samples")

    y_tr_f  = 1 - y_tr
    y_val_f = 1 - y_val

    n_neg = int((y_tr_f == 0).sum())
    n_pos = int((y_tr_f == 1).sum())
    spw   = n_neg / max(n_pos, 1)

    base = xgb.XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    cv = TimeSeriesSplit(n_splits=3)
    gs = GridSearchCV(
        base, _PARAM_GRID, scoring="recall", cv=cv,
        n_jobs=-1, refit=True, verbose=1,
    )
    gs.fit(X_tr, y_tr_f)

    best      = gs.best_estimator_
    val_proba = best.predict_proba(X_val)[:, 1]        # P(PE_BUY)
    val_preds_orig = (val_proba < 0.5).astype(int)     # back to original label space
    val_acc   = accuracy_score(y_val, val_preds_orig)
    val_rec   = float(np.sum((val_proba >= 0.5) & (y_val_f == 1)) / max(int((y_val_f == 1).sum()), 1))

    print(f"\n  Best params (APE) : {gs.best_params_}")
    print(f"  Best CV recall    : {gs.best_score_:.4f}")
    print(f"  Val accuracy      : {val_acc:.4f}")
    print(f"  Val PE recall     : {val_rec:.4f}")

    return best, {
        "best_cv_recall": gs.best_score_,
        "val_accuracy":   val_acc,
        "val_pe_recall":  val_rec,
        "best_params":    gs.best_params_,
    }


def _dual_model_predict(
    model_ace, model_ape,
    X: np.ndarray,
    conf_threshold: float = 0.55,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Generate final direction signal from dual ACE/APE classifiers.

      ce_conf = model_ace.predict_proba(X)[:,1]  — P(CE_BUY)
      pe_conf = model_ape.predict_proba(X)[:,1]  — P(PE_BUY)  [flipped-label space]

    Final signal: CE_BUY if ce_conf >= pe_conf, else PE_BUY.
    HIGH confidence if the winning model's confidence >= conf_threshold.

    Returns (preds, ce_confs, pe_confs, confidence_labels).
    """
    ce_confs  = model_ace.predict_proba(X)[:, 1]
    pe_confs  = model_ape.predict_proba(X)[:, 1]
    preds     = (ce_confs >= pe_confs).astype(int)   # 1=CE_BUY, 0=PE_BUY
    win_confs = np.where(preds == 1, ce_confs, pe_confs)
    conf_lbls = ["HIGH" if c >= conf_threshold else "LOW" for c in win_confs]
    return preds, ce_confs, pe_confs, conf_lbls


def _monthly_breakdown(
    preds:     np.ndarray,
    y_te:      np.ndarray,
    move_te:   np.ndarray,
    dates_te:  List[str],
    model_tag: str,
) -> None:
    """Print month-by-month PnL and win-rate table for the test period."""
    df = pd.DataFrame({
        "date":     pd.to_datetime(dates_te),
        "pred":     preds,
        "label":    y_te,
        "move_pts": move_te,
    })
    df["pnl"]     = np.where(df["pred"] == 1, df["move_pts"], -df["move_pts"])
    df["correct"] = (df["pred"] == df["label"]).astype(int)
    df["month"]   = df["date"].dt.to_period("M")

    grp = df.groupby("month").agg(
        trades=("pnl", "count"),
        wins=("correct", "sum"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
    ).reset_index()
    grp["win_rate"] = grp["wins"] / grp["trades"]

    print(f"\n  Monthly breakdown [{model_tag}]:")
    print(f"  {'Month':<10}  {'Trades':>6}  {'Wins':>5}  {'WinRate':>8}  {'TotalPnL':>10}  {'AvgPnL':>8}")
    print(f"  {'-'*60}")
    for _, r in grp.iterrows():
        print(f"  {str(r['month']):<10}  {int(r['trades']):>6}  {int(r['wins']):>5}  "
              f"{r['win_rate']:>8.1%}  {r['total_pnl']:>+10.1f}  {r['avg_pnl']:>+8.1f}")


def evaluate_dual_models(
    model_ace, model_ape,
    X_te:       np.ndarray,
    y_te:       np.ndarray,
    move_te:    np.ndarray,
    dates_te:   List[str],
    feature_cols: List[str],
    baseline_acc: float,
    baseline_pnl: float,
) -> dict:
    """Full test-set evaluation for the dual CE/PE model architecture."""
    _NEW = {"prev_day_gap_pts", "prev_5day_ml_accuracy",
            "nifty_5day_momentum", "vix_regime"}

    print("\n" + "=" * 64)
    print("[STEP 4b] Dual CE/PE Model Evaluation  (Improvement 2)")
    print("=" * 64)
    print(f"  Test dates: {dates_te[0]} to {dates_te[-1]}  ({len(dates_te)} samples)")

    preds, ce_confs, pe_confs, conf_lbls = _dual_model_predict(
        model_ace, model_ape, X_te
    )
    acc = accuracy_score(y_te, preds)
    delta_acc = acc - baseline_acc

    print(f"\n  Accuracy / win rate : {acc:.4f}  ({acc*100:.1f}%)  "
          f"[baseline {baseline_acc*100:.1f}%,  delta {delta_acc*100:+.1f}%]")
    print(f"\n  Classification report:")
    print(classification_report(y_te, preds,
                                target_names=["PE_BUY", "CE_BUY"], digits=4))
    cm = confusion_matrix(y_te, preds)
    print(f"  Confusion matrix:")
    print(f"    Predicted: PE_BUY  CE_BUY")
    print(f"    PE (true): {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"    CE (true): {cm[1,0]:6d}   {cm[1,1]:6d}")

    # Feature importances for ACE
    fi_ace = model_ace.feature_importances_
    top_ace = np.argsort(fi_ace)[::-1][:15]
    print(f"\n  Top 15 feature importances (Model A-CE):")
    for rank, idx in enumerate(top_ace, 1):
        new_tag = "  <-- NEW" if feature_cols[idx] in _NEW else ""
        print(f"    {rank:2d}. {feature_cols[idx]:<40} {fi_ace[idx]:.4f}{new_tag}")

    fi_ape = model_ape.feature_importances_
    top_ape = np.argsort(fi_ape)[::-1][:10]
    print(f"\n  Top 10 feature importances (Model A-PE):")
    for rank, idx in enumerate(top_ape, 1):
        new_tag = "  <-- NEW" if feature_cols[idx] in _NEW else ""
        print(f"    {rank:2d}. {feature_cols[idx]:<40} {fi_ape[idx]:.4f}{new_tag}")

    pnl_stats = _simulate_pnl(preds, y_te, move_te, dates_te, "Dual Model (ACE vs APE)")
    delta_pnl = pnl_stats["total_pnl"] - baseline_pnl
    print(f"\n  PnL comparison: dual={pnl_stats['total_pnl']:+.1f} pts  "
          f"baseline={baseline_pnl:+.1f} pts  delta={delta_pnl:+.1f} pts")

    # HIGH / LOW confidence split
    high_mask = np.array(conf_lbls) == "HIGH"
    low_mask  = ~high_mask
    print(f"\n  Confidence split: HIGH={high_mask.sum()}  LOW={low_mask.sum()}")
    pnl_h = {}
    if high_mask.sum() > 0:
        acc_h = accuracy_score(y_te[high_mask], preds[high_mask])
        pnl_h = _simulate_pnl(
            preds[high_mask], y_te[high_mask],
            move_te[high_mask],
            [d for d, h in zip(dates_te, high_mask) if h],
            "Dual HIGH confidence",
        )
        print(f"  HIGH confidence accuracy: {acc_h:.4f}")
    if low_mask.sum() > 0:
        acc_l = accuracy_score(y_te[low_mask], preds[low_mask])
        print(f"  LOW  confidence accuracy: {acc_l:.4f}")

    _monthly_breakdown(preds, y_te, move_te, dates_te, "Dual Model")

    return {
        "dual_accuracy": acc,
        "dual_pnl":      pnl_stats,
        "high_pnl":      pnl_h,
        "agree_rate":    float(high_mask.mean()),
    }


# ── Step 4: Evaluation ────────────────────────────────────────────────────────

def _model_b_direction(pred_move: np.ndarray) -> np.ndarray:
    """
    Convert regressor output to 0/1 direction.
    Values outside ±threshold still take the sign — never abstain.
    """
    return (pred_move > 0).astype(int)


def _simulate_pnl(
    preds:     np.ndarray,
    labels:    np.ndarray,
    move_pts:  np.ndarray,
    dates:     List[str],
    model_tag: str,
) -> dict:
    """
    Simulate per-trade PnL using actual move_pts.
      CE_BUY: PnL = +move_pts if correct (label=1), -|move_pts| if wrong
      PE_BUY: PnL = +|move_pts| if correct (label=0), -move_pts if wrong

    Since move_pts is signed (positive = market up):
      CE_BUY prediction: PnL =  move_pts
      PE_BUY prediction: PnL = -move_pts
    """
    pnls        = np.where(preds == 1, move_pts, -move_pts)
    cumulative  = np.cumsum(pnls)

    wins  = pnls > 0
    loses = pnls < 0

    # Max drawdown
    peak = cumulative[0]
    max_dd = 0.0
    for c in cumulative:
        if c > peak:
            peak = c
        dd = peak - c
        if dd > max_dd:
            max_dd = dd

    # Streaks
    best_streak = worst_streak = cur_streak = 0
    prev_win: Optional[bool] = None
    for w in wins:
        if prev_win is None or prev_win != w:
            cur_streak = 1
        else:
            cur_streak += 1
        if w and cur_streak > best_streak:
            best_streak = cur_streak
        if not w and cur_streak > worst_streak:
            worst_streak = cur_streak
        prev_win = bool(w)

    sharpe = float(np.mean(pnls) / np.std(pnls)) if np.std(pnls) > 0 else 0.0

    print(f"\n  [{model_tag}] Per-trade PnL simulation ({len(preds)} trades)")
    print(f"  {'Date':<12}  {'Pred':<7}  {'ActualMove':>10}  {'PnL':>8}  {'Running':>10}")
    print(f"  {'-'*60}")
    running = 0.0
    for i, (dt, pr, mv, pnl) in enumerate(zip(dates, preds, move_pts, pnls)):
        running += float(pnl)
        signal = "CE_BUY" if pr == 1 else "PE_BUY"
        if i < 15 or i >= len(preds) - 5:   # first 15 + last 5
            print(f"  {dt:<12}  {signal:<7}  {mv:>+10.1f}  {pnl:>+8.1f}  {running:>+10.1f}")
        elif i == 15:
            print(f"  ... ({len(preds)-20} trades omitted) ...")

    dec = int(wins.sum())
    print(f"\n  Total PnL       : {cumulative[-1]:+.1f} pts")
    print(f"  Win rate        : {dec}/{len(preds)} = {dec/len(preds)*100:.1f}%")
    print(f"  Avg win         : {pnls[wins].mean():+.1f} pts" if wins.any() else "  Avg win: N/A")
    print(f"  Avg loss        : {pnls[loses].mean():+.1f} pts" if loses.any() else "  Avg loss: N/A")
    wl = (abs(pnls[wins].mean()) / abs(pnls[loses].mean())
          if wins.any() and loses.any() else float("nan"))
    print(f"  Win/loss ratio  : {wl:.2f}")
    print(f"  Max drawdown    : -{max_dd:.1f} pts")
    print(f"  Best streak     : {best_streak} consecutive wins")
    print(f"  Worst streak    : {worst_streak} consecutive losses")
    print(f"  Sharpe-like     : {sharpe:.3f}  (mean/std of per-trade PnL)")

    return {
        "total_pnl":     float(cumulative[-1]),
        "win_rate":      float(dec / len(preds)),
        "avg_win":       float(pnls[wins].mean()) if wins.any() else 0.0,
        "avg_loss":      float(pnls[loses].mean()) if loses.any() else 0.0,
        "wl_ratio":      float(wl) if not np.isnan(wl) else 0.0,
        "max_drawdown":  float(-max_dd),
        "best_streak":   best_streak,
        "worst_streak":  worst_streak,
        "sharpe":        sharpe,
    }


def evaluate_models(
    model_a, model_b,
    X_te: np.ndarray, y_te: np.ndarray,
    move_te: np.ndarray, dates_te: List[str],
    feature_cols: List[str],
) -> dict:
    """Full test-set evaluation for both models."""
    print("\n" + "=" * 64)
    print("[STEP 4] Test-set evaluation")
    print("=" * 64)
    print(f"  Test dates: {dates_te[0]} to {dates_te[-1]}  ({len(dates_te)} samples)")

    metrics = {}

    for tag, preds in [
        ("Model A", model_a.predict(X_te)),
        ("Model B", _model_b_direction(model_b.predict(X_te))),
    ]:
        print(f"\n  -- {tag} --------------------------------------------------")
        acc = accuracy_score(y_te, preds)
        print(f"  Accuracy / win rate : {acc:.4f}  ({acc*100:.1f}%)")
        print(f"\n  Classification report:")
        print(classification_report(y_te, preds,
                                    target_names=["PE_BUY", "CE_BUY"],
                                    digits=4))
        cm = confusion_matrix(y_te, preds)
        print(f"  Confusion matrix:")
        print(f"    Predicted: PE_BUY  CE_BUY")
        print(f"    PE (true): {cm[0,0]:6d}   {cm[0,1]:6d}")
        print(f"    CE (true): {cm[1,0]:6d}   {cm[1,1]:6d}")

        # Feature importance top 15
        model = model_a if tag == "Model A" else model_b
        fi = model.feature_importances_
        top_idx = np.argsort(fi)[::-1][:15]
        print(f"\n  Top 15 feature importances ({tag}):")
        for rank, idx in enumerate(top_idx, 1):
            print(f"    {rank:2d}. {feature_cols[idx]:<40} {fi[idx]:.4f}")

        pnl_stats = _simulate_pnl(preds, y_te, move_te, dates_te, tag)
        metrics[tag] = {
            "test_accuracy": acc,
            "pnl":           pnl_stats,
        }

    return metrics


# ── Step 5: Ensemble ──────────────────────────────────────────────────────────

def ensemble_predict(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    HIGH CONFIDENCE: both models agree on direction.
    LOW  CONFIDENCE: models disagree — fall back to Model A.
    Returns (ensemble_preds, confidence_labels).
    """
    agree    = preds_a == preds_b
    ensemble = np.where(agree, preds_a, preds_a)   # A always provides direction
    conf     = ["HIGH" if a else "LOW" for a in agree]
    return ensemble, conf


def evaluate_ensemble(
    model_a, model_b,
    X_te: np.ndarray, y_te: np.ndarray,
    move_te: np.ndarray, dates_te: List[str],
) -> dict:
    print("\n[STEP 5] Ensemble evaluation on test set")
    preds_a   = model_a.predict(X_te)
    preds_b   = _model_b_direction(model_b.predict(X_te))
    ens, conf = ensemble_predict(preds_a, preds_b)

    high_mask = np.array(conf) == "HIGH"
    low_mask  = ~high_mask

    print(f"  Signals: {len(ens)}  |  HIGH: {high_mask.sum()}  |  LOW: {low_mask.sum()}")
    print(f"  Overall ensemble accuracy: {accuracy_score(y_te, ens):.4f}")

    if high_mask.sum() > 0:
        acc_h = accuracy_score(y_te[high_mask], ens[high_mask])
        pnl_h = _simulate_pnl(
            ens[high_mask], y_te[high_mask],
            move_te[high_mask], [d for d, h in zip(dates_te, high_mask) if h],
            "Ensemble HIGH",
        )
        print(f"  HIGH confidence accuracy: {acc_h:.4f}")
    else:
        pnl_h = {}

    if low_mask.sum() > 0:
        acc_l = accuracy_score(y_te[low_mask], ens[low_mask])
        print(f"  LOW  confidence accuracy: {acc_l:.4f}")

    return {"high_pnl": pnl_h, "agree_rate": float(high_mask.mean())}


# ── Step 6: Save ──────────────────────────────────────────────────────────────

def save_all(
    model_a, model_b,
    feature_cols: List[str],
    all_metrics: dict,
    X_te, dates_te, y_te, move_te,
    model_ace=None, model_ape=None,
) -> None:
    print("\n[STEP 6] Saving models, metrics, predictions...")
    _ML_DIR.mkdir(parents=True, exist_ok=True)

    model_a.save_model(str(_MODEL_A))
    model_b.save_model(str(_MODEL_B))

    if model_ace is not None:
        model_ace.save_model(str(_MODEL_ACE))
    if model_ape is not None:
        model_ape.save_model(str(_MODEL_APE))

    with _FEAT_COLS.open("w") as f:
        json.dump(feature_cols, f)

    with _METRICS.open("w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # Test predictions — use dual model if available, else legacy Model A
    if model_ace is not None and model_ape is not None:
        preds_final, ce_c, pe_c, conf_lbls = _dual_model_predict(model_ace, model_ape, X_te)
    else:
        preds_a   = model_a.predict(X_te)
        preds_b_r = model_b.predict(X_te)
        preds_b   = (preds_b_r > 0).astype(int)
        preds_final, conf_lbls = ensemble_predict(preds_a, preds_b)
        ce_c = model_a.predict_proba(X_te)[:, 1]
        pe_c = 1 - ce_c

    with _TEST_PREDS.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "date", "ce_conf", "pe_conf", "ensemble_pred",
            "ensemble_confidence", "actual_label", "move_pts", "pnl",
        ])
        w.writeheader()
        for i, dt in enumerate(dates_te):
            pe = int(preds_final[i])
            pnl = float(move_te[i]) if pe == 1 else float(-move_te[i])
            w.writerow({
                "date":               dt,
                "ce_conf":            round(float(ce_c[i]), 4),
                "pe_conf":            round(float(pe_c[i]), 4),
                "ensemble_pred":      "CE_BUY" if pe == 1 else "PE_BUY",
                "ensemble_confidence": conf_lbls[i],
                "actual_label":       "CE_BUY" if y_te[i] == 1 else "PE_BUY",
                "move_pts":           round(float(move_te[i]), 2),
                "pnl":                round(pnl, 2),
            })

    print(f"  Model A     → {_MODEL_A}")
    print(f"  Model B     → {_MODEL_B}")
    if model_ace is not None:
        print(f"  Model ACE   → {_MODEL_ACE}")
        print(f"  Model APE   → {_MODEL_APE}")
    print(f"  Feat cols   → {_FEAT_COLS}")
    print(f"  Metrics     → {_METRICS}")
    print(f"  Test preds  → {_TEST_PREDS}")


# ── Step 7: predict_today ─────────────────────────────────────────────────────

def predict_today(date_str: str) -> dict:
    """
    Generate a live CE/PE prediction for one date.

    Loads the saved models and feature column list, builds features for
    date_str from the rolling options cache, and returns a prediction dict.

    Returns
    -------
    {
      'date':               str,
      'signal':             'CE_BUY' or 'PE_BUY',
      'confidence':         float,   # Model A probability for CE_BUY
      'expected_move_pts':  float,   # Model B regression output
      'ensemble_confidence':'HIGH' or 'LOW',
      'top_features':       dict,    # top 5 feature → value
    }
    """
    if not _MODEL_A.exists() or not _FEAT_COLS.exists():
        raise RuntimeError(
            "Models not found. Run: python -m btst_engine.ml_predictor --train"
        )

    import logging as _log
    _log.getLogger("btst_engine.expiry_manager").setLevel(_log.ERROR)
    _log.getLogger("btst_engine.signal_builder").setLevel(_log.ERROR)

    from btst_engine.expiry_manager import resolve_expiry_context
    from btst_engine.signal_builder import build_snapshots

    d   = date.fromisoformat(date_str)
    ctx = resolve_expiry_context(d)

    try:
        snaps = build_snapshots(d, ctx)
    except Exception as exc:
        raise RuntimeError(f"Cannot build snapshot for {date_str}: {exc}")

    btst_lup = _btst_lookup()
    row = _build_row(d, snaps, ctx, btst_lup.get(date_str))
    row = _add_inference_features(row, d)

    with _FEAT_COLS.open() as f:
        feature_cols = json.load(f)

    feat_row = [row.get(c, 0.0) for c in feature_cols]
    X = np.array([feat_row], dtype=float)

    # Always load Model B for move magnitude
    mb = xgb.XGBRegressor()
    mb.load_model(str(_MODEL_B))
    pred_move = float(mb.predict(X)[0])

    if _MODEL_ACE.exists() and _MODEL_APE.exists():
        # ── Dual model path (Improvement 2) ──────────────────────────
        mac = xgb.XGBClassifier()
        map_ = xgb.XGBClassifier()
        mac.load_model(str(_MODEL_ACE))
        map_.load_model(str(_MODEL_APE))

        ce_conf = float(mac.predict_proba(X)[0, 1])
        pe_conf = float(map_.predict_proba(X)[0, 1])
        signal  = "CE_BUY" if ce_conf >= pe_conf else "PE_BUY"
        conf    = ce_conf if signal == "CE_BUY" else pe_conf
        ens_conf = "HIGH" if conf >= 0.55 else "LOW"

        fi      = mac.feature_importances_
        top_idx = np.argsort(fi)[::-1][:5]
    else:
        # ── Legacy single-model path ──────────────────────────────────
        ma = xgb.XGBClassifier()
        ma.load_model(str(_MODEL_A))

        ce_conf   = float(ma.predict_proba(X)[0, 1])
        pe_conf   = 1.0 - ce_conf
        pred_a    = 1 if ce_conf >= 0.5 else 0
        pred_b    = 1 if pred_move > 0 else 0
        signal    = "CE_BUY" if pred_a == 1 else "PE_BUY"
        conf      = ce_conf if pred_a == 1 else pe_conf
        ens_conf  = "HIGH" if pred_a == pred_b else "LOW"

        fi      = ma.feature_importances_
        top_idx = np.argsort(fi)[::-1][:5]

    top_feats = {feature_cols[i]: round(float(feat_row[i]), 4) for i in top_idx}

    return {
        "date":               date_str,
        "signal":             signal,
        "confidence":         round(conf, 4),
        "ce_conf":            round(ce_conf, 4),
        "pe_conf":            round(pe_conf, 4),
        "expected_move_pts":  round(pred_move, 2),
        "ensemble_confidence": ens_conf,
        "top_features":       top_feats,
    }


# ── Pipeline ──────────────────────────────────────────────────────────────────

def train_pipeline(rebuild: bool = False) -> None:
    """Full end-to-end training pipeline."""
    print("\n" + "=" * 64)
    print("  BTST ML Predictor — Training Pipeline")
    print("=" * 64)

    # ── 1. Features ────────────────────────────────────────────────
    feat_df = build_feature_matrix(rebuild=rebuild)

    # ── 2. Labels ──────────────────────────────────────────────────
    label_df = build_labels(feat_df, rebuild=rebuild)

    # ── Merge ──────────────────────────────────────────────────────
    merged = feat_df.merge(label_df, on="date", how="inner").dropna()
    print(f"\n[MERGE] Final dataset: {len(merged)} rows after merge + dropna")

    # Identify feature columns (everything except date, label, move_pts, move_pct)
    non_feat = {"date", "label", "move_pts", "move_pct"}
    feature_cols = [c for c in merged.columns if c not in non_feat]

    train_data, val_data, test_data = _split_data(merged, feature_cols)
    X_tr, y_tr, move_tr, dates_tr = train_data
    X_va, y_va, move_va, dates_va = val_data
    X_te, y_te, move_te, dates_te = test_data

    print(f"  Train: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_te)}")
    print(f"  Feature columns: {len(feature_cols)}")

    # ── 3. Train ───────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("[STEP 3] Model training")
    print("=" * 64)

    model_a, metrics_a = train_model_a(X_tr, y_tr, X_va, y_va)
    model_b, metrics_b = train_model_b(X_tr, move_tr, X_va, move_va)

    # ── 4. Evaluate single models (baseline) ──────────────────────
    test_metrics = evaluate_models(
        model_a, model_b, X_te, y_te, move_te, dates_te, feature_cols
    )

    # ── 5. Legacy ensemble ─────────────────────────────────────────
    ens_metrics = evaluate_ensemble(
        model_a, model_b, X_te, y_te, move_te, dates_te
    )

    # Monthly breakdown for single Model A (baseline)
    _monthly_breakdown(
        model_a.predict(X_te), y_te, move_te, dates_te, "Model A (single, baseline)"
    )

    # ── 3b. Train dual CE/PE models (Improvement 2) ────────────────
    print("\n" + "=" * 64)
    print("[STEP 3b] Dual CE/PE Model Training  (Improvement 2)")
    print("=" * 64)
    model_ace, metrics_ace = train_model_ace(X_tr, y_tr, X_va, y_va)
    model_ape, metrics_ape = train_model_ape(X_tr, y_tr, X_va, y_va)

    # ── 4b. Evaluate dual models ───────────────────────────────────
    baseline_acc = test_metrics["Model A"]["test_accuracy"]
    baseline_pnl = test_metrics["Model A"]["pnl"]["total_pnl"]
    dual_metrics = evaluate_dual_models(
        model_ace, model_ape,
        X_te, y_te, move_te, dates_te, feature_cols,
        baseline_acc, baseline_pnl,
    )

    # ── 6. Save ────────────────────────────────────────────────────
    all_metrics = {
        "model_a_training":   metrics_a,
        "model_b_training":   metrics_b,
        "model_ace_training": metrics_ace,
        "model_ape_training": metrics_ape,
        "test":               test_metrics,
        "ensemble":           ens_metrics,
        "dual":               dual_metrics,
    }
    save_all(model_a, model_b, feature_cols, all_metrics,
             X_te, dates_te, y_te, move_te,
             model_ace=model_ace, model_ape=model_ape)

    # ── Final summary ──────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  FINAL SUMMARY — Improvement 1 + 2")
    print("=" * 64)
    print(f"\n  {'Metric':<30}  {'Baseline (single Model A)':>25}  {'Dual ACE/APE':>15}")
    print(f"  {'-'*75}")
    da = dual_metrics["dual_accuracy"]
    dp = dual_metrics["dual_pnl"]
    ba = baseline_acc
    bp = baseline_pnl
    bwr  = test_metrics["Model A"]["pnl"]["win_rate"]
    dwr  = dp["win_rate"]
    baw  = test_metrics["Model A"]["pnl"]["avg_win"]
    daw  = dp["avg_win"]
    bsh  = test_metrics["Model A"]["pnl"]["sharpe"]
    dsh  = dp["sharpe"]
    print(f"  {'Win rate':<30}  {bwr*100:>24.1f}%  {dwr*100:>14.1f}%")
    print(f"  {'Total PnL (pts)':<30}  {bp:>+25.1f}  {dp['total_pnl']:>+15.1f}")
    print(f"  {'Avg win (pts)':<30}  {baw:>+25.1f}  {daw:>+15.1f}")
    print(f"  {'Avg loss (pts)':<30}  {test_metrics['Model A']['pnl']['avg_loss']:>+25.1f}  {dp['avg_loss']:>+15.1f}")
    print(f"  {'Sharpe-like':<30}  {bsh:>25.3f}  {dsh:>15.3f}")
    print(f"  {'Max drawdown (pts)':<30}  {test_metrics['Model A']['pnl']['max_drawdown']:>+25.1f}  {dp['max_drawdown']:>+15.1f}")
    high_acc_dual = accuracy_score(
        y_te[np.array([c == "HIGH" for c in _dual_model_predict(model_ace, model_ape, X_te)[3]])],
        _dual_model_predict(model_ace, model_ape, X_te)[0][np.array([c == "HIGH" for c in _dual_model_predict(model_ace, model_ape, X_te)[3]])],
    ) if dual_metrics["agree_rate"] > 0 else 0.0
    print(f"  {'HIGH-conf win rate':<30}  {ens_metrics.get('high_pnl', {}).get('win_rate', 0)*100:>24.1f}%  "
          f"{dual_metrics.get('high_pnl', {}).get('win_rate', 0)*100:>14.1f}%")
    print(f"  {'HIGH-conf count':<30}  "
          f"{ens_metrics.get('agree_rate',0)*163:>24.0f}  "
          f"{dual_metrics['agree_rate']*163:>14.0f}")
    print(f"\n  Next step: --predict YYYY-MM-DD for live inference")


# ── Walk-Forward Cross-Validation ─────────────────────────────────────────────

def walk_forward_cv() -> dict:
    """
    Expanding-window walk-forward CV — 10 quarterly folds, Jan 2023 → Apr 2026.
    Trains dual CE/PE classifiers with fixed hyperparameters on each growing
    training window; every test date appears in exactly one fold.

    Usage:
        python -m btst_engine.ml_predictor --walkforward
    """
    _REPORTS_DIR = _ROOT / "data" / "reports"
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _WF_PREDS  = _ML_DIR / "walkforward_predictions.csv"
    _WF_REPORT = _REPORTS_DIR / "walkforward_3yr_report.txt"

    # Fixed hyperparameters — best params from full-dataset GridSearchCV.
    # Using fixed params keeps each fold O(seconds) instead of O(minutes).
    _ACE_P = dict(max_depth=3, n_estimators=200, learning_rate=0.1,
                  subsample=0.7, colsample_bytree=0.9)
    _APE_P = dict(max_depth=3, n_estimators=100, learning_rate=0.01,
                  subsample=0.9, colsample_bytree=0.9)
    _CONF_THRESH = 0.55

    # ── Load merged dataset ───────────────────────────────────────────────────
    print("\n[WF-0] Loading feature + label data...", flush=True)
    feat_df  = build_feature_matrix(rebuild=False)
    label_df = build_labels(feat_df, rebuild=False)
    merged   = feat_df.merge(label_df, on="date", how="inner").dropna()
    merged["_dt"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values("_dt").reset_index(drop=True)

    non_feat     = {"date", "_dt", "label", "move_pts", "move_pct"}
    feature_cols = [c for c in merged.columns if c not in non_feat]
    print(f"  Dataset: {len(merged)} rows  |  {len(feature_cols)} features  "
          f"({merged['date'].iloc[0]} to {merged['date'].iloc[-1]})\n", flush=True)

    # ── Fold boundary definitions ─────────────────────────────────────────────
    # Each tuple: (train_end, test_start, test_end)
    FOLDS = [
        (date(2023,  9, 30), date(2023, 10,  1), date(2023, 12, 31)),  # F1
        (date(2023, 12, 31), date(2024,  1,  1), date(2024,  3, 31)),  # F2
        (date(2024,  3, 31), date(2024,  4,  1), date(2024,  6, 30)),  # F3
        (date(2024,  6, 30), date(2024,  7,  1), date(2024,  9, 30)),  # F4
        (date(2024,  9, 30), date(2024, 10,  1), date(2024, 12, 31)),  # F5
        (date(2024, 12, 31), date(2025,  1,  1), date(2025,  3, 31)),  # F6
        (date(2025,  3, 31), date(2025,  4,  1), date(2025,  6, 30)),  # F7
        (date(2025,  6, 30), date(2025,  7,  1), date(2025,  9, 30)),  # F8
        (date(2025,  9, 30), date(2025, 10,  1), date(2025, 12, 31)),  # F9
        (date(2025, 12, 31), date(2026,  1,  1), date(2026,  4,  8)),  # F10
    ]

    # ── Stats helper ──────────────────────────────────────────────────────────
    def _stats(pnls: np.ndarray, correct: np.ndarray) -> dict:
        if len(pnls) == 0:
            return {"n": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_win": 0.0,
                    "avg_loss": 0.0, "max_dd": 0.0, "sharpe": 0.0,
                    "wl": float("nan")}
        cumul = np.cumsum(pnls)
        wins  = pnls > 0
        loses = pnls < 0
        peak = cumul[0]; max_dd = 0.0
        for c in cumul:
            if c > peak: peak = c
            dd = peak - c
            if dd > max_dd: max_dd = dd
        sharpe = float(np.mean(pnls) / np.std(pnls)) if np.std(pnls) > 0 else 0.0
        wl = (abs(pnls[wins].mean()) / abs(pnls[loses].mean())
              if wins.any() and loses.any() else float("nan"))
        return {
            "n":         len(pnls),
            "win_rate":  float(correct.mean()),
            "total_pnl": float(cumul[-1]),
            "avg_win":   float(pnls[wins].mean()) if wins.any() else 0.0,
            "avg_loss":  float(pnls[loses].mean()) if loses.any() else 0.0,
            "max_dd":    float(max_dd),
            "sharpe":    sharpe,
            "wl":        wl,
        }

    # ── Run folds ─────────────────────────────────────────────────────────────
    print("=" * 64)
    print("  DATASHOTS — Walk-Forward CV — running 10 folds")
    print("=" * 64, flush=True)

    all_records: List[dict] = []
    fold_rows:   List[dict] = []

    for fold_num, (train_end, test_start, test_end) in enumerate(FOLDS, 1):
        te_ts  = pd.Timestamp(train_end)
        tst_ts = pd.Timestamp(test_start)
        tt_ts  = pd.Timestamp(test_end)

        tr = merged[merged["_dt"] <= te_ts].reset_index(drop=True)
        te = merged[(merged["_dt"] >= tst_ts) & (merged["_dt"] <= tt_ts)].reset_index(drop=True)

        if len(tr) < 30 or len(te) == 0:
            print(f"  Fold {fold_num:2d}  SKIPPED  (train={len(tr)}, test={len(te)})",
                  flush=True)
            continue

        X_tr  = tr[feature_cols].values.astype(float)
        y_tr  = tr["label"].values.astype(int)
        X_te  = te[feature_cols].values.astype(float)
        y_te  = te["label"].values.astype(int)
        mv_te = te["move_pts"].values.astype(float)
        dt_te = te["date"].tolist()

        print(f"\n  ── Fold {fold_num:2d}  "
              f"Train 2023-01 → {train_end}  (n={len(tr)})  "
              f"Test {test_start} → {test_end}  (n={len(te)})", flush=True)

        # Train ACE
        n_neg   = int((y_tr == 0).sum())
        n_pos   = int((y_tr == 1).sum())
        spw_ace = n_neg / max(n_pos, 1)
        ace = xgb.XGBClassifier(
            objective="binary:logistic", scale_pos_weight=spw_ace,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0, **_ACE_P,
        )
        ace.fit(X_tr, y_tr)
        print(f"     ACE trained (spw={spw_ace:.2f})  ", end="", flush=True)

        # Train APE — flipped labels
        y_tr_f  = 1 - y_tr
        spw_ape = int((y_tr_f == 0).sum()) / max(int((y_tr_f == 1).sum()), 1)
        ape = xgb.XGBClassifier(
            objective="binary:logistic", scale_pos_weight=spw_ape,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0, **_APE_P,
        )
        ape.fit(X_tr, y_tr_f)
        print(f"APE trained (spw={spw_ape:.2f})  ", end="", flush=True)

        # Predict
        ce_c  = ace.predict_proba(X_te)[:, 1]
        pe_c  = ape.predict_proba(X_te)[:, 1]
        preds = (ce_c >= pe_c).astype(int)           # 1=CE_BUY, 0=PE_BUY
        wconf = np.where(preds == 1, ce_c, pe_c)
        ec    = ["HIGH" if c >= _CONF_THRESH else "LOW" for c in wconf]

        pnls    = np.where(preds == 1, mv_te, -mv_te)
        correct = (preds == y_te).astype(int)
        s       = _stats(pnls, correct)

        print(f"→  win={s['win_rate']*100:.1f}%  PnL={s['total_pnl']:+.1f} pts",
              flush=True)

        for i in range(len(dt_te)):
            all_records.append({
                "fold":       fold_num,
                "date":       dt_te[i],
                "ce_conf":    round(float(ce_c[i]),  4),
                "pe_conf":    round(float(pe_c[i]),  4),
                "prediction": "CE_BUY" if preds[i] == 1 else "PE_BUY",
                "ens_conf":   ec[i],
                "actual":     "CE_BUY" if y_te[i] == 1 else "PE_BUY",
                "move_pts":   round(float(mv_te[i]), 2),
                "pnl":        round(float(pnls[i]),  2),
                "correct":    int(correct[i]),
            })

        fold_rows.append({
            "fold":       fold_num,
            "train_end":  str(train_end),
            "test_start": str(test_start),
            "test_end":   str(test_end),
            "train_size": len(tr),
            "test_size":  len(te),
            "win_rate":   s["win_rate"],
            "total_pnl":  s["total_pnl"],
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg = pd.DataFrame(all_records).sort_values("date").reset_index(drop=True)

    overall   = _stats(agg["pnl"].values, agg["correct"].values)
    high_mask = agg["ens_conf"] == "HIGH"
    high      = (_stats(agg.loc[high_mask, "pnl"].values,
                        agg.loc[high_mask, "correct"].values)
                 if high_mask.sum() > 0 else None)

    # ── Format report ─────────────────────────────────────────────────────────
    SEP = "=" * 64
    sep = "-" * 64

    L: List[str] = []
    L.append(SEP)
    L.append("  DATASHOTS — Walk-Forward Cross-Validation Report")
    L.append(f"  3 Years: {agg['date'].iloc[0]} to {agg['date'].iloc[-1]}")
    L.append("  Method: Expanding window, quarterly folds, 10 folds")
    L.append(SEP)
    L.append("")
    L.append(f"  {'Fold':<5}  {'Train size':>10}  {'Test size':>9}  {'Win%':>6}   {'PnL pts':>10}")
    L.append(f"  {'-'*48}")
    for s in fold_rows:
        L.append(
            f"  {s['fold']:<5}  {s['train_size']:>10}  {s['test_size']:>9}  "
            f"{s['win_rate']*100:>6.1f}%  {s['total_pnl']:>+10.1f}"
        )
    L.append("")
    L.append("  " + "-" * 60)
    L.append("  AGGREGATE OOS RESULTS (all 10 folds combined)")
    L.append("  " + "-" * 60)
    L.append(f"  Total OOS trading days    : {overall['n']}")
    L.append(f"  Overall win rate          : {overall['win_rate']*100:.1f}%")
    L.append(f"  Avg win                   : {overall['avg_win']:+.1f} pts")
    L.append(f"  Avg loss                  : {overall['avg_loss']:+.1f} pts")
    L.append(f"  Total PnL                 : {overall['total_pnl']:+.1f} pts")
    L.append(f"  Max drawdown              : -{overall['max_dd']:.1f} pts")
    L.append(f"  Sharpe                    : {overall['sharpe']:.3f}")
    wl_str = f"{overall['wl']:.2f}" if not np.isnan(overall["wl"]) else "N/A"
    L.append(f"  Win/loss ratio            : {wl_str}")
    L.append("")
    if high:
        hp = high["n"] / overall["n"] * 100
        L.append(f"  HIGH confidence only      : {high['n']} days ({hp:.0f}% of total)")
        L.append(f"  HIGH win rate             : {high['win_rate']*100:.1f}%")
        L.append(f"  HIGH total PnL            : {high['total_pnl']:+.1f} pts")
        L.append(f"  HIGH max drawdown         : -{high['max_dd']:.1f} pts")
        L.append(f"  HIGH Sharpe               : {high['sharpe']:.3f}")
        L.append("")
    L.append(SEP)

    report = "\n".join(L)
    print("\n\n" + report)

    # ── Save ──────────────────────────────────────────────────────────────────
    agg.to_csv(_WF_PREDS, index=False)
    _WF_REPORT.write_text(report, encoding="utf-8")
    print(f"\n  Predictions → {_WF_PREDS}")
    print(f"  Report      → {_WF_REPORT}\n")

    return {"overall": overall, "high": high, "fold_rows": fold_rows}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Force UTF-8 line-buffered stdout: needed for box-drawing chars on Windows
    # and to make progress prints appear immediately in background shells.
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
    logging.getLogger("btst_engine.expiry_manager").setLevel(logging.ERROR)
    logging.getLogger("dhanhq").setLevel(logging.CRITICAL)

    args = sys.argv[1:]

    if "--train" in args:
        rebuild = "--rebuild" in args
        train_pipeline(rebuild=rebuild)

    elif "--walkforward" in args:
        walk_forward_cv()

    elif "--predict" in args:
        idx = args.index("--predict")
        if idx + 1 >= len(args):
            print("[ERR] --predict requires a YYYY-MM-DD argument", file=sys.stderr)
            sys.exit(1)
        date_str = args[idx + 1]
        try:
            result = predict_today(date_str)
        except Exception as exc:
            print(f"[ERR] {exc}", file=sys.stderr)
            sys.exit(1)

        sep = "=" * 64
        print(f"\n{sep}")
        print(f"  ML Prediction  —  {result['date']}")
        print(sep)
        print(f"\n  Signal            : {result['signal']}")
        print(f"  Confidence        : {result['confidence']:.1%}  (winning model confidence)")
        print(f"  CE_BUY conf       : {result['ce_conf']:.1%}  |  PE_BUY conf: {result['pe_conf']:.1%}")
        print(f"  Expected move     : {result['expected_move_pts']:+.1f} pts  (model B)")
        print(f"  Ensemble          : {result['ensemble_confidence']} confidence")
        print(f"\n  Top 5 driving features:")
        for feat, val in result["top_features"].items():
            print(f"    {feat:<40} {val:>12.4f}")
        print(f"\n{sep}\n")

    else:
        print("Usage:")
        print("  python -m btst_engine.ml_predictor --train            # full pipeline")
        print("  python -m btst_engine.ml_predictor --train --rebuild  # force rebuild features")
        print("  python -m btst_engine.ml_predictor --walkforward      # 10-fold walk-forward CV")
        print("  python -m btst_engine.ml_predictor --predict YYYY-MM-DD")
        sys.exit(1)
