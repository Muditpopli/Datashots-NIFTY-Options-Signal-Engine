"""
intraday_engine/ml_predictor.py — Intraday Direction Predictor

Using the first 15 minutes of NIFTY + BANKNIFTY options data (9:15–9:30 AM),
predict whether NIFTY will close higher or lower at 3:00 PM.

Output: BULLISH or BEARISH — trader takes a hedged spread at 9:30 AM, exits 3 PM.

Steps
-----
  1  Build feature matrix v1 from rolling options cache (9:15–9:30 opening range)
  2  Build labels: NIFTY 15:00 close > 9:30 close → BULLISH (1), else BEARISH (0)
  3  Walk-forward CV — 10 expanding quarterly folds, dual Model-Bull / Model-Bear
  4  Walk-forward aggregate report (printed + saved)
  5  Feature importance — top 15 per model (last fold)
  6  Save models, predictions, report

  V2 extension (--walkforward-v2):
  Enriches v1 features with 7 previous-session context features, reruns same pipeline.

No BTST module imports — all file/data utilities are re-implemented here.
Only greeks.py (project root) is shared.

Usage
-----
  python -m intraday_engine.ml_predictor --walkforward          # v1 features
  python -m intraday_engine.ml_predictor --walkforward-v2       # v2 features
  python -m intraday_engine.ml_predictor --walkforward --rebuild # force feature rebuild
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT        = Path(__file__).parent.parent
_CACHE_DIR   = _ROOT / "data" / "backtest" / "cache" / "rolling_options"
_ML_DIR      = _ROOT / "data" / "ml"
_REPORTS_DIR = _ROOT / "data" / "reports"

# v1 outputs
_FEAT_CSV   = _ML_DIR / "intraday_features_raw.csv"
_LABEL_CSV  = _ML_DIR / "intraday_labels.csv"
_MODEL_BULL = _ML_DIR / "intraday_model_bull.json"
_MODEL_BEAR = _ML_DIR / "intraday_model_bear.json"
_FEAT_COLS  = _ML_DIR / "intraday_feature_columns.json"
_WF_PREDS   = _ML_DIR / "intraday_walkforward_predictions.csv"
_WF_REPORT  = _REPORTS_DIR / "intraday_walkforward_report.txt"

# v2 outputs
_FEAT_V2_CSV   = _ML_DIR / "intraday_features_v2.csv"
_MODEL_BULL_V2 = _ML_DIR / "intraday_model_bull_v2.json"
_MODEL_BEAR_V2 = _ML_DIR / "intraday_model_bear_v2.json"
_FEAT_COLS_V2  = _ML_DIR / "intraday_feature_columns_v2.json"
_WF_PREDS_V2   = _ML_DIR / "intraday_walkforward_predictions_v2.csv"
_WF_REPORT_V2  = _REPORTS_DIR / "intraday_walkforward_report_v2.txt"

# ── Time constants ─────────────────────────────────────────────────────────────

_OR_START_H,  _OR_START_M  = 9,  15
_OR_END_H,    _OR_END_M    = 9,  31    # exclusive → captures 9:15..9:30 (16 bars)
_OR_EXPECTED_BARS           = 16
_CLOSE_H, _CLOSE_M         = 15, 0
_MIN_BAR_FRAC               = 0.70

# ── Strike offset lists ────────────────────────────────────────────────────────

_ALL_OFFSETS: List[str] = (
    [f"ATM-{i}" for i in range(10, 0, -1)] + ["ATM"] +
    [f"ATM+{i}" for i in range(1, 11)]
)
_GREEK_OFFSETS: List[str] = (
    [f"ATM-{i}" for i in range(3, 0, -1)] + ["ATM"] +
    [f"ATM+{i}" for i in range(1, 4)]
)

# ── Walk-forward fold boundaries ───────────────────────────────────────────────

_FOLDS = [
    (date(2023,  9, 30), date(2023, 10,  1), date(2023, 12, 31)),
    (date(2023, 12, 31), date(2024,  1,  1), date(2024,  3, 31)),
    (date(2024,  3, 31), date(2024,  4,  1), date(2024,  6, 30)),
    (date(2024,  6, 30), date(2024,  7,  1), date(2024,  9, 30)),
    (date(2024,  9, 30), date(2024, 10,  1), date(2024, 12, 31)),
    (date(2024, 12, 31), date(2025,  1,  1), date(2025,  3, 31)),
    (date(2025,  3, 31), date(2025,  4,  1), date(2025,  6, 30)),
    (date(2025,  6, 30), date(2025,  7,  1), date(2025,  9, 30)),
    (date(2025,  9, 30), date(2025, 10,  1), date(2025, 12, 31)),
    (date(2025, 12, 31), date(2026,  1,  1), date(2026,  4,  8)),
]

# Fixed XGBoost hyperparameters for walk-forward (avoids per-fold grid search)
_BULL_P = dict(max_depth=3, n_estimators=200, learning_rate=0.05,
               subsample=0.8, colsample_bytree=0.8)
_BEAR_P = dict(max_depth=3, n_estimators=200, learning_rate=0.05,
               subsample=0.8, colsample_bytree=0.8)

_HIGH_GAP = 0.10

# v2 new feature names (overnight_gap_pct = gap_open_pct already in v1)
_V2_NEW_TAGS: List[str] = [
    "prev_day_close_vs_open_pct",
    "prev_day_930_to_close_pct",
    "prev_day_range_pct",
    "prev_day_pcr_close",
    "prev_day_bnf_pcr_close",
    "prev_5day_realized_vol",
    "prev_day_label",
]


# ── File utilities ─────────────────────────────────────────────────────────────

_FILE_IDX: Dict[str, Dict[str, list]] = {}


def _ts_ist(d: date, h: int, m: int) -> int:
    day_utc = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
    return day_utc + h * 3600 + m * 60 - 19800


def _find_file(
    index: str, date_str: str,
    expiry_type: str, e_code: str,
    offset: str, opt_type: str,
) -> Optional[Path]:
    index_dir = _CACHE_DIR / index
    suffix    = f"{expiry_type}_{e_code}_{offset}_{opt_type}.json"

    daily = index_dir / f"{date_str}_{date_str}_{suffix}"
    if daily.exists():
        return daily

    if index not in _FILE_IDX:
        _FILE_IDX[index] = {}
        for fp in sorted(index_dir.iterdir()):
            nm = fp.name
            if not nm.endswith(".json") or len(nm) < 22 or nm[10] != "_":
                continue
            suf    = nm[22:]
            from_d = nm[:10]
            to_d   = nm[11:21]
            if from_d == to_d:
                continue
            _FILE_IDX[index].setdefault(suf, []).append((from_d, to_d, fp))

    for from_d, to_d, fp in _FILE_IDX[index].get(suffix, []):
        if from_d <= date_str <= to_d:
            return fp
    return None


def _find_file_e1(index: str, date_str: str, offset: str, opt_type: str) -> Optional[Path]:
    fp = _find_file(index, date_str, "WEEK", "E1", offset, opt_type)
    return fp if fp is not None else _find_file(index, date_str, "MONTH", "E1", offset, opt_type)


def _load_leg(path: Path, opt_type: str) -> Optional[Dict]:
    try:
        with open(path) as fh:
            raw = json.load(fh)
        key = "ce" if opt_type == "CALL" else "pe"
        leg = raw.get("data", {}).get(key)
        if not isinstance(leg, dict):
            return None
        for k in ("timestamp", "iv", "oi", "spot"):
            if not leg.get(k):
                return None
        return leg
    except Exception:
        return None


def _slice_window(leg: Dict, ts_start: int, ts_end: int) -> Optional[Dict]:
    ts_arr = leg.get("timestamp") or []
    idxs   = [i for i, t in enumerate(ts_arr) if ts_start <= t < ts_end]
    if not idxs:
        return None

    def _pick(k: str) -> list:
        arr = leg.get(k) or []
        return [arr[i] for i in idxs if i < len(arr)]

    return {
        "timestamp": _pick("timestamp"),
        "iv":        _pick("iv"),
        "oi":        _pick("oi"),
        "strike":    _pick("strike"),
        "spot":      _pick("spot"),
        "close":     _pick("close"),
    }


# ── Available date discovery ──────────────────────────────────────────────────

def _collect_available_dates() -> List[date]:
    index_dir = _CACHE_DIR / "NIFTY"
    if not index_dir.exists():
        return []
    dates_set: set = set()
    for fp in index_dir.iterdir():
        nm = fp.name
        if not nm.endswith("_ATM_CALL.json"):
            continue
        if "_WEEK_E1_ATM" not in nm and "_MONTH_E1_ATM" not in nm:
            continue
        try:
            from_d = date.fromisoformat(nm[:10])
            to_d   = date.fromisoformat(nm[11:21])
        except ValueError:
            continue
        cur = from_d
        while cur <= to_d:
            if cur.weekday() < 5:
                dates_set.add(cur)
            cur += timedelta(days=1)
    return sorted(dates_set)


# ── Expiry / DTE resolution ────────────────────────────────────────────────────

def _resolve_expiry_date(index: str, date_str: str) -> Optional[date]:
    for et in ("WEEK", "MONTH"):
        fp = _find_file(index, date_str, et, "E1", "ATM", "CALL")
        if fp is not None:
            try:
                return date.fromisoformat(fp.name[11:21])
            except ValueError:
                continue
    return None


def _resolve_dte(index: str, date_str: str, signal_date: date) -> int:
    exp = _resolve_expiry_date(index, date_str)
    return max((exp - signal_date).days, 1) if exp else 5


# ── Opening range data loading ─────────────────────────────────────────────────

def _load_or_slices(index: str, date_str: str, ts_start: int, ts_end: int) -> Dict:
    slices: Dict = {"CALL": {}, "PUT": {}}
    for ot in ("CALL", "PUT"):
        for offset in _ALL_OFFSETS:
            fp  = _find_file_e1(index, date_str, offset, ot)
            leg = _load_leg(fp, ot) if fp else None
            slices[ot][offset] = _slice_window(leg, ts_start, ts_end) if leg else None
    return slices


def _n_bars(slices: Dict) -> int:
    atm = slices.get("CALL", {}).get("ATM") or slices.get("PUT", {}).get("ATM")
    return len(atm.get("timestamp", [])) if atm else 0


# ── Metric computation helpers ─────────────────────────────────────────────────

def _spot_ohlc(slices: Dict) -> Dict[str, float]:
    atm = slices.get("CALL", {}).get("ATM") or slices.get("PUT", {}).get("ATM")
    if not atm or not atm.get("spot"):
        return {"open": 0.0, "close": 0.0, "high": 0.0, "low": 0.0}
    sp = atm["spot"]
    return {"open": float(sp[0]), "close": float(sp[-1]),
            "high": float(max(sp)), "low": float(min(sp))}


def _oi_totals(slices: Dict) -> Tuple[float, float, float, float]:
    co = cc = po = pc = 0.0
    for offset in _ALL_OFFSETS:
        cs = slices.get("CALL", {}).get(offset)
        if cs and cs.get("oi"):
            oi = cs["oi"]; co += float(oi[0]); cc += float(oi[-1])
        ps = slices.get("PUT", {}).get(offset)
        if ps and ps.get("oi"):
            oi = ps["oi"]; po += float(oi[0]); pc += float(oi[-1])
    return co, cc, po, pc


def _greeks_net_delta(slices: Dict, spot_close: float, dte: int) -> float:
    if spot_close <= 0:
        return 0.0
    try:
        from greeks import GreeksCalculator
    except ImportError:
        return 0.0
    dte_safe = max(dte, 1)
    nd = 0.0
    for offset in _GREEK_OFFSETS:
        for ot, bs in (("CALL", "CE"), ("PUT", "PE")):
            sl = slices.get(ot, {}).get(offset)
            if not sl:
                continue
            iv_arr  = sl.get("iv")     or []
            oi_arr  = sl.get("oi")     or []
            str_arr = sl.get("strike") or []
            if not iv_arr or not oi_arr or not str_arr:
                continue
            iv_pct = float(iv_arr[-1])
            oi_val = float(oi_arr[-1])
            strike = float(str_arr[-1])
            if iv_pct <= 0 or oi_val <= 0 or strike <= 0:
                continue
            try:
                from greeks import GreeksCalculator
                g = GreeksCalculator.calculate_greeks(
                    spot=spot_close, strike=strike, dte=dte_safe,
                    iv=iv_pct / 100.0, option_type=bs,
                )
                nd += g["delta"] * oi_val
            except Exception:
                pass
    return nd


def _iv_skew(slices: Dict) -> float:
    cs = slices.get("CALL", {}).get("ATM")
    ps = slices.get("PUT",  {}).get("ATM")
    civ = float(cs["iv"][-1]) if cs and cs.get("iv") else 0.0
    piv = float(ps["iv"][-1]) if ps and ps.get("iv") else 0.0
    return piv - civ


# ── Spot / OI at a specific time ───────────────────────────────────────────────

def _spot_at_or_before(index: str, d: date, h: int, m: int) -> float:
    """Spot price at or just before h:m IST (must be within 5 min). Returns 0 on failure."""
    date_str  = d.strftime("%Y-%m-%d")
    fp        = _find_file_e1(index, date_str, "ATM", "CALL")
    if fp is None:
        return 0.0
    leg = _load_leg(fp, "CALL")
    if not leg or not leg.get("spot") or not leg.get("timestamp"):
        return 0.0
    ts_target = _ts_ist(d, h, m)
    ts_arr    = leg["timestamp"]
    sp_arr    = leg["spot"]
    best_ts, best_sp = -1, 0.0
    for i, t in enumerate(ts_arr):
        if t <= ts_target and t > best_ts and i < len(sp_arr) and sp_arr[i]:
            best_ts = t
            best_sp = float(sp_arr[i])
    return best_sp if best_ts >= ts_target - 300 else 0.0


def _get_full_day_high_low(index: str, d: date) -> Tuple[float, float]:
    """
    Full trading day high and low from ATM CALL spot array (9:15 to 15:00).
    Returns (high, low) or (0.0, 0.0) if data unavailable.
    """
    date_str = d.strftime("%Y-%m-%d")
    fp = _find_file_e1(index, date_str, "ATM", "CALL")
    if fp is None:
        return 0.0, 0.0
    leg = _load_leg(fp, "CALL")
    if not leg or not leg.get("spot") or not leg.get("timestamp"):
        return 0.0, 0.0

    ts_start  = _ts_ist(d, 9, 15)
    ts_end    = _ts_ist(d, 15, 1)   # include the 15:00 bar
    ts_arr    = leg["timestamp"]
    sp_arr    = leg["spot"]

    spots = [
        float(sp_arr[i])
        for i, t in enumerate(ts_arr)
        if ts_start <= t < ts_end and i < len(sp_arr) and sp_arr[i]
    ]
    return (float(max(spots)), float(min(spots))) if spots else (0.0, 0.0)


def _get_pcr_at_close(index: str, d: date) -> float:
    """
    PCR (put OI / call OI) at 15:00 IST using ATM±10 strikes.
    Uses a 5-minute window ending at 15:00; falls back to 15-minute window.
    """
    date_str = d.strftime("%Y-%m-%d")
    ts_end   = _ts_ist(d, 15, 1)     # exclusive: captures up to 15:00

    # Try narrow window first (14:56–15:00)
    ts_start = _ts_ist(d, 14, 56)
    slices   = _load_or_slices(index, date_str, ts_start, ts_end)
    _, cc, _, pc = _oi_totals(slices)

    if cc == 0:
        # Fallback: wider window (14:46–15:00)
        ts_start = _ts_ist(d, 14, 46)
        slices   = _load_or_slices(index, date_str, ts_start, ts_end)
        _, cc, _, pc = _oi_totals(slices)

    return pc / cc if cc > 0 else 0.0


# ── Step 1: Feature matrix (v1) ───────────────────────────────────────────────

def _extract_index_features(
    prefix: str, index: str, date_str: str,
    signal_date: date, ts_start: int, ts_end: int,
) -> Tuple[Optional[Dict], float]:
    slices = _load_or_slices(index, date_str, ts_start, ts_end)
    nb = _n_bars(slices)
    if nb < _OR_EXPECTED_BARS * _MIN_BAR_FRAC:
        return None, 0.0

    ohlc  = _spot_ohlc(slices)
    open_ = ohlc["open"]
    close = ohlc["close"]
    high  = ohlc["high"]
    low   = ohlc["low"]
    if open_ <= 0 or close <= 0:
        return None, 0.0

    move_pct  = (close - open_) / open_ * 100
    range_pct = (high  - low)   / open_ * 100
    co, cc, po, pc = _oi_totals(slices)
    pcr   = pc / cc if cc > 0 else 0.0
    oi_cc = cc - co
    oi_pc = pc - po
    dte   = _resolve_dte(index, date_str, signal_date)
    nd    = _greeks_net_delta(slices, close, dte)
    skew  = _iv_skew(slices)

    if prefix == "nifty":
        feats: Dict = {
            "nifty_open":               open_,
            "nifty_930_close":          close,
            "nifty_930_high":           high,
            "nifty_930_low":            low,
            "nifty_930_move_pct":       move_pct,
            "nifty_930_range_pct":      range_pct,
            "nifty_pcr_930":            pcr,
            "nifty_oi_change_call_930": oi_cc,
            "nifty_oi_change_put_930":  oi_pc,
            "nifty_net_delta_930":      nd,
            "nifty_iv_skew_930":        skew,
        }
    else:
        feats = {
            "bnf_open":                open_,
            "bnf_930_close":           close,
            "bnf_930_move_pct":        move_pct,
            "bnf_pcr_930":             pcr,
            "bnf_oi_change_call_930":  oi_cc,
            "bnf_oi_change_put_930":   oi_pc,
            "bnf_net_delta_930":       nd,
            "bnf_iv_skew_930":         skew,
        }
    return feats, close


def build_feature_matrix(rebuild: bool = False) -> pd.DataFrame:
    """Step 1 — Build v1 feature matrix (25 features, 9:15-9:30 opening range)."""
    _ML_DIR.mkdir(parents=True, exist_ok=True)

    if _FEAT_CSV.exists() and not rebuild:
        print(f"[STEP 1] Loading cached features from {_FEAT_CSV}")
        df = pd.read_csv(_FEAT_CSV)
        print(f"         Shape: {df.shape}  (use --rebuild to recompute)")
        print(df.head().to_string())
        return df

    print("[STEP 1] Building intraday feature matrix from rolling options cache...", flush=True)
    print("         Opening range: 9:15–9:30 AM  |  Window: 16 one-minute bars\n", flush=True)

    candidate_dates = _collect_available_dates()
    print(f"         Candidate dates from cache: {len(candidate_dates)}", flush=True)

    rows: List[Dict] = []
    skipped         = 0
    prev_1500_close = 0.0

    for i, d in enumerate(candidate_dates):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(candidate_dates)}  kept={len(rows)}  skipped={skipped}", flush=True)

        date_str = d.strftime("%Y-%m-%d")
        ts_start = _ts_ist(d, _OR_START_H, _OR_START_M)
        ts_end   = _ts_ist(d, _OR_END_H,   _OR_END_M)

        today_1500 = _spot_at_or_before("NIFTY", d, _CLOSE_H, _CLOSE_M)

        nifty_feats, nifty_930 = _extract_index_features(
            "nifty", "NIFTY", date_str, d, ts_start, ts_end)
        if nifty_feats is None:
            if today_1500 > 0: prev_1500_close = today_1500
            skipped += 1; continue

        bnf_feats, _ = _extract_index_features(
            "bnf", "BANKNIFTY", date_str, d, ts_start, ts_end)
        if bnf_feats is None:
            if today_1500 > 0: prev_1500_close = today_1500
            skipped += 1; continue

        n_pcr   = nifty_feats["nifty_pcr_930"]
        b_pcr   = bnf_feats["bnf_pcr_930"]
        n_delta = nifty_feats["nifty_net_delta_930"]
        b_delta = bnf_feats["bnf_net_delta_930"]

        nifty_exp = _resolve_expiry_date("NIFTY", date_str)
        dte       = max((nifty_exp - d).days, 0) if nifty_exp else 5
        is_exp    = int(nifty_exp == d) if nifty_exp else 0

        nifty_open = nifty_feats["nifty_open"]
        gap_pct    = (
            (nifty_open - prev_1500_close) / prev_1500_close * 100
            if prev_1500_close > 0 else 0.0
        )

        row = {"date": date_str}
        row.update(nifty_feats)
        row.update(bnf_feats)
        row["pcr_agreement_930"]   = int((n_pcr - 1.0) * (b_pcr - 1.0) > 0)
        row["delta_agreement_930"] = int(n_delta * b_delta > 0)
        row["day_of_week"]         = d.weekday()
        row["days_to_expiry"]      = dte
        row["is_expiry_day"]       = is_exp
        row["gap_open_pct"]        = round(gap_pct, 4)
        rows.append(row)

        if today_1500 > 0:
            prev_1500_close = today_1500

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df.to_csv(_FEAT_CSV, index=False)
    print(f"\n[STEP 1] Done.")
    print(f"  Shape      : {df.shape}")
    print(f"  Date range : {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print(f"  Kept       : {len(rows)}   Skipped: {skipped}")
    print(f"  Saved to   : {_FEAT_CSV}\n")
    print(df.head().to_string())
    return df


# ── Step 2: Labels ─────────────────────────────────────────────────────────────

def build_labels(feat_df: pd.DataFrame, rebuild: bool = False) -> pd.DataFrame:
    """Step 2 — Build intraday direction labels (NIFTY 15:00 vs 9:30 close)."""
    if _LABEL_CSV.exists() and not rebuild:
        print(f"\n[STEP 2] Loading cached labels from {_LABEL_CSV}")
        df = pd.read_csv(_LABEL_CSV)
        _print_label_stats(df)
        return df

    print("\n[STEP 2] Building labels (NIFTY 15:00 close vs 9:30 close)...", flush=True)
    records: List[Dict] = []
    skipped = 0

    for _, row in feat_df.iterrows():
        d        = date.fromisoformat(row["date"])
        spot_930 = float(row["nifty_930_close"])
        if spot_930 <= 0:
            skipped += 1; continue

        spot_1500 = _spot_at_or_before("NIFTY", d, _CLOSE_H, _CLOSE_M)
        if spot_1500 <= 0:
            skipped += 1; continue

        move_pts = spot_1500 - spot_930
        records.append({
            "date":               row["date"],
            "label":              1 if move_pts > 0 else 0,
            "move_pts_930to1500": round(move_pts, 2),
            "move_pct_930to1500": round(move_pts / spot_930 * 100, 4),
        })

    df = pd.DataFrame(records)
    df.to_csv(_LABEL_CSV, index=False)
    print(f"  Saved to: {_LABEL_CSV}\n")
    _print_label_stats(df)
    return df


def _print_label_stats(df: pd.DataFrame) -> None:
    bull = df[df["label"] == 1]
    bear = df[df["label"] == 0]
    n    = len(df)
    print(f"\n  Label distribution ({n} days):")
    print(f"    BULLISH (1) : {len(bull):4d}  ({len(bull)/n*100:.1f}%)")
    print(f"    BEARISH (0) : {len(bear):4d}  ({len(bear)/n*100:.1f}%)")
    if not bull.empty:
        print(f"  Avg BULLISH move : {bull['move_pts_930to1500'].mean():+.1f} pts")
    if not bear.empty:
        print(f"  Avg BEARISH move : {bear['move_pts_930to1500'].mean():+.1f} pts")
    print(f"\n  Top 5 biggest UP moves:   ", end="")
    for _, r in bull.nlargest(5, "move_pts_930to1500").iterrows():
        print(f"  {r['date']} +{r['move_pts_930to1500']:.0f}", end="")
    print(f"\n  Top 5 biggest DOWN moves: ", end="")
    for _, r in bear.nsmallest(5, "move_pts_930to1500").iterrows():
        print(f"  {r['date']} {r['move_pts_930to1500']:.0f}", end="")
    print()


# ── Step V2: Feature matrix v2 ────────────────────────────────────────────────

def build_feature_matrix_v2(
    feat_v1: pd.DataFrame,
    label_df: pd.DataFrame,
    rebuild: bool = False,
) -> pd.DataFrame:
    """
    Extend v1 features with 7 previous-session context features.

    New features added to v1 (25 → 32 total):
      prev_day_close_vs_open_pct  — yesterday's full-day return (open→15:00)
      prev_day_930_to_close_pct   — yesterday's 9:30→15:00 move (lagged label)
      prev_day_range_pct          — yesterday's intraday high-low / open
      prev_day_pcr_close          — NIFTY ATM±10 PCR at yesterday's 15:00
      prev_day_bnf_pcr_close      — BNF ATM±10 PCR at yesterday's 15:00
      prev_5day_realized_vol      — 5-day realized vol of daily open→close returns
      prev_day_label              — was yesterday BULLISH (1) or BEARISH (0)

    Confirmed: overnight_gap_pct == gap_open_pct (already in v1 — reused as-is).

    First row (no previous day) is dropped → v2 has one fewer row than v1∩labels.
    """
    _ML_DIR.mkdir(parents=True, exist_ok=True)

    if _FEAT_V2_CSV.exists() and not rebuild:
        print(f"\n[STEP V2] Loading cached v2 features from {_FEAT_V2_CSV}")
        df = pd.read_csv(_FEAT_V2_CSV)
        print(f"  Shape: {df.shape}")
        return df

    print("\n[STEP V2] Building v2 feature matrix with 7 previous-day context features...", flush=True)
    print("  Confirmed: gap_open_pct == overnight_gap_pct — reusing existing column.\n", flush=True)

    # Merge v1 + labels to get close_1500 = 930_close + move_pts
    base = feat_v1.merge(
        label_df[["date", "label", "move_pts_930to1500", "move_pct_930to1500"]],
        on="date", how="inner",
    ).sort_values("date").reset_index(drop=True)

    base["close_1500"]   = base["nifty_930_close"] + base["move_pts_930to1500"]
    base["daily_return"] = (base["close_1500"] - base["nifty_open"]) / base["nifty_open"]

    # ── Features derivable from existing CSVs (no cache reads) ────────────────
    base["prev_day_close_vs_open_pct"] = (base["daily_return"] * 100).shift(1)
    base["prev_day_930_to_close_pct"]  = base["move_pct_930to1500"].shift(1)
    base["prev_day_label"]             = base["label"].shift(1)
    base["prev_5day_realized_vol"]     = (
        base["daily_return"].rolling(5, min_periods=3).std().shift(1) * 100
    )

    # ── Features requiring cache reads ────────────────────────────────────────
    n = len(base)
    print(f"  Cache reads: prev_day_range_pct + 2× PCR-at-close  ({n} dates)", flush=True)
    print(f"  (~85 JSON files per date — may take a few minutes)\n", flush=True)

    prev_range  = [0.0] * n
    prev_pcr_n  = [0.0] * n
    prev_pcr_b  = [0.0] * n

    for i in range(1, n):
        if i % 100 == 0:
            print(f"  {i}/{n} cache reads done...", flush=True)

        prev_row  = base.iloc[i - 1]
        prev_d    = date.fromisoformat(prev_row["date"])
        prev_open = float(prev_row["nifty_open"])

        # Full-day high/low for range (1 file per index — cheap)
        high, low = _get_full_day_high_low("NIFTY", prev_d)
        if prev_open > 0 and high > 0 and low > 0:
            prev_range[i] = (high - low) / prev_open * 100

        # PCR at 15:00 (42 files per index — heavier; uses OS file cache)
        prev_pcr_n[i] = _get_pcr_at_close("NIFTY",     prev_d)
        prev_pcr_b[i] = _get_pcr_at_close("BANKNIFTY", prev_d)

    base["prev_day_range_pct"]     = prev_range
    base["prev_day_pcr_close"]     = prev_pcr_n
    base["prev_day_bnf_pcr_close"] = prev_pcr_b

    # Drop first row — no previous day data
    base = base.iloc[1:].reset_index(drop=True)

    # Drop temporary / label columns from the feature frame
    drop_cols = ["close_1500", "daily_return", "label",
                 "move_pts_930to1500", "move_pct_930to1500"]
    base = base.drop(columns=[c for c in drop_cols if c in base.columns])

    # NaN check
    print(f"\n  NaN check for new features:")
    total_nan = 0
    for col in _V2_NEW_TAGS:
        n_nan = int(base[col].isnull().sum())
        total_nan += n_nan
        status = "OK" if n_nan == 0 else f"WARNING {n_nan} NaN"
        print(f"    {col:<40}  {status}")

    if total_nan > 0:
        base[_V2_NEW_TAGS] = base[_V2_NEW_TAGS].fillna(0.0)
        print(f"  Filled {total_nan} NaN values with 0.0")

    base.to_csv(_FEAT_V2_CSV, index=False)
    print(f"\n  v2 shape      : {base.shape}")
    print(f"  Date range    : {base['date'].iloc[0]} to {base['date'].iloc[-1]}")
    print(f"  New features  : {_V2_NEW_TAGS}")
    print(f"  Saved to      : {_FEAT_V2_CSV}\n")
    return base


# ── Stats helper ───────────────────────────────────────────────────────────────

def _stats(pnls: np.ndarray, correct: np.ndarray) -> dict:
    if len(pnls) == 0:
        return {"n": 0, "win_rate": 0.0, "total_pnl": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "max_dd": 0.0,
                "sharpe": 0.0, "wl": float("nan")}
    cumul  = np.cumsum(pnls)
    wins   = pnls > 0
    loses  = pnls < 0
    peak   = cumul[0]; max_dd = 0.0
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
        "avg_win":   float(pnls[wins].mean())  if wins.any()  else 0.0,
        "avg_loss":  float(pnls[loses].mean()) if loses.any() else 0.0,
        "max_dd":    float(max_dd),
        "sharpe":    sharpe,
        "wl":        wl,
    }


# ── Dual model prediction ──────────────────────────────────────────────────────

def _dual_predict(
    bull_model, bear_model, X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    bull_conf = bull_model.predict_proba(X)[:, 1]
    bear_conf = bear_model.predict_proba(X)[:, 1]
    preds     = (bull_conf >= bear_conf).astype(int)
    gap       = np.abs(bull_conf - bear_conf)
    conf_lbls = ["HIGH" if g > _HIGH_GAP else "LOW" for g in gap]
    return preds, bull_conf, bear_conf, conf_lbls


# ── Core walk-forward engine (shared by v1 and v2) ────────────────────────────

def _run_walk_forward(
    feat_df:          pd.DataFrame,
    label_df:         pd.DataFrame,
    model_bull_path:  Path,
    model_bear_path:  Path,
    feat_cols_path:   Path,
    preds_path:       Path,
    report_path:      Path,
    v_tag:            str = "v1",
    new_feature_tags: Optional[List[str]] = None,
) -> None:
    """
    Generic expanding-window walk-forward CV.
    Trains dual Model-Bull / Model-Bear per fold; HIGH when |bull_conf-bear_conf| > 0.10.
    """
    _ML_DIR.mkdir(parents=True, exist_ok=True)
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[WF-{v_tag}] Merging features + labels...", flush=True)

    label_cols = ["label", "move_pts_930to1500", "move_pct_930to1500"]
    merged = feat_df.merge(
        label_df[["date"] + label_cols], on="date", how="inner"
    ).dropna()
    merged["_dt"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values("_dt").reset_index(drop=True)

    non_feat     = {"date", "_dt", "label", "move_pts_930to1500", "move_pct_930to1500"}
    feature_cols = [c for c in merged.columns if c not in non_feat]

    print(f"  Dataset  : {len(merged)} rows  |  {len(feature_cols)} features")
    print(f"  Range    : {merged['date'].iloc[0]} to {merged['date'].iloc[-1]}\n", flush=True)

    SEP = "=" * 64
    print(SEP)
    print(f"  DATASHOTS — Intraday Engine ({v_tag.upper()}) — Walk-Forward CV")
    print(f"  10 expanding quarterly folds")
    print(SEP, flush=True)

    all_records: List[Dict] = []
    fold_rows:   List[Dict] = []
    last_bull = last_bear = None

    for fold_num, (train_end, test_start, test_end) in enumerate(_FOLDS, 1):
        te_ts  = pd.Timestamp(train_end)
        tst_ts = pd.Timestamp(test_start)
        tt_ts  = pd.Timestamp(test_end)

        tr = merged[merged["_dt"] <= te_ts].reset_index(drop=True)
        te = merged[(merged["_dt"] >= tst_ts) & (merged["_dt"] <= tt_ts)].reset_index(drop=True)

        if len(tr) < 30 or len(te) == 0:
            print(f"  Fold {fold_num:2d}  SKIPPED  (train={len(tr)}, test={len(te)})", flush=True)
            continue

        X_tr  = tr[feature_cols].values.astype(float)
        y_tr  = tr["label"].values.astype(int)
        X_te  = te[feature_cols].values.astype(float)
        y_te  = te["label"].values.astype(int)
        mv_te = te["move_pts_930to1500"].values.astype(float)
        dt_te = te["date"].tolist()

        print(f"\n  ── Fold {fold_num:2d}  "
              f"Train 2023-01 → {train_end}  (n={len(tr)})  "
              f"Test {test_start} → {test_end}  (n={len(te)})", flush=True)

        n_neg = int((y_tr == 0).sum()); n_pos = int((y_tr == 1).sum())
        spw   = n_neg / max(n_pos, 1)
        bull  = xgb.XGBClassifier(
            objective="binary:logistic", scale_pos_weight=spw,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0, **_BULL_P,
        )
        bull.fit(X_tr, y_tr)

        y_tr_f  = 1 - y_tr
        n_neg_b = int((y_tr_f == 0).sum()); n_pos_b = int((y_tr_f == 1).sum())
        spw_b   = n_neg_b / max(n_pos_b, 1)
        bear    = xgb.XGBClassifier(
            objective="binary:logistic", scale_pos_weight=spw_b,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0, **_BEAR_P,
        )
        bear.fit(X_tr, y_tr_f)

        print(f"     Bull (spw={spw:.2f})  Bear (spw={spw_b:.2f})  ", end="", flush=True)

        preds, bull_c, bear_c, conf_lbls = _dual_predict(bull, bear, X_te)
        pnls    = np.where(preds == 1, mv_te, -mv_te)
        correct = (preds == y_te).astype(int)
        s       = _stats(pnls, correct)
        print(f"→  win={s['win_rate']*100:.1f}%  PnL={s['total_pnl']:+.1f} pts", flush=True)

        for i in range(len(dt_te)):
            all_records.append({
                "fold":       fold_num,
                "date":       dt_te[i],
                "bull_conf":  round(float(bull_c[i]), 4),
                "bear_conf":  round(float(bear_c[i]), 4),
                "prediction": "BULLISH" if preds[i] == 1 else "BEARISH",
                "confidence": "HIGH" if conf_lbls[i] == "HIGH" else "LOW",
                "actual":     "BULLISH" if y_te[i] == 1 else "BEARISH",
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
        last_bull = bull
        last_bear = bear

    if not all_records:
        print("\n[WF] No OOS records — check data availability.", flush=True)
        return

    agg       = pd.DataFrame(all_records).sort_values("date").reset_index(drop=True)
    overall   = _stats(agg["pnl"].values, agg["correct"].values)
    high_mask = agg["confidence"] == "HIGH"
    high      = (
        _stats(agg.loc[high_mask, "pnl"].values, agg.loc[high_mask, "correct"].values)
        if high_mask.sum() > 0 else None
    )

    report = _format_report(fold_rows, overall, high, agg, v_tag)
    print("\n\n" + report)

    if last_bull is not None and last_bear is not None:
        _print_feature_importance(
            last_bull, last_bear, feature_cols, new_feature_tags
        )

    _save_all(last_bull, last_bear, feature_cols,
              feat_cols_path, model_bull_path, model_bear_path,
              agg, report, preds_path, report_path)


# ── Report formatting ──────────────────────────────────────────────────────────

def _format_report(
    fold_rows: List[Dict],
    overall: dict,
    high: Optional[dict],
    agg: pd.DataFrame,
    v_tag: str = "v1",
) -> str:
    SEP = "=" * 64
    L: List[str] = []
    L.append(SEP)
    L.append(f"  DATASHOTS — Intraday Direction Engine — Walk-Forward Report ({v_tag.upper()})")
    L.append(f"  Dates: {agg['date'].iloc[0]}  to  {agg['date'].iloc[-1]}")
    L.append("  Method: Dual Model-Bull/Bear, expanding window, 10 folds")
    if v_tag == "v1":
        L.append("  Features: 25 opening-range features (9:15-9:30 window)")
    else:
        L.append("  Features: 32 features (25 opening-range + 7 previous-day context)")
    L.append(SEP)
    L.append("")
    L.append(f"  {'Fold':<5}  {'Train size':>10}  {'Test size':>9}  {'Win%':>7}   {'PnL pts':>10}")
    L.append(f"  {'-'*50}")
    for s in fold_rows:
        L.append(
            f"  {s['fold']:<5}  {s['train_size']:>10}  {s['test_size']:>9}  "
            f"{s['win_rate']*100:>6.1f}%  {s['total_pnl']:>+11.1f}"
        )
    L.append("")
    L.append("  " + "-" * 60)
    L.append("  AGGREGATE OOS RESULTS")
    L.append("  " + "-" * 60)
    L.append(f"  Total OOS days        : {overall['n']}")
    L.append(f"  Overall win rate       : {overall['win_rate']*100:.1f}%")
    L.append(f"  Avg win                : {overall['avg_win']:+.1f} pts")
    L.append(f"  Avg loss               : {overall['avg_loss']:+.1f} pts")
    L.append(f"  Total PnL              : {overall['total_pnl']:+.1f} pts")
    L.append(f"  Max drawdown           : -{overall['max_dd']:.1f} pts")
    L.append(f"  Sharpe                 : {overall['sharpe']:.3f}")
    wl_str = f"{overall['wl']:.2f}" if not np.isnan(overall["wl"]) else "N/A"
    L.append(f"  Win/loss ratio         : {wl_str}")
    L.append("")
    if high and high["n"] > 0:
        hp = high["n"] / overall["n"] * 100
        L.append(f"  HIGH confidence subset : {high['n']} days ({hp:.0f}%)")
        L.append(f"  HIGH win rate           : {high['win_rate']*100:.1f}%")
        L.append(
            f"  HIGH avg win / avg loss : {high['avg_win']:+.1f} / {high['avg_loss']:+.1f}"
        )
        L.append(f"  HIGH total PnL          : {high['total_pnl']:+.1f}")
        L.append(f"  HIGH max drawdown       : -{high['max_dd']:.1f} pts")
        L.append(f"  HIGH Sharpe             : {high['sharpe']:.3f}")
        L.append("")
    L.append("  " + "-" * 60)
    L.append("  HONESTY NOTE")
    L.append("  " + "-" * 60)
    wr = overall["win_rate"]
    if wr < 0.52:
        if v_tag == "v2":
            L.append("  Previous-day context does NOT fix the problem.")
            L.append("  Adding 7 prior-session features (yesterday's return,")
            L.append("  range, PCR at close, realized vol, prior direction)")
            L.append(f"  did not lift win rate above 52%. OOS win rate: {wr*100:.1f}%.")
            L.append("  Intraday direction prediction via this feature set")
            L.append("  should be abandoned.")
        else:
            L.append("  Opening range (9:15-9:30) shows NO meaningful predictive")
            L.append("  power for the 3:00 PM close in this dataset. Win rate")
            L.append(f"  {wr*100:.1f}% is below 52% — do not trade this signal.")
    elif wr < 0.54:
        L.append(f"  Win rate {wr*100:.1f}% is marginally above chance. Edge is")
        L.append("  weak; transaction costs will likely erase PnL. Caution advised.")
    else:
        L.append(f"  Win rate {wr*100:.1f}% suggests a real signal exists.")
        L.append("  Validate with live paper trading before deploying capital.")
    L.append(SEP)
    return "\n".join(L)


# ── Feature importance ─────────────────────────────────────────────────────────

def _print_feature_importance(
    bull_model,
    bear_model,
    feature_cols: List[str],
    new_feature_tags: Optional[List[str]] = None,
) -> None:
    print("\n" + "=" * 64)
    print("  STEP 5 — Feature Importance (last fold models)")
    print("=" * 64)

    bull_fi = bull_model.feature_importances_
    bear_fi = bear_model.feature_importances_

    for tag, fi in [("Model-Bull (predicts BULLISH)", bull_fi),
                    ("Model-Bear (predicts BEARISH)", bear_fi)]:
        top_idx = np.argsort(fi)[::-1][:15]
        print(f"\n  Top 15 features — {tag}:")
        print(f"  {'Rank':<5}  {'Feature':<42}  {'Importance':>10}")
        print(f"  {'-'*60}")
        for rank, idx in enumerate(top_idx, 1):
            name     = feature_cols[idx]
            is_new   = "  ← NEW" if (new_feature_tags and name in new_feature_tags) else ""
            print(f"  {rank:<5}  {name:<42}  {fi[idx]:>10.4f}{is_new}")

    if new_feature_tags:
        bull_top15 = {feature_cols[i] for i in np.argsort(bull_fi)[::-1][:15]}
        bear_top15 = {feature_cols[i] for i in np.argsort(bear_fi)[::-1][:15]}
        in_bull = sorted(bull_top15 & set(new_feature_tags))
        in_bear = sorted(bear_top15 & set(new_feature_tags))
        print(f"\n  New prev-day features in top-15 — Model-Bull : "
              f"{in_bull if in_bull else ['none']}")
        print(f"  New prev-day features in top-15 — Model-Bear : "
              f"{in_bear if in_bear else ['none']}")

    # Verdict on opening-range predictive value
    or_feats = {
        "nifty_930_move_pct", "nifty_930_range_pct", "nifty_pcr_930",
        "nifty_oi_change_call_930", "nifty_oi_change_put_930",
        "nifty_net_delta_930", "nifty_iv_skew_930", "gap_open_pct",
        "bnf_930_move_pct", "bnf_pcr_930",
    }
    bull_top5 = {feature_cols[i] for i in np.argsort(bull_fi)[::-1][:5]}
    or_in_top5 = bull_top5 & or_feats
    print(f"\n  Opening-range features in top-5 (Bull): "
          f"{sorted(or_in_top5) if or_in_top5 else 'none — likely noise'}")


# ── Save ───────────────────────────────────────────────────────────────────────

def _save_all(
    bull_model, bear_model,
    feature_cols: List[str],
    feat_cols_path: Path,
    model_bull_path: Path,
    model_bear_path: Path,
    agg: pd.DataFrame,
    report: str,
    preds_path: Path,
    report_path: Path,
) -> None:
    print("\n[STEP 6] Saving models, predictions, report...", flush=True)
    if bull_model is not None:
        bull_model.save_model(str(model_bull_path))
        print(f"  Model-Bull  → {model_bull_path}")
    if bear_model is not None:
        bear_model.save_model(str(model_bear_path))
        print(f"  Model-Bear  → {model_bear_path}")
    with feat_cols_path.open("w") as f:
        json.dump(feature_cols, f)
    print(f"  Feat cols   → {feat_cols_path}")
    agg.to_csv(preds_path, index=False)
    print(f"  Predictions → {preds_path}")
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report      → {report_path}")


# ── Public entry points ────────────────────────────────────────────────────────

def walk_forward_cv(rebuild: bool = False) -> None:
    """Walk-forward CV using v1 features (25 opening-range features)."""
    feat_df  = build_feature_matrix(rebuild=rebuild)
    label_df = build_labels(feat_df, rebuild=rebuild)
    _run_walk_forward(
        feat_df, label_df,
        _MODEL_BULL, _MODEL_BEAR, _FEAT_COLS,
        _WF_PREDS, _WF_REPORT,
        v_tag="v1",
        new_feature_tags=None,
    )


def walk_forward_cv_v2(rebuild: bool = False) -> None:
    """Walk-forward CV using v2 features (25 v1 + 7 previous-day context)."""
    feat_v1  = build_feature_matrix(rebuild=False)   # v1 must already exist
    label_df = build_labels(feat_v1, rebuild=False)
    feat_v2  = build_feature_matrix_v2(feat_v1, label_df, rebuild=rebuild)
    _run_walk_forward(
        feat_v2, label_df,
        _MODEL_BULL_V2, _MODEL_BEAR_V2, _FEAT_COLS_V2,
        _WF_PREDS_V2, _WF_REPORT_V2,
        v_tag="v2",
        new_feature_tags=_V2_NEW_TAGS,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")

    args = sys.argv[1:]

    if "--walkforward-v2" in args:
        rebuild = "--rebuild" in args
        walk_forward_cv_v2(rebuild=rebuild)

    elif "--walkforward" in args:
        rebuild = "--rebuild" in args
        walk_forward_cv(rebuild=rebuild)

    else:
        print("Usage:")
        print("  python -m intraday_engine.ml_predictor --walkforward          # v1 (opening range only)")
        print("  python -m intraday_engine.ml_predictor --walkforward-v2       # v2 (+ prev-day context)")
        print("  python -m intraday_engine.ml_predictor --walkforward --rebuild # force feature rebuild")
        sys.exit(1)
