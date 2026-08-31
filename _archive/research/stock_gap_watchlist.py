from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from backtest.data_loader import BacktestDataLoader
from datashots_gap.data_loader import RollingOptionsDataLoader
from datashots_gap.greek_calculator import calculate_greek_changes


def _pcr(chain: Dict) -> float:
    rows = chain.get("strikes", [])
    if not rows:
        return 0.0
    put_oi = 0.0
    call_oi = 0.0
    for r in rows:
        put_oi += float(r["pe"].get("oi", 0.0) or 0.0)
        call_oi += float(r["ce"].get("oi", 0.0) or 0.0)
    if call_oi <= 0:
        return 0.0
    return put_oi / call_oi


def _day_features(loader: RollingOptionsDataLoader, day: str, symbol: str) -> Optional[Dict[str, float]]:
    exp = loader.choose_expiry(day, phase="close_1515")
    if not exp:
        return None
    flag, code = exp
    open_chain = loader.build_chain(day, symbol, flag, code, "open_0915")
    pre_chain = loader.build_chain(day, symbol, flag, code, "preclose_1445")
    close_chain = loader.build_chain(day, symbol, flag, code, "close_1515")
    if not open_chain or not pre_chain or not close_chain:
        return None

    day_ch = calculate_greek_changes(open_chain, close_chain)
    l30_ch = calculate_greek_changes(pre_chain, close_chain)
    spot_day = float(close_chain["spot"]) - float(open_chain["spot"])
    spot_l30 = float(close_chain["spot"]) - float(pre_chain["spot"])
    oi_day = float(day_ch["put_oi"] - day_ch["call_oi"])
    oi_l30 = float(l30_ch["put_oi"] - l30_ch["call_oi"])
    pcr_day = float(_pcr(close_chain) - _pcr(open_chain))
    pcr_l30 = float(_pcr(close_chain) - _pcr(pre_chain))
    return {
        "spot_close": float(close_chain["spot"]),
        "spot_day": spot_day,
        "spot_l30": spot_l30,
        "oi_day": oi_day,
        "oi_l30": oi_l30,
        "pcr_day": pcr_day,
        "pcr_l30": pcr_l30,
    }


def _sign(v: float) -> int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def _score_row(row: Dict[str, float]) -> Tuple[float, str, float]:
    # Robust directional score from end-of-day + last-30m flow.
    # Positive => bullish odds for next open, negative => bearish odds.
    spot_day = row["spot_day"]
    spot_l30 = row["spot_l30"]
    oi_day = row["oi_day"]
    oi_l30 = row["oi_l30"]
    pcr_day = row["pcr_day"]
    pcr_l30 = row["pcr_l30"]
    spot_ref = max(1.0, abs(row["spot_close"]))

    # Normalize magnitudes to avoid bias from expensive stocks.
    s_day = np.clip(spot_day / (0.01 * spot_ref), -3.0, 3.0)
    s_l30 = np.clip(spot_l30 / (0.004 * spot_ref), -3.0, 3.0)
    o_day = np.clip(oi_day / 2_000_000.0, -3.0, 3.0)
    o_l30 = np.clip(oi_l30 / 500_000.0, -3.0, 3.0)
    p_day = np.clip(pcr_day / 0.10, -3.0, 3.0)
    p_l30 = np.clip(pcr_l30 / 0.05, -3.0, 3.0)

    score = 0.0
    score += 1.2 * s_day
    score += 1.5 * s_l30
    score += 0.9 * o_day
    score += 1.1 * o_l30
    score += 0.7 * p_day
    score += 0.9 * p_l30

    # Alignment bonuses/penalties.
    if _sign(spot_day) == _sign(oi_day) and _sign(spot_day) != 0:
        score += 0.7
    else:
        score -= 0.4
    if _sign(spot_l30) == _sign(oi_l30) and _sign(spot_l30) != 0:
        score += 1.0
    else:
        score -= 0.6
    if _sign(spot_l30) == _sign(pcr_l30) and _sign(spot_l30) != 0:
        score += 0.6
    else:
        score -= 0.3

    # Tagging thresholds.
    if score >= 2.0:
        tag = "GAP_UP_BIAS"
    elif score <= -2.0:
        tag = "GAP_DOWN_BIAS"
    else:
        tag = "NO_CLEAR_EDGE"

    confidence = float(max(0.0, min(100.0, abs(score) * 8.0)))
    return float(score), tag, confidence


def _default_symbols() -> List[str]:
    custom = getattr(config, "DHAN_UNDERLYING_MAP", {}) or {}
    return sorted([k for k in custom.keys() if k not in {"NIFTY", "BANKNIFTY"}])


def _fetch_if_needed(
    bdl: BacktestDataLoader,
    symbol: str,
    date_s: str,
    security_id: int,
    instrument: str,
    strike_range: int,
    interval: int,
) -> Dict:
    return bdl.fetch_dhan_rolling_options(
        index=symbol,
        from_date=date_s,
        to_date=date_s,
        security_id=security_id,
        instrument=instrument,
        strike_range=strike_range,
        interval=interval,
        expiry_flags=["WEEK", "MONTH"],
        expiry_codes=[1, 2, 3],
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Stock gap watchlist for next-day open bias.")
    p.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="Signal date (YYYY-MM-DD)")
    p.add_argument("--symbols", default=None, help="Comma-separated symbols. Default: keys from DHAN_UNDERLYING_MAP_JSON")
    p.add_argument("--fetch", action="store_true", help="Fetch rolling options data before scoring")
    p.add_argument("--strike-range", type=int, default=10)
    p.add_argument("--interval", type=int, default=1)
    p.add_argument("--top", type=int, default=20, help="Top names to print per side")
    p.add_argument("--out-csv", default=None)
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    mapping = getattr(config, "DHAN_UNDERLYING_MAP", {}) or {}
    if args.symbols:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    else:
        symbols = _default_symbols()

    if not symbols:
        raise SystemExit(
            "No symbols provided. Set --symbols or define DHAN_UNDERLYING_MAP_JSON in .env."
        )

    bdl = BacktestDataLoader()
    rows: List[Dict] = []

    for sym in symbols:
        meta = mapping.get(sym)
        if not meta:
            rows.append({"symbol": sym, "status": "MISSING_MAPPING"})
            continue

        sec_id = int(meta["security_id"])
        instr = str(meta.get("instrument", "OPTSTK")).upper()
        fetch_status = "SKIPPED"

        if args.fetch:
            try:
                man = _fetch_if_needed(
                    bdl=bdl,
                    symbol=sym,
                    date_s=args.date,
                    security_id=sec_id,
                    instrument=instr,
                    strike_range=args.strike_range,
                    interval=args.interval,
                )
                fetch_status = f"{man['success_requests']}/{man['total_requests']} success"
            except Exception as e:
                rows.append({"symbol": sym, "status": f"FETCH_FAIL: {e}"})
                continue

        ld = RollingOptionsDataLoader(cache_root=Path("data/backtest/cache/rolling_options"))
        ld.build_samples(index=sym, start_date=args.date, end_date=args.date)
        feat = _day_features(ld, args.date, sym)
        if not feat:
            rows.append({"symbol": sym, "status": f"NO_FEATURES ({fetch_status})"})
            continue

        score, tag, conf = _score_row(feat)
        rows.append(
            {
                "symbol": sym,
                "signal_date": args.date,
                "tag": tag,
                "score": round(score, 4),
                "confidence": round(conf, 2),
                "spot_close": round(feat["spot_close"], 2),
                "spot_day": round(feat["spot_day"], 2),
                "spot_l30": round(feat["spot_l30"], 2),
                "oi_day": round(feat["oi_day"], 2),
                "oi_l30": round(feat["oi_l30"], 2),
                "pcr_day": round(feat["pcr_day"], 5),
                "pcr_l30": round(feat["pcr_l30"], 5),
                "status": f"OK ({fetch_status})",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No rows produced.")

    # Output paths.
    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else Path("data/backtest/outputs") / f"stock_gap_watchlist_{args.date}.csv"
    )
    out_json = (
        Path(args.out_json)
        if args.out_json
        else Path("data/backtest/outputs") / f"stock_gap_watchlist_{args.date}.json"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Ranked output for quick action.
    scored = df[df["tag"].isin(["GAP_UP_BIAS", "GAP_DOWN_BIAS"])].copy()
    top_up = scored[scored["tag"] == "GAP_UP_BIAS"].sort_values("score", ascending=False).head(args.top)
    top_dn = scored[scored["tag"] == "GAP_DOWN_BIAS"].sort_values("score", ascending=True).head(args.top)

    payload = {
        "signal_date": args.date,
        "symbols_requested": symbols,
        "rows_total": int(len(df)),
        "rows_scored": int(len(scored)),
        "gap_up_count": int((df["tag"] == "GAP_UP_BIAS").sum()) if "tag" in df.columns else 0,
        "gap_down_count": int((df["tag"] == "GAP_DOWN_BIAS").sum()) if "tag" in df.columns else 0,
        "top_gap_up": top_up.to_dict(orient="records"),
        "top_gap_down": top_dn.to_dict(orient="records"),
        "csv_path": str(out_csv).replace("\\", "/"),
    }
    out_json.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
