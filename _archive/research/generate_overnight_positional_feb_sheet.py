from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datashots_gap.positional_v4 import run_positional_v4_backtest
from datashots_gap.data_loader import RollingOptionsDataLoader


IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class DayOHLC:
    date: str
    open: float
    high: float
    low: float
    close: float
    weekday: str


def _date_from_ts(ts: int) -> Tuple[str, str]:
    dt_utc = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    dt_ist = dt_utc.astimezone(IST)
    return dt_ist.date().isoformat(), dt_ist.strftime("%H:%M")


def _read_daily_ohlc_from_cache(path: Path) -> Dict[str, DayOHLC]:
    payload = pd.read_json(path)
    data = payload["data"]
    # layout: {"ce":{"spot":[...],"timestamp":[...]}, "pe":...}
    ce = data["ce"]
    spots = list(ce["spot"])
    ts = list(ce["timestamp"])
    rows: Dict[str, List[float]] = {}
    for t, s in zip(ts, spots):
        day, _hhmm = _date_from_ts(int(t))
        rows.setdefault(day, []).append(float(s))

    out: Dict[str, DayOHLC] = {}
    for day, arr in rows.items():
        if not arr:
            continue
        d = datetime.fromisoformat(day).strftime("%A")
        out[day] = DayOHLC(
            date=day,
            open=float(arr[0]),
            high=float(max(arr)),
            low=float(min(arr)),
            close=float(arr[-1]),
            weekday=d,
        )
    return out


def _trade_setup(
    sentiment: str,
    conviction: str,
    in_sync,
    weekday: str,
    score: float,
    min_abs_score: float,
) -> str:
    if weekday == "Thursday":
        return "Avoid/Expiry Day"
    # Keep a no-view zone for weak scores; tradeable setups need clear edge.
    if abs(float(score)) < float(min_abs_score):
        return "No View"
    # If sync is conflicting and confidence is low, avoid.
    if str(in_sync) == "False" and conviction == "LOW":
        return "No View"
    if sentiment == "BULLISH":
        return "Bullish"
    if sentiment == "BEARISH":
        return "Bearish"
    return "No View"


def _outcome_at_open(setup: str, open_px: float, prev_close: float) -> str:
    if setup in {"No View", "Avoid/Expiry Day"}:
        return "No Trade"
    if setup == "Bullish":
        return "Winning Trade" if open_px > prev_close else "Loss Trade"
    if setup == "Bearish":
        return "Winning Trade" if open_px < prev_close else "Loss Trade"
    return "No Trade"


def _outcome_open_hl(setup: str, open_px: float, high_px: float, low_px: float, prev_close: float) -> str:
    if setup in {"No View", "Avoid/Expiry Day"}:
        return "No Trade"
    if setup == "Bullish":
        # Combined opening gap + favorable extension.
        score = (open_px - prev_close) + (high_px - open_px)
        return "Winning Trade" if score > 0 else "Loss Trade"
    if setup == "Bearish":
        score = (prev_close - open_px) + (open_px - low_px)
        return "Winning Trade" if score > 0 else "Loss Trade"
    return "No Trade"


def _outcome_high_low(setup: str, open_px: float, high_px: float, low_px: float) -> str:
    if setup in {"No View", "Avoid/Expiry Day"}:
        return "No Trade"
    up_leg = high_px - open_px
    down_leg = open_px - low_px
    if setup == "Bullish":
        return "Winning Trade" if up_leg >= down_leg else "Loss Trade"
    if setup == "Bearish":
        return "Winning Trade" if down_leg >= up_leg else "Loss Trade"
    return "No Trade"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate commercial positional sheet.")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-02-28")
    parser.add_argument("--trade-start-date", default="2026-02-01")
    parser.add_argument("--trade-end-date", default="2026-02-28")
    parser.add_argument("--indices", default="NIFTY", help="Comma separated, e.g. NIFTY,BANKNIFTY")
    parser.add_argument("--min-abs-score", type=float, default=8.0)
    args = parser.parse_args()

    cache_root = Path("data/backtest/cache/rolling_options")
    loader = RollingOptionsDataLoader(cache_root=cache_root)
    indices = [i.strip().upper() for i in str(args.indices).split(",") if i.strip()]
    if not indices:
        raise RuntimeError("No indices provided.")

    # Build signals with prior-day context (positional v4).
    sig = run_positional_v4_backtest(
        loader=loader,
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        indices=indices,
    )
    if sig.empty:
        raise RuntimeError("No positional signals generated.")

    sig = sig[sig["next_day"].between(str(args.trade_start_date), str(args.trade_end_date))].copy()
    if sig.empty:
        raise RuntimeError("No Feb-2026 positional signals found.")

    # Read daily OHLC from cached ATM call series.
    ohlc_by_idx: Dict[str, Dict[str, DayOHLC]] = {}
    for idx in indices:
        p = cache_root / idx / f"{args.trade_start_date}_{args.trade_end_date}_WEEK_E1_ATM_CALL.json"
        if not p.exists():
            raise RuntimeError(f"Missing cache file: {p}")
        ohlc_by_idx[idx] = _read_daily_ohlc_from_cache(p)

    out_rows: List[Dict] = []
    for _, r in sig.iterrows():
        idx = str(r["index"])
        day = str(r["next_day"])
        d = ohlc_by_idx.get(idx, {}).get(day)
        if not d:
            continue

        setup = _trade_setup(
            sentiment=str(r["sentiment"]),
            conviction=str(r["conviction"]),
            in_sync=r.get("in_sync"),
            weekday=d.weekday,
            score=float(r["score"]),
            min_abs_score=float(args.min_abs_score),
        )
        prev_close = float(r["cmp"])

        row = {
            "Date": d.date,
            "Index": idx,
            "Open": round(d.open, 2),
            "High": round(d.high, 2),
            "Low": round(d.low, 2),
            "Close": round(d.close, 2),
            "WeekDay": d.weekday,
            "Trade Setup": setup,
            "At Opening": _outcome_at_open(setup, d.open, prev_close),
            "Open + H-L": _outcome_open_hl(setup, d.open, d.high, d.low, prev_close),
            "High - Low": _outcome_high_low(setup, d.open, d.high, d.low),
            "Signal Date": str(r["signal_date"]),
            "Prev Close (Signal CMP)": round(prev_close, 2),
            "Conviction": str(r["conviction"]),
            "Consensus": str(r.get("consensus_sentiment", "")),
            "Strength": round(float(r["strength"]), 2),
            "POS": round(float(r["pos"]), 2),
            "Score": round(float(r["score"]), 4),
        }
        out_rows.append(row)

    out = pd.DataFrame(out_rows).sort_values(["Date", "Index"]).reset_index(drop=True)
    out_dir = Path("data/backtest/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{args.trade_start_date}_to_{args.trade_end_date}_{'_'.join(indices)}".replace(",", "_")
    out_csv = out_dir / f"overnight_positional_commercial_sheet_{suffix}.csv"
    out_xlsx = out_dir / f"overnight_positional_commercial_sheet_{suffix}.xlsx"
    out.to_csv(out_csv, index=False)
    out.to_excel(out_xlsx, index=False)

    # Simple summary by column.
    def _rate(col: str) -> Dict[str, float]:
        s = out[col]
        traded = s[s != "No Trade"]
        if traded.empty:
            return {"trades": 0, "win_rate": 0.0}
        wr = (traded == "Winning Trade").mean() * 100.0
        return {"trades": int(len(traded)), "win_rate": round(float(wr), 2)}

    summary = pd.DataFrame(
        [
            {"Metric": "At Opening", **_rate("At Opening")},
            {"Metric": "Open + H-L", **_rate("Open + H-L")},
            {"Metric": "High - Low", **_rate("High - Low")},
        ]
    )
    summary_csv = out_dir / f"overnight_positional_commercial_summary_{suffix}.csv"
    summary.to_csv(summary_csv, index=False)

    print(f"sheet_csv={out_csv}")
    print(f"sheet_xlsx={out_xlsx}")
    print(f"summary_csv={summary_csv}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
