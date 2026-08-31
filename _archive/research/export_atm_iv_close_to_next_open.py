from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datashots_gap.data_loader import RollingOptionsDataLoader


def _default_start(end_date: str, years: int) -> str:
    end_d = date.fromisoformat(end_date)
    try:
        return end_d.replace(year=end_d.year - years).isoformat()
    except ValueError:
        # Handle leap-year edge by falling back to day-1.
        return (end_d.replace(year=end_d.year - years, day=end_d.day - 1)).isoformat()


def _next_day_map(days: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i in range(len(days) - 1):
        out[days[i]] = days[i + 1]
    return out


def _atm_row(chain: Dict) -> Optional[Dict]:
    atm = int(chain.get("atm", 0) or 0)
    for row in chain.get("strikes", []):
        if int(row.get("strike", 0) or 0) == atm:
            return row
    return None


def _extract_iv_snapshot(loader: RollingOptionsDataLoader, day_s: str, index: str, flag: str, code: int, phase: str) -> Optional[Dict]:
    chain = loader.build_chain(day_s, index, flag, code, phase)
    if not chain:
        return None
    row = _atm_row(chain)
    if not row:
        return None
    ce_iv = float(row.get("ce", {}).get("iv", 0.0) or 0.0)
    pe_iv = float(row.get("pe", {}).get("iv", 0.0) or 0.0)
    return {
        "spot": float(chain.get("spot", 0.0) or 0.0),
        "atm_strike": int(chain.get("atm", 0) or 0),
        "call_iv": ce_iv,
        "put_iv": pe_iv,
        "atm_iv": (ce_iv + pe_iv) / 2.0,
    }


def build_iv_change_table(
    cache_root: Path,
    index: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    loader = RollingOptionsDataLoader(cache_root=cache_root)
    loader.build_samples(index=index, start_date=start_date, end_date=end_date)
    days = loader.trading_days()
    nxt = _next_day_map(days)

    rows: List[Dict] = []
    for signal_day in days:
        next_day = nxt.get(signal_day)
        if not next_day:
            continue

        expiry = loader.choose_expiry(signal_day, phase="close_1515")
        if not expiry:
            continue
        flag, code = expiry

        close_snap = _extract_iv_snapshot(loader, signal_day, index, flag, code, "close_1515")
        open_snap = _extract_iv_snapshot(loader, next_day, index, flag, code, "next_open_0925")
        if not close_snap or not open_snap:
            continue

        rows.append(
            {
                "signal_date": signal_day,
                "next_open_date": next_day,
                "index": index,
                "expiry_flag": flag,
                "expiry_code": code,
                "close_spot_1515": round(close_snap["spot"], 2),
                "next_open_spot_0925": round(open_snap["spot"], 2),
                "close_atm_strike_1515": int(close_snap["atm_strike"]),
                "next_open_atm_strike_0925": int(open_snap["atm_strike"]),
                "close_call_iv_1515": round(close_snap["call_iv"], 4),
                "close_put_iv_1515": round(close_snap["put_iv"], 4),
                "close_atm_iv_1515": round(close_snap["atm_iv"], 4),
                "next_open_call_iv_0925": round(open_snap["call_iv"], 4),
                "next_open_put_iv_0925": round(open_snap["put_iv"], 4),
                "next_open_atm_iv_0925": round(open_snap["atm_iv"], 4),
                "call_iv_change": round(open_snap["call_iv"] - close_snap["call_iv"], 4),
                "put_iv_change": round(open_snap["put_iv"] - close_snap["put_iv"], 4),
                "atm_iv_change": round(open_snap["atm_iv"] - close_snap["atm_iv"], 4),
                "atm_iv_direction": (
                    "INCREASED"
                    if open_snap["atm_iv"] > close_snap["atm_iv"]
                    else "DECREASED"
                    if open_snap["atm_iv"] < close_snap["atm_iv"]
                    else "UNCHANGED"
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Export ATM IV change from close_1515 to next_open_0925 using rolling cache.")
    p.add_argument("--index", default="NIFTY")
    p.add_argument("--cache-root", default="data/backtest/cache/rolling_options")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=date.today().isoformat())
    p.add_argument("--years", type=int, default=3)
    p.add_argument("--output-csv", default=None)
    p.add_argument("--output-xlsx", default=None)
    args = p.parse_args()

    start_date = args.start_date or _default_start(args.end_date, args.years)
    index = str(args.index).upper()
    cache_root = Path(args.cache_root)

    out_dir = Path("data/backtest/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv) if args.output_csv else out_dir / f"{index.lower()}_atm_iv_close_to_next_open_{start_date}_to_{args.end_date}.csv"
    output_xlsx = Path(args.output_xlsx) if args.output_xlsx else out_dir / f"{index.lower()}_atm_iv_close_to_next_open_{start_date}_to_{args.end_date}.xlsx"

    df = build_iv_change_table(cache_root=cache_root, index=index, start_date=start_date, end_date=args.end_date)
    df.to_csv(output_csv, index=False)
    try:
        df.to_excel(output_xlsx, index=False)
    except Exception:
        output_xlsx = None

    summary = {
        "index": index,
        "start_date": start_date,
        "end_date": args.end_date,
        "rows": int(len(df)),
        "output_csv": str(output_csv),
        "output_xlsx": str(output_xlsx) if output_xlsx else None,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
