from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from backtest.data_loader import BacktestDataLoader
else:
    from .data_loader import BacktestDataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Positional-only backtest utility (rolling fetch + cache point check)."
    )
    parser.add_argument(
        "--mode",
        choices=["fetch_rolling", "point_check"],
        default="fetch_rolling",
    )
    parser.add_argument("--index", default="NIFTY")
    parser.add_argument("--security-id", type=int, default=None, help="Underlying securityId for custom symbols/stocks")
    parser.add_argument(
        "--instrument",
        default=None,
        help="Dhan instrument code (e.g., OPTIDX for indices, OPTSTK for stocks)",
    )
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--strike-range", type=int, default=10, help="ATM +/- N strikes for rolling fetch")
    parser.add_argument("--interval", type=int, default=1, help="Candle interval for rolling fetch")
    parser.add_argument(
        "--expiry-flags",
        default="WEEK,MONTH",
        help="Comma-separated expiry flags for rolling fetch",
    )
    parser.add_argument(
        "--expiry-codes",
        default="1,2,3",
        help="Comma-separated expiry codes for rolling fetch",
    )
    parser.add_argument("--date", default=None, help="Single YYYY-MM-DD date for point_check")
    parser.add_argument("--time", default=None, help="HH:MM for point_check mode (IST)")
    parser.add_argument("--expiry-flag", default="WEEK", help="WEEK or MONTH for point_check")
    parser.add_argument("--expiry-code", type=int, default=1, help="Expiry code for point_check")
    parser.add_argument("--strike-label", default="ATM", help="ATM / ATM+N / ATM-N for point_check")
    parser.add_argument(
        "--option-type",
        default="BOTH",
        choices=["CALL", "PUT", "BOTH"],
        help="Option leg for point_check",
    )
    return parser.parse_args()


def _resolve_closest_point(payload: dict, target_ts_utc: int):
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    if not isinstance(data, dict):
        return None
    candidates = [("CALL", data.get("ce")), ("PUT", data.get("pe"))]
    out = {}
    for name, leg in candidates:
        if not isinstance(leg, dict):
            continue
        ts = leg.get("timestamp") or []
        if not ts:
            continue
        i = min(range(len(ts)), key=lambda k: abs(int(ts[k]) - target_ts_utc))
        out[name] = {
            "ts": int(ts[i]),
            "close": (leg.get("close") or [None])[i],
            "spot": (leg.get("spot") or [None])[i],
            "iv": (leg.get("iv") or [None])[i],
            "oi": (leg.get("oi") or [None])[i],
            "strike": (leg.get("strike") or [None])[i],
        }
    return out


def _run_point_check(loader: BacktestDataLoader, args: argparse.Namespace) -> None:
    if not args.date or not args.time:
        raise ValueError("point_check mode requires --date YYYY-MM-DD and --time HH:MM")

    try:
        target_ist = datetime.strptime(f"{args.date} {args.time}", "%Y-%m-%d %H:%M")
    except ValueError as exc:
        raise ValueError("Invalid --date/--time. Use YYYY-MM-DD and HH:MM.") from exc

    target_ts_utc = int((target_ist - timedelta(hours=5, minutes=30)).timestamp())
    index = args.index.upper()
    option_types = ["CALL", "PUT"] if args.option_type == "BOTH" else [args.option_type]

    root = loader.cfg.cache_dir / "rolling_options" / index
    if not root.exists():
        print(f"Cache not found: {root}")
        return

    for option_type in option_types:
        fpath = root / (
            f"{args.date}_{args.date}_{args.expiry_flag.upper()}_E{int(args.expiry_code)}_"
            f"{args.strike_label}_{option_type}.json"
        )
        if not fpath.exists():
            print(f"Missing cache file: {fpath}")
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        points = _resolve_closest_point(payload, target_ts_utc)
        row = (points or {}).get(option_type)
        if not row:
            print(f"No points in file: {fpath}")
            continue

        actual_ist = datetime.fromtimestamp(row["ts"], UTC) + timedelta(hours=5, minutes=30)
        print(
            f"\n{index} {option_type} | {args.date} {args.time} IST target | "
            f"actual {actual_ist.strftime('%Y-%m-%d %H:%M:%S')} IST"
        )
        print(
            f"close={row['close']} spot={row['spot']} iv={row['iv']} "
            f"oi={row['oi']} strike={row['strike']}"
        )


def main() -> None:
    args = parse_args()
    loader = BacktestDataLoader()

    if args.mode == "fetch_rolling":
        if not args.start_date or not args.end_date:
            raise ValueError("fetch_rolling mode requires --start-date and --end-date")
        flags = [x.strip() for x in args.expiry_flags.split(",") if x.strip()]
        codes = [int(x.strip()) for x in args.expiry_codes.split(",") if x.strip()]
        manifest = loader.fetch_dhan_rolling_options(
            index=args.index,
            from_date=args.start_date,
            to_date=args.end_date,
            security_id=args.security_id,
            instrument=args.instrument,
            strike_range=args.strike_range,
            interval=args.interval,
            expiry_flags=flags,
            expiry_codes=codes,
        )
        print("\nRolling fetch complete")
        print(f"Index: {manifest['index']}")
        print(f"Range: {manifest['from_date']} -> {manifest['to_date']}")
        print(
            f"Requests: {manifest['success_requests']}/{manifest['total_requests']} "
            f"success, failed={manifest['failed_requests']}"
        )
        print(f"Cache: {manifest['cache_dir']}")
        return

    if args.mode == "point_check":
        _run_point_check(loader, args)
        return


if __name__ == "__main__":
    main()
