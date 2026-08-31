"""Live launcher for data-backed trading strategies only."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime


def _run(cmd: list[str]) -> int:
    print(f"[RUN] {' '.join(cmd)}")
    env = os.environ.copy()
    if any("get_locked_tag_for_open.py" in part for part in cmd):
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, env=env)
    return int(proc.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live launcher for data-backed positional strategies",
    )
    parser.add_argument("--mode", choices=["fetch", "tag", "routine"], default="routine")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="YYYY-MM-DD")
    parser.add_argument("--open-date", default=None, help="Optional target open date for tag mode")
    parser.add_argument("--lock-file", default="data/backtest/outputs/final_daily_tag_lock.json")
    parser.add_argument("--primary-index", default="NIFTY")
    parser.add_argument("--confirm-index", default="BANKNIFTY")
    parser.add_argument("--expiry-flags", default="WEEK,MONTH")
    parser.add_argument("--expiry-codes", default="1,2,3")
    parser.add_argument("--strike-range", type=int, default=10)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--primary-security-id", type=int, default=None)
    parser.add_argument("--primary-instrument", default=None)
    parser.add_argument("--confirm-security-id", type=int, default=None)
    parser.add_argument("--confirm-instrument", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    py = sys.executable

    fetch_primary = [
        py,
        "-m",
        "backtest.run",
        "--mode",
        "fetch_rolling",
        "--index",
        args.primary_index,
        "--start-date",
        args.date,
        "--end-date",
        args.date,
        "--expiry-flags",
        args.expiry_flags,
        "--expiry-codes",
        args.expiry_codes,
        "--strike-range",
        str(args.strike_range),
        "--interval",
        str(args.interval),
    ]
    if args.primary_security_id is not None:
        fetch_primary += ["--security-id", str(args.primary_security_id)]
    if args.primary_instrument:
        fetch_primary += ["--instrument", str(args.primary_instrument)]

    fetch_confirm = [
        py,
        "-m",
        "backtest.run",
        "--mode",
        "fetch_rolling",
        "--index",
        args.confirm_index,
        "--start-date",
        args.date,
        "--end-date",
        args.date,
        "--expiry-flags",
        args.expiry_flags,
        "--expiry-codes",
        args.expiry_codes,
        "--strike-range",
        str(args.strike_range),
        "--interval",
        str(args.interval),
    ]
    if args.confirm_security_id is not None:
        fetch_confirm += ["--security-id", str(args.confirm_security_id)]
    if args.confirm_instrument:
        fetch_confirm += ["--instrument", str(args.confirm_instrument)]

    tag_cmd = [
        py,
        "research/get_locked_tag_for_open.py",
        "--lock-file",
        args.lock_file,
        "--primary-index",
        args.primary_index,
        "--confirm-index",
        args.confirm_index,
    ]
    if args.open_date:
        tag_cmd += ["--open-date", args.open_date]

    if args.mode in {"fetch", "routine"}:
        rc = _run(fetch_primary)
        if rc != 0:
            return rc
        rc = _run(fetch_confirm)
        if rc != 0:
            return rc

    if args.mode in {"tag", "routine"}:
        rc = _run(tag_cmd)
        if rc != 0:
            return rc

    print("[DONE] Data-backed live flow complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
