"""
backtest_runner.py -- BTST Historical Backtest

Replays the full BTST engine on every trading date available in the
rolling options cache (2023-2026).  One BTSTSignal per date, logged to
data/tracker/btst_signals.csv via btst_logger.

Date coverage:
  - All dates where BOTH NIFTY and BANKNIFTY have W2 data (not MISSING).
  - Dates where both indices expire today are logged as NO_TRADE and skipped.
  - Expiry eve is a valid trading day (signal uses next expiry + rollover).

Resume support:
  - Dates already present in btst_signals.csv are skipped automatically.

Error handling:
  - One bad date never aborts the run; errors go to btst_backtest_errors.csv.

Usage:
  python -m btst_engine.backtest_runner                   full run
  python -m btst_engine.backtest_runner --dry-run         first 10 dates, no writes
  python -m btst_engine.backtest_runner --from 2025-01-01 start from a date
"""

from __future__ import annotations

import csv
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Set, Tuple

import config
from btst_engine.expiry_manager import (
    _CACHE_DIR,
    _FNAME_PAT,
    resolve_expiry_context,
)
from btst_engine.signal_builder import build_snapshots
from btst_engine.btst_signal import BTSTSignal, generate_signal
import btst_engine.btst_logger as _logger

log = logging.getLogger(__name__)

_IST              = timezone(timedelta(hours=5, minutes=30))
_TRACKER_DIR      = Path(__file__).parent.parent / "data" / "tracker"
_ERRORS_CSV       = _TRACKER_DIR / "btst_backtest_errors.csv"
_ERROR_COLS       = ["date", "error"]
_DRY_RUN_LIMIT    = 10
_PROGRESS_EVERY   = 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_trading_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    return d.strftime("%Y-%m-%d") not in config.MARKET_HOLIDAYS


def _now_ist() -> str:
    return datetime.now(tz=_IST).strftime("%Y-%m-%d %H:%M:%S IST")


# ── Cache date enumeration ────────────────────────────────────────────────────

def _collect_dates_for_index(index: str) -> Set[date]:
    """
    Enumerate every trading date covered by the rolling options cache for `index`.

    Range files (from_date != to_date) contain data for every trading day in
    their interval — we expand them to all non-holiday weekdays in [from, to].
    Daily files (from_date == to_date) cover exactly that one date.
    """
    index_dir = _CACHE_DIR / index
    if not index_dir.is_dir():
        log.warning("[RUNNER] Cache directory not found: %s", index_dir)
        return set()

    dates: Set[date] = set()

    for fp in index_dir.glob("*.json"):
        m = _FNAME_PAT.match(fp.name)
        if not m:
            continue

        from_d = date.fromisoformat(m.group(1))
        to_d   = date.fromisoformat(m.group(2))

        if from_d == to_d:
            if _is_trading_day(from_d):
                dates.add(from_d)
        else:
            cur = from_d
            while cur <= to_d:
                if _is_trading_day(cur):
                    dates.add(cur)
                cur += timedelta(days=1)

    return dates


def collect_backtest_dates(from_date: Optional[date] = None) -> List[date]:
    """
    Return sorted list of dates where BOTH NIFTY and BANKNIFTY have cache data.
    Optionally filtered to on/after from_date.
    """
    nifty_dates = _collect_dates_for_index("NIFTY")
    bnf_dates   = _collect_dates_for_index("BANKNIFTY")
    common      = sorted(nifty_dates & bnf_dates)
    if from_date:
        common = [d for d in common if d >= from_date]
    return common


# ── Error logging ─────────────────────────────────────────────────────────────

def _log_error(date_str: str, error: str) -> None:
    _TRACKER_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not _ERRORS_CSV.exists()
    with _ERRORS_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_ERROR_COLS)
        if write_header:
            w.writeheader()
        w.writerow({"date": date_str, "error": error})


# ── Per-date processing ───────────────────────────────────────────────────────

def _make_no_trade(date_str: str, reason: str, ctx) -> BTSTSignal:
    return BTSTSignal(
        date=date_str,
        signal="NO_TRADE",
        reason=reason,
        confidence=0.0,
        direction="NONE",
        trade_expiry=ctx.trade_expiry,
        strike_ce_pe="",
        exit_rule={},
        nifty_snapshot={},
        banknifty_snapshot={},
        rollover_used=False,
        generated_at=_now_ist(),
    )


def _process_date(
    d: date,
    dry_run: bool,
) -> Tuple[Optional[BTSTSignal], Optional[str]]:
    """
    Run the full BTST pipeline for one date.

    Returns:
        (BTSTSignal, None)  — on success (signal may be NO_TRADE)
        (None,       None)  — silently skipped (W2 MISSING on both indices)
        (None,       str)   — error occurred; string is the error message
    """
    date_str = d.strftime("%Y-%m-%d")

    try:
        ctx = resolve_expiry_context(d)

        # Dates where both indices expire today: log NO_TRADE, no snapshot needed.
        if ctx.today_is_nifty_expiry and ctx.today_is_bnf_expiry:
            sig = _make_no_trade(
                date_str,
                "EXPIRY_DAY: both NIFTY and BANKNIFTY expire today -- no overnight hold",
                ctx,
            )
            if not dry_run:
                _logger.log_signal(sig)
            return sig, None

        snapshots  = build_snapshots(d, ctx)
        nifty_snap = snapshots["NIFTY"]
        bnf_snap   = snapshots["BANKNIFTY"]

        # Silently skip dates with no W2 data on either index.
        # (These aren't trading days in the engine's eyes; no point cluttering the CSV.)
        if (nifty_snap.w2.data_quality == "MISSING"
                or bnf_snap.w2.data_quality == "MISSING"):
            return None, None

        sig = generate_signal(nifty_snap, bnf_snap, ctx, vix=None)

        if not dry_run:
            _logger.log_signal(sig)

        return sig, None

    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        if not dry_run:
            _log_error(date_str, err)
        return None, err


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_backtest(
    dry_run:   bool = False,
    from_date: Optional[date] = None,
) -> None:
    """
    Replay the BTST engine over all available historical cache dates.

    dry_run   : Process first 10 dates only; print results but write nothing.
    from_date : Skip all dates before this date (resume / partial rerun).
    """
    all_dates = collect_backtest_dates(from_date)

    if not all_dates:
        print("[WARN] No backtest dates found. Check that the rolling_options "
              "cache exists at data/backtest/cache/rolling_options/.")
        return

    # Resume: skip dates already in the CSV (first write wins in btst_logger)
    existing_dates: Set[str] = set()
    if not dry_run:
        existing_dates = {r["date"] for r in _logger._read_all()}

    fresh_dates   = [d for d in all_dates if d.strftime("%Y-%m-%d") not in existing_dates]
    process_dates = fresh_dates[:_DRY_RUN_LIMIT] if dry_run else fresh_dates

    n_total = len(all_dates)
    n_skip  = n_total - len(fresh_dates)
    n_run   = len(process_dates)

    print(f"\n{'=' * 64}")
    print(f"  BTST Backtest Runner")
    print(f"{'=' * 64}")
    print(f"  Cache dates (both indices) : {n_total}")
    print(f"  Already logged (skip)      : {n_skip}")
    print(f"  To process                 : {len(fresh_dates)}")
    if dry_run:
        print(f"  DRY-RUN: showing first {_DRY_RUN_LIMIT} of {len(fresh_dates)} fresh dates -- no writes")
    if from_date:
        print(f"  Starting from              : {from_date}")
    print(f"{'=' * 64}\n")

    processed    = 0
    no_data_skip = 0
    errors       = 0

    for i, d in enumerate(process_dates):
        date_str = d.strftime("%Y-%m-%d")
        sig, err = _process_date(d, dry_run)

        if err:
            errors += 1
            print(f"  [ERR] {date_str}  {err}")
            continue

        if sig is None:
            no_data_skip += 1
            continue

        processed += 1

        badge = {
            "BULLISH":  "[BULL]",
            "BEARISH":  "[BEAR]",
            "NO_TRADE": "[SKIP]",
        }.get(sig.signal, f"[{sig.signal[:4]}]")

        conf_str = f"{sig.confidence:.0%}" if sig.signal != "NO_TRADE" else "---"
        reason_short = sig.reason[:55] + ".." if len(sig.reason) > 57 else sig.reason

        print(f"  {date_str}  {badge}  conf={conf_str:<4}  {reason_short}")

        if not dry_run and processed % _PROGRESS_EVERY == 0:
            done = n_skip + processed
            print(f"\n  --- Progress: {processed} processed / {n_run} queued "
                  f"({done}/{n_total} total) ---\n")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print(f"  Run complete")
    print(f"  Signals generated  : {processed}")
    print(f"  No-data skipped    : {no_data_skip}")
    print(f"  Errors             : {errors}")
    print(f"{'=' * 64}")

    if not dry_run:
        _print_summary()


def _print_summary() -> None:
    s = _logger.generate_summary()
    print(f"\n{'=' * 64}")
    print(f"  BTST Aggregate Summary  ({s['total_signals']} signals in CSV)")
    print(f"{'=' * 64}")
    print(f"  NO_TRADE             : {s['no_trade_count']}")
    print(f"  Trade signals        : {s['trade_signals']}")
    print(f"  Outcomes filled      : {s['outcomes_filled']}")

    wr = s["win_rate"]
    ac = s["avg_confidence"]
    print(f"  Win rate (W/W+L)     : {wr:.0%}" if wr is not None
          else "  Win rate            : N/A  (outcomes not filled yet)")
    print(f"  Avg signal confidence: {ac:.0%}" if ac is not None else "  Avg confidence      : N/A")
    print(f"{'=' * 64}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
    # Suppress Dhan API "thin list" warnings during backtest — the fallback
    # calendar is always used here; these warnings are noise at this volume.
    logging.getLogger("btst_engine.expiry_manager").setLevel(logging.ERROR)

    dry_run: bool = "--dry-run" in sys.argv
    from_date: Optional[date] = None

    if "--from" in sys.argv:
        idx = sys.argv.index("--from")
        if idx + 1 >= len(sys.argv):
            print("[ERR] --from requires a YYYY-MM-DD argument", file=sys.stderr)
            sys.exit(1)
        try:
            from_date = date.fromisoformat(sys.argv[idx + 1])
        except ValueError:
            print(f"[ERR] Bad date '{sys.argv[idx + 1]}' -- expected YYYY-MM-DD",
                  file=sys.stderr)
            sys.exit(1)

    run_backtest(dry_run=dry_run, from_date=from_date)
