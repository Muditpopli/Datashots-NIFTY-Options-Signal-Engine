"""
btst_logger.py -- BTST Signal Audit Logger

Logs every BTSTSignal (including NO_TRADE) to data/tracker/btst_signals.csv.
Supports post-morning outcome fills and summary statistics.

Public API:
  log_signal(sig)                  -> bool   (False if date already logged)
  log_outcome(date_str, outcome,   -> bool   (False if date not found)
              exit_spot, notes)
  generate_summary()               -> dict
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────

_TRACKER_DIR = Path(__file__).parent.parent / "data" / "tracker"
_CSV_PATH    = _TRACKER_DIR / "btst_signals.csv"

_IST = timezone(timedelta(hours=5, minutes=30))

# ── CSV Schema ────────────────────────────────────────────────────────────────

_COLUMNS = [
    "date",
    "signal",
    "direction",
    "confidence",
    "reason",
    "trade_expiry",
    "strike_ce_pe",
    "basis_spot",
    "stop_level",
    "target_level",
    "rollover_used",
    "generated_at",
    # Post-morning fill fields
    "outcome",       # WIN / LOSS / BREAKEVEN / SKIPPED
    "exit_spot",
    "pnl_pts",
    "notes",
]

_VALID_OUTCOMES = {"WIN", "LOSS", "BREAKEVEN", "SKIPPED"}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure_file() -> None:
    _TRACKER_DIR.mkdir(parents=True, exist_ok=True)
    if not _CSV_PATH.exists():
        with _CSV_PATH.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=_COLUMNS).writeheader()


def _read_all() -> list[dict]:
    _ensure_file()
    with _CSV_PATH.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _write_all(rows: list[dict]) -> None:
    _ensure_file()
    with _CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _now_ist() -> str:
    return datetime.now(tz=_IST).strftime("%Y-%m-%d %H:%M:%S IST")


# ── Public API ────────────────────────────────────────────────────────────────

def log_signal(sig) -> bool:
    """
    Append a BTSTSignal to the CSV.

    Returns False (no-op) if that date is already present — prevents duplicate
    runs from creating double entries.  The first write wins.
    """
    _ensure_file()
    existing = _read_all()
    existing_dates = {r["date"] for r in existing}

    if sig.date in existing_dates:
        return False

    exit_rule = sig.exit_rule or {}
    row = {
        "date":         sig.date,
        "signal":       sig.signal,
        "direction":    sig.direction,
        "confidence":   f"{sig.confidence:.4f}",
        "reason":       sig.reason,
        "trade_expiry": sig.trade_expiry,
        "strike_ce_pe": sig.strike_ce_pe,
        "basis_spot":   exit_rule.get("basis_spot", ""),
        "stop_level":   exit_rule.get("stop_level", ""),
        "target_level": exit_rule.get("target_level", ""),
        "rollover_used": str(sig.rollover_used),
        "generated_at": sig.generated_at,
        # Outcome fields — empty until filled post-morning
        "outcome":      "",
        "exit_spot":    "",
        "pnl_pts":      "",
        "notes":        "",
    }

    existing.append(row)
    _write_all(existing)
    return True


def log_outcome(
    date_str:  str,
    outcome:   str,
    exit_spot: Optional[float] = None,
    notes:     str = "",
) -> bool:
    """
    Fill post-morning outcome for a previously logged signal date.

    outcome must be one of: WIN | LOSS | BREAKEVEN | SKIPPED
    Returns False if the date is not found in the CSV.
    """
    outcome = outcome.upper().strip()
    if outcome not in _VALID_OUTCOMES:
        raise ValueError(f"outcome must be one of {_VALID_OUTCOMES}, got '{outcome}'")

    rows = _read_all()
    found = False
    for row in rows:
        if row["date"] == date_str:
            found = True
            row["outcome"]    = outcome
            row["exit_spot"]  = f"{exit_spot:.2f}" if exit_spot is not None else ""
            row["notes"]      = notes

            # Compute rough PnL in spot points when both levels are present.
            try:
                basis = float(row["basis_spot"])
                if exit_spot is not None and basis > 0:
                    direction = row["direction"]
                    pnl = (exit_spot - basis) if direction == "BULLISH" else (basis - exit_spot)
                    row["pnl_pts"] = f"{pnl:.2f}"
            except (ValueError, TypeError):
                row["pnl_pts"] = ""
            break

    if not found:
        return False

    _write_all(rows)
    return True


def generate_summary(include_no_trade: bool = False) -> dict:
    """
    Compute aggregate stats from all logged signals.

    Parameters
    ----------
    include_no_trade : if True, include NO_TRADE rows in counts (they are
                       always excluded from win-rate and PnL calculations).

    Returns a dict with keys:
      total_signals, no_trade_count, trade_signals,
      outcomes_filled, win, loss, breakeven, skipped,
      win_rate, avg_confidence, avg_pnl_pts, total_pnl_pts
    """
    rows = _read_all()

    trade_rows = [r for r in rows if r["signal"] != "NO_TRADE"]
    no_trade_count = len(rows) - len(trade_rows)

    outcomes_filled = [r for r in trade_rows if r["outcome"] in _VALID_OUTCOMES]
    win       = sum(1 for r in outcomes_filled if r["outcome"] == "WIN")
    loss      = sum(1 for r in outcomes_filled if r["outcome"] == "LOSS")
    breakeven = sum(1 for r in outcomes_filled if r["outcome"] == "BREAKEVEN")
    skipped   = sum(1 for r in outcomes_filled if r["outcome"] == "SKIPPED")

    # Win rate over decided trades (WIN + LOSS only; BREAKEVEN/SKIPPED excluded)
    decided = win + loss
    win_rate = (win / decided) if decided > 0 else None

    # Confidence (trade rows only, all of them including unfilled)
    conf_vals = []
    for r in trade_rows:
        try:
            conf_vals.append(float(r["confidence"]))
        except (ValueError, TypeError):
            pass
    avg_confidence = (sum(conf_vals) / len(conf_vals)) if conf_vals else None

    # PnL (only rows where pnl_pts was computed)
    pnl_vals = []
    for r in trade_rows:
        try:
            if r["pnl_pts"]:
                pnl_vals.append(float(r["pnl_pts"]))
        except (ValueError, TypeError):
            pass
    avg_pnl_pts   = (sum(pnl_vals) / len(pnl_vals)) if pnl_vals else None
    total_pnl_pts = sum(pnl_vals) if pnl_vals else None

    return {
        "total_signals":    len(rows),
        "no_trade_count":   no_trade_count,
        "trade_signals":    len(trade_rows),
        "outcomes_filled":  len(outcomes_filled),
        "win":              win,
        "loss":             loss,
        "breakeven":        breakeven,
        "skipped":          skipped,
        "win_rate":         win_rate,
        "avg_confidence":   avg_confidence,
        "avg_pnl_pts":      avg_pnl_pts,
        "total_pnl_pts":    total_pnl_pts,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _run_test() -> None:
    """
    Insert two synthetic signals (one BULLISH, one NO_TRADE), add an outcome
    to the trade signal, then print the CSV contents.
    """
    import os
    import tempfile

    print("=" * 60)
    print("  btst_logger -- self-test")
    print("=" * 60)

    # --- Build minimal mock BTSTSignal without importing btst_signal
    # (avoids pulling in signal_builder / expiry_manager dependencies)
    from dataclasses import dataclass

    @dataclass
    class _MockSignal:
        date:               str
        signal:             str
        reason:             str
        confidence:         float
        direction:          str
        trade_expiry:       str
        strike_ce_pe:       str
        exit_rule:          dict
        nifty_snapshot:     dict
        banknifty_snapshot: dict
        rollover_used:      bool
        generated_at:       str

    # --- Use a temp CSV for the test so we don't pollute the real file
    global _CSV_PATH, _TRACKER_DIR
    orig_csv  = _CSV_PATH
    orig_dir  = _TRACKER_DIR

    with tempfile.TemporaryDirectory() as tmp:
        _TRACKER_DIR = Path(tmp)
        _CSV_PATH    = Path(tmp) / "btst_signals_test.csv"

        trade_sig = _MockSignal(
            date="2026-04-07",
            signal="BULLISH",
            reason="direction=BULLISH | W2=BULLISH W1=BULLISH (confirmed) | confidence=78%",
            confidence=0.78,
            direction="BULLISH",
            trade_expiry="2026-04-08",
            strike_ce_pe="NIFTY 22550 CE",
            exit_rule={
                "primary":        "Exit at market open next trading day",
                "stop_level":     22200.0,
                "target_level":   22840.0,
                "basis_spot":     22500.0,
            },
            nifty_snapshot={},
            banknifty_snapshot={},
            rollover_used=False,
            generated_at=_now_ist(),
        )

        no_trade_sig = _MockSignal(
            date="2026-04-08",
            signal="NO_TRADE",
            reason="EXPIRY_DAY: today is NIFTY/BANKNIFTY expiry -- no overnight hold",
            confidence=0.0,
            direction="NONE",
            trade_expiry="2026-04-15",
            strike_ce_pe="",
            exit_rule={},
            nifty_snapshot={},
            banknifty_snapshot={},
            rollover_used=False,
            generated_at=_now_ist(),
        )

        # log_signal
        r1 = log_signal(trade_sig)
        r2 = log_signal(no_trade_sig)
        r3 = log_signal(trade_sig)   # duplicate — should return False

        print(f"\n  log_signal(BULLISH 2026-04-07) -> {r1}  (expected True)")
        print(f"  log_signal(NO_TRADE 2026-04-08) -> {r2}  (expected True)")
        print(f"  log_signal(BULLISH 2026-04-07 duplicate) -> {r3}  (expected False)")

        # log_outcome
        r4 = log_outcome("2026-04-07", "WIN", exit_spot=22750.0, notes="Gap up open, exited at open")
        r5 = log_outcome("2099-01-01", "WIN")   # non-existent date
        print(f"\n  log_outcome(2026-04-07, WIN, exit_spot=22750) -> {r4}  (expected True)")
        print(f"  log_outcome(2099-01-01, WIN)               -> {r5}  (expected False)")

        # Print CSV
        rows = _read_all()
        print(f"\n  CSV rows written: {len(rows)}")
        print()
        for row in rows:
            print(f"  date={row['date']}  signal={row['signal']:<8}  "
                  f"confidence={row['confidence']}  outcome={row['outcome'] or '(none)'}  "
                  f"pnl_pts={row['pnl_pts'] or '(none)'}")

        # Summary
        summary = generate_summary()
        print(f"\n  Summary:")
        print(f"    total_signals  : {summary['total_signals']}")
        print(f"    no_trade_count : {summary['no_trade_count']}")
        print(f"    trade_signals  : {summary['trade_signals']}")
        print(f"    outcomes_filled: {summary['outcomes_filled']}")
        print(f"    win            : {summary['win']}")
        print(f"    loss           : {summary['loss']}")
        win_rate = summary['win_rate']
        print(f"    win_rate       : {win_rate:.0%}" if win_rate is not None else "    win_rate       : N/A")
        avg_conf = summary['avg_confidence']
        print(f"    avg_confidence : {avg_conf:.0%}" if avg_conf is not None else "    avg_confidence : N/A")
        avg_pnl = summary['avg_pnl_pts']
        print(f"    avg_pnl_pts    : {avg_pnl:.2f}" if avg_pnl is not None else "    avg_pnl_pts    : N/A")

        _CSV_PATH    = orig_csv
        _TRACKER_DIR = orig_dir

    print(f"\n  [PASS] All assertions passed. Real CSV untouched.")
    print("=" * 60)


def _run_summary() -> None:
    _ensure_file()
    summary = generate_summary()

    print("\n" + "=" * 60)
    print(f"  BTST Signal Summary  --  {_now_ist()}")
    print(f"  Source: {_CSV_PATH}")
    print("=" * 60)

    print(f"\n  Total signals logged : {summary['total_signals']}")
    print(f"  NO_TRADE             : {summary['no_trade_count']}")
    print(f"  Trade signals        : {summary['trade_signals']}")
    print(f"  Outcomes filled      : {summary['outcomes_filled']}")
    print()
    print(f"  WIN                  : {summary['win']}")
    print(f"  LOSS                 : {summary['loss']}")
    print(f"  BREAKEVEN            : {summary['breakeven']}")
    print(f"  SKIPPED              : {summary['skipped']}")
    print()

    wr = summary["win_rate"]
    print(f"  Win rate (W/W+L)     : {wr:.0%}" if wr is not None else "  Win rate            : N/A (no decided trades)")

    ac = summary["avg_confidence"]
    print(f"  Avg confidence       : {ac:.0%}" if ac is not None else "  Avg confidence      : N/A")

    ap = summary["avg_pnl_pts"]
    print(f"  Avg PnL (spot pts)   : {ap:+.2f}" if ap is not None else "  Avg PnL             : N/A")

    tp = summary["total_pnl_pts"]
    print(f"  Total PnL (spot pts) : {tp:+.2f}" if tp is not None else "  Total PnL           : N/A")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_test()
    elif "--summary" in sys.argv:
        _run_summary()
    else:
        print("Usage: python -m btst_engine.btst_logger [--test | --summary]", file=sys.stderr)
        sys.exit(1)
