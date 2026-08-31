"""
btst_engine/orchestrator.py — BTST Daily Signal Orchestrator

Runs every trading day at 3:15 PM IST after market close.
Combines the rule-based BTST signal engine with the XGBoost ML predictor
into a single daily CE_BUY / PE_BUY signal with a confidence tier.

Confidence tiers
  A — Rule engine fired AND ML ensemble = HIGH  →  full position
  B — One engine fired / partial agreement      →  half position
  C — ML LOW confidence only                   →  display only

Usage
  python -m btst_engine.orchestrator               auto: today or last cache date
  python -m btst_engine.orchestrator --date YYYY-MM-DD
  python -m btst_engine.orchestrator --outcome YYYY-MM-DD --open 22150.5
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import config
from btst_engine.expiry_manager import find_last_cache_date, resolve_expiry_context
from btst_engine.signal_builder  import build_snapshots
from btst_engine.btst_signal     import BTSTSignal, generate_signal
import btst_engine.btst_logger   as _logger

logger = logging.getLogger(__name__)

_IST         = timezone(timedelta(hours=5, minutes=30))
_ROOT        = Path(__file__).parent.parent
_REPORTS_DIR = _ROOT / "data" / "reports"


# ── Utilities ─────────────────────────────────────────────────────────────────

def _now_ist() -> datetime:
    return datetime.now(tz=_IST)


def _is_trading_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    return d.strftime("%Y-%m-%d") not in config.MARKET_HOLIDAYS


def _has_cache_data(d: date) -> bool:
    """True if the rolling options cache has any NIFTY data for date d."""
    from btst_engine.signal_builder import _find_file
    return _find_file("NIFTY", d.strftime("%Y-%m-%d"), "WEEK", "E1", "ATM", "CALL") is not None


def _find_run_date() -> tuple[date, str]:
    """
    Auto-detect the date to run for.
    Returns (run_date, mode) where mode is "LIVE" or "CACHE_ONLY".
    """
    today = _now_ist().date()
    if _is_trading_day(today) and _has_cache_data(today):
        return today, "LIVE"

    last = find_last_cache_date()
    if last:
        return last, "CACHE_ONLY"

    raise RuntimeError(
        "No cache data found. "
        "Check data/backtest/cache/rolling_options/ is populated."
    )


# ── Step 2-3: Market context ──────────────────────────────────────────────────

def _build_market_context(d: date):
    """
    Resolve expiry context and build NIFTY + BANKNIFTY snapshots.
    Dhan API is bypassed — uses cache-derived expiry calendar.
    """
    import logging as _log
    _log.getLogger("btst_engine.expiry_manager").setLevel(_log.ERROR)
    _log.getLogger("btst_engine.signal_builder").setLevel(_log.ERROR)
    _log.getLogger("dhanhq").setLevel(_log.CRITICAL)

    import btst_engine.expiry_manager as _em
    _em._fetch_expiries_from_dhan = lambda index: None   # skip live API

    ctx  = resolve_expiry_context(d)
    snaps = build_snapshots(d, ctx)
    return ctx, snaps


# ── Step 5: ML signal ─────────────────────────────────────────────────────────

def _run_ml_signal(date_str: str) -> Optional[dict]:
    """Run ML predictor. Returns None if models are not trained."""
    try:
        from btst_engine.ml_predictor import predict_today
        return predict_today(date_str)
    except FileNotFoundError:
        logger.warning("[ML] Models not found — run: python -m btst_engine.ml_predictor --train")
        return None
    except Exception as exc:
        logger.warning("[ML] Prediction failed: %s", exc)
        return None


# ── Signal combination ────────────────────────────────────────────────────────

def _expiry_regime_str(ctx) -> str:
    if ctx.today_is_expiry:
        return "EXPIRY DAY"
    if ctx.is_expiry_eve:
        return "expiry eve  [rollover active]"
    return "normal"


def _final_direction(btst_sig: BTSTSignal, ml_result: Optional[dict]) -> str:
    """
    Rule engine takes priority when it fires; ML is the fallback.
    Returns "CE_BUY" or "PE_BUY".
    """
    if btst_sig.signal == "BULLISH":
        return "CE_BUY"
    if btst_sig.signal == "BEARISH":
        return "PE_BUY"
    # NO_TRADE from rule engine — use ML
    if ml_result:
        return ml_result["signal"]
    return "CE_BUY"   # last-resort default (shouldn't be reached in practice)


def _signals_agree(btst_sig: BTSTSignal, ml_result: Optional[dict]) -> bool:
    """True when rule engine and ML engine point in the same direction."""
    if not ml_result:
        return False
    rule_ce = btst_sig.signal == "BULLISH"
    rule_pe = btst_sig.signal == "BEARISH"
    ml_ce   = ml_result["signal"] == "CE_BUY"
    return (rule_ce and ml_ce) or (rule_pe and not ml_ce)


def _confidence_tier(btst_sig: BTSTSignal, ml_result: Optional[dict]) -> str:
    """
    A  — Rule fired + ML ensemble HIGH + both agree       → full position
    A- — Rule confidence >= 85% regardless of ML          → full position (rule dominates)
    B  — Rule fired + ML disagrees OR ML ensemble LOW     → half position
    C  — Rule NO_TRADE + ML LOW confidence                → display only
    """
    rule_fired = btst_sig.signal in ("BULLISH", "BEARISH")
    ml_high    = ml_result and ml_result.get("ensemble_confidence") == "HIGH"
    agree      = _signals_agree(btst_sig, ml_result)

    if rule_fired and ml_high and agree:
        return "A"
    if rule_fired and btst_sig.confidence >= 0.85:
        return "A-"
    if rule_fired or ml_high:
        return "B"
    return "C"


def _compute_strike(final_dir: str, nifty_snap, btst_sig: BTSTSignal) -> str:
    """
    Prefer rule engine's pre-computed strike; fall back to ATM±1 from W2 close.
    """
    if btst_sig.signal != "NO_TRADE" and btst_sig.strike_ce_pe:
        return btst_sig.strike_ce_pe

    spot = nifty_snap.w2.spot_close or nifty_snap.w1.spot_close
    if spot <= 0:
        opt = "CE" if final_dir == "CE_BUY" else "PE"
        return f"NIFTY ATM {opt}"

    gap = config.STRIKE_GAPS.get("NIFTY", 50)
    atm = round(spot / gap) * gap
    if final_dir == "CE_BUY":
        return f"NIFTY {int(atm + gap)} CE"
    return f"NIFTY {int(atm - gap)} PE"


def _stop_target_lines(btst_sig: BTSTSignal, final_dir: str, spot: float) -> tuple[str, str]:
    """Return (stop_line, target_line) for the report."""
    if btst_sig.signal != "NO_TRADE" and btst_sig.exit_rule:
        er = btst_sig.exit_rule
        stop_lvl   = er.get("stop_level")
        target_lvl = er.get("target_level")
        if stop_lvl and target_lvl:
            if final_dir == "CE_BUY":
                return (
                    f"Spot opens below {stop_lvl:.0f}  →  exit immediately",
                    f"Spot opens above {target_lvl:.0f}  →  exit at open",
                )
            else:
                return (
                    f"Spot opens above {stop_lvl:.0f}  →  exit immediately",
                    f"Spot opens below {target_lvl:.0f}  →  exit at open",
                )

    # Compute from spot
    if spot > 0:
        if final_dir == "CE_BUY":
            return (
                f"Spot opens below {spot * 0.992:.0f}  →  exit immediately",
                f"Spot opens above {spot * 1.015:.0f}  →  exit at open",
            )
        else:
            return (
                f"Spot opens above {spot * 1.008:.0f}  →  exit immediately",
                f"Spot opens below {spot * 0.985:.0f}  →  exit at open",
            )
    return ("Gap 0.8% against position  →  exit immediately",
            "Gap 1.5% in favour  →  exit at open")


# ── Report formatting ─────────────────────────────────────────────────────────

_TIER_DESC = {
    "A":  "Rule + ML agree + ensemble HIGH  →  full position recommended",
    "A-": "Rule >= 85% confidence           →  full position (rule dominates)",
    "B":  "Partial agreement                →  half position recommended",
    "C":  "ML LOW confidence                →  display only, client decides",
}


def _format_report(
    d: date,
    mode: str,
    ctx,
    btst_sig:  BTSTSignal,
    ml_result: Optional[dict],
    nifty_snap,
) -> str:
    """
    Format the client-facing signal report.
    Clean enough to screenshot and send directly.
    """
    final_dir = _final_direction(btst_sig, ml_result)
    strike    = _compute_strike(final_dir, nifty_snap, btst_sig)
    tier      = _confidence_tier(btst_sig, ml_result)
    spot      = nifty_snap.w2.spot_close or nifty_snap.w1.spot_close
    stop_line, tgt_line = _stop_target_lines(btst_sig, final_dir, spot)

    # Rule engine line
    if btst_sig.signal == "NO_TRADE":
        rule_str = f"NO_TRADE  [{btst_sig.reason[:48]}]"
    else:
        rule_str = f"{btst_sig.signal} @ {btst_sig.confidence:.0%}"
        if btst_sig.rollover_used:
            rule_str += "  [rollover active]"

    # ML engine lines
    if ml_result:
        agree_note = ""
        if btst_sig.signal != "NO_TRADE":
            agree_note = "  [AGREE]" if _signals_agree(btst_sig, ml_result) else "  [DISAGREE — rule takes precedence]"
        ml_str  = (f"{ml_result['signal']} @ {ml_result['confidence']:.1%}"
                   f"  [ensemble: {ml_result['ensemble_confidence']}]{agree_note}")
        exp_str = f"{ml_result['expected_move_pts']:+.1f} pts  (Model B magnitude estimate)"
    else:
        ml_str  = "models not trained — run: python -m btst_engine.ml_predictor --train"
        exp_str = "N/A"

    sep   = "=" * 64
    blank = ""

    lines = [
        blank,
        sep,
        f"  DATASHOTS — Daily BTST Signal — {d.strftime('%Y-%m-%d')}",
        sep,
        f"  Mode              : {mode}",
        f"  Generated         : {_now_ist().strftime('%Y-%m-%d %H:%M IST')}",
        blank,
        f"  Expiry regime     : {_expiry_regime_str(ctx)}",
        f"  Trade expiry      : {ctx.trade_expiry}",
        f"  NIFTY spot close  : {spot:.2f}",
        blank,
        f"  RULE ENGINE       : {rule_str}",
        f"  ML ENGINE         : {ml_str}",
        f"  Expected move     : {exp_str}",
        blank,
        f"  FINAL SIGNAL      : {final_dir}",
        f"  Strike            : {strike}",
        f"  Entry window      : 3:20 PM – 3:25 PM IST (today)",
        f"  Exit window       : 9:15 AM – 9:20 AM IST (next morning)",
        f"  Stop loss         : {stop_line}",
        f"  Target            : {tgt_line}",
        blank,
        f"  Confidence tier   : {tier}",
        f"    {_TIER_DESC[tier]}",
        sep,
        blank,
    ]
    return "\n".join(lines)


# ── Orchestrator state (for outcome tracking) ─────────────────────────────────

def _state_path(d: date) -> Path:
    return _REPORTS_DIR / f"btst_{d.strftime('%Y-%m-%d')}.json"


def _save_state(
    d:          date,
    final_dir:  str,
    basis_spot: float,
    tier:       str,
    strike:     str,
    stop_level: Optional[float] = None,
    target_level: Optional[float] = None,
    ml_signal:  str = "",
    ml_conf:    float = 0.0,
    ensemble:   str = "",
) -> None:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with _state_path(d).open("w") as f:
        json.dump({
            "date":         d.strftime("%Y-%m-%d"),
            "final_dir":    final_dir,
            "basis_spot":   basis_spot,
            "tier":         tier,
            "strike":       strike,
            "stop_level":   stop_level,
            "target_level": target_level,
            "ml_signal":    ml_signal,
            "ml_conf":      ml_conf,
            "ensemble":     ensemble,
            "generated":    _now_ist().strftime("%Y-%m-%d %H:%M:%S IST"),
        }, f, indent=2)


# ── Main orchestrator function ────────────────────────────────────────────────

def run_orchestrator(date_str: Optional[str] = None) -> None:
    """
    Full daily signal pipeline.
    date_str: "YYYY-MM-DD" or None (auto-detect today / last cache date).
    """

    # ── [1/8] Pre-flight ──────────────────────────────────────────
    print(f"\n{'─'*64}")
    print("[1/8] Pre-flight checks")
    print(f"{'-'*64}")

    if date_str:
        d    = date.fromisoformat(date_str)
        mode = "LIVE" if _has_cache_data(d) else "CACHE_ONLY"
    else:
        d, mode = _find_run_date()

    ds = d.strftime("%Y-%m-%d")
    dow = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d.weekday()]
    print(f"  Date      : {ds} ({dow})")
    print(f"  Mode      : {mode}")
    print(f"  IST time  : {_now_ist().strftime('%H:%M:%S')}")

    if not _is_trading_day(d):
        print(f"  [NOTE] {ds} is a non-trading day — running in CACHE_ONLY mode.")
        mode = "CACHE_ONLY"

    print(f"\n  [DATASHOTS] Running for {ds} — mode: {mode}")

    # ── [2/8] Expiry context ──────────────────────────────────────
    print(f"\n{'─'*64}")
    print("[2/8] Building expiry context")
    print(f"{'-'*64}")

    ctx, snaps = _build_market_context(d)
    print(f"  Regime          : {_expiry_regime_str(ctx)}")
    print(f"  Trade expiry    : {ctx.trade_expiry}")
    print(f"  Confirm expiry  : {ctx.confirm_expiry}")
    print(f"  Rollover mode   : {ctx.rollover_mode}")
    print(f"  Source          : {ctx.source}")

    nifty_snap = snaps["NIFTY"]
    bnf_snap   = snaps["BANKNIFTY"]

    # ── [3/8] Data quality ────────────────────────────────────────
    print(f"\n{'─'*64}")
    print("[3/8] Snapshot data quality")
    print(f"{'-'*64}")

    for label, snap in [("NIFTY    ", nifty_snap), ("BANKNIFTY", bnf_snap)]:
        w1, w2 = snap.w1, snap.w2
        print(f"  {label}  W1: {w1.data_quality:8s}  ({w1.n_bars} bars)"
              f"  W2: {w2.data_quality:8s}  ({w2.n_bars} bars)")
        if w2.data_quality != "MISSING":
            print(f"           PCR W2={w2.pcr:.4f}  delta={w2.net_delta:+.0f}"
                  f"  spot_close={w2.spot_close:.2f}")

    if nifty_snap.w2.data_quality == "MISSING":
        print("\n[ABORT] NIFTY W2 data is MISSING — cannot generate signal for this date.")
        return
    if bnf_snap.w2.data_quality == "MISSING":
        print("\n[ABORT] BANKNIFTY W2 data is MISSING — cannot generate signal for this date.")
        return

    # ── [4/8] Rule-based BTST signal ─────────────────────────────
    print(f"\n{'─'*64}")
    print("[4/8] Rule-based BTST signal engine")
    print(f"{'-'*64}")

    btst_sig = generate_signal(nifty_snap, bnf_snap, ctx, vix=None)

    print(f"  Signal          : {btst_sig.signal}")
    print(f"  Direction       : {btst_sig.direction}")
    print(f"  Confidence      : {btst_sig.confidence:.0%}")
    print(f"  Reason          : {btst_sig.reason}")
    if btst_sig.signal != "NO_TRADE":
        print(f"  Strike          : {btst_sig.strike_ce_pe}")
        print(f"  Trade expiry    : {btst_sig.trade_expiry}")
        print(f"  Rollover used   : {btst_sig.rollover_used}")

    # ── [5/8] ML signal ───────────────────────────────────────────
    print(f"\n{'─'*64}")
    print("[5/8] ML signal engine")
    print(f"{'-'*64}")

    ml_result = _run_ml_signal(ds)

    if ml_result:
        print(f"  Signal          : {ml_result['signal']}  (Model A direction)")
        print(f"  Confidence      : {ml_result['confidence']:.1%}")
        print(f"  Expected move   : {ml_result['expected_move_pts']:+.1f} pts  (Model B — magnitude only)")
        print(f"  Ensemble        : {ml_result['ensemble_confidence']} confidence")
        print(f"  Top 3 features  :")
        for feat, val in list(ml_result["top_features"].items())[:3]:
            print(f"    {feat:<40} {val:>10.4f}")
    else:
        print("  [SKIP] ML models not found.")
        print("         Train first: python -m btst_engine.ml_predictor --train")

    # ── [6/8] Final combined report ───────────────────────────────
    print(f"\n{'─'*64}")
    print("[6/8] Final combined signal report")
    print(f"{'-'*64}")

    final_dir = _final_direction(btst_sig, ml_result)
    strike    = _compute_strike(final_dir, nifty_snap, btst_sig)
    tier      = _confidence_tier(btst_sig, ml_result)
    spot      = nifty_snap.w2.spot_close or nifty_snap.w1.spot_close
    agree     = _signals_agree(btst_sig, ml_result)

    print(f"  Final direction : {final_dir}")
    print(f"  Strike          : {strike}")
    print(f"  Tier            : {tier}  — {_TIER_DESC[tier]}")
    if ml_result:
        print(f"  Rule+ML agree   : {'YES' if agree else 'NO'}")

    report = _format_report(d, mode, ctx, btst_sig, ml_result, nifty_snap)
    print(report)

    # ── [7/8] Log and save ────────────────────────────────────────
    print(f"{'-'*64}")
    print("[7/8] Logging signal and saving report")
    print(f"{'-'*64}")

    logged = _logger.log_signal(btst_sig)
    if logged:
        print(f"  [OK] Signal logged to btst_signals.csv")
    else:
        print(f"  [SKIP] Date {ds} already in btst_signals.csv (backtest data present)")

    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _REPORTS_DIR / f"btst_{ds}.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  [OK] Report saved  → {report_path}")

    # Compute stop/target for the JSON state (needed by dashboard)
    stop_l, tgt_l = _stop_target_lines(btst_sig, final_dir, spot)
    def _extract_level(line: str) -> Optional[float]:
        import re
        m = re.search(r'(\d[\d.]+)', line)
        return float(m.group(1)) if m else None

    _save_state(
        d, final_dir, float(spot), tier, strike,
        stop_level   = _extract_level(stop_l),
        target_level = _extract_level(tgt_l),
        ml_signal    = ml_result["signal"] if ml_result else "",
        ml_conf      = ml_result["confidence"] if ml_result else 0.0,
        ensemble     = ml_result["ensemble_confidence"] if ml_result else "",
    )
    print(f"  [OK] State saved   → {_state_path(d)}")

    # ── [8/8] Summary ─────────────────────────────────────────────
    summary = _logger.generate_summary()
    print(f"\n{'─'*64}")
    print("[8/8] Running totals from btst_signals.csv")
    print(f"{'-'*64}")
    print(f"  Total signals logged : {summary['total_signals']}")
    print(f"  Trade signals        : {summary['trade_signals']}")
    print(f"  NO_TRADE             : {summary['no_trade_count']}")
    wr = summary["win_rate"]
    print(f"  Win rate (W/W+L)     : {wr:.0%}" if wr else "  Win rate: N/A (no outcomes filled yet)")
    ac = summary["avg_confidence"]
    print(f"  Avg confidence       : {ac:.0%}" if ac else "  Avg confidence: N/A")
    print()


# ── Outcome logging ───────────────────────────────────────────────────────────

def log_today_outcome(date_str: str, next_open: float) -> None:
    """
    Log the actual outcome for a previously signalled date.

    Parameters
    ----------
    date_str  : "YYYY-MM-DD" — the signal date (yesterday)
    next_open : actual NIFTY spot open price next morning
    """
    print(f"\n{'─'*64}")
    print(f"[OUTCOME] Loading state for {date_str}")
    print(f"{'-'*64}")

    d  = date.fromisoformat(date_str)
    sp = _state_path(d)

    if sp.exists():
        with sp.open() as f:
            state = json.load(f)
        print(f"  Source    : orchestrator state file")
    else:
        # Fallback: reconstruct from btst_signals.csv
        import csv
        csv_path = _ROOT / "data" / "tracker" / "btst_signals.csv"
        state = None
        if csv_path.exists():
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    if row["date"] != date_str:
                        continue
                    sig = row.get("signal", "")
                    final_dir = (
                        "CE_BUY" if sig == "BULLISH"
                        else "PE_BUY" if sig == "BEARISH"
                        else None
                    )
                    try:
                        basis = float(row.get("basis_spot") or 0)
                    except (ValueError, TypeError):
                        basis = 0.0
                    if final_dir and basis > 0:
                        state = {"date": date_str, "final_dir": final_dir,
                                 "basis_spot": basis, "tier": "?", "strike": ""}
                    break

        if not state:
            print(f"  [ERR] No state or CSV entry found for {date_str}.")
            print(f"        Run orchestrator for that date first.")
            return
        print(f"  Source    : btst_signals.csv fallback")

    final_dir  = state["final_dir"]
    basis_spot = float(state["basis_spot"])
    move_pts   = round(next_open - basis_spot, 2)

    if final_dir == "CE_BUY":
        pnl_pts = move_pts
        outcome = ("WIN" if pnl_pts > 0.5
                   else "LOSS" if pnl_pts < -0.5
                   else "BREAKEVEN")
    else:                               # PE_BUY
        pnl_pts = -move_pts
        outcome = ("WIN" if pnl_pts > 0.5
                   else "LOSS" if pnl_pts < -0.5
                   else "BREAKEVEN")

    notes  = f"tier={state.get('tier','?')} strike={state.get('strike','')} basis={basis_spot:.1f}"
    ok     = _logger.log_outcome(date_str, outcome, exit_spot=next_open, notes=notes)

    arrow  = "+" if pnl_pts >= 0 else ""
    print(f"  Signal date : {date_str}")
    print(f"  Direction   : {final_dir}  (tier {state.get('tier','?')})")
    print(f"  Strike      : {state.get('strike','?')}")
    print(f"  Basis spot  : {basis_spot:.2f}")
    print(f"  Next open   : {next_open:.2f}")
    print(f"  Move        : {move_pts:+.2f} pts")
    print(f"  PnL         : {arrow}{pnl_pts:.2f} pts")
    print(f"  Outcome     : {outcome}")

    if ok:
        print(f"\n  [OK] Outcome logged for {date_str}: {outcome} ({arrow}{pnl_pts:.1f} pts)")
    else:
        print(f"\n  [WARN] Date {date_str} not found in btst_signals.csv — cannot update.")

    # Print updated summary
    summary = _logger.generate_summary()
    wr = summary["win_rate"]
    tp = summary["total_pnl_pts"]
    if wr is not None:
        print(f"\n  Running W/L: {wr:.0%}  |  Total PnL: {tp:+.1f} pts")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
    logging.getLogger("btst_engine.expiry_manager").setLevel(logging.ERROR)
    logging.getLogger("dhanhq").setLevel(logging.CRITICAL)

    args = sys.argv[1:]

    if "--outcome" in args:
        idx = args.index("--outcome")
        if idx + 1 >= len(args):
            print("[ERR] --outcome requires YYYY-MM-DD", file=sys.stderr)
            sys.exit(1)
        out_date = args[idx + 1]

        if "--open" not in args:
            print("[ERR] --outcome requires --open <price>", file=sys.stderr)
            sys.exit(1)
        oi = args.index("--open")
        if oi + 1 >= len(args):
            print("[ERR] --open requires a numeric value", file=sys.stderr)
            sys.exit(1)
        try:
            open_price = float(args[oi + 1])
        except ValueError:
            print(f"[ERR] --open value must be a number, got '{args[oi+1]}'", file=sys.stderr)
            sys.exit(1)

        log_today_outcome(out_date, open_price)

    elif "--date" in args:
        idx = args.index("--date")
        if idx + 1 >= len(args):
            print("[ERR] --date requires YYYY-MM-DD", file=sys.stderr)
            sys.exit(1)
        run_orchestrator(args[idx + 1])

    else:
        run_orchestrator()
