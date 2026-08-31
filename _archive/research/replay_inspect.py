"""
Replay inspection utility for manual signal diagnostics.

Usage:
    python replay_inspect.py
    python replay_inspect.py --date 2026-02-13
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from backtest.rolling_cache_backtester import RollingCacheBacktester
from greeks import GreeksCalculator
from vega_theta_engine import VegaThetaEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect 09:20 vs 10:00 replay signal for one date.")
    parser.add_argument("--date", type=str, default="", help="Trade date in YYYY-MM-DD format.")
    parser.add_argument("--index", type=str, default="NIFTY", help="Index symbol (default: NIFTY).")
    return parser.parse_args()


def ask_date_if_missing(date_arg: str) -> str:
    if date_arg:
        return date_arg.strip()
    date_input = input("Enter trade date (YYYY-MM-DD): ").strip()
    return date_input


def choose_expiry(bt: RollingCacheBacktester, trade_date: str) -> Optional[Tuple[str, int]]:
    """
    Select a primary expiry using simple preference order.
    current: WEEK E1 -> WEEK E2 -> MONTH E1 -> MONTH E2 -> MONTH E3
    """
    candidates = [("WEEK", 1), ("WEEK", 2), ("MONTH", 1), ("MONTH", 2), ("MONTH", 3)]
    for flag, code in candidates:
        labels = bt.available_labels.get((trade_date, flag, code), set())
        if "ATM" in labels:
            return flag, code
    return None


def choose_next_expiry(bt: RollingCacheBacktester, trade_date: str, current_flag: str, current_code: int) -> Optional[Tuple[str, int]]:
    """
    Pick the next available expiry after current.
    """
    ordered = [("WEEK", 1), ("WEEK", 2), ("WEEK", 3), ("MONTH", 1), ("MONTH", 2), ("MONTH", 3)]
    try:
        start_idx = ordered.index((current_flag, current_code)) + 1
    except ValueError:
        start_idx = 0

    for flag, code in ordered[start_idx:]:
        labels = bt.available_labels.get((trade_date, flag, code), set())
        if "ATM" in labels:
            return flag, code
    return None


def build_chain_for_phase(
    bt: RollingCacheBacktester,
    trade_date: str,
    index: str,
    flag: str,
    code: int,
    phase: str,
) -> Optional[Dict]:
    """
    Build a VegaThetaEngine-compatible option chain snapshot from rolling cache.
    """
    labels = bt.available_labels.get((trade_date, flag, code), set())
    if not labels:
        return None

    strike_gap = config.STRIKE_GAPS.get(index, 50)
    dte = {1: 7, 2: 14, 3: 30}.get(code, 7)

    strikes = []
    spot_ref = None

    for label in sorted(labels):
        ce = bt._get(trade_date, flag, code, label, "CALL", phase)
        pe = bt._get(trade_date, flag, code, label, "PUT", phase)
        if not ce or not pe:
            continue
        if ce.iv is None or pe.iv is None:
            continue

        strike = int(round(((ce.strike + pe.strike) / 2.0) / strike_gap) * strike_gap)
        spot = float((ce.spot + pe.spot) / 2.0)
        spot_ref = spot if spot_ref is None else spot_ref

        ce_g = GreeksCalculator.calculate_greeks(
            spot=spot,
            strike=strike,
            dte=dte,
            iv=float(ce.iv) / 100.0,
            option_type="CE",
        )
        pe_g = GreeksCalculator.calculate_greeks(
            spot=spot,
            strike=strike,
            dte=dte,
            iv=float(pe.iv) / 100.0,
            option_type="PE",
        )

        strikes.append(
            {
                "strike": strike,
                "ce": {
                    "premium": float(ce.close),
                    "iv": float(ce.iv),
                    "delta": float(ce_g["delta"]),
                    "gamma": float(ce_g["gamma"]),
                    "theta": float(ce_g["theta"]),
                    "vega": float(ce_g["vega"]),
                    "volume": 0.0,
                    "oi": float(ce.oi or 0.0),
                },
                "pe": {
                    "premium": float(pe.close),
                    "iv": float(pe.iv),
                    "delta": float(pe_g["delta"]),
                    "gamma": float(pe_g["gamma"]),
                    "theta": float(pe_g["theta"]),
                    "vega": float(pe_g["vega"]),
                    "volume": 0.0,
                    "oi": float(pe.oi or 0.0),
                },
            }
        )

    if not strikes or spot_ref is None:
        return None

    strikes.sort(key=lambda x: int(x["strike"]))
    atm = int(round(float(spot_ref) / strike_gap) * strike_gap)

    return {
        "index": index,
        "spot": float(spot_ref),
        "atm": atm,
        "expiry": f"{flag}_E{code}",
        "strikes": strikes,
    }


def print_signal_block(title: str, signal: Dict) -> None:
    raw = signal.get("analytics_features", {}).get("raw", {})
    print("\n" + "=" * 72)
    print(f"{title}")
    print("=" * 72)
    print(f"Regime: {signal.get('regime', 'NA')}")
    print(f"Direction: {signal.get('direction', 'NA')} | Trade Allowed: {signal.get('trade_allowed', False)}")
    print(f"Confidence: {signal.get('confidence', 0):.2f} | Strength: {signal.get('strength', 0):+.2f}")
    print("-" * 72)
    print("Raw Structural Features")
    print(f"  flow_delta_true       : {float(raw.get('flow_delta_true', 0.0)):+.4f}")
    print(f"  gamma_skew_shift_true : {float(raw.get('gamma_skew_shift_true', 0.0)):+.4f}")
    print(f"  pos_delta_now         : {float(raw.get('pos_delta_now', 0.0)):+.4f}")
    print(f"  spot_change_pct       : {float(raw.get('spot_change_pct', 0.0)):+.4f}%")
    print("-" * 72)
    print("Interpretation")
    print(f"  {signal.get('interpretation', '')}")
    stack = signal.get("rule_stack", [])
    print("Rule Stack")
    if stack:
        for item in stack:
            print(f"  - {item}")
    else:
        print("  - (none)")


def main() -> int:
    args = parse_args()
    trade_date = ask_date_if_missing(args.date)
    index = args.index.strip().upper()

    cache_root = Path("data") / "backtest" / "cache" / "rolling_options" / index
    bt = RollingCacheBacktester(cache_root=cache_root)
    bt._build_samples(index=index, start_date=trade_date, end_date=trade_date)

    current_expiry = choose_expiry(bt, trade_date)
    if not current_expiry:
        print(f"\nERROR: No usable cached expiry found for {index} on {trade_date}.")
        print(f"Searched under: {cache_root}")
        return 1

    cur_flag, cur_code = current_expiry
    next_expiry = choose_next_expiry(bt, trade_date, cur_flag, cur_code)

    baseline_current = build_chain_for_phase(bt, trade_date, index, cur_flag, cur_code, phase="baseline")
    entry_current = build_chain_for_phase(bt, trade_date, index, cur_flag, cur_code, phase="entry")

    if baseline_current is None or entry_current is None:
        print(f"\nERROR: Required snapshots not found for {trade_date}.")
        print("Expected two phase snapshots around baseline (~09:20) and entry (~10:00).")
        print(f"Resolved current expiry: {cur_flag}_E{cur_code}")
        return 1

    baseline_next = None
    entry_next = None
    if next_expiry:
        nflag, ncode = next_expiry
        baseline_next = build_chain_for_phase(bt, trade_date, index, nflag, ncode, phase="baseline")
        entry_next = build_chain_for_phase(bt, trade_date, index, nflag, ncode, phase="entry")
        if baseline_next is None or entry_next is None:
            baseline_next = None
            entry_next = None

    engine = VegaThetaEngine()
    # Set baseline in-memory to avoid writing files during inspection.
    engine.baseline[index] = {
        "index": index,
        "timestamp": "replay_inspect",
        "current": baseline_current,
        "next": baseline_next,
    }

    signal_0920 = engine.generate_signal(
        index=index,
        current_chain=baseline_current,
        next_chain=baseline_next,
        market_context={},
    )
    signal_1000 = engine.generate_signal(
        index=index,
        current_chain=entry_current,
        next_chain=entry_next,
        market_context={},
    )

    if not signal_0920 or not signal_1000:
        print("\nERROR: Could not generate one or both signals from cached snapshots.")
        return 1

    print("\nReplay Inspection")
    print(f"Date: {trade_date} | Index: {index}")
    print(f"Current expiry used: {cur_flag}_E{cur_code}")
    if next_expiry and baseline_next and entry_next:
        print(f"Next expiry used: {next_expiry[0]}_E{next_expiry[1]}")
    else:
        print("Next expiry used: (missing) -> current only")

    print_signal_block("09:20 Snapshot (Baseline vs Baseline)", signal_0920)
    print_signal_block("10:00 Snapshot (Baseline vs 10:00)", signal_1000)

    print("\n" + "=" * 72)
    print("Quick Comparison (09:20 -> 10:00)")
    print("=" * 72)
    print(f"Regime: {signal_0920.get('regime')} -> {signal_1000.get('regime')}")
    print(f"Direction: {signal_0920.get('direction')} -> {signal_1000.get('direction')}")
    print(f"Trade Allowed: {signal_0920.get('trade_allowed')} -> {signal_1000.get('trade_allowed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
