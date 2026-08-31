"""
Manual replay reader for one date (09:20 vs 10:00) using raw structural metrics.

This script is intentionally simple:
- loads two cached snapshots
- computes raw baseline-relative metrics
- prints human-readable inspection
- appends a manual validation log row
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from analytics.greek_flow import compute_flow_metrics
from backtest.rolling_cache_backtester import RollingCacheBacktester
from greeks import GreeksCalculator
from vega_theta_engine import VegaThetaEngine


def _log_error(message: str) -> None:
    """
    Append runtime errors to a local system log for post-mortem inspection.
    """
    log_path = Path("data") / "backtest" / "manual_reader_errors.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {message}\n")


def _ask_inputs() -> tuple[str, str]:
    date_str = input("Enter date (YYYY-MM-DD): ").strip()
    index = input("Enter index (e.g., NIFTY): ").strip().upper()
    return date_str, index


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_chain(payload: Dict, index: str) -> Optional[Dict]:
    """
    Accept common snapshot shapes and return chain with keys:
    index, spot, atm, strikes
    """
    if not isinstance(payload, dict):
        return None

    # Direct chain shape.
    if {"spot", "atm", "strikes"}.issubset(payload.keys()):
        chain = dict(payload)
        chain.setdefault("index", index)
        return chain

    # Wrapped shape.
    if isinstance(payload.get("current_chain"), dict):
        chain = dict(payload["current_chain"])
        chain.setdefault("index", index)
        return chain

    if isinstance(payload.get("chain"), dict):
        chain = dict(payload["chain"])
        chain.setdefault("index", index)
        return chain

    return None


def _choose_expiry(bt: RollingCacheBacktester, trade_date: str) -> Optional[Tuple[str, int]]:
    candidates = [("WEEK", 1), ("WEEK", 2), ("WEEK", 3), ("MONTH", 1), ("MONTH", 2), ("MONTH", 3)]
    for flag, code in candidates:
        labels = bt.available_labels.get((trade_date, flag, code), set())
        if "ATM" in labels:
            return flag, code
    return None


def _build_chain_from_rolling_cache(
    bt: RollingCacheBacktester,
    trade_date: str,
    index: str,
    flag: str,
    code: int,
    phase: str,
) -> Optional[Dict]:
    labels = bt.available_labels.get((trade_date, flag, code), set())
    if not labels:
        return None

    strike_gap = config.STRIKE_GAPS.get(index, 50)
    dte = {1: 7, 2: 14, 3: 30}.get(code, 7)
    rows = []
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
        if spot_ref is None:
            spot_ref = spot

        ce_g = GreeksCalculator.calculate_greeks(
            spot=spot, strike=strike, dte=dte, iv=float(ce.iv) / 100.0, option_type="CE"
        )
        pe_g = GreeksCalculator.calculate_greeks(
            spot=spot, strike=strike, dte=dte, iv=float(pe.iv) / 100.0, option_type="PE"
        )

        rows.append(
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

    if not rows or spot_ref is None:
        return None

    rows.sort(key=lambda x: int(x["strike"]))
    atm = int(round(float(spot_ref) / strike_gap) * strike_gap)

    return {
        "index": index,
        "spot": float(spot_ref),
        "atm": atm,
        "expiry": f"{flag}_E{code}",
        "strikes": rows,
    }


def _sum_oi(chain: Dict) -> Tuple[float, float]:
    call_oi = 0.0
    put_oi = 0.0
    for row in chain.get("strikes", []):
        try:
            call_oi += float(row.get("ce", {}).get("oi", 0.0) or 0.0)
            put_oi += float(row.get("pe", {}).get("oi", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return call_oi, put_oi


def _atm_iv(chain: Dict) -> float:
    atm = int(chain.get("atm", 0) or 0)
    for row in chain.get("strikes", []):
        try:
            if int(row.get("strike", 0) or 0) != atm:
                continue
            ce_iv = float(row.get("ce", {}).get("iv", 0.0) or 0.0)
            pe_iv = float(row.get("pe", {}).get("iv", 0.0) or 0.0)
            return (ce_iv + pe_iv) / 2.0
        except (TypeError, ValueError):
            continue
    return 0.0


def _metric_block(base_chain: Dict, now_chain: Dict) -> Dict[str, float]:
    flow = compute_flow_metrics(base_chain=base_chain, now_chain=now_chain, window_strikes_each_side=6)
    call_oi, put_oi = _sum_oi(now_chain)
    base_call_oi, base_put_oi = _sum_oi(base_chain)
    total_oi = call_oi + put_oi
    oi_imbalance = ((put_oi - call_oi) / total_oi) if total_oi > 0 else 0.0

    return {
        "spot": float(now_chain.get("spot", 0.0)),
        "flow_delta_true": float(flow.flow_delta_true),
        "flow_vega_true": float(flow.flow_vega_true),
        "gamma_shift_true": float(flow.gamma_shift_true),
        "gamma_skew_shift_true": float(flow.gamma_skew_shift_true),
        "pos_delta_now": float(flow.pos_delta_now),
        "spot_change_pct": float(flow.spot_change_pct),
        "total_call_oi": float(call_oi),
        "total_put_oi": float(put_oi),
        "call_oi_change": float(call_oi - base_call_oi),
        "put_oi_change": float(put_oi - base_put_oi),
        "oi_imbalance": float(oi_imbalance),
        "atm_iv": float(_atm_iv(now_chain)),
    }


def _print_block(label: str, m: Dict[str, float]) -> None:
    print(f"\n🕒 {label}")
    print("-" * 72)
    print(f"🔹 Spot: {m['spot']:.2f}")
    print(f"🔹 flow_delta_true: {m['flow_delta_true']:+.4f}")
    print(f"🔹 flow_vega_true: {m['flow_vega_true']:+.4f}")
    print(f"🔹 gamma_shift_true: {m['gamma_shift_true']:+.4f}")
    print(f"🔹 gamma_skew_shift_true: {m['gamma_skew_shift_true']:+.4f}")
    print(f"🔹 pos_delta_now: {m['pos_delta_now']:+.4f}")
    print(f"🔹 spot_change_pct: {m['spot_change_pct']:+.4f}%")
    print(f"🔹 total_call_oi: {m['total_call_oi']:.0f}")
    print(f"🔹 total_put_oi: {m['total_put_oi']:.0f}")
    print(f"🔹 Call OI Change: {m['call_oi_change']:+.0f}")
    print(f"🔹 Put OI Change: {m['put_oi_change']:+.0f}")
    print(f"🔹 OI put/call imbalance: {m['oi_imbalance']:+.4f}")
    print(f"🔹 ATM IV: {m['atm_iv']:.4f}")

    # Human-readable quick interpretation.
    if m["flow_delta_true"] > 0:
        print("  🟢 Delta flow: positive (bullish pressure)")
    elif m["flow_delta_true"] < 0:
        print("  🔴 Delta flow: negative (bearish pressure)")
    else:
        print("  🟡 Delta flow: flat")

    if m["flow_vega_true"] > 0:
        print("  🔹 Vega flow: expanding on call-dominant side")
    elif m["flow_vega_true"] < 0:
        print("  🔹 Vega flow: expanding on put-dominant side")
    else:
        print("  🔹 Vega flow: neutral")

    if abs(m["gamma_shift_true"]) >= 30:
        print("  🔥 Gamma: elevated structural stress")
    else:
        print("  🔹 Gamma: moderate structural stress")

    if m["oi_imbalance"] > 0:
        print("  🔹 OI build-up: put side heavier")
    elif m["oi_imbalance"] < 0:
        print("  🔹 OI build-up: call side heavier")
    else:
        print("  🔹 OI build-up: balanced")



def _print_diffs(m1: Dict[str, float], m2: Dict[str, float]) -> None:
    print("\n📍 Metric Differences (10:00 - 09:20)")
    print("-" * 72)
    keys = [
        "spot",
        "flow_delta_true",
        "flow_vega_true",
        "gamma_shift_true",
        "gamma_skew_shift_true",
        "pos_delta_now",
        "spot_change_pct",
        "total_call_oi",
        "total_put_oi",
        "call_oi_change",
        "put_oi_change",
        "oi_imbalance",
        "atm_iv",
    ]
    for k in keys:
        print(f"{k}: {m2[k] - m1[k]:+.6f}")


def _append_log(
    csv_path: Path,
    date_str: str,
    index: str,
    m10: Dict[str, float],
    regime: str,
    engine_direction: str,
    actual_move_pct: float,
    actual_label: str,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fieldnames = [
        "date",
        "index",
        "flow_delta_true",
        "flow_vega_true",
        "gamma_shift_true",
        "gamma_skew_shift_true",
        "pos_delta_now",
        "spot_change_pct",
        "call_oi",
        "put_oi",
        "regime",
        "engine_direction",
        "actual_move_pct",
        "actual_label",
    ]
    row = {
        "date": date_str,
        "index": index,
        "flow_delta_true": f"{m10['flow_delta_true']:.6f}",
        "flow_vega_true": f"{m10['flow_vega_true']:.6f}",
        "gamma_shift_true": f"{m10['gamma_shift_true']:.6f}",
        "gamma_skew_shift_true": f"{m10['gamma_skew_shift_true']:.6f}",
        "pos_delta_now": f"{m10['pos_delta_now']:.6f}",
        "spot_change_pct": f"{m10['spot_change_pct']:.6f}",
        "call_oi": f"{m10['total_call_oi']:.0f}",
        "put_oi": f"{m10['total_put_oi']:.0f}",
        "regime": regime,
        "engine_direction": engine_direction,
        "actual_move_pct": f"{actual_move_pct:.6f}",
        "actual_label": actual_label,
    }

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    date_str, index = _ask_inputs()

    p0920 = Path("data") / "backtest" / "cache" / "rolling_options" / index / date_str / "09:20.json"
    p1000 = Path("data") / "backtest" / "cache" / "rolling_options" / index / date_str / "10:00.json"

    chain_0920 = None
    chain_1000 = None
    if p0920.exists() and p1000.exists():
        payload_0920 = _load_json(p0920)
        payload_1000 = _load_json(p1000)
        chain_0920 = _extract_chain(payload_0920, index=index)
        chain_1000 = _extract_chain(payload_1000, index=index)
        if chain_0920 is None or chain_1000 is None:
            print("\n❌ ERROR: Could not parse one or both direct snapshot files into option-chain format.")
            return 1
    else:
        # Fallback: use existing rolling cache format and nearest phase samples.
        cache_root = Path("data") / "backtest" / "cache" / "rolling_options" / index
        bt = RollingCacheBacktester(cache_root=cache_root)
        bt._build_samples(index=index, start_date=date_str, end_date=date_str)
        expiry = _choose_expiry(bt, date_str)
        if not expiry:
            msg = (
                f"Direct snapshot files missing and no rolling-cache samples found for "
                f"index={index} date={date_str}"
            )
            _log_error(msg)
            print("\n❌ ERROR: Direct snapshot files missing and no rolling-cache samples found.")
            print(f"Expected direct files:\n  {p0920}\n  {p1000}")
            print(f"Searched rolling cache in: {cache_root}")
            return 1
        flag, code = expiry
        chain_0920 = _build_chain_from_rolling_cache(bt, date_str, index, flag, code, phase="baseline")
        chain_1000 = _build_chain_from_rolling_cache(bt, date_str, index, flag, code, phase="entry")
        if chain_0920 is None or chain_1000 is None:
            _log_error(
                f"Could not reconstruct 09:20/10:00 snapshots from rolling cache "
                f"for index={index} date={date_str}"
            )
            print("\n❌ ERROR: Could not reconstruct 09:20/10:00 snapshots from rolling cache.")
            return 1

    m0920 = _metric_block(base_chain=chain_0920, now_chain=chain_0920)
    m1000 = _metric_block(base_chain=chain_0920, now_chain=chain_1000)

    engine = VegaThetaEngine()
    engine.baseline[index] = {
        "index": index,
        "timestamp": "manual_reader",
        "current": chain_0920,
        "next": None,
    }
    signal_1000 = engine.generate_signal(index=index, current_chain=chain_1000, next_chain=None, market_context={})
    if not signal_1000:
        _log_error(f"Engine signal generation failed for index={index} date={date_str}")
        print("\n❌ ERROR: Engine signal generation failed for 10:00 snapshot.")
        return 1

    regime_1000 = str(signal_1000.get("regime", "NA"))
    direction_1000 = str(signal_1000.get("direction", "NA"))

    print("\n" + "=" * 72)
    print(f"DATE: {date_str}")
    print(f"INDEX: {index}")
    print("=" * 72)
    _print_block("09:20 Snapshot", m0920)
    _print_block("10:00 Snapshot", m1000)
    print(f"\n📌 Engine 10:00 Regime: {regime_1000}")
    print(f"📌 Engine 10:00 Direction: {direction_1000}")
    _print_diffs(m0920, m1000)

    move_str = input("\nEnter actual net move % after 10:00: ").strip()
    label_str = input("Enter actual direction label (Bullish/Bearish/Sideways): ").strip().upper()
    if label_str not in {"BULLISH", "BEARISH", "SIDEWAYS"}:
        _log_error(
            f"Invalid actual label input index={index} date={date_str} value={label_str}"
        )
        print("❌ ERROR: actual label must be Bullish/Bearish/Sideways.")
        return 1
    try:
        actual_move_pct = float(move_str)
    except ValueError:
        _log_error(
            f"Invalid actual move pct input index={index} date={date_str} value={move_str}"
        )
        print("❌ ERROR: actual move % must be numeric.")
        return 1

    log_path = Path("data") / "backtest" / "manual_validation_log.csv"
    _append_log(
        csv_path=log_path,
        date_str=date_str,
        index=index,
        m10=m1000,
        regime=regime_1000,
        engine_direction=direction_1000,
        actual_move_pct=actual_move_pct,
        actual_label=label_str,
    )
    print(f"✅ Saved log row to: {log_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        _log_error(f"Unhandled exception: {exc}")
        raise
