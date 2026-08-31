"""
Single-date replay tool for 09:20 vs 10:00 structural inspection.

This wrapper is intentionally simple and deterministic:
- loads historical cached snapshots
- computes raw baseline-relative metrics
- runs VegaThetaEngine
- prints readable interpretation
- logs manual validation outcome
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from analytics.greek_flow import compute_flow_metrics
from analytics.regime import VolatilityRegimeDetector
from analytics.rules import RuleBasedClassifier
from backtest.rolling_cache_backtester import RollingCacheBacktester
from greeks import GreeksCalculator
from vega_theta_engine import VegaThetaEngine


def _ask_inputs() -> tuple[str, str, str]:
    date_str = input("Enter DATE (YYYY-MM-DD): ").strip()
    index = input("Enter INDEX (default NIFTY): ").strip().upper() or "NIFTY"
    check_time = input("Enter CHECK TIME HH:MM (default 10:00): ").strip() or "10:00"
    return date_str, index, check_time


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_chain(payload: Dict, index: str) -> Optional[Dict]:
    if not isinstance(payload, dict):
        return None
    if {"spot", "atm", "strikes"}.issubset(payload.keys()):
        chain = dict(payload)
        chain.setdefault("index", index)
        return chain
    if isinstance(payload.get("current_chain"), dict):
        chain = dict(payload["current_chain"])
        chain.setdefault("index", index)
        return chain
    if isinstance(payload.get("chain"), dict):
        chain = dict(payload["chain"])
        chain.setdefault("index", index)
        return chain
    return None


def _choose_expiry(
    bt: RollingCacheBacktester,
    trade_date: str,
    min_premium_ratio: float = 0.01,
) -> Optional[Tuple[str, int]]:
    for flag, code in [("WEEK", 1), ("WEEK", 2), ("WEEK", 3), ("MONTH", 1), ("MONTH", 2), ("MONTH", 3)]:
        labels = bt.available_labels.get((trade_date, flag, code), set())
        if "ATM" not in labels:
            continue

        ce = bt._get(trade_date, flag, code, "ATM", "CALL", "entry")
        pe = bt._get(trade_date, flag, code, "ATM", "PUT", "entry")
        if not ce or not pe or not ce.spot:
            continue

        spot = float(ce.spot)
        straddle = float(ce.close) + float(pe.close)
        premium_ratio = (straddle / spot) if spot > 0 else 0.0
        if premium_ratio >= min_premium_ratio:
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
    return {"index": index, "spot": float(spot_ref), "atm": atm, "expiry": f"{flag}_E{code}", "strikes": rows}


def _closest_idx_from_ts(ts_list: list, target_ts: int) -> Optional[int]:
    if not ts_list:
        return None
    try:
        return min(range(len(ts_list)), key=lambda i: abs(int(ts_list[i]) - int(target_ts)))
    except Exception:
        return None


def _find_rolling_file(
    cache_root: Path,
    trade_date: str,
    flag: str,
    code: int,
    strike_label: str,
    option_type: str,
) -> Optional[Path]:
    pattern = f"*_{flag}_E{code}_{strike_label}_{option_type}.json"
    target = datetime.strptime(trade_date, "%Y-%m-%d").date()
    for path in cache_root.glob(pattern):
        m = RollingCacheBacktester.FILE_RE.match(path.name)
        if not m:
            continue
        start = datetime.strptime(m.group("from"), "%Y-%m-%d").date()
        end = datetime.strptime(m.group("to"), "%Y-%m-%d").date()
        if start <= target <= end:
            return path
    return None


def _build_chain_at_time_from_rolling_cache(
    cache_root: Path,
    bt: RollingCacheBacktester,
    trade_date: str,
    index: str,
    flag: str,
    code: int,
    hhmm: str,
) -> Optional[Dict]:
    labels = bt.available_labels.get((trade_date, flag, code), set())
    if not labels:
        return None

    try:
        target_dt_ist = datetime.strptime(f"{trade_date} {hhmm}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    target_ts_utc = int((target_dt_ist - timedelta(hours=5, minutes=30)).timestamp())

    strike_gap = config.STRIKE_GAPS.get(index, 50)
    dte = {1: 7, 2: 14, 3: 30}.get(code, 7)
    rows = []
    spot_ref = None

    for label in sorted(labels):
        ce_path = _find_rolling_file(cache_root, trade_date, flag, code, label, "CALL")
        pe_path = _find_rolling_file(cache_root, trade_date, flag, code, label, "PUT")
        if not ce_path or not pe_path:
            continue

        ce_payload = _load_json(ce_path).get("data", {}).get("ce", {})
        pe_payload = _load_json(pe_path).get("data", {}).get("pe", {})

        ce_ts = ce_payload.get("timestamp") or []
        pe_ts = pe_payload.get("timestamp") or []
        ce_i = _closest_idx_from_ts(ce_ts, target_ts_utc)
        pe_i = _closest_idx_from_ts(pe_ts, target_ts_utc)
        if ce_i is None or pe_i is None:
            continue

        try:
            ce_close = float((ce_payload.get("close") or [])[ce_i])
            pe_close = float((pe_payload.get("close") or [])[pe_i])
            ce_iv = float((ce_payload.get("iv") or [])[ce_i])
            pe_iv = float((pe_payload.get("iv") or [])[pe_i])
            ce_oi = float((ce_payload.get("oi") or [])[ce_i] or 0.0)
            pe_oi = float((pe_payload.get("oi") or [])[pe_i] or 0.0)
            ce_strike = float((ce_payload.get("strike") or [])[ce_i])
            pe_strike = float((pe_payload.get("strike") or [])[pe_i])
            ce_spot = float((ce_payload.get("spot") or [])[ce_i])
            pe_spot = float((pe_payload.get("spot") or [])[pe_i])
        except (TypeError, ValueError, IndexError):
            continue

        strike = int(round(((ce_strike + pe_strike) / 2.0) / strike_gap) * strike_gap)
        spot = float((ce_spot + pe_spot) / 2.0)
        if spot_ref is None:
            spot_ref = spot

        ce_g = GreeksCalculator.calculate_greeks(
            spot=spot,
            strike=strike,
            dte=dte,
            iv=ce_iv / 100.0,
            option_type="CE",
        )
        pe_g = GreeksCalculator.calculate_greeks(
            spot=spot,
            strike=strike,
            dte=dte,
            iv=pe_iv / 100.0,
            option_type="PE",
        )

        rows.append(
            {
                "strike": strike,
                "ce": {
                    "premium": ce_close,
                    "iv": ce_iv,
                    "delta": float(ce_g["delta"]),
                    "gamma": float(ce_g["gamma"]),
                    "theta": float(ce_g["theta"]),
                    "vega": float(ce_g["vega"]),
                    "volume": 0.0,
                    "oi": ce_oi,
                },
                "pe": {
                    "premium": pe_close,
                    "iv": pe_iv,
                    "delta": float(pe_g["delta"]),
                    "gamma": float(pe_g["gamma"]),
                    "theta": float(pe_g["theta"]),
                    "vega": float(pe_g["vega"]),
                    "volume": 0.0,
                    "oi": pe_oi,
                },
            }
        )

    if not rows or spot_ref is None:
        return None
    rows.sort(key=lambda x: int(x["strike"]))
    atm = int(round(float(spot_ref) / strike_gap) * strike_gap)
    return {"index": index, "spot": float(spot_ref), "atm": atm, "expiry": f"{flag}_E{code}", "strikes": rows}


def _load_snapshots(date_str: str, index: str, check_time: str) -> tuple[Optional[Dict], Optional[Dict], str]:
    p0920 = Path("data") / "backtest" / "cache" / "rolling_options" / index / date_str / "09:20.json"
    p1000 = Path("data") / "backtest" / "cache" / "rolling_options" / index / date_str / "10:00.json"
    if p0920.exists() and p1000.exists():
        c0920 = _extract_chain(_load_json(p0920), index=index)
        c1000 = _extract_chain(_load_json(p1000), index=index)
        return c0920, c1000, "direct-json"

    cache_root = Path("data") / "backtest" / "cache" / "rolling_options" / index
    bt = RollingCacheBacktester(cache_root=cache_root)
    bt._build_samples(index=index, start_date=date_str, end_date=date_str)
    expiry = _choose_expiry(bt, date_str)
    if not expiry:
        return None, None, "missing"

    flag, code = expiry
    c0920 = _build_chain_from_rolling_cache(bt, date_str, index, flag, code, phase="baseline")
    cache_root = Path("data") / "backtest" / "cache" / "rolling_options" / index
    c_now = _build_chain_at_time_from_rolling_cache(
        cache_root=cache_root,
        bt=bt,
        trade_date=date_str,
        index=index,
        flag=flag,
        code=code,
        hhmm=check_time,
    )
    return c0920, c_now, f"rolling-cache({flag}_E{code})"


def _ask_float(prompt: str, default: float = 0.0) -> float:
    raw = input(prompt).strip()
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _manual_mode(date_str: str, index: str) -> int:
    print("\n⚠️ No cache found. Running MANUAL MODE so tool remains usable.")
    print("🔹 Enter raw values; press Enter to use 0.")

    spot_0920 = _ask_float("09:20 spot: ", 0.0)
    spot_1000 = _ask_float("10:00 spot: ", spot_0920)

    flow_delta = _ask_float("10:00 flow_delta_true: ", 0.0)
    flow_vega = _ask_float("10:00 flow_vega_true: ", 0.0)
    gamma_shift = _ask_float("10:00 gamma_shift_true: ", 0.0)
    gamma_skew = _ask_float("10:00 gamma_skew_shift_true: ", 0.0)
    pos_delta = _ask_float("10:00 pos_delta_now: ", 0.0)
    call_oi = _ask_float("10:00 total_call_oi: ", 0.0)
    put_oi = _ask_float("10:00 total_put_oi: ", 0.0)

    total_oi = call_oi + put_oi
    oi_imbalance = ((put_oi - call_oi) / total_oi) if total_oi > 0 else 0.0
    spot_change_pct = ((spot_1000 - spot_0920) / spot_0920 * 100.0) if spot_0920 else 0.0

    raw_0920 = {
        "spot": spot_0920,
        "flow_delta_true": 0.0,
        "flow_vega_true": 0.0,
        "gamma_shift_true": 0.0,
        "gamma_skew_shift_true": 0.0,
        "pos_delta_now": 0.0,
        "spot_change_pct": 0.0,
        "total_call_oi": call_oi,
        "total_put_oi": put_oi,
        "oi_imbalance": oi_imbalance,
        "total_iv_change": 0.0,
        "call_iv_diff": 0.0,
        "put_iv_diff": 0.0,
    }
    raw_1000 = {
        "spot": spot_1000,
        "flow_delta_true": flow_delta,
        "flow_vega_true": flow_vega,
        "gamma_shift_true": gamma_shift,
        "gamma_skew_shift_true": gamma_skew,
        "pos_delta_now": pos_delta,
        "spot_change_pct": spot_change_pct,
        "total_call_oi": call_oi,
        "total_put_oi": put_oi,
        "oi_imbalance": oi_imbalance,
        "total_iv_change": 0.0,
        "call_iv_diff": 0.0,
        "put_iv_diff": 0.0,
    }

    detector = VolatilityRegimeDetector()
    classifier = RuleBasedClassifier()

    regime_0920 = detector.detect({}, {}, structural_features=raw_0920)
    regime_1000 = detector.detect({}, {}, structural_features=raw_1000)
    decision_0920 = classifier.classify(raw_features=raw_0920, regime=regime_0920.label)
    decision_1000 = classifier.classify(raw_features=raw_1000, regime=regime_1000.label)

    print("\n" + "=" * 72)
    print(f"DATE: {date_str}")
    print(f"INDEX: {index}")
    print("SOURCE: manual-input")
    print("=" * 72)
    _print_snapshot(
        "09:20",
        raw_0920,
        regime_0920.label,
        decision_0920.direction,
        decision_0920.interpretation,
        decision_0920.strategy_bias,
        decision_0920.confidence,
        decision_0920.vega_state,
        decision_0920.iv_state,
        decision_0920.delta_state,
        decision_0920.gamma_stress,
    )
    _print_snapshot(
        "10:00",
        raw_1000,
        regime_1000.label,
        decision_1000.direction,
        decision_1000.interpretation,
        decision_1000.strategy_bias,
        decision_1000.confidence,
        decision_1000.vega_state,
        decision_1000.iv_state,
        decision_1000.delta_state,
        decision_1000.gamma_stress,
    )

    spot_move = raw_1000["spot"] - raw_0920["spot"]
    print(f"\n📍 Spot Change (09:20 -> 10:00): {spot_move:+.2f} ({spot_change_pct:+.4f}%)")

    move_input = input("\nWhat was actual move % after 10:00? ").strip()
    bias_input = input("Actual bias? (BULLISH/BEARISH/SIDEWAYS): ").strip().upper()
    if bias_input not in {"BULLISH", "BEARISH", "SIDEWAYS"}:
        print("❌ ERROR: bias must be one of BULLISH, BEARISH, SIDEWAYS.")
        return 1
    try:
        actual_move_pct = float(move_input)
    except ValueError:
        print("❌ ERROR: actual move % must be numeric.")
        return 1

    _append_accuracy_log(
        date_str=date_str,
        regime=regime_1000.label,
        direction=decision_1000.direction,
        actual_move_pct=actual_move_pct,
        actual_bias=bias_input,
    )
    print("✅ Saved log row to: data/backtest/accuracy_log.csv")
    return 0


def _oi_summary(chain: Dict) -> tuple[float, float, float]:
    call_oi = 0.0
    put_oi = 0.0
    for row in chain.get("strikes", []):
        try:
            call_oi += float(row.get("ce", {}).get("oi", 0.0) or 0.0)
            put_oi += float(row.get("pe", {}).get("oi", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    total = call_oi + put_oi
    imbalance = ((put_oi - call_oi) / total) if total > 0 else 0.0
    return call_oi, put_oi, imbalance


def _oi_weighted_leg_iv(chain: Dict) -> tuple[float, float]:
    call_num = 0.0
    call_den = 0.0
    put_num = 0.0
    put_den = 0.0
    for row in chain.get("strikes", []):
        try:
            ce = row.get("ce", {})
            pe = row.get("pe", {})
            call_iv = float(ce.get("iv", 0.0) or 0.0)
            put_iv = float(pe.get("iv", 0.0) or 0.0)
            call_oi = float(ce.get("oi", 0.0) or 0.0)
            put_oi = float(pe.get("oi", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        call_num += call_iv * call_oi
        call_den += call_oi
        put_num += put_iv * put_oi
        put_den += put_oi
    call_avg = (call_num / call_den) if call_den > 0 else 0.0
    put_avg = (put_num / put_den) if put_den > 0 else 0.0
    return call_avg, put_avg


def _atm_avg_iv(chain: Dict) -> float:
    atm = int(chain.get("atm", 0) or 0)
    for row in chain.get("strikes", []):
        try:
            strike = int(row.get("strike", 0) or 0)
            if strike != atm:
                continue
            ce_iv = float(row.get("ce", {}).get("iv", 0.0) or 0.0)
            pe_iv = float(row.get("pe", {}).get("iv", 0.0) or 0.0)
            return (ce_iv + pe_iv) / 2.0
        except (TypeError, ValueError):
            continue
    return 0.0


def _metrics(base_chain: Dict, now_chain: Dict) -> Dict[str, float]:
    flow = compute_flow_metrics(base_chain=base_chain, now_chain=now_chain, window_strikes_each_side=6)
    call_oi, put_oi, oi_imbalance = _oi_summary(now_chain)
    base_call_oi, base_put_oi, _ = _oi_summary(base_chain)
    call_iv_now, put_iv_now = _oi_weighted_leg_iv(now_chain)
    call_iv_base, put_iv_base = _oi_weighted_leg_iv(base_chain)
    atm_iv_now = _atm_avg_iv(now_chain)
    atm_iv_base = _atm_avg_iv(base_chain)
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
        "call_iv_diff": float(call_iv_now - call_iv_base),
        "put_iv_diff": float(put_iv_now - put_iv_base),
        "total_iv_change": float(atm_iv_now - atm_iv_base),
    }


def _print_snapshot(
    label: str,
    metrics: Dict[str, float],
    regime: str,
    direction: str,
    interpretation: str,
    strategy_bias: str,
    confidence: float,
    vega_state: str,
    iv_state: str,
    delta_state: str,
    gamma_stress: str,
) -> None:
    print(f"\n🕒 {label}")
    print(f"🔹 Spot: {metrics['spot']:.2f}")
    print(f"🔹 flow_delta_true: {metrics['flow_delta_true']:+.4f}")
    print(f"🔹 flow_vega_true: {metrics['flow_vega_true']:+.4f}")
    print(f"🔹 gamma_shift_true: {metrics['gamma_shift_true']:+.4f}")
    print(f"🔹 gamma_skew_shift_true: {metrics['gamma_skew_shift_true']:+.4f}")
    print(f"🔹 pos_delta_now: {metrics['pos_delta_now']:+.4f}")
    print(f"🔹 spot_change_pct: {metrics['spot_change_pct']:+.4f}%")
    print(f"🔹 total_call_oi: {metrics['total_call_oi']:.0f}")
    print(f"🔹 total_put_oi: {metrics['total_put_oi']:.0f}")
    print(f"🔹 Call OI Change: {metrics['call_oi_change']:+.0f}")
    print(f"🔹 Put OI Change: {metrics['put_oi_change']:+.0f}")
    print(f"🔹 OI imbalance: {metrics['oi_imbalance']:+.4f}")
    print(f"🔹 Call IV Change: {metrics['call_iv_diff']:+.4f}")
    print(f"🔹 Put IV Change: {metrics['put_iv_diff']:+.4f}")
    print(f"Vega State: {vega_state}")
    print(f"IV State: {iv_state}")
    print(f"Delta State: {delta_state}")
    print(f"🔥 Gamma Stress: {gamma_stress}")
    print(f"📊 Confidence: {confidence:.2f}")
    print(f"📌 regime: {regime}")
    print(f"📌 direction: {direction}")
    print(f"📌 Strategy Bias: {strategy_bias}")
    print(f"📝 interpretation: {interpretation}")


def _append_accuracy_log(
    date_str: str,
    regime: str,
    direction: str,
    actual_move_pct: float,
    actual_bias: str,
) -> None:
    out = Path("data") / "backtest" / "accuracy_log.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()

    correct = direction == actual_bias
    row = {
        "date": date_str,
        "regime": regime,
        "direction": direction,
        "actual_move_pct": f"{actual_move_pct:.6f}",
        "actual_bias": actual_bias,
        "correct_or_not": str(correct),
    }
    fields = ["date", "regime", "direction", "actual_move_pct", "actual_bias", "correct_or_not"]
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    date_str, index, check_time = _ask_inputs()
    chain_0920, chain_now, source = _load_snapshots(
        date_str=date_str,
        index=index,
        check_time=check_time,
    )
    if chain_0920 is None or chain_now is None:
        return _manual_mode(date_str=date_str, index=index)

    engine = VegaThetaEngine()
    engine.baseline[index] = {"index": index, "timestamp": "backtest_snapshot", "current": chain_0920, "next": None}
    signal_0920 = engine.generate_signal(index=index, current_chain=chain_0920, next_chain=None, market_context={})
    signal_now = engine.generate_signal(index=index, current_chain=chain_now, next_chain=None, market_context={})
    if not signal_0920 or not signal_now:
        print("\n? ERROR: VegaThetaEngine failed to generate one or both signals.")
        return 1

    m0920 = _metrics(base_chain=chain_0920, now_chain=chain_0920)
    m_now = _metrics(base_chain=chain_0920, now_chain=chain_now)

    print("\n" + "=" * 72)
    print(f"DATE: {date_str}")
    print(f"INDEX: {index}")
    print(f"SOURCE: {source}")
    print("=" * 72)

    _print_snapshot(
        label="09:20",
        metrics=m0920,
        regime=str(signal_0920.get("regime", "NA")),
        direction=str(signal_0920.get("direction", "NA")),
        interpretation=str(signal_0920.get("interpretation", "")),
        strategy_bias=str(signal_0920.get("strategy_bias", "NA")),
        confidence=float(signal_0920.get("confidence", 0.0) or 0.0),
        vega_state=str(signal_0920.get("vega_state", "NA")),
        iv_state=str(signal_0920.get("iv_state", "NA")),
        delta_state=str(signal_0920.get("delta_state", "NA")),
        gamma_stress=str(signal_0920.get("gamma_stress", "NA")),
    )
    _print_snapshot(
        label=check_time,
        metrics=m_now,
        regime=str(signal_now.get("regime", "NA")),
        direction=str(signal_now.get("direction", "NA")),
        interpretation=str(signal_now.get("interpretation", "")),
        strategy_bias=str(signal_now.get("strategy_bias", "NA")),
        confidence=float(signal_now.get("confidence", 0.0) or 0.0),
        vega_state=str(signal_now.get("vega_state", "NA")),
        iv_state=str(signal_now.get("iv_state", "NA")),
        delta_state=str(signal_now.get("delta_state", "NA")),
        gamma_stress=str(signal_now.get("gamma_stress", "NA")),
    )

    spot_move = m_now["spot"] - m0920["spot"]
    spot_move_pct = ((spot_move / m0920["spot"]) * 100.0) if m0920["spot"] else 0.0
    print(f"\n?? Spot Change (09:20 -> {check_time}): {spot_move:+.2f} ({spot_move_pct:+.4f}%)")

    move_input = input(f"\nWhat was actual move % after {check_time}? ").strip()
    bias_input = input("Actual bias? (BULLISH/BEARISH/SIDEWAYS): ").strip().upper()
    if bias_input not in {"BULLISH", "BEARISH", "SIDEWAYS"}:
        print("? ERROR: bias must be one of BULLISH, BEARISH, SIDEWAYS.")
        return 1
    try:
        actual_move_pct = float(move_input)
    except ValueError:
        print("? ERROR: actual move % must be numeric.")
        return 1

    _append_accuracy_log(
        date_str=date_str,
        regime=str(signal_now.get("regime", "NA")),
        direction=str(signal_now.get("direction", "NA")),
        actual_move_pct=actual_move_pct,
        actual_bias=bias_input,
    )
    print("? Saved log row to: data/backtest/accuracy_log.csv")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())

