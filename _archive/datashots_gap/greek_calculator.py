from __future__ import annotations

from typing import Dict


def calculate_greek_changes(opening_chain: Dict, current_chain: Dict, delta_min: float = 0.05, delta_max: float = 0.5) -> Dict[str, float]:
    by_strike_open = {int(r["strike"]): r for r in opening_chain.get("strikes", [])}
    by_strike_now = {int(r["strike"]): r for r in current_chain.get("strikes", [])}
    common = sorted(set(by_strike_open.keys()) & set(by_strike_now.keys()))

    changes = {
        "put_vega": 0.0,
        "call_vega": 0.0,
        "put_delta": 0.0,
        "call_delta": 0.0,
        "put_theta": 0.0,
        "call_theta": 0.0,
        "put_oi": 0.0,
        "call_oi": 0.0,
    }
    if not common:
        return changes

    for strike in common:
        o = by_strike_open[strike]
        n = by_strike_now[strike]

        call_delta_abs = abs(float(n.get("ce", {}).get("delta", 0.0) or 0.0))
        put_delta_abs = abs(float(n.get("pe", {}).get("delta", 0.0) or 0.0))

        if delta_min <= call_delta_abs <= delta_max:
            changes["call_vega"] += float(n["ce"]["vega"]) - float(o["ce"]["vega"])
            changes["call_delta"] += float(n["ce"]["delta"]) - float(o["ce"]["delta"])
            changes["call_theta"] += float(n["ce"]["theta"]) - float(o["ce"]["theta"])
            changes["call_oi"] += float(n["ce"].get("oi", 0.0) or 0.0) - float(o["ce"].get("oi", 0.0) or 0.0)

        if delta_min <= put_delta_abs <= delta_max:
            changes["put_vega"] += float(n["pe"]["vega"]) - float(o["pe"]["vega"])
            changes["put_delta"] += float(n["pe"]["delta"]) - float(o["pe"]["delta"])
            changes["put_theta"] += float(n["pe"]["theta"]) - float(o["pe"]["theta"])
            changes["put_oi"] += float(n["pe"].get("oi", 0.0) or 0.0) - float(o["pe"].get("oi", 0.0) or 0.0)

    return changes

