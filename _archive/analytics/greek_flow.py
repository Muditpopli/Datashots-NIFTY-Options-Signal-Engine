"""
Greek flow decomposition with signed OI-weighted aggregation.

Architecture diagram (high-level)
--------------------------------
1. Baseline chain + current chain -> aligned strike universe near ATM.
2. Compute signed exposure metrics (delta/vega/gamma/theta) with OI weights.
3. Estimate mechanical Greek drift from spot movement via moneyness-shift interpolation.
4. True flow = observed flow - mechanical flow.
5. Return directional and structural metrics for rule engine consumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import math


@dataclass
class FlowMetrics:
    """Container for raw, baseline-relative Greek flow metrics."""

    pos_delta_now: float
    pos_delta_baseline: float
    flow_delta_raw: float
    flow_delta_mechanical: float
    flow_delta_true: float

    pos_vega_now: float
    pos_vega_baseline: float
    flow_vega_raw: float
    flow_vega_mechanical: float
    flow_vega_true: float

    gamma_total_now: float
    gamma_total_baseline: float
    gamma_shift_raw: float
    gamma_shift_mechanical: float
    gamma_shift_true: float

    gamma_skew_now: float
    gamma_skew_baseline: float
    gamma_skew_shift_raw: float
    gamma_skew_shift_mechanical: float
    gamma_skew_shift_true: float

    theta_skew_now: float
    theta_skew_baseline: float
    theta_skew_shift: float

    spot_change: float
    spot_change_pct: float
    strike_count: int
    oi_total_now: float


def _safe_float(value, default=0.0) -> float:
    try:
        num = float(value)
        return num if math.isfinite(num) else default
    except (TypeError, ValueError):
        return default


def _build_strike_map(chain: Dict) -> Dict[int, Dict]:
    return {int(_safe_float(row.get("strike"), -1)): row for row in chain.get("strikes", [])}


def _linear_interp(points: List[Tuple[float, float]], x: float) -> float:
    """Linear interpolation over strike-value points with edge clamping."""
    if not points:
        return 0.0
    points = sorted(points, key=lambda p: p[0])
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]

    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            ratio = (x - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return points[-1][1]


def _collect_points(strike_map: Dict[int, Dict], side: str, field: str) -> List[Tuple[float, float]]:
    points = []
    for strike, row in strike_map.items():
        leg = row.get(side, {})
        points.append((float(strike), _safe_float(leg.get(field))))
    return points


def _iter_window_strikes(
    now_chain: Dict,
    base_map: Dict[int, Dict],
    window_strikes_each_side: int,
) -> Iterable[Tuple[int, Dict, Dict]]:
    """Yield aligned strikes around current ATM where baseline and current both exist."""
    atm = int(_safe_float(now_chain.get("atm"), 0))
    if atm <= 0:
        return

    rows_now = _build_strike_map(now_chain)
    strike_values = sorted(rows_now.keys())

    # Infer gap from chain itself to stay API-format agnostic.
    gaps = [b - a for a, b in zip(strike_values[:-1], strike_values[1:]) if b > a]
    strike_gap = min(gaps) if gaps else 50
    bound = window_strikes_each_side * strike_gap

    for strike, now_row in rows_now.items():
        if abs(strike - atm) > bound:
            continue
        if strike not in base_map:
            continue
        yield strike, base_map[strike], now_row


def compute_flow_metrics(
    base_chain: Dict,
    now_chain: Dict,
    window_strikes_each_side: int = 6,
) -> FlowMetrics:
    """
    Compute signed OI-weighted position and flow metrics.

    Mechanical-effect isolation:
    - Uses baseline Greek surface interpolation shifted by spot change.
    - Expected mechanical Greek at strike k is approximated as baseline Greek at (k - dS).
    - True flow = observed baseline-to-now flow - mechanical flow.
    """

    base_map = _build_strike_map(base_chain)
    base_ce_delta_points = _collect_points(base_map, "ce", "delta")
    base_pe_delta_points = _collect_points(base_map, "pe", "delta")
    base_ce_vega_points = _collect_points(base_map, "ce", "vega")
    base_pe_vega_points = _collect_points(base_map, "pe", "vega")
    base_ce_gamma_points = _collect_points(base_map, "ce", "gamma")
    base_pe_gamma_points = _collect_points(base_map, "pe", "gamma")

    spot_now = _safe_float(now_chain.get("spot"))
    spot_base = _safe_float(base_chain.get("spot"), 1.0)
    spot_change = spot_now - spot_base
    spot_change_pct = (spot_change / spot_base * 100.0) if spot_base else 0.0

    pos_delta_now = 0.0
    pos_delta_baseline = 0.0
    flow_delta_mechanical = 0.0

    pos_vega_now = 0.0
    pos_vega_baseline = 0.0
    flow_vega_mechanical = 0.0

    gamma_total_now = 0.0
    gamma_total_baseline = 0.0
    gamma_shift_mechanical = 0.0

    gamma_skew_now = 0.0
    gamma_skew_baseline = 0.0
    gamma_skew_shift_mechanical = 0.0

    theta_skew_now = 0.0
    theta_skew_baseline = 0.0

    strike_count = 0
    oi_total_now = 0.0

    for strike, base_row, now_row in _iter_window_strikes(
        now_chain=now_chain,
        base_map=base_map,
        window_strikes_each_side=window_strikes_each_side,
    ):
        strike_count += 1

        ce_oi_now = _safe_float(now_row.get("ce", {}).get("oi"))
        pe_oi_now = _safe_float(now_row.get("pe", {}).get("oi"))
        ce_oi_base = _safe_float(base_row.get("ce", {}).get("oi"))
        pe_oi_base = _safe_float(base_row.get("pe", {}).get("oi"))

        oi_total_now += ce_oi_now + pe_oi_now

        ce_delta_now = _safe_float(now_row.get("ce", {}).get("delta"))
        pe_delta_now = _safe_float(now_row.get("pe", {}).get("delta"))
        ce_delta_base = _safe_float(base_row.get("ce", {}).get("delta"))
        pe_delta_base = _safe_float(base_row.get("pe", {}).get("delta"))

        ce_vega_now = _safe_float(now_row.get("ce", {}).get("vega"))
        pe_vega_now = _safe_float(now_row.get("pe", {}).get("vega"))
        ce_vega_base = _safe_float(base_row.get("ce", {}).get("vega"))
        pe_vega_base = _safe_float(base_row.get("pe", {}).get("vega"))

        ce_gamma_now = _safe_float(now_row.get("ce", {}).get("gamma"))
        pe_gamma_now = _safe_float(now_row.get("pe", {}).get("gamma"))
        ce_gamma_base = _safe_float(base_row.get("ce", {}).get("gamma"))
        pe_gamma_base = _safe_float(base_row.get("pe", {}).get("gamma"))

        ce_theta_now = _safe_float(now_row.get("ce", {}).get("theta"))
        pe_theta_now = _safe_float(now_row.get("pe", {}).get("theta"))
        ce_theta_base = _safe_float(base_row.get("ce", {}).get("theta"))
        pe_theta_base = _safe_float(base_row.get("pe", {}).get("theta"))

        # Signed positional metrics (direction-aware) using actual side deltas.
        pos_delta_now += ce_delta_now * ce_oi_now + pe_delta_now * pe_oi_now
        pos_delta_baseline += ce_delta_base * ce_oi_base + pe_delta_base * pe_oi_base

        # Directional vega bias: call-side minus put-side exposure.
        pos_vega_now += ce_vega_now * ce_oi_now - pe_vega_now * pe_oi_now
        pos_vega_baseline += ce_vega_base * ce_oi_base - pe_vega_base * pe_oi_base

        # Gamma total is structural convexity load; skew is directional asymmetry.
        gamma_total_now += ce_gamma_now * ce_oi_now + pe_gamma_now * pe_oi_now
        gamma_total_baseline += ce_gamma_base * ce_oi_base + pe_gamma_base * pe_oi_base
        gamma_skew_now += ce_gamma_now * ce_oi_now - pe_gamma_now * pe_oi_now
        gamma_skew_baseline += ce_gamma_base * ce_oi_base - pe_gamma_base * pe_oi_base

        # Theta skew tells which side is decaying faster.
        theta_skew_now += ce_theta_now * ce_oi_now - pe_theta_now * pe_oi_now
        theta_skew_baseline += ce_theta_base * ce_oi_base - pe_theta_base * pe_oi_base

        # Mechanical effect via moneyness-preserving strike shift.
        shifted_strike = float(strike) - spot_change

        exp_ce_delta = _linear_interp(base_ce_delta_points, shifted_strike)
        exp_pe_delta = _linear_interp(base_pe_delta_points, shifted_strike)
        flow_delta_mechanical += (exp_ce_delta - ce_delta_base) * ce_oi_base
        flow_delta_mechanical += (exp_pe_delta - pe_delta_base) * pe_oi_base

        exp_ce_vega = _linear_interp(base_ce_vega_points, shifted_strike)
        exp_pe_vega = _linear_interp(base_pe_vega_points, shifted_strike)
        flow_vega_mechanical += (exp_ce_vega - ce_vega_base) * ce_oi_base
        flow_vega_mechanical += -(exp_pe_vega - pe_vega_base) * pe_oi_base

        exp_ce_gamma = _linear_interp(base_ce_gamma_points, shifted_strike)
        exp_pe_gamma = _linear_interp(base_pe_gamma_points, shifted_strike)
        gamma_shift_mechanical += (exp_ce_gamma - ce_gamma_base) * ce_oi_base
        gamma_shift_mechanical += (exp_pe_gamma - pe_gamma_base) * pe_oi_base
        gamma_skew_shift_mechanical += (exp_ce_gamma - ce_gamma_base) * ce_oi_base
        gamma_skew_shift_mechanical += -(exp_pe_gamma - pe_gamma_base) * pe_oi_base

    flow_delta_raw = pos_delta_now - pos_delta_baseline
    flow_delta_true = flow_delta_raw - flow_delta_mechanical

    flow_vega_raw = pos_vega_now - pos_vega_baseline
    flow_vega_true = flow_vega_raw - flow_vega_mechanical

    gamma_shift_raw = gamma_total_now - gamma_total_baseline
    gamma_shift_true = gamma_shift_raw - gamma_shift_mechanical

    gamma_skew_shift_raw = gamma_skew_now - gamma_skew_baseline
    gamma_skew_shift_true = gamma_skew_shift_raw - gamma_skew_shift_mechanical

    theta_skew_shift = theta_skew_now - theta_skew_baseline

    return FlowMetrics(
        pos_delta_now=pos_delta_now,
        pos_delta_baseline=pos_delta_baseline,
        flow_delta_raw=flow_delta_raw,
        flow_delta_mechanical=flow_delta_mechanical,
        flow_delta_true=flow_delta_true,
        pos_vega_now=pos_vega_now,
        pos_vega_baseline=pos_vega_baseline,
        flow_vega_raw=flow_vega_raw,
        flow_vega_mechanical=flow_vega_mechanical,
        flow_vega_true=flow_vega_true,
        gamma_total_now=gamma_total_now,
        gamma_total_baseline=gamma_total_baseline,
        gamma_shift_raw=gamma_shift_raw,
        gamma_shift_mechanical=gamma_shift_mechanical,
        gamma_shift_true=gamma_shift_true,
        gamma_skew_now=gamma_skew_now,
        gamma_skew_baseline=gamma_skew_baseline,
        gamma_skew_shift_raw=gamma_skew_shift_raw,
        gamma_skew_shift_mechanical=gamma_skew_shift_mechanical,
        gamma_skew_shift_true=gamma_skew_shift_true,
        theta_skew_now=theta_skew_now,
        theta_skew_baseline=theta_skew_baseline,
        theta_skew_shift=theta_skew_shift,
        spot_change=spot_change,
        spot_change_pct=spot_change_pct,
        strike_count=strike_count,
        oi_total_now=oi_total_now,
    )
