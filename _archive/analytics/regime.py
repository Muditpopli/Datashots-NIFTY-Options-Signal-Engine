"""Deterministic regime detection from raw structural Greek/OI features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RegimeState:
    """
    Instantaneous structural regime payload.

    Regime is deterministic and single-pass by design:
    - no history
    - no persistence counters
    - no statistical normalization
    """

    label: str
    reasons: List[str]
    metrics: Dict[str, float]


class VolatilityRegimeDetector:
    """
    Structural regime detector with fixed human-readable thresholds only.

    Deterministic rules:
    - CALM: vega contracting/flat (non-expanding environment).
    - TRENDING: vega expanding with directional delta flow.
    - VOLATILE: vega expanding with gamma/IV stress or unstable near-zero delta.
    """

    def __init__(
        self,
        iv_flat_tolerance: float = 0.05,
        delta_near_zero_threshold: float = 5.0,
        gamma_stress_threshold: float = 50000.0,
        iv_spike_threshold: float = 0.8,
        vega_flat_tolerance: float | None = None,
    ):
        # Backward-compatible: old callers pass iv_flat_tolerance.
        self.vega_flat_tolerance = (
            float(vega_flat_tolerance)
            if vega_flat_tolerance is not None
            else float(iv_flat_tolerance)
        )
        self.delta_near_zero_threshold = float(delta_near_zero_threshold)
        self.gamma_stress_threshold = float(gamma_stress_threshold)
        self.iv_spike_threshold = float(iv_spike_threshold)

    @staticmethod
    def _sign(x: float) -> int:
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    def detect(
        self,
        base_chain: Dict,
        now_chain: Dict,
        market_context: Optional[Dict] = None,
        structural_features: Optional[Dict[str, float]] = None,
        recent_rows: Optional[List[Dict[str, str]]] = None,  # kept for interface compatibility
    ) -> RegimeState:
        """
        Determine regime from raw structural features only.

        The function is intentionally single-pass and stateless:
        no history, no smoothing, no persistence counters.
        """
        del recent_rows  # Explicitly unused: no persistence or history logic.
        del market_context  # Explicitly unused: regime is internal-only.
        del base_chain
        del now_chain

        features = structural_features or {}

        flow_delta = float(features.get("flow_delta_true", 0.0))
        flow_vega = float(features.get("flow_vega_true", 0.0))
        total_iv_change = float(features.get("total_iv_change", 0.0))
        gamma_shift = float(features.get("gamma_shift_true", 0.0))
        gamma_skew_shift = float(features.get("gamma_skew_shift_true", 0.0))
        spot_change_pct = float(features.get("spot_change_pct", 0.0))
        total_call_oi = float(features.get("total_call_oi", 0.0))
        total_put_oi = float(features.get("total_put_oi", 0.0))
        oi_imbalance = float(features.get("oi_imbalance", 0.0))
        call_iv_diff = float(features.get("call_iv_diff", 0.0))
        put_iv_diff = float(features.get("put_iv_diff", 0.0))
        pos_delta_now = float(features.get("pos_delta_now", 0.0))

        # Vega state is defined by IV movement from baseline (not by vega exposure flow).
        if total_iv_change > self.vega_flat_tolerance:
            vega_state = "VEGA_EXPANDING"
        elif total_iv_change < -self.vega_flat_tolerance:
            vega_state = "VEGA_CONTRACTING"
        else:
            vega_state = "VEGA_FLAT"

        if flow_delta > self.delta_near_zero_threshold:
            delta_state = "DELTA_RISING"
        elif flow_delta < -self.delta_near_zero_threshold:
            delta_state = "DELTA_FALLING"
        else:
            delta_state = "DELTA_NEAR_ZERO"

        gamma_high_stress = abs(gamma_shift) > self.gamma_stress_threshold
        iv_spike = abs(total_iv_change) > self.iv_spike_threshold
        aligned_flow_skew = self._sign(flow_delta) != 0 and self._sign(flow_delta) == self._sign(gamma_skew_shift)

        reasons: List[str] = []

        if vega_state in {"VEGA_CONTRACTING", "VEGA_FLAT"}:
            label = "CALM"
            reasons.append("calm:vega_contracting_or_flat")
        elif gamma_high_stress or iv_spike:
            label = "VOLATILE"
            if gamma_high_stress:
                reasons.append("volatile:high_gamma_stress")
            if iv_spike:
                reasons.append("volatile:iv_spike")
        elif delta_state != "DELTA_NEAR_ZERO" and aligned_flow_skew:
            label = "TRENDING"
            reasons.append("trending:vega_expanding_aligned_flow_skew")
        elif delta_state != "DELTA_NEAR_ZERO":
            label = "TRENDING"
            reasons.append("trending:vega_expanding_directional_flow")
        else:
            label = "VOLATILE"
            reasons.append("volatile:vega_expanding_unstable_alignment")

        metrics = {
            "flow_delta_true": round(flow_delta, 4),
            "flow_vega_true": round(flow_vega, 4),
            "total_iv_change": round(total_iv_change, 4),
            "call_iv_diff": round(call_iv_diff, 4),
            "put_iv_diff": round(put_iv_diff, 4),
            "gamma_shift_true": round(gamma_shift, 4),
            "gamma_skew_shift_true": round(gamma_skew_shift, 4),
            "spot_change_pct": round(spot_change_pct, 4),
            "pos_delta_now": round(pos_delta_now, 4),
            "total_call_oi": round(total_call_oi, 2),
            "total_put_oi": round(total_put_oi, 2),
            "oi_imbalance": round(oi_imbalance, 4),
            "vega_state": 1.0 if vega_state == "VEGA_EXPANDING" else -1.0 if vega_state == "VEGA_CONTRACTING" else 0.0,
            "delta_state": 1.0 if delta_state == "DELTA_RISING" else -1.0 if delta_state == "DELTA_FALLING" else 0.0,
            "gamma_high_stress": 1.0 if gamma_high_stress else 0.0,
            "iv_spike": 1.0 if iv_spike else 0.0,
        }

        return RegimeState(
            label=label,
            reasons=reasons,
            metrics=metrics,
        )
