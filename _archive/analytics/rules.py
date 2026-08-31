"""Deterministic rule classifier using raw structural signals only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class RuleDecision:
    """Final classification output for one aggregated snapshot."""

    direction: str
    strategy_bias: str
    directional_bias: str
    structural_bias: str
    trade_allowed: bool
    confidence: float
    vega_state: str
    iv_state: str
    delta_state: str
    gamma_stress: str
    interpretation: str
    rule_stack: list[str]
    directional_strength: float


class RuleBasedClassifier:
    """
    Minimal and transparent vega-led classifier.

    Primary directional logic:
    - Vega contracting -> SIDEWAYS / DELTA_NEUTRAL.
    - Vega expanding with delta sign -> directional path.
    - Vega flat near zero -> SIDEWAYS / DELTA_NEUTRAL.

    For vega-expanding + delta-rising branch, direction is finalized only after
    OI-change + IV-change consultation.
    """

    def __init__(
        self,
        iv_flat_tolerance: float = 0.05,
        delta_near_zero_threshold: float = 5.0,
        gamma_stress_threshold: float = 50000.0,
        iv_strong_expansion_threshold: float = 0.6,
        directional_flow_min: float = 1_000_000.0,
        directional_vega_min: float = 10_000_000.0,
        directional_iv_spread_min: float = 0.15,
        vega_flat_tolerance: float | None = None,
        vega_flow_flat_tolerance: float = 100000.0,
    ):
        # Backward-compatible: old callers pass iv_flat_tolerance.
        self.vega_flat_tolerance = (
            float(vega_flat_tolerance)
            if vega_flat_tolerance is not None
            else float(iv_flat_tolerance)
        )
        self.delta_near_zero_threshold = float(delta_near_zero_threshold)
        self.gamma_stress_threshold = float(gamma_stress_threshold)
        self.iv_strong_expansion_threshold = float(iv_strong_expansion_threshold)
        self.directional_flow_min = float(directional_flow_min)
        self.directional_vega_min = float(directional_vega_min)
        self.directional_iv_spread_min = float(directional_iv_spread_min)
        self.vega_flow_flat_tolerance = float(vega_flow_flat_tolerance)

    def classify(
        self,
        raw_features: Dict[str, float],
        regime: str,
        event_risk: bool = False,  # retained for backward-compatible call sites
        regime_meta: Dict | None = None,  # retained for interface compatibility
    ) -> RuleDecision:
        del regime_meta
        del regime
        del event_risk

        flow_delta = float(raw_features.get("flow_delta_true", 0.0))
        flow_vega = float(raw_features.get("flow_vega_true", 0.0))
        total_iv_change = float(raw_features.get("total_iv_change", 0.0))
        gamma_shift = float(raw_features.get("gamma_shift_true", 0.0))
        call_oi_change = float(raw_features.get("total_call_oi_change", 0.0))
        put_oi_change = float(raw_features.get("total_put_oi_change", 0.0))
        call_iv_diff = float(raw_features.get("call_iv_diff", 0.0))
        put_iv_diff = float(raw_features.get("put_iv_diff", 0.0))

        rule_stack: list[str] = []

        # Keep IV-state and vega-flow-state separate:
        # - iv_state is used for structural logic
        # - vega_state is shown to users and follows flow_vega_true
        if total_iv_change > self.vega_flat_tolerance:
            iv_state = "NET_IV_EXPANDING"
        elif total_iv_change < -self.vega_flat_tolerance:
            iv_state = "NET_IV_CONTRACTING"
        else:
            iv_state = "NET_IV_FLAT"

        if flow_vega > self.vega_flow_flat_tolerance:
            vega_state = "VEGA_EXPANDING"
        elif flow_vega < -self.vega_flow_flat_tolerance:
            vega_state = "VEGA_CONTRACTING"
        else:
            vega_state = "VEGA_FLAT"

        if flow_delta > self.delta_near_zero_threshold:
            delta_state = "DELTA_RISING"
        elif flow_delta < -self.delta_near_zero_threshold:
            delta_state = "DELTA_FALLING"
        else:
            delta_state = "DELTA_NEAR_ZERO"

        if iv_state in {"NET_IV_CONTRACTING", "NET_IV_FLAT"}:
            direction = "SIDEWAYS"
            strategy_bias = "DELTA_NEUTRAL"
            trade_allowed = True
            rule_stack.append("vega_contracting_neutral")
            # If both legs are seeing IV expansion despite vega contraction/flat reading,
            # structure is unstable for neutral premium-decay deployment.
            if call_iv_diff > 0 and put_iv_diff > 0:
                trade_allowed = False
                strategy_bias = "UNSTABLE"
                rule_stack.append("both_legs_iv_rising_avoid")
        else:
            if delta_state == "DELTA_RISING":
                # Primary direction is IV-led here; OI is secondary confidence only.
                bullish_iv = call_iv_diff > put_iv_diff
                bearish_iv = put_iv_diff > call_iv_diff
                iv_spread = abs(call_iv_diff - put_iv_diff)

                directional_filter_ok = (
                    abs(flow_delta) >= self.directional_flow_min
                    and flow_vega >= self.directional_vega_min
                    and iv_spread >= self.directional_iv_spread_min
                )

                if call_iv_diff < 0 and put_iv_diff < 0:
                    direction = "SIDEWAYS"
                    strategy_bias = "UNSTABLE"
                    trade_allowed = False
                    rule_stack.append("vega_up_delta_up_both_legs_iv_falling_avoid")
                elif not directional_filter_ok:
                    direction = "SIDEWAYS"
                    strategy_bias = "UNSTABLE"
                    trade_allowed = False
                    rule_stack.append("directional_threshold_filter_block")
                elif bullish_iv:
                    direction = "BULLISH"
                    strategy_bias = "BULLISH"
                    trade_allowed = True
                    rule_stack.append("vega_up_delta_up_call_iv_confirm_bull")
                elif bearish_iv:
                    direction = "BEARISH"
                    strategy_bias = "BEARISH"
                    trade_allowed = True
                    rule_stack.append("vega_up_delta_up_put_iv_confirm_bear")
                else:
                    direction = "SIDEWAYS"
                    strategy_bias = "UNSTABLE"
                    trade_allowed = False
                    rule_stack.append("vega_up_delta_up_iv_unclear")

            elif delta_state == "DELTA_FALLING":
                direction = "SIDEWAYS"
                strategy_bias = "UNSTABLE"
                trade_allowed = False
                rule_stack.append("vega_up_delta_down_unclear")
            else:
                direction = "SIDEWAYS"
                strategy_bias = "UNSTABLE"
                trade_allowed = False
                rule_stack.append("vega_up_delta_near_zero_unstable")

        # Keep OI/IV context in rule stack for diagnostics only.
        if call_oi_change > put_oi_change:
            rule_stack.append("oi_tilt_call_side")
        elif put_oi_change > call_oi_change:
            rule_stack.append("oi_tilt_put_side")

        if call_iv_diff > put_iv_diff:
            rule_stack.append("iv_tilt_call_side")
        elif put_iv_diff > call_iv_diff:
            rule_stack.append("iv_tilt_put_side")

        gamma_stress = "HIGH" if abs(gamma_shift) > self.gamma_stress_threshold else "NORMAL"
        if (
            gamma_stress == "HIGH"
            and iv_state == "NET_IV_EXPANDING"
            and total_iv_change > self.iv_strong_expansion_threshold
        ):
            rule_stack.append("gamma_stress_high_reduce_confidence")

        directional_bias = direction if direction in {"BULLISH", "BEARISH"} else "NEUTRAL"
        structural_bias = strategy_bias

        if direction == "SIDEWAYS":
            interpretation = "Signal is non-directional or structurally unclear for directional deployment."
        elif direction == "BULLISH":
            interpretation = "Vega expansion + delta rise with call-side OI/IV confirmation supports bullish bias."
        else:
            interpretation = "Vega expansion + delta rise with put-side OI/IV confirmation supports bearish bias."

        if "both_legs_iv_rising_avoid" in rule_stack:
            interpretation = (
                "Vega is contracting/flat but both call and put IV are rising; "
                "avoid trade for the day."
            )
        if "vega_up_delta_up_both_legs_iv_falling_avoid" in rule_stack:
            interpretation = (
                "Vega and delta are rising but both call and put IV are falling; "
                "avoid trade for the day."
            )
        if "directional_threshold_filter_block" in rule_stack:
            interpretation = (
                "Directional branch blocked by threshold filter "
                "(flow/vega/IV-spread not strong enough)."
            )

        if "gamma_stress_high_reduce_confidence" in rule_stack:
            interpretation = f"{interpretation} Gamma stress is high under IV expansion; confidence reduced."

        directional_strength = 0.0
        flow_abs = abs(flow_delta)
        if direction == "BULLISH":
            directional_strength = min(flow_abs / 20.0, 2.0)
        elif direction == "BEARISH":
            directional_strength = -min(flow_abs / 20.0, 2.0)

        confidence = 0.65
        if direction in {"BULLISH", "BEARISH"}:
            confidence = 0.72
            if (direction == "BULLISH" and call_oi_change > put_oi_change) or (
                direction == "BEARISH" and put_oi_change > call_oi_change
            ):
                confidence = min(0.85, confidence + 0.05)
            elif (direction == "BULLISH" and put_oi_change > call_oi_change) or (
                direction == "BEARISH" and call_oi_change > put_oi_change
            ):
                confidence = max(0.45, confidence - 0.1)
        if direction == "SIDEWAYS" and strategy_bias == "UNSTABLE":
            confidence = 0.4
        if "gamma_stress_high_reduce_confidence" in rule_stack:
            confidence = max(0.35, confidence - 0.15)
        if not trade_allowed:
            confidence = min(confidence, 0.45)

        return RuleDecision(
            direction=direction,
            strategy_bias=strategy_bias,
            directional_bias=directional_bias,
            structural_bias=structural_bias,
            trade_allowed=trade_allowed,
            confidence=round(confidence, 2),
            vega_state=vega_state,
            iv_state=iv_state,
            delta_state=delta_state,
            gamma_stress=gamma_stress,
            interpretation=interpretation,
            rule_stack=rule_stack,
            directional_strength=round(directional_strength, 4),
        )
