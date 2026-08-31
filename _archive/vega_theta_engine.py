"""
Vega-Theta Engine (Simplified Deterministic Regime + Rules)

Architecture (single pass):
1) Baseline vs current flow decomposition (raw features).
2) Deterministic regime detection from structural raw signals.
3) Raw-signal rule classifier with regime as trade filter.

No statistical normalization or persistence state is used in classification.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional
import json
import os

import config
from analytics.greek_flow import compute_flow_metrics
from analytics.regime import VolatilityRegimeDetector
from analytics.rules import RuleBasedClassifier


class VegaThetaEngine:
    """Rule-based multi-expiry sentiment engine with baseline-relative math."""

    def __init__(self):
        self.baseline = {}  # index -> {"current": chain, "next": chain}
        self.regime_detector = VolatilityRegimeDetector(
            iv_flat_tolerance=float(getattr(config, "REGIME_IV_FLAT_TOLERANCE", 0.05)),
            delta_near_zero_threshold=float(getattr(config, "REGIME_DELTA_NEAR_ZERO_THRESHOLD", 5.0)),
            gamma_stress_threshold=float(getattr(config, "REGIME_GAMMA_STRESS_THRESHOLD", 50000.0)),
            iv_spike_threshold=float(getattr(config, "REGIME_IV_SPIKE_THRESHOLD", 0.8)),
        )
        self.classifier = RuleBasedClassifier(
            iv_flat_tolerance=float(getattr(config, "RULE_IV_FLAT_TOLERANCE", 0.05)),
            delta_near_zero_threshold=float(getattr(config, "RULE_DELTA_NEAR_ZERO_THRESHOLD", 5.0)),
            gamma_stress_threshold=float(getattr(config, "RULE_GAMMA_STRESS_THRESHOLD", 50000.0)),
            iv_strong_expansion_threshold=float(getattr(config, "RULE_IV_STRONG_EXPANSION_THRESHOLD", 0.6)),
            directional_flow_min=float(getattr(config, "RULE_DIRECTIONAL_FLOW_MIN", 1_000_000.0)),
            directional_vega_min=float(getattr(config, "RULE_DIRECTIONAL_VEGA_MIN", 10_000_000.0)),
            directional_iv_spread_min=float(getattr(config, "RULE_DIRECTIONAL_IV_SPREAD_MIN", 0.15)),
        )

    def capture_baseline(self, index: str, current_chain: Dict, next_chain: Optional[Dict]) -> bool:
        """Store baseline snapshots for later baseline-relative flow computation."""
        if not current_chain:
            return False

        payload = {
            "index": index,
            "timestamp": datetime.now().isoformat(),
            "current": current_chain,
            "next": next_chain,
        }
        self.baseline[index] = payload
        return self._save_baseline_to_file(index, payload)

    def load_latest_baseline(self, index: str) -> bool:
        """Load latest baseline from disk for continuity after process restarts."""
        try:
            pattern = f"vega_baseline_{index}_"
            candidates = []
            for name in os.listdir(config.BASELINE_DIR):
                if name.startswith(pattern) and name.endswith(".json"):
                    candidates.append(os.path.join(config.BASELINE_DIR, name))

            if not candidates:
                return False

            latest_file = max(candidates, key=os.path.getmtime)
            with open(latest_file, "r") as f:
                payload = json.load(f)

            if payload.get("index") != index or "current" not in payload:
                return False

            self.baseline[index] = payload
            print(f"[INFO] Loaded vega baseline: {latest_file}")
            return True
        except Exception as e:
            print(f"[WARN] Could not load vega baseline for {index}: {e}")
            return False

    def generate_signal(
        self,
        index: str,
        current_chain: Dict,
        next_chain: Optional[Dict],
        market_context: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Generate rule-based signal from raw structural features."""
        if index not in self.baseline:
            print("[ERR] No vega baseline captured")
            return None

        baseline = self.baseline[index]
        base_current = baseline.get("current")
        base_next = baseline.get("next")

        if not base_current:
            print("[ERR] Invalid vega baseline")
            return None

        current_metrics = self._compute_metrics(base_current, current_chain)
        next_metrics = (
            self._compute_metrics(base_next, next_chain)
            if base_next and next_chain
            else None
        )

        combined_raw = self._combine_raw_features(current_metrics, next_metrics)
        regime_state = self.regime_detector.detect(
            base_chain=base_current,
            now_chain=current_chain,
            market_context=market_context or {},
            structural_features=combined_raw,
        )

        decision = self.classifier.classify(
            raw_features=combined_raw,
            regime=regime_state.label,
        )

        # Keep compatibility with existing logs/prints.
        strength = max(min(decision.directional_strength * 10.0, 20.0), -20.0)
        components = {
            "delta_flow": round(combined_raw.get("flow_delta_true", 0.0), 2),
            "vega_flow": round(combined_raw.get("flow_vega_true", 0.0), 2),
            "gamma_shift": round(combined_raw.get("gamma_shift_true", 0.0), 2),
        }

        signal = {
            "index": index,
            "direction": decision.direction,
            "strength": round(strength, 2),
            "confidence": decision.confidence,
            "interpretation": decision.interpretation,
            "spot": current_chain["spot"],
            "atm": current_chain["atm"],
            "baseline_spot": base_current["spot"],
            "baseline_atm": base_current["atm"],
            "baseline_timestamp": baseline.get("timestamp"),
            "spot_change": round(current_chain["spot"] - base_current["spot"], 2),
            "spot_change_pct": round(
                (current_chain["spot"] - base_current["spot"]) / base_current["spot"] * 100, 2
            ),
            "components": components,
            "directional_bias": decision.directional_bias,
            "strategy_bias": decision.strategy_bias,
            "structural_bias": decision.structural_bias,
            "trade_allowed": decision.trade_allowed,
            "vega_state": decision.vega_state,
            "iv_state": decision.iv_state,
            "delta_state": decision.delta_state,
            "gamma_stress": decision.gamma_stress,
            "regime": regime_state.label,
            "regime_detail": {
                "reasons": regime_state.reasons,
                "structural_metrics": regime_state.metrics,
            },
            "rule_stack": decision.rule_stack,
            "analytics_features": {
                "raw": combined_raw,
            },
            "expiry_metrics": {
                "current": current_metrics,
                "next": next_metrics,
            },
        }
        return signal

    def _compute_metrics(self, base_chain: Dict, now_chain: Dict) -> Dict:
        """Compute raw baseline-relative metrics for one expiry."""
        flow = compute_flow_metrics(
            base_chain=base_chain,
            now_chain=now_chain,
            window_strikes_each_side=int(getattr(config, "FLOW_WINDOW_STRIKES", 6)),
        )
        call_oi, put_oi, oi_imbalance = self._oi_summary(now_chain)
        base_call_oi, base_put_oi, _ = self._oi_summary(base_chain)
        call_iv_now, put_iv_now = self._oi_weighted_leg_iv(now_chain)
        call_iv_base, put_iv_base = self._oi_weighted_leg_iv(base_chain)
        atm_iv_now = self._atm_avg_iv(now_chain)
        atm_iv_base = self._atm_avg_iv(base_chain)
        call_oi_change = call_oi - base_call_oi
        put_oi_change = put_oi - base_put_oi

        return {
            "base_expiry": base_chain.get("expiry"),
            "now_expiry": now_chain.get("expiry"),
            "spot_change": round(flow.spot_change, 4),
            "spot_change_pct": round(flow.spot_change_pct, 4),
            "strike_count": flow.strike_count,
            "oi_total_now": round(flow.oi_total_now, 2),
            "raw": {
                "flow_delta_true": flow.flow_delta_true,
                "flow_vega_true": flow.flow_vega_true,
                "gamma_shift_true": flow.gamma_shift_true,
                "gamma_skew_shift_true": flow.gamma_skew_shift_true,
                "pos_delta_now": flow.pos_delta_now,
                "theta_skew_shift": flow.theta_skew_shift,
                "spot_change_pct": flow.spot_change_pct,
                "total_call_oi": call_oi,
                "total_put_oi": put_oi,
                "total_call_oi_change": call_oi_change,
                "total_put_oi_change": put_oi_change,
                "call_iv_now": call_iv_now,
                "put_iv_now": put_iv_now,
                "call_iv_base": call_iv_base,
                "put_iv_base": put_iv_base,
                "call_iv_diff": call_iv_now - call_iv_base,
                "put_iv_diff": put_iv_now - put_iv_base,
                "atm_iv_now": atm_iv_now,
                "atm_iv_base": atm_iv_base,
                "total_iv_change": atm_iv_now - atm_iv_base,
                "oi_imbalance": oi_imbalance,
            },
            "debug": {
                "pos_delta_baseline": flow.pos_delta_baseline,
                "flow_delta_raw": flow.flow_delta_raw,
                "flow_delta_mechanical": flow.flow_delta_mechanical,
                "pos_vega_now": flow.pos_vega_now,
                "pos_vega_baseline": flow.pos_vega_baseline,
                "flow_vega_raw": flow.flow_vega_raw,
                "flow_vega_mechanical": flow.flow_vega_mechanical,
                "gamma_total_now": flow.gamma_total_now,
                "gamma_total_baseline": flow.gamma_total_baseline,
                "gamma_shift_raw": flow.gamma_shift_raw,
                "gamma_shift_mechanical": flow.gamma_shift_mechanical,
                "gamma_skew_now": flow.gamma_skew_now,
                "gamma_skew_baseline": flow.gamma_skew_baseline,
                "gamma_skew_shift_raw": flow.gamma_skew_shift_raw,
                "gamma_skew_shift_mechanical": flow.gamma_skew_shift_mechanical,
                "theta_skew_now": flow.theta_skew_now,
                "theta_skew_baseline": flow.theta_skew_baseline,
            },
        }

    @staticmethod
    def _combine_raw_features(current_metrics: Dict, next_metrics: Optional[Dict]) -> Dict[str, float]:
        """Average raw feature block across current/next expiries when available."""
        keys = [
            "flow_delta_true",
            "flow_vega_true",
            "gamma_shift_true",
            "gamma_skew_shift_true",
            "pos_delta_now",
            "theta_skew_shift",
            "spot_change_pct",
            "total_call_oi",
            "total_put_oi",
            "total_call_oi_change",
            "total_put_oi_change",
            "call_iv_now",
            "put_iv_now",
            "call_iv_base",
            "put_iv_base",
            "call_iv_diff",
            "put_iv_diff",
            "atm_iv_now",
            "atm_iv_base",
            "total_iv_change",
            "oi_imbalance",
        ]

        combined = {}
        for key in keys:
            cur_val = float(current_metrics["raw"].get(key, 0.0))
            if next_metrics:
                nxt_val = float(next_metrics["raw"].get(key, 0.0))
                combined[key] = (cur_val + nxt_val) / 2.0
            else:
                combined[key] = cur_val
        return combined

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    def _save_baseline_to_file(self, index: str, payload: Dict) -> bool:
        """Persist captured baseline for replay and audit."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                config.BASELINE_DIR,
                f"vega_baseline_{index}_{timestamp}.json",
            )
            with open(filename, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[OK] Vega baseline saved: {filename}")
            return True
        except Exception as e:
            print(f"[WARN] Could not save vega baseline: {e}")
            return False
