import unittest

from analytics.greek_flow import compute_flow_metrics
from analytics.regime import VolatilityRegimeDetector
from analytics.rules import RuleBasedClassifier


def _mk_chain(spot, atm, strikes):
    return {
        "index": "NIFTY",
        "spot": spot,
        "atm": atm,
        "expiry": "2026-02-17",
        "strikes": strikes,
    }


class TestGreekAggregation(unittest.TestCase):
    def test_signed_pos_delta_and_flow(self):
        base = _mk_chain(
            100.0,
            100,
            [
                {"strike": 90, "ce": {"delta": 0.8, "vega": 2.0, "gamma": 0.01, "theta": -1.0, "oi": 100}, "pe": {"delta": -0.2, "vega": 1.0, "gamma": 0.02, "theta": -0.5, "oi": 150}},
                {"strike": 100, "ce": {"delta": 0.5, "vega": 3.0, "gamma": 0.03, "theta": -1.2, "oi": 200}, "pe": {"delta": -0.5, "vega": 3.1, "gamma": 0.03, "theta": -1.1, "oi": 200}},
                {"strike": 110, "ce": {"delta": 0.2, "vega": 1.1, "gamma": 0.02, "theta": -0.6, "oi": 140}, "pe": {"delta": -0.8, "vega": 2.2, "gamma": 0.01, "theta": -1.0, "oi": 90}},
            ],
        )
        now = _mk_chain(
            101.0,
            100,
            [
                {"strike": 90, "ce": {"delta": 0.82, "vega": 2.1, "gamma": 0.011, "theta": -0.95, "oi": 105}, "pe": {"delta": -0.18, "vega": 0.95, "gamma": 0.019, "theta": -0.45, "oi": 145}},
                {"strike": 100, "ce": {"delta": 0.55, "vega": 3.2, "gamma": 0.032, "theta": -1.1, "oi": 210}, "pe": {"delta": -0.45, "vega": 2.9, "gamma": 0.029, "theta": -1.0, "oi": 195}},
                {"strike": 110, "ce": {"delta": 0.23, "vega": 1.2, "gamma": 0.021, "theta": -0.55, "oi": 150}, "pe": {"delta": -0.77, "vega": 2.1, "gamma": 0.011, "theta": -0.95, "oi": 100}},
            ],
        )

        m = compute_flow_metrics(base, now, window_strikes_each_side=3)
        self.assertAlmostEqual(m.flow_delta_raw, m.pos_delta_now - m.pos_delta_baseline, places=6)
        self.assertAlmostEqual(m.gamma_shift_raw, m.gamma_total_now - m.gamma_total_baseline, places=6)


class TestRegimeAndClassifier(unittest.TestCase):
    def setUp(self):
        self.regime = VolatilityRegimeDetector()
        self.classifier = RuleBasedClassifier()
        self.base_chain = _mk_chain(100, 100, [{"strike": 100, "ce": {"iv": 10}, "pe": {"iv": 10}}])
        self.now_chain = _mk_chain(100, 100, [{"strike": 100, "ce": {"iv": 10.2}, "pe": {"iv": 10.1}}])

    def test_vega_contracting_forces_delta_neutral(self):
        raw = {
            "flow_delta_true": -25.0,
            "flow_vega_true": -6.0,
            "total_iv_change": -0.4,
            "gamma_skew_shift_true": -9.0,
            "gamma_shift_true": 4.0,
            "pos_delta_now": -12.0,
            "spot_change_pct": 0.10,
            "total_call_oi_change": 100.0,
            "total_put_oi_change": 200.0,
            "call_iv_diff": 0.2,
            "put_iv_diff": 0.1,
        }
        regime = self.regime.detect(self.base_chain, self.now_chain, {}, raw)
        self.assertIn(regime.label, {"CALM", "TRENDING", "VOLATILE"})

        decision = self.classifier.classify(raw_features=raw, regime=regime.label)
        self.assertEqual(decision.direction, "SIDEWAYS")
        self.assertTrue(decision.trade_allowed)
        self.assertEqual(decision.strategy_bias, "DELTA_NEUTRAL")
        self.assertEqual(decision.vega_state, "VEGA_CONTRACTING")
        self.assertIn("vega_contracting_neutral", decision.rule_stack)

    def test_vega_expansion_with_positive_delta_put_dominance_is_bullish(self):
        raw = {
            "flow_delta_true": 22.0,
            "flow_vega_true": 12.0,
            "total_iv_change": 1.1,
            "gamma_skew_shift_true": 12.0,
            "gamma_shift_true": 15.0,
            "pos_delta_now": 19.0,
            "spot_change_pct": 0.20,
            "total_call_oi_change": 80.0,
            "total_put_oi_change": 160.0,
            "call_iv_diff": 0.3,
            "put_iv_diff": 0.7,
        }
        regime = self.regime.detect(self.base_chain, self.now_chain, {}, raw)
        self.assertIn(regime.label, {"CALM", "TRENDING", "VOLATILE"})

        decision = self.classifier.classify(raw_features=raw, regime=regime.label)
        self.assertEqual(decision.direction, "BULLISH")
        self.assertTrue(decision.trade_allowed)
        self.assertEqual(decision.strategy_bias, "BULLISH")
        self.assertEqual(decision.delta_state, "DELTA_RISING")
        self.assertIn("vega_up_delta_up_put_iv_dominate", decision.rule_stack)

    def test_vega_expansion_with_positive_delta_call_dominance_is_bearish(self):
        raw = {
            "flow_delta_true": 18.0,
            "flow_vega_true": 11.0,
            "total_iv_change": 0.9,
            "gamma_skew_shift_true": 7.0,
            "gamma_shift_true": 30.0,
            "pos_delta_now": 8.0,
            "spot_change_pct": 0.12,
            "total_call_oi_change": 210.0,
            "total_put_oi_change": 90.0,
            "call_iv_diff": 0.9,
            "put_iv_diff": 0.4,
        }
        regime = self.regime.detect(self.base_chain, self.now_chain, {}, raw)
        decision = self.classifier.classify(raw_features=raw, regime=regime.label)
        self.assertEqual(decision.direction, "BEARISH")
        self.assertTrue(decision.trade_allowed)
        self.assertEqual(decision.strategy_bias, "BEARISH")
        self.assertIn("vega_up_delta_up_call_oi_dominate", decision.rule_stack)

    def test_expanding_delta_near_zero_is_unstable_no_trade(self):
        raw = {
            "flow_delta_true": 1.0,
            "flow_vega_true": 7.0,
            "total_iv_change": 0.7,
            "gamma_skew_shift_true": 0.5,
            "gamma_shift_true": 120.0,
            "pos_delta_now": 0.5,
            "spot_change_pct": 0.02,
            "total_call_oi_change": 100.0,
            "total_put_oi_change": 110.0,
            "call_iv_diff": 0.8,
            "put_iv_diff": 0.75,
        }
        regime = self.regime.detect(self.base_chain, self.now_chain, {}, raw)
        decision = self.classifier.classify(raw_features=raw, regime=regime.label)
        self.assertEqual(decision.direction, "SIDEWAYS")
        self.assertFalse(decision.trade_allowed)
        self.assertEqual(decision.strategy_bias, "UNSTABLE")
        self.assertIn("vega_up_delta_near_zero_unstable", decision.rule_stack)

    def test_gamma_stress_reduces_confidence_not_autoblock(self):
        raw = {
            "flow_delta_true": -24.0,
            "flow_vega_true": 18.0,
            "total_iv_change": 1.2,
            "gamma_skew_shift_true": -13.0,
            "gamma_shift_true": 80000.0,
            "pos_delta_now": -18.0,
            "spot_change_pct": -0.20,
            "total_call_oi_change": 300.0,
            "total_put_oi_change": 100.0,
            "call_iv_diff": 1.2,
            "put_iv_diff": 0.5,
        }
        regime = self.regime.detect(self.base_chain, self.now_chain, {}, raw)
        decision = self.classifier.classify(raw_features=raw, regime=regime.label)
        self.assertEqual(decision.direction, "BEARISH")
        self.assertTrue(decision.trade_allowed)
        self.assertIn("gamma_stress_high_reduce_confidence", decision.rule_stack)
        self.assertLess(decision.confidence, 0.72)


if __name__ == "__main__":
    unittest.main()
