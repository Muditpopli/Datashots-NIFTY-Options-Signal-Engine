"""
Accuracy Tracker
Stores model predictions and end-of-day outcomes for performance tracking.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional
import csv
import os

import config


class AccuracyTracker:
    def __init__(self):
        self.predictions_file = os.path.join(config.TRACKER_DIR, "predictions.csv")
        self.detailed_predictions_file = os.path.join(
            config.TRACKER_DIR, "predictions_detailed.csv"
        )
        self.outcomes_file = os.path.join(config.TRACKER_DIR, "outcomes.csv")
        self.gap_predictions_file = os.path.join(
            config.TRACKER_DIR, "gap_predictions.csv"
        )
        self.gap_outcomes_file = os.path.join(config.TRACKER_DIR, "gap_outcomes.csv")
        self._ensure_files()

    def _ensure_files(self) -> None:
        if not os.path.exists(self.predictions_file):
            with open(self.predictions_file, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "signal_id",
                        "timestamp",
                        "index",
                        "pred_direction",
                        "strength",
                        "confidence",
                        "spot",
                        "baseline_spot",
                        "spot_change",
                        "spot_change_pct",
                        "delta_flow",
                        "vega_flow",
                        "gamma_shift",
                    ],
                )
                writer.writeheader()

        if not os.path.exists(self.detailed_predictions_file):
            with open(self.detailed_predictions_file, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "signal_id",
                        "timestamp",
                        "index",
                        "pred_direction",
                        "strength",
                        "confidence",
                        "spot",
                        "baseline_spot",
                        "spot_change",
                        "spot_change_pct",
                        "delta_flow",
                        "vega_flow",
                        "gamma_shift",
                        "regime",
                        "vega_state",
                        "delta_state",
                        "gamma_stress",
                        "directional_bias",
                        "strategy_bias",
                        "structural_bias",
                        "trade_allowed",
                        "rule_stack",
                        "flow_delta_true",
                        "flow_vega_true",
                        "gamma_shift_true",
                        "gamma_skew_shift_true",
                        "pos_delta_now",
                        "theta_skew_shift",
                        "total_call_oi",
                        "total_put_oi",
                        "total_call_oi_change",
                        "total_put_oi_change",
                        "call_iv_diff",
                        "put_iv_diff",
                        "total_iv_change",
                        "oi_imbalance",
                        "intraday_has_reference",
                        "intraday_prior_count",
                        "intraday_status",
                        "intraday_agreement_ratio",
                        "intraday_range_state",
                        "intraday_anchor_direction",
                        "intraday_anchor_strength",
                        "intraday_latest_direction",
                        "intraday_latest_strength",
                    ],
                )
                writer.writeheader()

        if not os.path.exists(self.outcomes_file):
            with open(self.outcomes_file, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["signal_id", "actual_direction", "labeled_at"],
                )
                writer.writeheader()

        if not os.path.exists(self.gap_predictions_file):
            with open(self.gap_predictions_file, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "gap_id",
                        "timestamp",
                        "index",
                        "pred_direction",
                        "recommendation",
                        "score",
                        "confidence",
                        "spot",
                    ],
                )
                writer.writeheader()

        if not os.path.exists(self.gap_outcomes_file):
            with open(self.gap_outcomes_file, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["gap_id", "actual_direction", "labeled_at"]
                )
                writer.writeheader()

    def record_prediction(self, signal: Dict) -> str:
        now = datetime.now()
        signal_id = f"{signal['index']}_{now.strftime('%Y%m%d_%H%M%S')}"

        base_row = {
            "signal_id": signal_id,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "index": signal["index"],
            "pred_direction": signal["direction"],
            "strength": signal["strength"],
            "confidence": signal["confidence"],
            "spot": signal["spot"],
            "baseline_spot": signal["baseline_spot"],
            "spot_change": signal["spot_change"],
            "spot_change_pct": signal["spot_change_pct"],
            "delta_flow": signal["components"]["delta_flow"],
            "vega_flow": signal["components"]["vega_flow"],
            "gamma_shift": signal["components"]["gamma_shift"],
        }

        with open(self.predictions_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(base_row.keys()))
            writer.writerow(base_row)

        raw = signal.get("analytics_features", {}).get("raw", {})
        dual = signal.get("dual_context", {})
        detailed_row = {
            **base_row,
            "regime": signal.get("regime", ""),
            "vega_state": signal.get("vega_state", ""),
            "delta_state": signal.get("delta_state", ""),
            "gamma_stress": signal.get("gamma_stress", ""),
            "directional_bias": signal.get("directional_bias", ""),
            "strategy_bias": signal.get("strategy_bias", ""),
            "structural_bias": signal.get("structural_bias", ""),
            "trade_allowed": signal.get("trade_allowed", ""),
            "rule_stack": "|".join(signal.get("rule_stack", []) or []),
            "flow_delta_true": raw.get("flow_delta_true", ""),
            "flow_vega_true": raw.get("flow_vega_true", ""),
            "gamma_shift_true": raw.get("gamma_shift_true", ""),
            "gamma_skew_shift_true": raw.get("gamma_skew_shift_true", ""),
            "pos_delta_now": raw.get("pos_delta_now", ""),
            "theta_skew_shift": raw.get("theta_skew_shift", ""),
            "total_call_oi": raw.get("total_call_oi", ""),
            "total_put_oi": raw.get("total_put_oi", ""),
            "total_call_oi_change": raw.get("total_call_oi_change", ""),
            "total_put_oi_change": raw.get("total_put_oi_change", ""),
            "call_iv_diff": raw.get("call_iv_diff", ""),
            "put_iv_diff": raw.get("put_iv_diff", ""),
            "total_iv_change": raw.get("total_iv_change", ""),
            "oi_imbalance": raw.get("oi_imbalance", ""),
            "intraday_has_reference": dual.get("has_intraday_reference", False),
            "intraday_prior_count": dual.get("prior_count", 0),
            "intraday_status": dual.get("status", ""),
            "intraday_agreement_ratio": dual.get("agreement_ratio", ""),
            "intraday_range_state": dual.get("range_state", ""),
            "intraday_anchor_direction": dual.get("anchor_direction", ""),
            "intraday_anchor_strength": dual.get("anchor_strength", ""),
            "intraday_latest_direction": dual.get("latest_direction", ""),
            "intraday_latest_strength": dual.get("latest_strength", ""),
        }
        with open(self.detailed_predictions_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(detailed_row.keys()))
            writer.writerow(detailed_row)

        return signal_id

    def label_outcome(self, signal_id: str, actual_direction: str) -> bool:
        actual_direction = actual_direction.upper()
        if actual_direction not in {"BULLISH", "BEARISH", "SIDEWAYS"}:
            print("Invalid actual_direction. Use: BULLISH | BEARISH | SIDEWAYS")
            return False

        predictions = self._read_csv(self.predictions_file)
        if not any(p["signal_id"] == signal_id for p in predictions):
            print(f"Signal ID not found: {signal_id}")
            return False

        outcomes = self._read_csv(self.outcomes_file)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated = False

        for row in outcomes:
            if row["signal_id"] == signal_id:
                row["actual_direction"] = actual_direction
                row["labeled_at"] = now
                updated = True
                break

        if not updated:
            outcomes.append(
                {
                    "signal_id": signal_id,
                    "actual_direction": actual_direction,
                    "labeled_at": now,
                }
            )

        with open(self.outcomes_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["signal_id", "actual_direction", "labeled_at"]
            )
            writer.writeheader()
            writer.writerows(outcomes)

        return True

    def summary(self) -> Dict:
        predictions = self._read_csv(self.predictions_file)
        outcomes = self._read_csv(self.outcomes_file)
        outcomes_by_id = {row["signal_id"]: row for row in outcomes}

        labeled = []
        for p in predictions:
            out = outcomes_by_id.get(p["signal_id"])
            if out:
                labeled.append((p, out))

        if not labeled:
            return {
                "total_predictions": len(predictions),
                "labeled_predictions": 0,
                "exact_accuracy": 0.0,
                "structure_accuracy": 0.0,
                "directional_accuracy": 0.0,
            }

        exact_correct = 0
        structure_correct = 0
        directional_total = 0
        directional_correct = 0

        for pred, out in labeled:
            p_dir = pred["pred_direction"]
            a_dir = out["actual_direction"]

            if p_dir == a_dir:
                exact_correct += 1

            p_struct = "DIRECTIONAL" if p_dir in {"BULLISH", "BEARISH"} else "SIDEWAYS"
            a_struct = "DIRECTIONAL" if a_dir in {"BULLISH", "BEARISH"} else "SIDEWAYS"
            if p_struct == a_struct:
                structure_correct += 1

            if p_struct == "DIRECTIONAL" and a_struct == "DIRECTIONAL":
                directional_total += 1
                if p_dir == a_dir:
                    directional_correct += 1

        labeled_count = len(labeled)
        return {
            "total_predictions": len(predictions),
            "labeled_predictions": labeled_count,
            "exact_accuracy": round(exact_correct / labeled_count, 4),
            "structure_accuracy": round(structure_correct / labeled_count, 4),
            "directional_accuracy": round(
                (directional_correct / directional_total) if directional_total else 0.0,
                4,
            ),
        }

    def get_unlabeled_predictions(self):
        predictions = self._read_csv(self.predictions_file)
        outcomes = self._read_csv(self.outcomes_file)
        labeled_ids = {row["signal_id"] for row in outcomes}
        return [p for p in predictions if p["signal_id"] not in labeled_ids]

    def get_latest_prediction(self, index: str, trade_date: Optional[str] = None):
        """
        Fetch latest prediction row for a given index.
        Optionally filter by trade_date in YYYY-MM-DD.
        """
        predictions = self._read_csv(self.predictions_file)
        filtered = [p for p in predictions if p.get("index") == index]

        if trade_date:
            filtered = [
                p
                for p in filtered
                if str(p.get("timestamp", "")).startswith(trade_date)
            ]

        if not filtered:
            return None

        return filtered[-1]

    def get_latest_detailed_prediction(self, index: str, trade_date: Optional[str] = None):
        """
        Fetch latest detailed prediction row for a given index.
        Optionally filter by trade_date in YYYY-MM-DD.
        """
        predictions = self._read_csv(self.detailed_predictions_file)
        filtered = [p for p in predictions if p.get("index") == index]

        if trade_date:
            filtered = [
                p
                for p in filtered
                if str(p.get("timestamp", "")).startswith(trade_date)
            ]

        if not filtered:
            return None

        return filtered[-1]

    def get_first_detailed_prediction(self, index: str, trade_date: Optional[str] = None):
        """
        Fetch first detailed prediction row for a given index.
        Optionally filter by trade_date in YYYY-MM-DD.
        """
        predictions = self._read_csv(self.detailed_predictions_file)
        filtered = [p for p in predictions if p.get("index") == index]

        if trade_date:
            filtered = [
                p
                for p in filtered
                if str(p.get("timestamp", "")).startswith(trade_date)
            ]

        if not filtered:
            return None

        return filtered[0]

    def record_gap_prediction(self, gap_result: Dict) -> str:
        now = datetime.now()
        gap_id = f"{gap_result['index']}_GAP_{now.strftime('%Y%m%d_%H%M%S')}"

        row = {
            "gap_id": gap_id,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "index": gap_result["index"],
            "pred_direction": gap_result["direction"],
            "recommendation": gap_result["recommendation"],
            "score": gap_result["score"],
            "confidence": gap_result["confidence"],
            "spot": gap_result["spot"],
        }

        with open(self.gap_predictions_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

        return gap_id

    def get_unlabeled_gap_predictions(self):
        predictions = self._read_csv(self.gap_predictions_file)
        outcomes = self._read_csv(self.gap_outcomes_file)
        labeled_ids = {row["gap_id"] for row in outcomes}
        return [p for p in predictions if p["gap_id"] not in labeled_ids]

    def label_gap_outcome(self, gap_id: str, actual_direction: str) -> bool:
        actual_direction = actual_direction.upper()
        if actual_direction not in {"BULLISH", "BEARISH", "NEUTRAL"}:
            print("Invalid actual_direction. Use: BULLISH | BEARISH | NEUTRAL")
            return False

        predictions = self._read_csv(self.gap_predictions_file)
        if not any(p["gap_id"] == gap_id for p in predictions):
            print(f"Gap ID not found: {gap_id}")
            return False

        outcomes = self._read_csv(self.gap_outcomes_file)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated = False

        for row in outcomes:
            if row["gap_id"] == gap_id:
                row["actual_direction"] = actual_direction
                row["labeled_at"] = now
                updated = True
                break

        if not updated:
            outcomes.append(
                {
                    "gap_id": gap_id,
                    "actual_direction": actual_direction,
                    "labeled_at": now,
                }
            )

        with open(self.gap_outcomes_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["gap_id", "actual_direction", "labeled_at"]
            )
            writer.writeheader()
            writer.writerows(outcomes)
        return True

    def gap_summary(self) -> Dict:
        predictions = self._read_csv(self.gap_predictions_file)
        outcomes = self._read_csv(self.gap_outcomes_file)
        outcomes_by_id = {row["gap_id"]: row for row in outcomes}

        labeled = []
        for p in predictions:
            out = outcomes_by_id.get(p["gap_id"])
            if out:
                labeled.append((p, out))

        if not labeled:
            return {
                "total_gap_predictions": len(predictions),
                "labeled_gap_predictions": 0,
                "gap_exact_accuracy": 0.0,
            }

        exact_correct = 0
        for pred, out in labeled:
            if pred["pred_direction"] == out["actual_direction"]:
                exact_correct += 1

        labeled_count = len(labeled)
        return {
            "total_gap_predictions": len(predictions),
            "labeled_gap_predictions": labeled_count,
            "gap_exact_accuracy": round(exact_correct / labeled_count, 4),
        }

    @staticmethod
    def _read_csv(path: str):
        if not os.path.exists(path):
            return []
        with open(path, "r", newline="") as f:
            return list(csv.DictReader(f))
