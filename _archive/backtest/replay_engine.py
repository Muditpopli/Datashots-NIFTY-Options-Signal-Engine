from __future__ import annotations

from typing import Dict, List


class ReplayEngine:
    """
    Minimal replay engine over recorded signal events.
    """

    def run(self, events: List[Dict], outcomes_by_signal_id: Dict[str, Dict]) -> Dict:
        if not events:
            return {
                "total_events": 0,
                "labeled_events": 0,
                "exact_accuracy": 0.0,
                "avg_strength": 0.0,
                "avg_confidence": 0.0,
                "direction_counts": {},
            }

        direction_counts: Dict[str, int] = {}
        strength_values: List[float] = []
        conf_values: List[float] = []
        labeled = 0
        exact = 0

        for e in events:
            direction = e.get("direction", "")
            if direction:
                direction_counts[direction] = direction_counts.get(direction, 0) + 1

            if isinstance(e.get("strength"), float):
                strength_values.append(e["strength"])
            if isinstance(e.get("confidence"), float):
                conf_values.append(e["confidence"])

            sid = e.get("signal_id", "")
            out = outcomes_by_signal_id.get(sid)
            if out:
                labeled += 1
                if out.get("actual_direction") == direction:
                    exact += 1

        return {
            "total_events": len(events),
            "labeled_events": labeled,
            "exact_accuracy": round((exact / labeled) if labeled else 0.0, 4),
            "avg_strength": round(sum(strength_values) / len(strength_values), 4)
            if strength_values
            else 0.0,
            "avg_confidence": round(sum(conf_values) / len(conf_values), 4)
            if conf_values
            else 0.0,
            "direction_counts": direction_counts,
        }

