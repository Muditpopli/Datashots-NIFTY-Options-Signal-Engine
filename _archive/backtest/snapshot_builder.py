from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


class SnapshotBuilder:
    """
    Builds replay-ready event rows from recorded live signal logs.
    """

    @staticmethod
    def build_intraday_events(rows: List[Dict]) -> List[Dict]:
        # Sort by timestamp string (format is lexicographically sortable).
        ordered = sorted(rows, key=lambda r: r.get("timestamp", ""))
        by_day = defaultdict(list)
        for row in ordered:
            trade_date = str(row.get("timestamp", ""))[:10]
            by_day[trade_date].append(row)

        events: List[Dict] = []
        for trade_date, day_rows in sorted(by_day.items()):
            if not day_rows:
                continue

            baseline_spot = SnapshotBuilder._to_float(day_rows[0].get("baseline_spot"))
            for i, row in enumerate(day_rows):
                current_spot = SnapshotBuilder._to_float(row.get("spot"))
                events.append(
                    {
                        "trade_date": trade_date,
                        "sequence": i + 1,
                        "signal_id": row.get("signal_id", ""),
                        "timestamp": row.get("timestamp", ""),
                        "direction": row.get("pred_direction", ""),
                        "strength": SnapshotBuilder._to_float(row.get("strength")),
                        "confidence": SnapshotBuilder._to_float(row.get("confidence")),
                        "spot": current_spot,
                        "baseline_spot": baseline_spot,
                        "spot_change_from_baseline": (
                            (current_spot - baseline_spot)
                            if current_spot is not None and baseline_spot is not None
                            else None
                        ),
                    }
                )

        return events

    @staticmethod
    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

