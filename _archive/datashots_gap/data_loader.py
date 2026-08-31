from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
import re

import config
from greeks import GreeksCalculator

IST_OFFSET = timedelta(hours=5, minutes=30)


@dataclass
class Point:
    ts_utc: int
    dt_ist: datetime
    close: float
    spot: float
    strike: float
    iv: Optional[float] = None
    oi: Optional[float] = None


class RollingOptionsDataLoader:
    FILE_RE = re.compile(
        r"^(?P<from>\d{4}-\d{2}-\d{2})_(?P<to>\d{4}-\d{2}-\d{2})_"
        r"(?P<flag>[A-Z]+)_E(?P<code>\d+)_(?P<strike>ATM(?:[+-]\d+)?)_"
        r"(?P<otype>CALL|PUT)\.json$"
    )

    PHASE_TARGETS = {
        "open_0915": time(9, 15),
        "preclose_1445": time(14, 45),
        "close_1515": time(15, 15),
        "next_open_0925": time(9, 25),
    }
    PHASE_WINDOWS = {
        "open_0915": (time(9, 10), time(9, 30)),
        "preclose_1445": (time(14, 30), time(15, 0)),
        "close_1515": (time(14, 45), time(15, 30)),
        "next_open_0925": (time(9, 20), time(9, 35)),
    }

    def __init__(self, cache_root: Path):
        self.cache_root = cache_root
        self.samples: Dict[Tuple[str, str, int, str, str, str], Point] = {}
        self.available_labels: Dict[Tuple[str, str, int], set] = {}
        self.loaded_index: Optional[str] = None

    @staticmethod
    def _time_diff_seconds(a: time, b: time) -> float:
        da = a.hour * 3600 + a.minute * 60 + a.second
        db = b.hour * 3600 + b.minute * 60 + b.second
        return abs(da - db)

    def _update_phase(
        self,
        day_s: str,
        flag: str,
        code: int,
        strike_label: str,
        otype: str,
        p: Point,
    ) -> None:
        t = p.dt_ist.time()
        for phase, target in self.PHASE_TARGETS.items():
            ws, we = self.PHASE_WINDOWS[phase]
            if not (ws <= t <= we):
                continue
            key = (day_s, flag, code, strike_label, otype, phase)
            cur = self.samples.get(key)
            if cur is None:
                self.samples[key] = p
                continue
            cur_diff = self._time_diff_seconds(cur.dt_ist.time(), target)
            new_diff = self._time_diff_seconds(t, target)
            if new_diff < cur_diff:
                self.samples[key] = p

    def build_samples(self, index: str, start_date: str, end_date: str) -> None:
        self.samples.clear()
        self.available_labels.clear()
        self.loaded_index = index

        idx_root = self.cache_root / index
        if not idx_root.exists():
            return
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)

        for f in idx_root.glob("*.json"):
            m = self.FILE_RE.match(f.name)
            if not m:
                continue
            d0 = date.fromisoformat(m.group("from"))
            d1 = date.fromisoformat(m.group("to"))
            if d1 < start or d0 > end:
                continue

            flag = m.group("flag")
            code = int(m.group("code"))
            strike_label = m.group("strike")
            otype = m.group("otype")

            with f.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            leg = payload.get("data", {}).get("ce" if otype == "CALL" else "pe")
            if not isinstance(leg, dict):
                continue

            ts_list = leg.get("timestamp") or []
            close_list = leg.get("close") or []
            spot_list = leg.get("spot") or []
            strike_list = leg.get("strike") or []
            iv_list = leg.get("iv") or []
            oi_list = leg.get("oi") or []

            n = min(len(ts_list), len(close_list), len(spot_list), len(strike_list))
            if n <= 0:
                continue

            for i in range(n):
                try:
                    ts = int(ts_list[i])
                    dt_ist = datetime.utcfromtimestamp(ts) + IST_OFFSET
                    day = dt_ist.date()
                    if day < start or day > end:
                        continue
                    p = Point(
                        ts_utc=ts,
                        dt_ist=dt_ist,
                        close=float(close_list[i]),
                        spot=float(spot_list[i]),
                        strike=float(strike_list[i]),
                        iv=float(iv_list[i]) if i < len(iv_list) and iv_list[i] is not None else None,
                        oi=float(oi_list[i]) if i < len(oi_list) and oi_list[i] is not None else None,
                    )
                except (TypeError, ValueError, IndexError):
                    continue

                day_s = day.isoformat()
                self.available_labels.setdefault((day_s, flag, code), set()).add(strike_label)
                self._update_phase(day_s, flag, code, strike_label, otype, p)

    def get(self, day_s: str, flag: str, code: int, strike_label: str, otype: str, phase: str) -> Optional[Point]:
        return self.samples.get((day_s, flag, code, strike_label, otype, phase))


    def choose_expiry(self, day_s: str, phase: str = "close_1515", min_premium_ratio: float = 0.003) -> Optional[Tuple[str, int]]:
        # Prefer weeklies first for gap trades.
        for flag, code in [("WEEK", 1), ("WEEK", 2), ("WEEK", 3), ("MONTH", 1), ("MONTH", 2), ("MONTH", 3)]:
            labels = self.available_labels.get((day_s, flag, code), set())
            if "ATM" not in labels:
                continue
            ce = self.get(day_s, flag, code, "ATM", "CALL", phase)
            pe = self.get(day_s, flag, code, "ATM", "PUT", phase)
            if not ce or not pe or ce.spot <= 0:
                continue
            ratio = (ce.close + pe.close) / ce.spot
            if ratio >= min_premium_ratio:
                return flag, code
        return None

    def build_chain(self, day_s: str, index: str, flag: str, code: int, phase: str) -> Optional[Dict]:
        labels = self.available_labels.get((day_s, flag, code), set())
        if not labels:
            return None

        strike_gap = config.STRIKE_GAPS.get(index, 50)
        dte = {1: 7, 2: 14, 3: 30}.get(code, 7)
        rows: List[Dict] = []
        spot_ref: Optional[float] = None

        for label in sorted(labels):
            ce = self.get(day_s, flag, code, label, "CALL", phase)
            pe = self.get(day_s, flag, code, label, "PUT", phase)
            if not ce or not pe or ce.iv is None or pe.iv is None:
                continue

            strike = int(round(((ce.strike + pe.strike) / 2.0) / strike_gap) * strike_gap)
            spot = float((ce.spot + pe.spot) / 2.0)
            if spot_ref is None:
                spot_ref = spot

            ce_g = GreeksCalculator.calculate_greeks(spot=spot, strike=strike, dte=dte, iv=float(ce.iv) / 100.0, option_type="CE")
            pe_g = GreeksCalculator.calculate_greeks(spot=spot, strike=strike, dte=dte, iv=float(pe.iv) / 100.0, option_type="PE")

            rows.append(
                {
                    "strike": strike,
                    "ce": {
                        "premium": float(ce.close),
                        "iv": float(ce.iv),
                        "delta": float(ce_g["delta"]),
                        "theta": float(ce_g["theta"]),
                        "vega": float(ce_g["vega"]),
                        "oi": float(ce.oi or 0.0),
                    },
                    "pe": {
                        "premium": float(pe.close),
                        "iv": float(pe.iv),
                        "delta": float(pe_g["delta"]),
                        "theta": float(pe_g["theta"]),
                        "vega": float(pe_g["vega"]),
                        "oi": float(pe.oi or 0.0),
                    },
                }
            )

        if not rows or spot_ref is None:
            return None
        rows.sort(key=lambda x: int(x["strike"]))
        atm = int(round(float(spot_ref) / strike_gap) * strike_gap)
        return {"index": index, "spot": float(spot_ref), "atm": atm, "expiry": f"{flag}_E{code}", "strikes": rows}

    def spot_at(self, day_s: str, index: str, phase: str, flag: str = "WEEK", code: int = 1) -> Optional[float]:
        p = self.get(day_s, flag, code, "ATM", "CALL", phase)
        if p:
            return float(p.spot)
        # fallback
        for (d, _f, _c, _s, _o, ph), point in self.samples.items():
            if d == day_s and ph == phase:
                return float(point.spot)
        return None

    def trading_days(self) -> List[str]:
        return sorted({k[0] for k in self.samples.keys()})
