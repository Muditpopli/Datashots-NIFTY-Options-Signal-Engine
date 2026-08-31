from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import math
import re

import config
from greeks import GreeksCalculator


IST_OFFSET = timedelta(hours=5, minutes=30)


@dataclass
class Point:
    ts: int
    dt: datetime
    close: float
    spot: float
    strike: float
    iv: Optional[float] = None
    oi: Optional[float] = None


class RollingCacheBacktester:
    """
    Backtest engine over cached Dhan rolling options files.
    """

    FILE_RE = re.compile(
        r"^(?P<from>\d{4}-\d{2}-\d{2})_(?P<to>\d{4}-\d{2}-\d{2})_"
        r"(?P<flag>[A-Z]+)_E(?P<code>\d+)_(?P<strike>ATM(?:[+-]\d+)?)_"
        r"(?P<otype>CALL|PUT)\.json$"
    )

    def __init__(self, cache_root: Path):
        self.cache_root = cache_root
        self.samples: Dict[Tuple[str, str, int, str, str, str], Point] = {}
        self.available_labels: Dict[Tuple[str, str, int], set] = {}

    def run(
        self,
        index: str,
        start_date: str,
        end_date: str,
        margin: float,
        max_daily_loss_pct: float,
        min_premium_ratio: float = 0.01,
        sideways_spot_band_pct: float = 0.35,
    ) -> Tuple[Dict, List[Dict]]:
        self._build_samples(index=index, start_date=start_date, end_date=end_date)

        max_daily_loss = margin * max_daily_loss_pct
        lot_size = config.LOT_SIZES.get(index, 50)
        strike_gap = config.STRIKE_GAPS.get(index, 50)

        days = sorted({k[0] for k in self.samples.keys()})
        trades: List[Dict] = []
        equity = 0.0
        max_equity = 0.0
        max_dd = 0.0

        for day in days:
            baseline_spot = self._ref_spot(day, phase="baseline")
            entry_spot = self._ref_spot(day, phase="entry")
            if baseline_spot is None or entry_spot is None or baseline_spot <= 0:
                continue

            spot_move_pct = (entry_spot - baseline_spot) / baseline_spot * 100.0
            is_sideways = abs(spot_move_pct) <= sideways_spot_band_pct
            if not is_sideways:
                continue

            best = None
            for flag in ("WEEK", "MONTH"):
                for code in (1, 2, 3):
                    ce_e = self._get(day, flag, code, "ATM", "CALL", "entry")
                    pe_e = self._get(day, flag, code, "ATM", "PUT", "entry")
                    ce_x = self._get(day, flag, code, "ATM", "CALL", "exit")
                    pe_x = self._get(day, flag, code, "ATM", "PUT", "exit")
                    if not ce_e or not pe_e or not ce_x or not pe_x:
                        continue

                    straddle_premium = ce_e.close + pe_e.close
                    premium_ratio = straddle_premium / entry_spot if entry_spot > 0 else 0.0
                    if premium_ratio < min_premium_ratio:
                        continue

                    atm_strike = int(round((ce_e.strike + pe_e.strike) / 2.0 / strike_gap) * strike_gap)
                    dte = self._dte_from_code(code)
                    synthetic = GreeksCalculator.calculate_synthetic_future(
                        call_premium=ce_e.close,
                        put_premium=pe_e.close,
                        strike=atm_strike,
                        dte=dte,
                    )
                    center = int(round(synthetic / strike_gap) * strike_gap)
                    width = max(int(round(straddle_premium / strike_gap) * strike_gap), strike_gap)

                    target_pe = center - width
                    target_ce = center + width
                    labels = self.available_labels.get((day, flag, code), set())
                    pe_label = self._nearest_label(target_pe, atm_strike, strike_gap, labels)
                    ce_label = self._nearest_label(target_ce, atm_strike, strike_gap, labels)
                    if not pe_label or not ce_label:
                        continue

                    pe_entry = self._get(day, flag, code, pe_label, "PUT", "entry")
                    ce_entry = self._get(day, flag, code, ce_label, "CALL", "entry")
                    pe_exit = self._get(day, flag, code, pe_label, "PUT", "exit")
                    ce_exit = self._get(day, flag, code, ce_label, "CALL", "exit")
                    if not pe_entry or not ce_entry or not pe_exit or not ce_exit:
                        continue

                    entry_credit = pe_entry.close + ce_entry.close
                    exit_debit = pe_exit.close + ce_exit.close
                    per_lot_pnl = (entry_credit - exit_debit) * lot_size

                    # Conservative position sizing using entry credit as risk proxy.
                    per_lot_loss_proxy = max(entry_credit * lot_size, 1.0)
                    lots = int(max_daily_loss // per_lot_loss_proxy)
                    if lots < 1:
                        continue

                    pnl = per_lot_pnl * lots
                    pnl_capped = max(pnl, -max_daily_loss)

                    row = {
                        "trade_date": day,
                        "flag": flag,
                        "expiry_code": code,
                        "baseline_spot": round(baseline_spot, 2),
                        "entry_spot": round(entry_spot, 2),
                        "spot_move_pct_to_entry": round(spot_move_pct, 4),
                        "atm_strike": atm_strike,
                        "synthetic_center": center,
                        "width_points": width,
                        "pe_strike_label": pe_label,
                        "ce_strike_label": ce_label,
                        "entry_credit": round(entry_credit, 4),
                        "exit_debit": round(exit_debit, 4),
                        "lots": lots,
                        "lot_size": lot_size,
                        "premium_ratio": round(premium_ratio, 6),
                        "pnl": round(pnl, 2),
                        "pnl_capped": round(pnl_capped, 2),
                        "daily_loss_cap": round(max_daily_loss, 2),
                    }

                    # Pick richer premium ratio, tie-break by higher entry credit.
                    rank = (row["premium_ratio"], row["entry_credit"])
                    if best is None or rank > best[0]:
                        best = (rank, row)

            if not best:
                continue

            trade = best[1]
            equity += trade["pnl_capped"]
            max_equity = max(max_equity, equity)
            max_dd = max(max_dd, max_equity - equity)
            trade["equity_curve"] = round(equity, 2)
            trades.append(trade)

        wins = sum(1 for t in trades if t["pnl_capped"] > 0)
        losses = sum(1 for t in trades if t["pnl_capped"] < 0)
        summary = {
            "mode": "rolling_sideways_strangle",
            "index": index,
            "start_date": start_date,
            "end_date": end_date,
            "trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": round((wins / len(trades)) if trades else 0.0, 4),
            "total_pnl": round(sum(t["pnl_capped"] for t in trades), 2),
            "avg_pnl": round((sum(t["pnl_capped"] for t in trades) / len(trades)) if trades else 0.0, 2),
            "max_drawdown": round(max_dd, 2),
            "margin": margin,
            "max_daily_loss_pct": max_daily_loss_pct,
            "max_daily_loss_value": round(max_daily_loss, 2),
            "sideways_spot_band_pct": sideways_spot_band_pct,
            "min_premium_ratio": min_premium_ratio,
            "notes": (
                "Sideways proxy is based on |spot move| from ~09:19 to ~10:00 being <= configured band. "
                "Entry is ~10:00, exit is ~15:00 using nearest minute in rolling cache."
            ),
        }
        return summary, trades

    def _build_samples(self, index: str, start_date: str, end_date: str) -> None:
        self.samples.clear()
        self.available_labels.clear()
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        if not self.cache_root.exists():
            return

        for file in self.cache_root.glob("*.json"):
            m = self.FILE_RE.match(file.name)
            if not m:
                continue

            # Fast pre-filter: skip cache chunks whose file date range does not
            # intersect the requested backtest window. This avoids loading/parsing
            # large JSON files that cannot contribute any sample points.
            file_from = datetime.strptime(m.group("from"), "%Y-%m-%d").date()
            file_to = datetime.strptime(m.group("to"), "%Y-%m-%d").date()
            if file_to < start or file_from > end:
                continue

            flag = m.group("flag")
            code = int(m.group("code"))
            strike_label = m.group("strike")
            otype = m.group("otype")

            with open(file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            data = payload.get("data", {})
            leg = data.get("ce") if otype == "CALL" else data.get("pe")
            if not isinstance(leg, dict):
                continue

            ts_list = leg.get("timestamp") or []
            close_list = leg.get("close") or []
            spot_list = leg.get("spot") or []
            iv_list = leg.get("iv") or []
            oi_list = leg.get("oi") or []
            strike_list = leg.get("strike") or []
            n = min(len(ts_list), len(close_list), len(spot_list), len(strike_list))
            if n == 0:
                continue

            for i in range(n):
                ts = int(ts_list[i])
                dt_ist = datetime.utcfromtimestamp(ts) + IST_OFFSET
                day = dt_ist.date()
                if day < start or day > end:
                    continue
                day_s = day.isoformat()

                self.available_labels.setdefault((day_s, flag, code), set()).add(strike_label)

                point = Point(
                    ts=ts,
                    dt=dt_ist,
                    close=float(close_list[i]),
                    spot=float(spot_list[i]),
                    strike=float(strike_list[i]),
                    iv=float(iv_list[i]) if i < len(iv_list) and iv_list[i] is not None else None,
                    oi=float(oi_list[i]) if i < len(oi_list) and oi_list[i] is not None else None,
                )
                self._update_phase(day_s, flag, code, strike_label, otype, point)

    def _update_phase(self, day: str, flag: str, code: int, strike_label: str, otype: str, p: Point) -> None:
        targets = {
            "baseline": time(9, 19),
            "entry": time(10, 0),
            "exit": time(15, 0),
        }
        windows = {
            "baseline": (time(9, 15), time(9, 30)),
            "entry": (time(10, 0), time(10, 15)),
            "exit": (time(14, 45), time(15, 10)),
        }
        t = p.dt.time()
        for phase, target in targets.items():
            ws, we = windows[phase]
            if not (ws <= t <= we):
                continue
            key = (day, flag, code, strike_label, otype, phase)
            cur = self.samples.get(key)
            if cur is None:
                self.samples[key] = p
                continue
            cur_diff = self._time_diff_seconds(cur.dt.time(), target)
            new_diff = self._time_diff_seconds(t, target)
            if new_diff < cur_diff:
                self.samples[key] = p

    def _get(
        self,
        day: str,
        flag: str,
        code: int,
        strike_label: str,
        otype: str,
        phase: str,
    ) -> Optional[Point]:
        return self.samples.get((day, flag, code, strike_label, otype, phase))

    def _ref_spot(self, day: str, phase: str) -> Optional[float]:
        # Prefer WEEK/E1 ATM CALL as stable reference.
        p = self._get(day, "WEEK", 1, "ATM", "CALL", phase)
        if p:
            return p.spot
        # Fallback to any available sample for the day+phase.
        for (d, _f, _c, _s, _o, ph), point in self.samples.items():
            if d == day and ph == phase:
                return point.spot
        return None

    @staticmethod
    def _time_diff_seconds(a: time, b: time) -> float:
        da = a.hour * 3600 + a.minute * 60 + a.second
        db = b.hour * 3600 + b.minute * 60 + b.second
        return abs(da - db)

    @staticmethod
    def _dte_from_code(code: int) -> int:
        return {1: 7, 2: 14, 3: 30}.get(code, 7)

    @staticmethod
    def _label_from_n(n: int) -> str:
        if n == 0:
            return "ATM"
        return f"ATM+{n}" if n > 0 else f"ATM{n}"

    def _nearest_label(
        self,
        target_strike: int,
        atm_strike: int,
        gap: int,
        labels: set,
    ) -> Optional[str]:
        if not labels:
            return None
        n_float = (target_strike - atm_strike) / float(gap)
        n = int(round(n_float))
        preferred = self._label_from_n(n)
        if preferred in labels:
            return preferred

        # Fallback to nearest available label by n distance.
        best = None
        for label in labels:
            if label == "ATM":
                ln = 0
            elif label.startswith("ATM+"):
                ln = int(label.replace("ATM+", ""))
            elif label.startswith("ATM-"):
                ln = int(label.replace("ATM", ""))
            else:
                continue
            dist = abs(ln - n)
            if best is None or dist < best[0]:
                best = (dist, label)
        return best[1] if best else None
