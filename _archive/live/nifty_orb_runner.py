from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from dhanhq import dhanhq

import config


IST_OFFSET = timedelta(hours=5, minutes=30)


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float

    @property
    def color(self) -> str:
        if self.close > self.open:
            return "green"
        if self.close < self.open:
            return "red"
        return "doji"


@dataclass
class OrbRange:
    first: Candle
    second: Candle
    high: float
    low: float
    ready_after: datetime


class NiftyOrbRunner:
    def __init__(self, args: argparse.Namespace):
        token = config.DHAN_ACCESS_TOKEN.strip('"').strip("'")
        client = config.DHAN_CLIENT_ID.strip('"').strip("'")
        if not token or not client:
            raise ValueError("Missing DHAN_CLIENT_ID or DHAN_ACCESS_TOKEN in environment.")

        self.args = args
        self.dhan = dhanhq(client, token)
        self.spot_security_id = int(args.spot_security_id)
        self.trade_futures_security_id = args.futures_security_id
        self.triggered = False

    def run(self) -> int:
        print(
            f"[START] NIFTY ORB runner | mode={self.args.mode} | interval={self.args.interval}m | "
            f"session={self.args.session_date}"
        )
        if self.args.mode == "live" and not self.trade_futures_security_id:
            raise ValueError("--futures-security-id is required in live mode.")

        session_date = datetime.strptime(self.args.session_date, "%Y-%m-%d").date()
        pattern = self._wait_for_pattern(session_date)
        if pattern is None:
            print("[STOP] Pattern not found before cutoff.")
            return 1

        return self._monitor_breakout(pattern)

    def _wait_for_pattern(self, session_date: date) -> OrbRange | None:
        cutoff = self._combine_ist(session_date, self.args.entry_cutoff)
        while datetime.utcnow() + IST_OFFSET <= cutoff:
            candles = self._fetch_candles(session_date)
            pattern = self._build_pattern(candles, session_date)
            if pattern is not None:
                self._print_pattern(pattern)
                return pattern
            print("[WAIT] ORB pattern not ready yet. Polling again...")
            time.sleep(self.args.poll_seconds)
        return None

    def _monitor_breakout(self, pattern: OrbRange) -> int:
        cutoff = self._combine_ist(pattern.ready_after.date(), self.args.entry_cutoff)
        while datetime.utcnow() + IST_OFFSET <= cutoff and not self.triggered:
            candles = self._fetch_candles(pattern.ready_after.date())
            breakout = self._check_breakout(candles, pattern)
            if breakout is None:
                print("[WAIT] No breakout yet. Polling again...")
                time.sleep(self.args.poll_seconds)
                continue

            side, trigger_candle = breakout
            self._fire_trade(side=side, trigger_candle=trigger_candle, pattern=pattern)
            self.triggered = True
            return 0

        print("[STOP] No breakout triggered before cutoff.")
        return 1

    def _fetch_candles(self, session_date: date) -> list[Candle]:
        resp = self.dhan.intraday_minute_data(
            security_id=str(self.spot_security_id),
            exchange_segment=dhanhq.IDX_I if hasattr(dhanhq, "IDX_I") else "IDX_I",
            instrument_type="INDEX",
            from_date=session_date.strftime("%Y-%m-%d"),
            to_date=session_date.strftime("%Y-%m-%d"),
            interval=int(self.args.interval),
        )
        data = resp.get("data") if isinstance(resp, dict) else None
        candles = self._parse_candles(data)
        candles.sort(key=lambda row: row.timestamp)
        return candles

    def _parse_candles(self, payload: Any) -> list[Candle]:
        rows: list[Candle] = []
        if payload is None:
            return rows

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    candle = self._candle_from_row(item)
                    if candle is not None:
                        rows.append(candle)
            return rows

        if isinstance(payload, dict):
            if {"open", "high", "low", "close", "timestamp"}.issubset(payload.keys()):
                lengths = []
                for key in ("open", "high", "low", "close", "timestamp"):
                    val = payload.get(key)
                    if isinstance(val, list):
                        lengths.append(len(val))
                if lengths:
                    count = min(lengths)
                    for idx in range(count):
                        candle = self._candle_from_row(
                            {
                                "open": payload.get("open", [None])[idx],
                                "high": payload.get("high", [None])[idx],
                                "low": payload.get("low", [None])[idx],
                                "close": payload.get("close", [None])[idx],
                                "timestamp": payload.get("timestamp", [None])[idx],
                            }
                        )
                        if candle is not None:
                            rows.append(candle)
                    return rows

            for value in payload.values():
                rows.extend(self._parse_candles(value))
            return rows

        return rows

    def _candle_from_row(self, row: dict[str, Any]) -> Candle | None:
        try:
            ts_value = row.get("timestamp")
            if ts_value is None:
                return None
            ts_int = int(float(ts_value))
            ts_ist = datetime.utcfromtimestamp(ts_int) + IST_OFFSET
            return Candle(
                timestamp=ts_ist,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _build_pattern(self, candles: list[Candle], session_date: date) -> OrbRange | None:
        start = self._combine_ist(session_date, self.args.start_time)
        end = start + timedelta(minutes=self.args.interval * 2)
        window = [c for c in candles if start <= c.timestamp < end]
        if len(window) < 2:
            return None

        first, second = window[0], window[1]
        if first.color != "green" or second.color != "red":
            print(
                f"[INFO] First two candles do not match green-red pattern: "
                f"{first.color}, {second.color}"
            )
            return None

        return OrbRange(
            first=first,
            second=second,
            high=max(first.high, second.high),
            low=min(first.low, second.low),
            ready_after=second.timestamp + timedelta(minutes=self.args.interval),
        )

    def _check_breakout(self, candles: list[Candle], pattern: OrbRange) -> tuple[str, Candle] | None:
        for candle in candles:
            if candle.timestamp < pattern.ready_after:
                continue
            if self.args.breakout_basis == "close":
                if candle.close > pattern.high:
                    return ("LONG", candle)
                if candle.close < pattern.low:
                    return ("SHORT", candle)
            else:
                if candle.high > pattern.high:
                    return ("LONG", candle)
                if candle.low < pattern.low:
                    return ("SHORT", candle)
        return None

    def _fire_trade(self, side: str, trigger_candle: Candle, pattern: OrbRange) -> None:
        payload = {
            "session_date": self.args.session_date,
            "symbol": self.args.symbol,
            "side": side,
            "mode": self.args.mode,
            "breakout_basis": self.args.breakout_basis,
            "range_high": round(pattern.high, 2),
            "range_low": round(pattern.low, 2),
            "trigger_time_ist": trigger_candle.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "trigger_candle": {
                "open": trigger_candle.open,
                "high": trigger_candle.high,
                "low": trigger_candle.low,
                "close": trigger_candle.close,
            },
            "suggested_stop_loss": round(pattern.low if side == "LONG" else pattern.high, 2),
            "quantity": self.args.quantity,
            "futures_security_id": self.trade_futures_security_id,
        }

        print("[SIGNAL]")
        print(json.dumps(payload, indent=2))

        if self.args.mode == "signal":
            return

        if self.args.mode == "paper":
            print("[PAPER] Trade simulated. No broker order sent.")
            return

        txn = dhanhq.BUY if side == "LONG" else dhanhq.SELL
        response = self.dhan.place_order(
            security_id=str(int(self.trade_futures_security_id)),
            exchange_segment=dhanhq.NSE_FNO,
            transaction_type=txn,
            quantity=int(self.args.quantity),
            order_type=dhanhq.MARKET,
            product_type=dhanhq.INTRA,
            price=0,
            tag=self.args.order_tag,
        )
        print("[LIVE] Order response:")
        print(json.dumps(response, indent=2, default=str))

    def _print_pattern(self, pattern: OrbRange) -> None:
        print("[PATTERN] Green + Red confirmed")
        print(
            json.dumps(
                {
                    "first_candle_time": pattern.first.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "second_candle_time": pattern.second.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "range_high": round(pattern.high, 2),
                    "range_low": round(pattern.low, 2),
                    "breakout_watch_starts": pattern.ready_after.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )

    @staticmethod
    def _combine_ist(session_date: date, hhmm: str) -> datetime:
        parsed = datetime.strptime(hhmm, "%H:%M").time()
        return datetime.combine(session_date, parsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate a green-then-red opening range breakout strategy for NIFTY."
    )
    parser.add_argument("--mode", choices=["signal", "paper", "live"], default="signal")
    parser.add_argument("--session-date", default=(datetime.utcnow() + IST_OFFSET).strftime("%Y-%m-%d"))
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--spot-security-id", type=int, default=13, help="Dhan security id for NIFTY spot index.")
    parser.add_argument("--futures-security-id", type=int, default=None, help="Dhan security id for tradable NIFTY futures contract.")
    parser.add_argument("--interval", type=int, default=5, help="Candle interval in minutes.")
    parser.add_argument("--start-time", default="09:15", help="Session candle start time in IST.")
    parser.add_argument("--entry-cutoff", default="11:00", help="Stop monitoring for new entries after this IST time.")
    parser.add_argument("--breakout-basis", choices=["high_low", "close"], default="high_low")
    parser.add_argument("--quantity", type=int, default=75, help="Order quantity for paper/live mode.")
    parser.add_argument("--poll-seconds", type=int, default=20, help="Polling interval while waiting for candles/breakout.")
    parser.add_argument("--order-tag", default="NIFTY_ORB")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = NiftyOrbRunner(args)
    return runner.run()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.")
        raise SystemExit(130)
