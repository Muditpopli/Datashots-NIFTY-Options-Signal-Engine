"""
signal_builder.py -- BTST Cache Data Extractor

Given a date and an ExpiryContext, reads the rolling options cache and
produces a SignalSnapshot with two session windows of OI, Greeks, IV,
and PCR data for NIFTY and BANKNIFTY.

No API calls.  All data is read from data/backtest/cache/rolling_options/.

Window 1 (W1): 09:15 IST -- 14:45 IST  (full session, 330 expected bars)
Window 2 (W2): 14:45 IST -- 15:15 IST  (last 30 min,   30 expected bars)
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config
from greeks import GreeksCalculator
from btst_engine.expiry_manager import ExpiryContext, resolve_expiry_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CACHE_DIR: Path = (
    Path(__file__).parent.parent
    / "data" / "backtest" / "cache" / "rolling_options"
)
_IST_OFFSET = timedelta(hours=5, minutes=30)
_IST = timezone(_IST_OFFSET)

# ---------------------------------------------------------------------------
# Strike offset lists
# ---------------------------------------------------------------------------

_ALL_OFFSETS: List[str] = (
    [f"ATM-{i}" for i in range(10, 0, -1)]
    + ["ATM"]
    + [f"ATM+{i}" for i in range(1, 11)]
)  # 21 labels: ATM-10 ... ATM ... ATM+10

_GREEK_OFFSETS: List[str] = (
    [f"ATM-{i}" for i in range(3, 0, -1)]
    + ["ATM"]
    + [f"ATM+{i}" for i in range(1, 4)]
)  # 7 labels: ATM-3 ... ATM ... ATM+3

# ---------------------------------------------------------------------------
# Window time boundaries (IST hh, mm)
# ---------------------------------------------------------------------------

_W1_START = (9, 15)
_W1_END   = (14, 45)
_W2_START = (14, 45)
_W2_END   = (15, 15)

_W1_EXPECTED_BARS = 330   # 09:15-14:44 = 330 one-minute bars
_W2_EXPECTED_BARS = 30    # 14:45-15:14 = 30  one-minute bars

# A window must have >= this fraction of expected bars to be PARTIAL (not MISSING).
_PARTIAL_FLOOR = 0.30

# Near-ATM files present fraction threshold for FULL quality.
_FULL_COVERAGE_BARS  = 0.90   # >= 90% of expected bars
_FULL_STRIKE_FRAC    = 0.80   # >= 80% of ATM+-3 files present


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WindowData:
    """Aggregated option chain metrics for one session window."""

    # Spot OHLC (derived from the ATM file's spot array)
    spot_open:  float = 0.0
    spot_close: float = 0.0
    spot_high:  float = 0.0
    spot_low:   float = 0.0

    # Aggregate OI across all available strikes (ATM-10 to ATM+10)
    total_call_oi:  float = 0.0   # total call OI at window close
    total_put_oi:   float = 0.0   # total put OI at window close
    pcr:            float = 0.0   # put OI / call OI at window close
    oi_change_call: float = 0.0   # close OI - open OI (all calls)
    oi_change_put:  float = 0.0   # close OI - open OI (all puts)

    # OI-weighted net Greeks at window close (ATM-3 to ATM+3 only)
    # Positive net_delta => call OI delta exposure > put OI delta exposure => bullish skew
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_vega:  float = 0.0
    net_theta: float = 0.0

    # ATM implied volatility at window close
    atm_call_iv: float = 0.0
    atm_put_iv:  float = 0.0
    iv_skew:     float = 0.0   # put IV - call IV  (positive => put demand)

    # Max pain strike at window close
    max_pain_strike: float = 0.0

    # Diagnostics
    n_bars:       int = 0
    data_quality: str = "MISSING"   # FULL | PARTIAL | MISSING

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RolloverData:
    """OI rollover metrics comparing current (E1) vs next (E2) expiry."""

    # E1 OI at W2 close
    e1_call_oi: float = 0.0
    e1_put_oi:  float = 0.0
    e1_pcr:     float = 0.0

    # E2 OI at W2 close
    e2_call_oi: float = 0.0
    e2_put_oi:  float = 0.0
    e2_pcr:     float = 0.0

    # Migration ratios: how much of current OI has moved to next expiry
    rollover_ratio_call: float = 0.0   # e2_call_oi / e1_call_oi
    rollover_ratio_put:  float = 0.0   # e2_put_oi  / e1_put_oi

    # Direction signal (per spec: E2 PCR > E1 PCR => BULLISH rollover)
    rollover_direction: str = "NEUTRAL"   # BULLISH | BEARISH | NEUTRAL
    rollover_note:      str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SignalSnapshot:
    """Complete BTST signal data for one index on one date."""

    date:           str
    index:          str
    w1:             WindowData
    w2:             WindowData
    rollover:       Optional[RolloverData]
    expiry_context: dict

    def to_dict(self) -> dict:
        return {
            "date":           self.date,
            "index":          self.index,
            "w1":             self.w1.to_dict(),
            "w2":             self.w2.to_dict(),
            "rollover":       self.rollover.to_dict() if self.rollover else None,
            "expiry_context": self.expiry_context,
        }


# ---------------------------------------------------------------------------
# Low-level file utilities
# ---------------------------------------------------------------------------

def _ts_ist(d: date, h: int, m: int) -> int:
    """
    Return the Unix timestamp (integer seconds) for h:m IST on date d.
    IST = UTC+5:30, so UTC = IST - 19800 s.
    """
    day_start_utc = int(
        datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()
    )
    return day_start_utc + h * 3600 + m * 60 - 19800


def _find_file(
    index: str,
    date_str: str,
    expiry_type: str,   # "WEEK" or "MONTH"
    e_code: str,        # "E1", "E2", "E3"
    offset: str,        # "ATM", "ATM+1", "ATM-3", ...
    opt_type: str,      # "CALL" or "PUT"
) -> Optional[Path]:
    """
    Locate a cache file for the given combination.

    Checks for an exact daily file (from==to==date_str) first; then scans
    range files whose [from, to] bracket date_str.  Returns None if absent.

    Performance: uses a module-level directory index so each index directory
    is scanned at most once (first call), turning O(17k-file glob) per lookup
    into a one-time build + O(range_count) dict lookup thereafter.
    """
    index_dir = _CACHE_DIR / index
    suffix = f"{expiry_type}_{e_code}_{offset}_{opt_type}.json"

    # Fast path: exact daily file (avoids index lookup entirely)
    daily = index_dir / f"{date_str}_{date_str}_{suffix}"
    if daily.exists():
        return daily

    # Build or retrieve the per-index file index.
    # _FILE_IDX[index][suffix] = list of (from_date_str, to_date_str, Path)
    # Built lazily on first access and kept for the process lifetime.
    if index not in _FILE_IDX:
        _FILE_IDX[index] = {}
        for fp in sorted(index_dir.iterdir()):
            nm = fp.name
            if not nm.endswith(".json") or len(nm) < 22:
                continue
            suf = nm[22:]          # everything after "YYYY-MM-DD_YYYY-MM-DD_"
            if nm[10] != "_":
                continue
            from_d, to_d = nm[:10], nm[11:21]
            if from_d == to_d:     # daily files already handled by fast path
                continue
            _FILE_IDX[index].setdefault(suf, []).append((from_d, to_d, fp))

    entries = _FILE_IDX[index].get(suffix)
    if not entries:
        return None

    for from_d, to_d, fp in entries:
        if from_d <= date_str <= to_d:
            return fp

    return None


# Module-level directory index: built once per process per index directory.
_FILE_IDX: Dict[str, Dict[str, list]] = {}


def _load_leg(path: Path, opt_type: str) -> Optional[Dict]:
    """
    Load the CE (CALL) or PE (PUT) data dict from a cache JSON file.
    Returns None if the file is corrupt, empty, or missing required arrays.
    """
    try:
        with open(path) as fh:
            raw = json.load(fh)
        leg_key = "ce" if opt_type == "CALL" else "pe"
        leg = raw.get("data", {}).get(leg_key)
        if not isinstance(leg, dict):
            return None
        # Require the four essential arrays.
        for k in ("timestamp", "iv", "oi", "spot"):
            if not leg.get(k):
                return None
        return leg
    except Exception as exc:
        logger.debug("[BUILDER] Cannot load %s: %s", path, exc)
        return None


def _slice_window(leg: Dict, ts_start: int, ts_end: int) -> Optional[Dict]:
    """
    Return a sub-dict containing only rows where ts_start <= ts < ts_end.
    Returns None if no rows fall within the window.
    """
    ts_arr = leg.get("timestamp") or []
    idxs = [i for i, t in enumerate(ts_arr) if ts_start <= t < ts_end]
    if not idxs:
        return None

    def _pick(key: str) -> list:
        arr = leg.get(key) or []
        return [arr[i] for i in idxs if i < len(arr)]

    return {
        "timestamp": _pick("timestamp"),
        "iv":        _pick("iv"),
        "oi":        _pick("oi"),
        "strike":    _pick("strike"),
        "spot":      _pick("spot"),
        "close":     _pick("close"),
    }


# ---------------------------------------------------------------------------
# Bulk data loading
# ---------------------------------------------------------------------------

def _load_all_slices(
    index: str,
    date_str: str,
    e_code: str,
    ts_w1_start: int, ts_w1_end: int,
    ts_w2_start: int, ts_w2_end: int,
) -> Tuple[Dict, Dict]:
    """
    Load every strike file for (index, date, WEEK, e_code), then slice
    each into W1 and W2 time windows.

    Returns:
        w1_slices, w2_slices  -- each has shape:
            { "CALL": { offset: sliced_dict | None, ... },
              "PUT":  { offset: sliced_dict | None, ... } }
    """
    w1: Dict = {"CALL": {}, "PUT": {}}
    w2: Dict = {"CALL": {}, "PUT": {}}

    for opt_type in ("CALL", "PUT"):
        for offset in _ALL_OFFSETS:
            fp = _find_file(index, date_str, "WEEK", e_code, offset, opt_type)
            if fp is None:
                w1[opt_type][offset] = None
                w2[opt_type][offset] = None
                continue

            leg = _load_leg(fp, opt_type)
            if leg is None:
                w1[opt_type][offset] = None
                w2[opt_type][offset] = None
                continue

            w1[opt_type][offset] = _slice_window(leg, ts_w1_start, ts_w1_end)
            w2[opt_type][offset] = _slice_window(leg, ts_w2_start, ts_w2_end)

    return w1, w2


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

def _spot_ohlc(slices: Dict) -> Dict[str, float]:
    """Derive spot OHLC from the ATM CALL slice (fallback: ATM PUT)."""
    atm = (
        slices.get("CALL", {}).get("ATM")
        or slices.get("PUT",  {}).get("ATM")
    )
    if not atm or not atm.get("spot"):
        return {"open": 0.0, "close": 0.0, "high": 0.0, "low": 0.0}
    sp = atm["spot"]
    return {
        "open":  float(sp[0]),
        "close": float(sp[-1]),
        "high":  float(max(sp)),
        "low":   float(min(sp)),
    }


def _oi_totals(slices: Dict) -> Tuple[float, float, float, float]:
    """
    Sum OI across all offset files.

    Returns: (call_oi_open, call_oi_close, put_oi_open, put_oi_close)
    Open  = first bar in the slice.
    Close = last  bar in the slice.
    """
    co, cc, po, pc = 0.0, 0.0, 0.0, 0.0

    for offset in _ALL_OFFSETS:
        cs = slices.get("CALL", {}).get(offset)
        if cs and cs.get("oi"):
            oi = cs["oi"]
            co += float(oi[0])
            cc += float(oi[-1])

        ps = slices.get("PUT", {}).get(offset)
        if ps and ps.get("oi"):
            oi = ps["oi"]
            po += float(oi[0])
            pc += float(oi[-1])

    return co, cc, po, pc


def _greeks_oi_weighted(
    slices: Dict,
    spot_close: float,
    dte: int,
) -> Tuple[float, float, float, float]:
    """
    Compute OI-weighted net Greeks at window close across ATM-3 to ATM+3.

    Each call/put offset contributes:
        delta * oi,  gamma * oi,  vega * oi,  theta * oi

    where Greek values are from Black-Scholes (greeks.py) and OI is the
    number of contracts open at the last bar of the window.

    Returns: (net_delta, net_gamma, net_vega, net_theta)
    """
    if spot_close <= 0:
        return 0.0, 0.0, 0.0, 0.0

    dte_safe = max(dte, 1)   # BS requires T > 0
    nd = ng = nv = nt = 0.0

    for offset in _GREEK_OFFSETS:
        for opt_type, bs_type in (("CALL", "CE"), ("PUT", "PE")):
            sliced = slices.get(opt_type, {}).get(offset)
            if not sliced:
                continue

            iv_arr  = sliced.get("iv")     or []
            oi_arr  = sliced.get("oi")     or []
            str_arr = sliced.get("strike") or []

            if not iv_arr or not oi_arr or not str_arr:
                continue

            iv_pct = float(iv_arr[-1])
            oi_val = float(oi_arr[-1])
            strike = float(str_arr[-1])

            if iv_pct <= 0 or oi_val <= 0 or strike <= 0:
                continue

            try:
                g = GreeksCalculator.calculate_greeks(
                    spot=spot_close,
                    strike=strike,
                    dte=dte_safe,
                    iv=iv_pct / 100.0,   # greeks.py expects decimal (0.18 not 18)
                    option_type=bs_type,
                )
                nd += g["delta"] * oi_val
                ng += g["gamma"] * oi_val
                nv += g["vega"]  * oi_val
                nt += g["theta"] * oi_val
            except Exception as exc:
                logger.debug("[BUILDER] Greeks failed offset=%s %s: %s", offset, opt_type, exc)

    return nd, ng, nv, nt


def _compute_max_pain(slices: Dict) -> float:
    """
    Compute the max pain strike at window close.

    Max pain = strike S that minimises total payout to option buyers:
        pain(S) = sum_K[ max(S - K_c, 0) * oi_c ]   (ITM calls)
                + sum_K[ max(K_p - S, 0) * oi_p ]   (ITM puts)
    """
    call_oi: Dict[float, float] = {}
    put_oi:  Dict[float, float] = {}

    for offset in _ALL_OFFSETS:
        cs = slices.get("CALL", {}).get(offset)
        if cs and cs.get("strike") and cs.get("oi"):
            k = float(cs["strike"][-1])
            o = float(cs["oi"][-1])
            call_oi[k] = call_oi.get(k, 0.0) + o

        ps = slices.get("PUT", {}).get(offset)
        if ps and ps.get("strike") and ps.get("oi"):
            k = float(ps["strike"][-1])
            o = float(ps["oi"][-1])
            put_oi[k] = put_oi.get(k, 0.0) + o

    all_strikes = sorted(set(call_oi) | set(put_oi))
    if not all_strikes:
        return 0.0

    min_pain    = float("inf")
    mp_strike   = all_strikes[0]

    for s in all_strikes:
        pain = sum(max(s - k, 0.0) * o for k, o in call_oi.items())
        pain += sum(max(k - s, 0.0) * o for k, o in put_oi.items())
        if pain < min_pain:
            min_pain  = pain
            mp_strike = s

    return mp_strike


def _atm_iv(slices: Dict) -> Tuple[float, float, float]:
    """
    Return (atm_call_iv, atm_put_iv, iv_skew) at window close.
    iv_skew = put_iv - call_iv  (positive => elevated put demand).
    """
    cs = slices.get("CALL", {}).get("ATM")
    ps = slices.get("PUT",  {}).get("ATM")

    call_iv = float(cs["iv"][-1]) if cs and cs.get("iv") else 0.0
    put_iv  = float(ps["iv"][-1]) if ps and ps.get("iv") else 0.0

    return call_iv, put_iv, put_iv - call_iv


def _data_quality(slices: Dict, expected_bars: int) -> Tuple[str, int]:
    """
    Assess the data quality for a window.

    Rules:
      MISSING  -- ATM CALL and PUT both absent, or zero bars in window.
      FULL     -- >= 90% of expected bars AND >= 80% of ATM+-3 files present.
      PARTIAL  -- >= 30% of expected bars (but not FULL).
      MISSING  -- < 30% of expected bars.
    """
    atm_c = slices.get("CALL", {}).get("ATM")
    atm_p = slices.get("PUT",  {}).get("ATM")

    if atm_c is None and atm_p is None:
        return "MISSING", 0

    n_bars = max(
        len(atm_c.get("timestamp", [])) if atm_c else 0,
        len(atm_p.get("timestamp", [])) if atm_p else 0,
    )
    if n_bars == 0:
        return "MISSING", 0

    coverage = n_bars / max(expected_bars, 1)

    if coverage >= _FULL_COVERAGE_BARS:
        total_near_atm = len(_GREEK_OFFSETS) * 2
        present = sum(
            1
            for off in _GREEK_OFFSETS
            for ot in ("CALL", "PUT")
            if slices.get(ot, {}).get(off) is not None
        )
        if present / total_near_atm >= _FULL_STRIKE_FRAC:
            return "FULL", n_bars

    if coverage >= _PARTIAL_FLOOR:
        return "PARTIAL", n_bars

    return "MISSING", n_bars


# ---------------------------------------------------------------------------
# Window data builder
# ---------------------------------------------------------------------------

def _build_window(
    slices: Dict,
    expected_bars: int,
    spot_fallback: float,
    dte: int,
) -> WindowData:
    """Assemble a WindowData from pre-sliced strike data."""

    quality, n_bars = _data_quality(slices, expected_bars)

    if quality == "MISSING":
        return WindowData(data_quality="MISSING", n_bars=0)

    ohlc = _spot_ohlc(slices)
    spot_close = ohlc["close"] if ohlc["close"] > 0 else spot_fallback

    co, cc, po, pc = _oi_totals(slices)
    pcr = pc / cc if cc > 0 else 0.0

    nd, ng, nv, nt = _greeks_oi_weighted(slices, spot_close, dte)
    call_iv, put_iv, skew = _atm_iv(slices)
    max_pain = _compute_max_pain(slices)

    return WindowData(
        spot_open=ohlc["open"],
        spot_close=spot_close,
        spot_high=ohlc["high"],
        spot_low=ohlc["low"],
        total_call_oi=cc,
        total_put_oi=pc,
        pcr=pcr,
        oi_change_call=cc - co,
        oi_change_put=pc - po,
        net_delta=nd,
        net_gamma=ng,
        net_vega=nv,
        net_theta=nt,
        atm_call_iv=call_iv,
        atm_put_iv=put_iv,
        iv_skew=skew,
        max_pain_strike=max_pain,
        n_bars=n_bars,
        data_quality=quality,
    )


# ---------------------------------------------------------------------------
# Rollover computation
# ---------------------------------------------------------------------------

def _compute_rollover(
    index: str,
    date_str: str,
    e1_w2_slices: Dict,
    ts_w2_start: int,
    ts_w2_end: int,
) -> Optional[RolloverData]:
    """
    Compare E1 and E2 OI at the close of W2.

    Rollover direction (per spec):
        E2 PCR > E1 PCR  =>  BULLISH  (protective put buying rolling forward)
        E2 PCR < E1 PCR  =>  BEARISH
        otherwise        =>  NEUTRAL
    """
    # Load E2 W2 slices
    e2_slices: Dict = {"CALL": {}, "PUT": {}}
    for opt_type in ("CALL", "PUT"):
        for offset in _ALL_OFFSETS:
            fp = _find_file(index, date_str, "WEEK", "E2", offset, opt_type)
            if fp is None:
                e2_slices[opt_type][offset] = None
                continue
            leg = _load_leg(fp, opt_type)
            if leg is None:
                e2_slices[opt_type][offset] = None
                continue
            e2_slices[opt_type][offset] = _slice_window(leg, ts_w2_start, ts_w2_end)

    # E1 OI at W2 close
    _, e1_call, _, e1_put = _oi_totals(e1_w2_slices)
    # E2 OI at W2 close
    _, e2_call, _, e2_put = _oi_totals(e2_slices)

    if e1_call + e1_put + e2_call + e2_put == 0:
        return None

    e1_pcr = e1_put / e1_call if e1_call > 0 else 0.0
    e2_pcr = e2_put / e2_call if e2_call > 0 else 0.0

    rr_call = e2_call / e1_call if e1_call > 0 else 0.0
    rr_put  = e2_put  / e1_put  if e1_put  > 0 else 0.0

    # Direction: 5% relative change in PCR required to call directional
    if e1_pcr > 0 and abs(e2_pcr - e1_pcr) / e1_pcr > 0.05:
        if e2_pcr > e1_pcr:
            direction = "BULLISH"
            note = (
                f"E2 PCR {e2_pcr:.3f} > E1 PCR {e1_pcr:.3f}: "
                "more protective puts rolling forward"
            )
        else:
            direction = "BEARISH"
            note = (
                f"E2 PCR {e2_pcr:.3f} < E1 PCR {e1_pcr:.3f}: "
                "more calls rolling forward"
            )
    else:
        direction = "NEUTRAL"
        note = f"E1 PCR={e1_pcr:.3f}  E2 PCR={e2_pcr:.3f}: no significant difference"

    return RolloverData(
        e1_call_oi=e1_call,
        e1_put_oi=e1_put,
        e1_pcr=e1_pcr,
        e2_call_oi=e2_call,
        e2_put_oi=e2_put,
        e2_pcr=e2_pcr,
        rollover_ratio_call=rr_call,
        rollover_ratio_put=rr_put,
        rollover_direction=direction,
        rollover_note=note,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_snapshot(
    index: str,
    signal_date: date,
    ctx: ExpiryContext,
) -> SignalSnapshot:
    """
    Build a SignalSnapshot for `index` on `signal_date`.

    Reads only from the rolling options cache (no API calls).
    ExpiryContext must be pre-resolved for `signal_date`.
    """
    date_str = signal_date.strftime("%Y-%m-%d")

    # DTE for Black-Scholes (use current expiry date for the index)
    cur_expiry_str = (
        ctx.current_nifty_expiry if index == "NIFTY" else ctx.current_bnf_expiry
    )
    try:
        dte = max((date.fromisoformat(cur_expiry_str) - signal_date).days, 0)
    except (ValueError, AttributeError):
        dte = config.DEFAULT_DTE

    # Unix timestamp boundaries for W1 and W2
    ts_w1_start = _ts_ist(signal_date, *_W1_START)
    ts_w1_end   = _ts_ist(signal_date, *_W1_END)
    ts_w2_start = _ts_ist(signal_date, *_W2_START)
    ts_w2_end   = _ts_ist(signal_date, *_W2_END)

    # Load and slice all E1 strike files
    w1_slices, w2_slices = _load_all_slices(
        index, date_str, "E1",
        ts_w1_start, ts_w1_end,
        ts_w2_start, ts_w2_end,
    )

    # Spot fallback: use W1 close spot if W2 has no ATM data
    spot_w1 = _spot_ohlc(w1_slices).get("close", 0.0)

    w1 = _build_window(w1_slices, _W1_EXPECTED_BARS, 0.0,    dte)
    w2 = _build_window(w2_slices, _W2_EXPECTED_BARS, spot_w1, dte)

    # Rollover: compare E1 and E2 OI at W2 close
    rollover: Optional[RolloverData] = None
    if ctx.rollover_mode:
        rollover = _compute_rollover(
            index, date_str, w2_slices,
            ts_w2_start, ts_w2_end,
        )

    return SignalSnapshot(
        date=date_str,
        index=index,
        w1=w1,
        w2=w2,
        rollover=rollover,
        expiry_context=ctx.to_dict(),
    )


def build_snapshots(
    signal_date: date,
    ctx: Optional[ExpiryContext] = None,
) -> Dict[str, SignalSnapshot]:
    """
    Build SignalSnapshots for NIFTY and BANKNIFTY on `signal_date`.
    Auto-resolves ExpiryContext when not supplied.
    """
    if ctx is None:
        ctx = resolve_expiry_context(signal_date)
    return {
        "NIFTY":     build_snapshot("NIFTY",     signal_date, ctx),
        "BANKNIFTY": build_snapshot("BANKNIFTY", signal_date, ctx),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_window(label: str, w: WindowData) -> None:
    badge = {"FULL": "[FULL]", "PARTIAL": "[PARTIAL]", "MISSING": "[MISSING]"}.get(
        w.data_quality, "[?]"
    )
    print(f"\n  {label}  {badge}  {w.n_bars} bars")
    if w.data_quality == "MISSING":
        print("    No data in this window.")
        return
    print(
        f"    Spot        open={w.spot_open:.2f}  close={w.spot_close:.2f}"
        f"  high={w.spot_high:.2f}  low={w.spot_low:.2f}"
    )
    print(
        f"    OI          call={w.total_call_oi/1e6:.3f}M  put={w.total_put_oi/1e6:.3f}M"
        f"  PCR={w.pcr:.3f}"
    )
    print(
        f"    OI change   call={w.oi_change_call/1e6:+.3f}M  put={w.oi_change_put/1e6:+.3f}M"
    )
    print(
        f"    IV (ATM)    call={w.atm_call_iv:.2f}%  put={w.atm_put_iv:.2f}%"
        f"  skew={w.iv_skew:+.2f}%"
    )
    print(
        f"    Greeks(+-3) delta={w.net_delta:+.0f}  gamma={w.net_gamma:.4f}"
        f"  vega={w.net_vega:.0f}  theta={w.net_theta:.0f}"
    )
    print(f"    Max pain    {w.max_pain_strike:.0f}")


def _print_rollover(r: RolloverData) -> None:
    print(f"\n  Rollover  [{r.rollover_direction}]")
    print(
        f"    E1  call={r.e1_call_oi/1e6:.3f}M  put={r.e1_put_oi/1e6:.3f}M"
        f"  PCR={r.e1_pcr:.3f}"
    )
    print(
        f"    E2  call={r.e2_call_oi/1e6:.3f}M  put={r.e2_put_oi/1e6:.3f}M"
        f"  PCR={r.e2_pcr:.3f}"
    )
    print(
        f"    Ratio  call={r.rollover_ratio_call:.3f}  put={r.rollover_ratio_put:.3f}"
    )
    print(f"    {r.rollover_note}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if len(sys.argv) < 2:
        print(
            "Usage: python -m btst_engine.signal_builder YYYY-MM-DD",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        signal_date = date.fromisoformat(sys.argv[1])
    except ValueError:
        print(f"[ERR] Bad date '{sys.argv[1]}' -- expected YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)

    ctx = resolve_expiry_context(signal_date)
    snapshots = build_snapshots(signal_date, ctx)

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  BTST Signal Builder  --  {signal_date}")
    print(f"  Expiry  trade={ctx.trade_expiry}  confirm={ctx.confirm_expiry}")
    flags = []
    if ctx.today_is_expiry:  flags.append("EXPIRY_DAY")
    if ctx.is_expiry_eve:    flags.append("EXPIRY_EVE")
    if ctx.rollover_mode:    flags.append("ROLLOVER")
    if flags:
        print(f"  Flags   {' | '.join(flags)}")
    print(sep)

    for idx, snap in snapshots.items():
        print(f"\n{'-' * 64}")
        print(f"  {idx}")
        _print_window("W1  09:15-14:45", snap.w1)
        _print_window("W2  14:45-15:15", snap.w2)
        if snap.rollover:
            _print_rollover(snap.rollover)

    print(f"\n{sep}\n")
