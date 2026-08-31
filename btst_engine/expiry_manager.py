"""
expiry_manager.py — BTST Expiry Context Manager

Given today's date this module resolves everything the BTST signal engine
needs to know about expiry state:

  - Upcoming NIFTY and BANKNIFTY expiry calendars
  - Whether today or tomorrow is an expiry day for either index
  - Which expiry to use for the overnight options trade (skip expiring
    contracts — overnight holds in expiry-day options are reckless)
  - Whether rollover mode is active (OI migration across expiries is
    a confirmatory signal when expiry is imminent)

Expiry calendar priority:
  1. Live Dhan API (most accurate)
  2. Cache-derived schedule — scans data/backtest/cache/rolling_options/
     to infer the expiry weekday from observed boundary dates
  3. Last-known documented defaults (only used when cache signal is tied)

Expiry weekday is NOT hardcoded. It is derived from the rolling options
cache so that any future NSE schedule change is reflected automatically.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Set, Tuple

import config

logger = logging.getLogger(__name__)


# ─── Cache location ───────────────────────────────────────────────────────────

# Path resolved relative to this file so it works regardless of CWD.
_CACHE_DIR: Path = (
    Path(__file__).parent.parent / "data" / "backtest" / "cache" / "rolling_options"
)

# ─── Documented schedule defaults ─────────────────────────────────────────────
#
# These are the LAST KNOWN expiry weekdays, captured from observed NSE schedule.
# They are used ONLY when the cache scan cannot produce a clear winner (e.g.,
# tied counts, insufficient data, or cache absent).  If NSE changes the expiry
# day, the cache-derived path will detect it first; these values will lag by
# at most the cache's coverage period.
#
#   0 = Monday, 1 = Tuesday, 2 = Wednesday, 3 = Thursday, 4 = Friday
#
# NSE schedule as of March 2026 (confirmed from cache data and exchange notices):
#   NIFTY     — weekly expiry moved from Thursday to Tuesday
#   BANKNIFTY — monthly expiry on Tuesday (last Tuesday of the month)
#
# Evidence from cache: March 5 2026 (Thursday) is the last date with no
# WEEK_E1_ATM_CALL file → final Thursday expiry.  Daily-file era starts
# March 6 2026.  Range file from_date 2026-01-27 (Tuesday) marks the first
# cycle under the new Tuesday schedule.
_FALLBACK_EXPIRY_WEEKDAY: dict[str, int] = {
    "NIFTY":     1,   # Tuesday — current schedule as of early 2026
    "BANKNIFTY": 1,   # Tuesday — current schedule as of early 2026
}

# How many weeks ahead to build the forward calendar.
_CALENDAR_WEEKS_AHEAD = 10


# ─── Output contract ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExpiryContext:
    """
    Fully resolved expiry state for one trading day.

    trade_expiry    — NIFTY expiry for overnight option entry.
                      Skips to next if current expires today or tomorrow.
    confirm_expiry  — BANKNIFTY expiry for directional confirmation.
                      Same skip logic applied independently.
    is_expiry_eve   — True when tomorrow is a NIFTY or BANKNIFTY expiry.
    today_is_expiry — True when today itself is a NIFTY or BANKNIFTY expiry.
    rollover_mode   — True when either flag above is set; activates OI
                      migration tracking across current and next expiry.
    """

    # ── Primary trading output ────────────────────────────────────────────────
    trade_expiry: str
    confirm_expiry: str

    # ── Regime flags ──────────────────────────────────────────────────────────
    is_expiry_eve: bool
    today_is_expiry: bool
    rollover_mode: bool

    # ── Per-index current / next expiry ───────────────────────────────────────
    current_nifty_expiry: str
    next_nifty_expiry: str
    current_bnf_expiry: str
    next_bnf_expiry: str

    # ── Per-index granular flags ───────────────────────────────────────────────
    tomorrow_is_nifty_expiry: bool
    tomorrow_is_bnf_expiry: bool
    today_is_nifty_expiry: bool
    today_is_bnf_expiry: bool

    # ── Forward calendars ─────────────────────────────────────────────────────
    nifty_expiries: List[str] = field(default_factory=list)
    bnf_expiries: List[str] = field(default_factory=list)

    # ── Provenance ────────────────────────────────────────────────────────────
    source: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        parts = []
        if self.today_is_expiry:
            parts.append("[EXPIRY DAY]")
        elif self.is_expiry_eve:
            parts.append("[EXPIRY EVE]")
        else:
            parts.append("[NORMAL]")
        if self.rollover_mode:
            parts.append("+ROLLOVER")
        parts.append(
            f"trade={self.trade_expiry} confirm={self.confirm_expiry} | "
            f"NIFTY {self.current_nifty_expiry}->{self.next_nifty_expiry} | "
            f"BNF {self.current_bnf_expiry}->{self.next_bnf_expiry} "
            f"[{self.source}]"
        )
        return " ".join(parts)


# ─── Holiday / trading-day utilities ─────────────────────────────────────────

def _is_market_holiday(d: date) -> bool:
    if d.weekday() >= 5:
        return True
    return d.strftime("%Y-%m-%d") in config.MARKET_HOLIDAYS


def _prev_trading_day(d: date) -> date:
    while _is_market_holiday(d):
        d -= timedelta(days=1)
    return d


# ─── Cache scanning ───────────────────────────────────────────────────────────

# Match only the canonical sentinel file per (index, date-range):
#   WEEK_E1_ATM_CALL — one file type is enough; avoids counting the same
#   date multiple times from the many strike/type variants.
_FNAME_PAT = re.compile(
    r"^(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_WEEK_E1_ATM_CALL\.json$"
)


def _scan_cache_filenames(index: str) -> Tuple[List[str], List[str], Set[str]]:
    """
    Parse WEEK_E1_ATM_CALL filenames for `index`.

    Returns:
        range_from_dates   — from_dates of multi-day range files (period starts)
        boundary_to_dates  — to_dates of multi-day range files (period ends)
        single_dates       — dates from files where from == to (daily files)

    Range-file from_dates are the primary expiry-date candidates: each range
    file marks a new contract period starting on (or immediately after) an
    expiry date.  They are more reliable than to_dates because the to_date is
    the last data day of the previous cycle, while the from_date is the first
    day of the next cycle — which equals or immediately follows the expiry.
    """
    index_dir = _CACHE_DIR / index
    if not index_dir.is_dir():
        return [], [], set()

    range_from_dates: List[str] = []
    boundary_to_dates: List[str] = []
    single_dates: Set[str] = set()

    for fp in index_dir.glob("*.json"):
        m = _FNAME_PAT.match(fp.name)
        if not m:
            continue
        from_d, to_d = m.group(1), m.group(2)
        if from_d != to_d:
            range_from_dates.append(from_d)
            boundary_to_dates.append(to_d)
        else:
            single_dates.add(from_d)

    return sorted(set(range_from_dates)), sorted(boundary_to_dates), single_dates


def _all_cache_dates() -> Set[str]:
    """
    Collect every date string present in any WEEK_E1_ATM_CALL filename
    across both NIFTY and BANKNIFTY.  Used to find the last available date.
    """
    all_dates: Set[str] = set()
    for index in ("NIFTY", "BANKNIFTY"):
        from_dates, to_dates, single = _scan_cache_filenames(index)
        all_dates.update(from_dates)
        all_dates.update(to_dates)
        all_dates.update(single)
    return all_dates


def find_last_cache_date() -> Optional[date]:
    """
    Scan the rolling options cache and return the latest date found
    across all NIFTY and BANKNIFTY filenames.  Returns None if the cache
    is absent or empty.
    """
    all_dates = _all_cache_dates()
    if not all_dates:
        return None
    return date.fromisoformat(max(all_dates))


# ─── Expiry weekday derivation ────────────────────────────────────────────────

def _derive_expiry_weekday(index: str) -> Optional[int]:
    """
    Infer the weekly expiry day-of-week for `index` from the cache filenames.

    Method: uses the from_date of each range (multi-day) file as a direct
    expiry-date candidate.  Each range file marks when a new contract period
    began being tracked; that date equals or immediately follows the expiry
    date of the previous contract.  Among these candidates, the weekday that
    appears in >40% of the most recent 12 months of data (and is strictly
    ahead of all other weekdays) is returned as the derived expiry weekday.

    Why this can still be inconclusive: the cache stores data in monthly
    chunks rather than per-expiry-cycle chunks.  Range files therefore start
    on the 1st (or 2nd) of each month regardless of the actual expiry day,
    causing from_dates to rotate through all five weekdays uniformly over a
    year.  When the signal is weak or tied, None is returned and the caller
    falls back to _FALLBACK_EXPIRY_WEEKDAY.

    A future improvement would be to detect cycle boundaries by comparing
    the ATM strike or OI across consecutive daily files — that would give a
    clean, unambiguous signal without relying on range-file structure.
    """
    from_dates, _, _ = _scan_cache_filenames(index)
    if not from_dates:
        return None

    max_d  = date.fromisoformat(from_dates[-1])
    cutoff = max_d - timedelta(days=365)

    # Filter to weekday dates in the last 12 months only.
    recent = [
        d_str for d_str in from_dates
        if date.fromisoformat(d_str) >= cutoff
        and date.fromisoformat(d_str).weekday() < 5
    ]

    if len(recent) < 3:
        return None

    wd_counts = Counter(date.fromisoformat(d).weekday() for d in recent)
    ranked    = wd_counts.most_common()
    total     = sum(wd_counts.values())

    if not ranked:
        return None

    top_wd, top_count = ranked[0]

    # Require >40% dominance AND a strict lead over the runner-up.
    if top_count / total < 0.40:
        logger.debug(
            "[EXPIRY] Weekday signal too weak for %s (%.0f%% < 40%%): %s — "
            "using documented default",
            index, top_count / total * 100, dict(ranked[:3]),
        )
        return None

    if len(ranked) >= 2 and ranked[1][1] == top_count:
        logger.debug(
            "[EXPIRY] Weekday derivation tied for %s: %s — using documented default",
            index, ranked[:3],
        )
        return None

    logger.info(
        "[EXPIRY] Cache-derived expiry weekday for %s: %d (%.0f%% of %d dates)",
        index, top_wd, top_count / total * 100, total,
    )
    return top_wd


def _get_expiry_weekday(index: str) -> Tuple[int, str]:
    """
    Return (weekday_int, source_label) for `index`.

    Tries cache derivation first; falls back to last-known documented default.
    """
    derived = _derive_expiry_weekday(index)
    if derived is not None:
        return derived, "cache_derived"
    fallback = _FALLBACK_EXPIRY_WEEKDAY.get(index, 3)
    logger.info(
        "[EXPIRY] Using documented default weekday for %s: %d", index, fallback
    )
    return fallback, "documented_default"


# ─── Fallback calendar construction ──────────────────────────────────────────

def _build_fallback_calendar(index: str, ref_date: date) -> Tuple[List[str], str]:
    """
    Build a weekly expiry calendar for `index` starting from `ref_date`.

    Weekday is derived from the rolling options cache; falls back to the
    last-known documented schedule when the cache signal is inconclusive.

    NSE rule: if the nominal expiry falls on a market holiday, it moves to
    the previous trading day.  Duplicates (two nominations collapsing onto
    the same trading day) are silently dropped.
    """
    target_wd, weekday_source = _get_expiry_weekday(index)
    days_ahead = (target_wd - ref_date.weekday()) % 7
    nominal = ref_date + timedelta(days=days_ahead)

    seen: Set[str] = set()
    expiries: List[str] = []

    for _ in range(_CALENDAR_WEEKS_AHEAD):
        actual = _prev_trading_day(nominal)
        key = actual.strftime("%Y-%m-%d")
        if key not in seen:
            seen.add(key)
            expiries.append(key)
        nominal += timedelta(weeks=1)

    return sorted(expiries), weekday_source


# ─── Dhan API integration ─────────────────────────────────────────────────────

def _fetch_expiries_from_dhan(index: str) -> Optional[List[str]]:
    try:
        from dhan_api import get_dhan
        api = get_dhan()
        expiries = api.get_expiry_list(index)
        if expiries and len(expiries) >= 2:
            logger.info("[EXPIRY] Dhan returned %d expiries for %s", len(expiries), index)
            return expiries
        logger.warning("[EXPIRY] Dhan returned thin list for %s: %s", index, expiries)
        return None
    except Exception as exc:
        logger.warning("[EXPIRY] Dhan API unavailable for %s: %s", index, exc)
        return None


# ─── Expiry list resolution ───────────────────────────────────────────────────

def _resolve_expiry_list(index: str, today: date) -> Tuple[List[str], str]:
    """
    Return (sorted_expiries_from_today, source) for `index`.

    Source is one of: "dhan_api" | "cache_derived" | "documented_default"
    """
    expiries = _fetch_expiries_from_dhan(index)
    if expiries:
        today_str = today.strftime("%Y-%m-%d")
        expiries = sorted(e for e in expiries if e >= today_str)
        return expiries, "dhan_api"

    expiries, weekday_source = _build_fallback_calendar(index, today)
    today_str = today.strftime("%Y-%m-%d")
    expiries = sorted(e for e in expiries if e >= today_str)
    return expiries, weekday_source


def _current_expiry(expiries: List[str], today_str: str) -> Optional[str]:
    for exp in expiries:
        if exp >= today_str:
            return exp
    return expiries[0] if expiries else None


def _next_expiry_after(expiries: List[str], current: str) -> Optional[str]:
    for exp in expiries:
        if exp > current:
            return exp
    return None


# ─── Public API ──────────────────────────────────────────────────────────────

def resolve_expiry_context(today: Optional[date] = None) -> ExpiryContext:
    """
    Resolve all expiry state for `today` (defaults to date.today()).

    Raises RuntimeError if the expiry calendar cannot be constructed for
    either index — the BTST engine must not trade without a valid context.
    """
    if today is None:
        today = date.today()

    today_str = today.strftime("%Y-%m-%d")
    tomorrow_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    nifty_expiries, nifty_source = _resolve_expiry_list("NIFTY", today)
    bnf_expiries, bnf_source = _resolve_expiry_list("BANKNIFTY", today)

    if len(nifty_expiries) < 2:
        raise RuntimeError(
            f"[EXPIRY] NIFTY calendar too short: {nifty_expiries}. "
            "Check Dhan API credentials and config.MARKET_HOLIDAYS."
        )
    if len(bnf_expiries) < 2:
        raise RuntimeError(
            f"[EXPIRY] BANKNIFTY calendar too short: {bnf_expiries}. "
            "Check Dhan API credentials and config.MARKET_HOLIDAYS."
        )

    source = nifty_source if nifty_source == "dhan_api" else f"fallback:{nifty_source}"

    cur_nifty = _current_expiry(nifty_expiries, today_str)
    nxt_nifty = _next_expiry_after(nifty_expiries, cur_nifty)
    cur_bnf = _current_expiry(bnf_expiries, today_str)
    nxt_bnf = _next_expiry_after(bnf_expiries, cur_bnf)

    if not cur_nifty or not nxt_nifty:
        raise RuntimeError(f"[EXPIRY] Cannot resolve NIFTY current/next from {nifty_expiries}")
    if not cur_bnf or not nxt_bnf:
        raise RuntimeError(f"[EXPIRY] Cannot resolve BANKNIFTY current/next from {bnf_expiries}")

    today_is_nifty = cur_nifty == today_str
    today_is_bnf = cur_bnf == today_str
    tomorrow_is_nifty = tomorrow_str in nifty_expiries
    tomorrow_is_bnf = tomorrow_str in bnf_expiries

    is_expiry_eve = tomorrow_is_nifty or tomorrow_is_bnf
    today_is_expiry = today_is_nifty or today_is_bnf
    rollover_mode = is_expiry_eve or today_is_expiry

    if today_is_nifty or tomorrow_is_nifty:
        trade_expiry = nxt_nifty
        logger.info(
            "[EXPIRY] NIFTY: skipping %s (expires %s), using %s for BTST",
            cur_nifty, "today" if today_is_nifty else "tomorrow", nxt_nifty,
        )
    else:
        trade_expiry = cur_nifty

    if today_is_bnf or tomorrow_is_bnf:
        confirm_expiry = nxt_bnf
        logger.info(
            "[EXPIRY] BANKNIFTY: skipping %s (expires %s), using %s for confirmation",
            cur_bnf, "today" if today_is_bnf else "tomorrow", nxt_bnf,
        )
    else:
        confirm_expiry = cur_bnf

    ctx = ExpiryContext(
        trade_expiry=trade_expiry,
        confirm_expiry=confirm_expiry,
        is_expiry_eve=is_expiry_eve,
        today_is_expiry=today_is_expiry,
        rollover_mode=rollover_mode,
        current_nifty_expiry=cur_nifty,
        next_nifty_expiry=nxt_nifty,
        current_bnf_expiry=cur_bnf,
        next_bnf_expiry=nxt_bnf,
        tomorrow_is_nifty_expiry=tomorrow_is_nifty,
        tomorrow_is_bnf_expiry=tomorrow_is_bnf,
        today_is_nifty_expiry=today_is_nifty,
        today_is_bnf_expiry=today_is_bnf,
        nifty_expiries=nifty_expiries[:8],
        bnf_expiries=bnf_expiries[:8],
        source=source,
    )

    logger.info("[EXPIRY] %s", ctx)
    return ctx


# ─── Convenience helpers ──────────────────────────────────────────────────────

def is_expiry_eve_today(today: Optional[date] = None) -> bool:
    return resolve_expiry_context(today).is_expiry_eve


def get_trade_expiry(index: str = "NIFTY", today: Optional[date] = None) -> str:
    ctx = resolve_expiry_context(today)
    return ctx.confirm_expiry if index == "BANKNIFTY" else ctx.trade_expiry


# ─── CLI / debug ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Fix 2: default to the last date present in the rolling options cache.
    last_cache_date = find_last_cache_date()
    ref: date = last_cache_date or date.today()

    if len(sys.argv) > 1:
        try:
            ref = date.fromisoformat(sys.argv[1])
        except ValueError:
            print(f"[ERR] Bad date '{sys.argv[1]}' — expected YYYY-MM-DD", file=sys.stderr)
            sys.exit(1)

    print(f"\n{'='*62}")
    print(f"  BTST Expiry Manager  —  {ref}")
    if last_cache_date:
        print(f"  Last available date in cache: {last_cache_date}")
    print(f"{'='*62}")

    try:
        ctx = resolve_expiry_context(ref)
    except RuntimeError as err:
        print(f"\n[ERR] {err}", file=sys.stderr)
        sys.exit(2)

    print(f"\n  Status          : {ctx}")
    print()
    print(f"  trade_expiry    : {ctx.trade_expiry}")
    print(f"  confirm_expiry  : {ctx.confirm_expiry}")
    print()
    print(f"  is_expiry_eve   : {ctx.is_expiry_eve}")
    print(f"  today_is_expiry : {ctx.today_is_expiry}")
    print(f"  rollover_mode   : {ctx.rollover_mode}")
    print()
    print(f"  NIFTY  current  : {ctx.current_nifty_expiry}")
    print(f"  NIFTY  next     : {ctx.next_nifty_expiry}")
    print(f"  BNF    current  : {ctx.current_bnf_expiry}")
    print(f"  BNF    next     : {ctx.next_bnf_expiry}")
    print()
    print(f"  tomorrow_is_nifty_expiry : {ctx.tomorrow_is_nifty_expiry}")
    print(f"  tomorrow_is_bnf_expiry   : {ctx.tomorrow_is_bnf_expiry}")
    print(f"  today_is_nifty_expiry    : {ctx.today_is_nifty_expiry}")
    print(f"  today_is_bnf_expiry      : {ctx.today_is_bnf_expiry}")
    print()
    print(f"  Source          : {ctx.source}")
    print(f"  NIFTY calendar  : {ctx.nifty_expiries}")
    print(f"  BNF   calendar  : {ctx.bnf_expiries}")
    print(f"\n{'='*62}\n")
