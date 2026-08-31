"""
Overnight Gap Signal Engine
============================
Generates a BULLISH / BEARISH / DIVERGE signal at 3:20 PM for the
next trading session's opening gap direction.

Signal Architecture — 8-factor directional alignment
------------------------------------------------------
For each index (Nifty primary, Banknifty confirmation), four checks:

  Check 1 — Spot day direction   : sign(close_1515 − open_0915)
  Check 2 — Spot L30 direction   : sign(close_1515 − preclose_1445)
  Check 3 — OI skew day          : sign(Δput_oi − Δcall_oi) full day
  Check 4 — OI skew L30          : sign(Δput_oi − Δcall_oi) last 30 min

OI Skew Rationale
-----------------
  Δput_oi − Δcall_oi > 0 on up day  → put writers adding positions
                                       = smart money expects floor to hold → BULLISH
  Δput_oi − Δcall_oi < 0 on down day → call writers adding positions
                                       = smart money expects ceiling to hold → BEARISH

Decision Rules
--------------
  1. Nifty must have ≥ nifty_min_aligned (default 3) of its 4 checks agree.
  2. Banknifty must have ≥ bn_min_aligned (default 3) checks agree in the SAME direction.
  3. Total aligned checks (out of 8) determine conviction:
       HIGH   : ≥ 7 checks aligned
       MEDIUM : ≥ 5 checks aligned
       DIVERGE: < 5 checks → no trade

Reversal Warning
----------------
  Flagged when Spot-L30 direction opposes the Spot-day direction (profit booking
  into close). The signal remains valid but conviction is not upgraded to HIGH
  when a reversal warning is active; it caps at MEDIUM.

Usage — Live (3:20 PM)
-----------------------
  loader_n = RollingOptionsDataLoader("data/backtest/cache/rolling_options")
  loader_n.build_samples("NIFTY", today, today)
  loader_b = RollingOptionsDataLoader("data/backtest/cache/rolling_options")
  loader_b.build_samples("BANKNIFTY", today, today)

  engine = OvernightSignalEngine()
  signal = engine.generate(loader_n, loader_b, today)
  # → {"direction": "BULLISH", "conviction": "HIGH", "aligned_of_8": 8, ...}

Usage — Backtest
-----------------
  df = run_backtest(cache_root, "2024-01-01", "2025-03-31")
  print(summarize(df))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .data_loader import RollingOptionsDataLoader
from .greek_calculator import calculate_greek_changes


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OvernightConfig:
    """All tunable thresholds in one place — change here, affects everything."""

    # Magnitude filters: moves smaller than these are treated as "flat" (direction = 0)
    # Prevents taking a signal on a 10-point drift day
    nifty_min_day_pts: float = 30.0       # Nifty full-day move minimum (pts)
    nifty_min_l30_pts: float = 15.0       # Nifty last-30-min move minimum (pts)
    bn_min_day_pts: float = 100.0         # Banknifty full-day move minimum (pts)
    bn_min_l30_pts: float = 50.0          # Banknifty last-30-min move minimum (pts)
    min_oi_skew_chg: float = 5_000.0      # Minimum OI skew change to treat as directional

    # Per-index alignment gate (out of 4 checks each)
    nifty_min_aligned: int = 3            # Nifty needs ≥ 3/4 to establish direction
    bn_min_aligned: int = 3              # Banknifty needs ≥ 3/4 to confirm

    # Final conviction thresholds (out of 8 total checks)
    high_conviction_min: int = 7          # 7 or 8 → HIGH
    medium_conviction_min: int = 5        # 5 or 6 → MEDIUM  (< 5 → DIVERGE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dir(value: float, threshold: float = 0.0) -> int:
    """Sign of value relative to threshold. Returns +1, -1, or 0."""
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def _majority_dir(checks: List[int]) -> Tuple[int, int]:
    """
    Returns (direction, aligned_count).
    direction: +1, -1, or 0 (split / all flat).
    aligned_count: how many checks agree with that direction.
    """
    pos = sum(1 for c in checks if c == 1)
    neg = sum(1 for c in checks if c == -1)
    if pos > neg:
        return 1, pos
    if neg > pos:
        return -1, neg
    return 0, 0


def _next_day_map(days: Sequence[str]) -> Dict[str, str]:
    return {days[i]: days[i + 1] for i in range(len(days) - 1)}


def _signed_points(direction: str, actual_gap: float, threshold: float = 1.0) -> Tuple[str, float]:
    if direction == "BULLISH":
        return ("WIN", actual_gap) if actual_gap >= threshold else ("LOSS", -abs(actual_gap))
    if direction == "BEARISH":
        return ("WIN", abs(actual_gap)) if actual_gap <= -threshold else ("LOSS", -abs(actual_gap))
    return ("SKIP", 0.0)


# ---------------------------------------------------------------------------
# Per-index feature extraction
# ---------------------------------------------------------------------------

def _index_features(
    loader: RollingOptionsDataLoader,
    day: str,
    index: str,
    min_day_pts: float,
    min_l30_pts: float,
    min_oi_skew: float,
) -> Optional[Dict]:
    """
    Extract 4 directional checks for one index on one trading day.
    Returns None if chain data is missing.
    """
    exp = loader.choose_expiry(day, phase="close_1515")
    if not exp:
        return None
    flag, code = exp

    open_chain = loader.build_chain(day, index, flag, code, "open_0915")
    pre_chain  = loader.build_chain(day, index, flag, code, "preclose_1445")
    close_chain = loader.build_chain(day, index, flag, code, "close_1515")
    if not open_chain or not pre_chain or not close_chain:
        return None

    # ── Spot movements ────────────────────────────────────────────────────
    spot_open  = float(open_chain["spot"])
    spot_pre   = float(pre_chain["spot"])
    spot_close = float(close_chain["spot"])

    day_move = spot_close - spot_open    # 9:15 → 15:15 full session
    l30_move = spot_close - spot_pre     # 14:45 → 15:15 closing window

    spot_day_dir = _dir(day_move, min_day_pts)
    spot_l30_dir = _dir(l30_move, min_l30_pts)

    # ── OI skew changes ────────────────────────────────────────────────────
    # Filters to near-ATM strikes (delta 0.05–0.50) to exclude far-OTM noise
    day_ch = calculate_greek_changes(open_chain, close_chain)
    l30_ch = calculate_greek_changes(pre_chain, close_chain)

    # Positive = more put writing relative to calls = bullish bias
    day_oi_skew = float(day_ch["put_oi"] - day_ch["call_oi"])
    l30_oi_skew = float(l30_ch["put_oi"] - l30_ch["call_oi"])

    oi_day_dir = _dir(day_oi_skew, min_oi_skew)
    oi_l30_dir = _dir(l30_oi_skew, min_oi_skew)

    # ── 4-check alignment ─────────────────────────────────────────────────
    checks = [spot_day_dir, spot_l30_dir, oi_day_dir, oi_l30_dir]
    primary_dir, aligned = _majority_dir(checks)

    # Reversal warning: last-30-min spot move opposes the full-day trend
    # Often signals profit booking into close — weakens overnight conviction
    reversal_warning = (
        spot_day_dir != 0
        and spot_l30_dir != 0
        and spot_day_dir != spot_l30_dir
    )

    return {
        "flag": flag,
        "code": code,
        "spot_open":    round(spot_open, 2),
        "spot_preclose": round(spot_pre, 2),
        "spot_close":   round(spot_close, 2),
        "day_move":     round(day_move, 2),
        "l30_move":     round(l30_move, 2),
        "day_oi_skew":  round(day_oi_skew, 0),
        "l30_oi_skew":  round(l30_oi_skew, 0),
        "spot_day_dir": spot_day_dir,
        "spot_l30_dir": spot_l30_dir,
        "oi_day_dir":   oi_day_dir,
        "oi_l30_dir":   oi_l30_dir,
        "checks":       checks,
        "primary_dir":  primary_dir,
        "aligned":      aligned,
        "reversal_warning": reversal_warning,
    }


# ---------------------------------------------------------------------------
# Signal engine
# ---------------------------------------------------------------------------

class OvernightSignalEngine:
    """
    Core overnight gap signal generator.

    Two loaders are required (one per index, each pre-built via build_samples)
    so the engine can be used both in live mode and inside backtests without
    re-loading data on every call.
    """

    def __init__(self, cfg: OvernightConfig = OvernightConfig()):
        self.cfg = cfg

    def generate(
        self,
        loader_n: RollingOptionsDataLoader,
        loader_b: RollingOptionsDataLoader,
        day: str,
        primary_index: str = "NIFTY",
        confirm_index: str = "BANKNIFTY",
    ) -> Dict:
        """
        Generate overnight signal for `day`.
        Call at 3:20 PM after preclose_1445 and close_1515 data are available.

        Returns a dict with keys:
          direction    : "BULLISH" | "BEARISH" | "DIVERGE" | "NO_DATA"
          conviction   : "HIGH" | "MEDIUM" | "NONE"
          aligned_of_8 : int (0–8)
          reason       : human-readable explanation
          + all raw index features for transparency
        """
        cfg = self.cfg

        nf = _index_features(
            loader_n, day, primary_index,
            cfg.nifty_min_day_pts, cfg.nifty_min_l30_pts, cfg.min_oi_skew_chg,
        )
        bf = _index_features(
            loader_b, day, confirm_index,
            cfg.bn_min_day_pts, cfg.bn_min_l30_pts, cfg.min_oi_skew_chg,
        )

        base = self._base_row(nf, bf, day)

        if not nf or not bf:
            return {**base, "direction": "NO_DATA", "conviction": "NONE",
                    "aligned_of_8": 0, "reason": "missing_chain_data"}

        n_dir = nf["primary_dir"]
        b_dir = bf["primary_dir"]

        # ── Gate 1: Nifty must establish a clear direction ────────────────
        if n_dir == 0 or nf["aligned"] < cfg.nifty_min_aligned:
            return {**base, "direction": "DIVERGE", "conviction": "NONE",
                    "aligned_of_8": nf["aligned"],
                    "reason": f"nifty_no_consensus ({nf['aligned']}/4 aligned)"}

        # ── Gate 2: Banknifty must confirm the same direction ─────────────
        if b_dir != n_dir or bf["aligned"] < cfg.bn_min_aligned:
            return {**base, "direction": "DIVERGE", "conviction": "NONE",
                    "aligned_of_8": nf["aligned"] + (bf["aligned"] if b_dir == n_dir else 0),
                    "reason": (
                        f"bn_diverge (nifty={n_dir} bn={b_dir} "
                        f"nifty_aligned={nf['aligned']}/4 bn_aligned={bf['aligned']}/4)"
                    )}

        # ── Conviction: count all 8 checks aligned with final direction ───
        total_aligned = sum(1 for c in (nf["checks"] + bf["checks"]) if c == n_dir)

        final_dir = "BULLISH" if n_dir > 0 else "BEARISH"
        reversal = nf["reversal_warning"] or bf["reversal_warning"]

        if total_aligned >= cfg.high_conviction_min and not reversal:
            conviction = "HIGH"
        elif total_aligned >= cfg.medium_conviction_min:
            conviction = "MEDIUM"
        else:
            # Both indices agreed directionally but not enough checks aligned
            conviction = "NONE"
            final_dir = "DIVERGE"

        reason_parts = ["nifty_bn_confirmed"]
        if reversal:
            reason_parts.append("reversal_warning_active")
        if conviction == "NONE":
            reason_parts.append(f"low_alignment ({total_aligned}/8)")

        return {
            **base,
            "direction":    final_dir,
            "conviction":   conviction,
            "aligned_of_8": total_aligned,
            "reason":       " | ".join(reason_parts),
        }

    @staticmethod
    def _base_row(nf: Optional[Dict], bf: Optional[Dict], day: str) -> Dict:
        """Flat row of all raw features for full auditability."""
        def _extract(f: Optional[Dict], prefix: str) -> Dict:
            if not f:
                return {f"{prefix}_{k}": None for k in [
                    "day_move", "l30_move", "day_oi_skew", "l30_oi_skew",
                    "spot_day_dir", "spot_l30_dir", "oi_day_dir", "oi_l30_dir",
                    "aligned", "reversal_warning",
                ]}
            return {
                f"{prefix}_day_move":        f["day_move"],
                f"{prefix}_l30_move":        f["l30_move"],
                f"{prefix}_day_oi_skew":     f["day_oi_skew"],
                f"{prefix}_l30_oi_skew":     f["l30_oi_skew"],
                f"{prefix}_spot_day_dir":    f["spot_day_dir"],
                f"{prefix}_spot_l30_dir":    f["spot_l30_dir"],
                f"{prefix}_oi_day_dir":      f["oi_day_dir"],
                f"{prefix}_oi_l30_dir":      f["oi_l30_dir"],
                f"{prefix}_aligned":         f["aligned"],
                f"{prefix}_reversal_warning": f["reversal_warning"],
            }
        return {
            "signal_date": day,
            "cmp": nf["spot_close"] if nf else None,
            **_extract(nf, "n"),
            **_extract(bf, "b"),
        }


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    cache_root: Union[str, Path],
    start_date: str,
    end_date: str,
    primary_index: str = "NIFTY",
    confirm_index: str = "BANKNIFTY",
    cfg: OvernightConfig = OvernightConfig(),
    gap_win_threshold_pts: float = 1.0,
) -> pd.DataFrame:
    """
    Backtest the overnight signal over a date range.

    Parameters
    ----------
    cache_root          : path to rolling_options cache directory
    start_date, end_date: "YYYY-MM-DD" inclusive range
    gap_win_threshold_pts: minimum gap (pts) to count as a WIN

    Returns
    -------
    DataFrame with one row per trading day, columns:
      signal_date, direction, conviction, aligned_of_8, reason,
      cmp, next_open, actual_gap, result, points, trade,
      cumulative_points, + all raw index feature columns
    """
    engine = OvernightSignalEngine(cfg)
    root = Path(cache_root)

    loader_n = RollingOptionsDataLoader(cache_root=root)
    loader_n.build_samples(index=primary_index, start_date=start_date, end_date=end_date)

    loader_b = RollingOptionsDataLoader(cache_root=root)
    loader_b.build_samples(index=confirm_index, start_date=start_date, end_date=end_date)

    days = loader_n.trading_days()
    if not days:
        return pd.DataFrame()

    nxt = _next_day_map(days)
    rows: List[Dict] = []

    for d in days:
        nd = nxt.get(d)
        if not nd:
            continue

        sig = engine.generate(loader_n, loader_b, d, primary_index, confirm_index)

        # Resolve next-day open for actual gap measurement
        exp = loader_n.choose_expiry(d, phase="close_1515")
        if not exp:
            continue
        flag, code = exp
        next_open = loader_n.spot_at(nd, primary_index, "next_open_0925", flag=flag, code=code)
        if next_open is None:
            continue

        cmp = sig.get("cmp") or 0.0
        actual_gap = float(next_open) - float(cmp)
        result, points = _signed_points(sig["direction"], actual_gap, gap_win_threshold_pts)

        rows.append({
            **sig,
            "next_day":   nd,
            "next_open":  round(float(next_open), 2),
            "actual_gap": round(actual_gap, 2),
            "result":     result,
            "points":     round(points, 2),
            "trade":      sig["direction"] in {"BULLISH", "BEARISH"},
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)
    out["cumulative_points"] = out["points"].cumsum()
    return out


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> Dict:
    """
    Accuracy metrics for the backtest output, broken down by conviction level.

    Returns a dict:
      {
        "overall":           {trades, wins, losses, win_rate, expectancy, avg_win, avg_loss},
        "conviction_HIGH":   {...},
        "conviction_MEDIUM": {...},
        "diverge_rate":      float   # fraction of days with no trade signal
      }
    """
    if df.empty:
        return {}

    def _metrics(sub: pd.DataFrame) -> Dict:
        traded = sub[sub["trade"] == True]
        if traded.empty:
            return {"trades": 0, "wins": 0, "losses": 0,
                    "win_rate": 0.0, "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
        pts = traded["points"].to_numpy()
        wins = pts[pts > 0]
        losses = pts[pts <= 0]
        return {
            "trades":      int(len(traded)),
            "wins":        int((pts > 0).sum()),
            "losses":      int((pts <= 0).sum()),
            "win_rate":    round(float((pts > 0).mean()), 3),
            "expectancy":  round(float(pts.mean()), 2),
            "avg_win":     round(float(wins.mean()), 2) if len(wins) else 0.0,
            "avg_loss":    round(float(losses.mean()), 2) if len(losses) else 0.0,
        }

    total_days = len(df)
    traded_days = int(df["trade"].sum())
    diverge_rate = round(1.0 - traded_days / total_days, 3) if total_days else 0.0

    out: Dict = {
        "overall":           _metrics(df),
        "conviction_HIGH":   _metrics(df[df["conviction"] == "HIGH"]),
        "conviction_MEDIUM": _metrics(df[df["conviction"] == "MEDIUM"]),
        "diverge_rate":      diverge_rate,
        "total_days":        total_days,
        "traded_days":       traded_days,
    }
    return out
