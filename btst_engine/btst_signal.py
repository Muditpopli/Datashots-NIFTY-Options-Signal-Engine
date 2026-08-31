"""
btst_signal.py -- BTST Final Decision Layer

Consumes two SignalSnapshot objects (NIFTY + BANKNIFTY) and an ExpiryContext,
applies a gated decision sequence, and emits a single BTSTSignal.

Decision order:
  1. Data quality gate  -- W2 MISSING on either index -> NO_TRADE
  2. VIX gate           -- India VIX > 18 -> NO_TRADE
  3. Per-index direction -- derived from W2 (primary) + W1 (confirmation)
  4. Cross-index check  -- both must agree; both NEUTRAL is also a block
  5. Rollover modifier  -- adjusts confidence; never hard-blocks the signal
  6. Confidence scoring -- composite 0-95 % score
  7. Strike & exit rule -- ATM +/-1 strike, standard BTST exit logic

Public API:
  generate_signal(nifty_snap, bnf_snap, ctx, vix) -> BTSTSignal
  run_btst_signal(signal_date, vix, ctx)           -> BTSTSignal (full pipeline)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import config
from btst_engine.expiry_manager import ExpiryContext, resolve_expiry_context
from btst_engine.signal_builder import SignalSnapshot, WindowData, build_snapshots

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_PCR_BULLISH  = 1.15   # PCR above this => put-writer regime => bullish
_PCR_BEARISH  = 0.85   # PCR below this => call-writer regime => bearish
_VIX_GATE     = 18.0   # India VIX hard block above this level

# Distance from PCR 1.0 at which maximum PCR confidence bonus is awarded.
# A distance of 0.5 (e.g., PCR = 1.5 or 0.5) earns the full 15 %.
_PCR_DIST_MAX  = 0.50
_PCR_BONUS_MAX = 0.15   # 15 %

# Exit rule thresholds (relative to prior close spot)
_GAP_STOP_PCT   = 0.008   # 0.8 % adverse gap -> exit at open
_GAP_TARGET_PCT = 0.015   # 1.5 % favourable gap -> exit at open (bank profit)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class BTSTSignal:
    """
    Final BTST trade decision for one calendar date.

    Attributes
    ----------
    signal      : "BULLISH" | "BEARISH" | "NO_TRADE"
    reason      : Human-readable explanation for the signal or block.
    confidence  : Float in [0, 0.95].  Meaningful only when signal != NO_TRADE.
    direction   : Agreed direction ("BULLISH"/"BEARISH") or "NEUTRAL"/"NONE"
                  when the signal is blocked.
    strike_ce_pe: Recommended entry string, e.g. "NIFTY 22500 CE".
                  Empty string when signal is NO_TRADE.
    exit_rule   : Dict describing primary, stop and target rules.
                  Empty dict when signal is NO_TRADE.
    rollover_used: True when rollover data influenced the confidence score.
    """

    date:               str
    signal:             str
    reason:             str
    confidence:         float
    direction:          str
    trade_expiry:       str
    strike_ce_pe:       str
    exit_rule:          dict
    nifty_snapshot:     dict
    banknifty_snapshot: dict
    rollover_used:      bool
    generated_at:       str   # "YYYY-MM-DD HH:MM:SS IST"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Direction logic
# ---------------------------------------------------------------------------

def _oi_change_bullish(w: WindowData) -> bool:
    """
    True when OI flow favours bulls in a high-PCR (put-writer) regime.

    "Bullish OI change" = put OI is building faster than call OI, so the
    PCR is trending higher -- consistent with continued put-writing activity.
    """
    return w.oi_change_put >= w.oi_change_call


def _oi_change_bearish(w: WindowData) -> bool:
    """
    True when OI flow favours bears in a low-PCR (call-writer) regime.

    "Bearish OI change" = call OI is building faster than put OI, so the
    PCR is trending lower -- consistent with continued call-writing activity.
    """
    return w.oi_change_call >= w.oi_change_put


def _window_direction(w: WindowData) -> str:
    """
    Derive directional bias from one window's aggregated metrics.

    Returns "BULLISH", "BEARISH", or "NEUTRAL".

    BULLISH requires ALL three conditions (conjunction, not disjunction):
      - PCR > 1.15  (put-writer dominance)
      - Put OI building faster than call OI (PCR trending higher)
      - Net OI-weighted delta < 0 (put OI dominates ATM±3 delta exposure,
        consistent with high-PCR / put-writer regime)

    BEARISH is the symmetric mirror (net_delta > 0 for call-writer regime).
    If neither or both fire, NEUTRAL.
    """
    if w.data_quality == "MISSING":
        return "NEUTRAL"

    is_bullish = (
        w.pcr > _PCR_BULLISH
        and _oi_change_bullish(w)
        and w.net_delta < 0    # put OI dominates ATM±3 → negative net delta = correct for high-PCR regime
    )
    is_bearish = (
        w.pcr < _PCR_BEARISH
        and _oi_change_bearish(w)
        and w.net_delta > 0    # call OI dominates ATM±3 → positive net delta = correct for low-PCR regime
    )

    if is_bullish and not is_bearish:
        return "BULLISH"
    if is_bearish and not is_bullish:
        return "BEARISH"
    return "NEUTRAL"


def _index_direction(snap: SignalSnapshot) -> Tuple[str, str]:
    """
    Derive the final directional call for one index.

    W2 (last 30 min) is the primary signal.
    W1 (full session) is the confirmation: if it fires in the OPPOSITE
    direction to W2 (not neutral), W2 is downgraded to NEUTRAL.

    Returns (direction_str, diagnostic_note).
    """
    w2_dir = _window_direction(snap.w2)
    w1_dir = _window_direction(snap.w1)

    if w2_dir == "NEUTRAL":
        return "NEUTRAL", f"W2=NEUTRAL (W1={w1_dir})"

    # W1 must not actively contradict W2.
    if w1_dir not in ("NEUTRAL", w2_dir):
        return "NEUTRAL", f"W1({w1_dir}) contradicts W2({w2_dir}) -- downgraded"

    confirmation = "confirmed" if w1_dir == w2_dir else "no contradiction"
    return w2_dir, f"W2={w2_dir} W1={w1_dir} ({confirmation})"


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _pcr_strength_bonus(pcr: float) -> float:
    """
    Scale the PCR's distance from 1.0 into a 0-15 % confidence bonus.

    Distance |PCR - 1.0| = 0.50 earns the maximum 15 %.
    """
    distance = abs(pcr - 1.0)
    return min(_PCR_BONUS_MAX, distance / _PCR_DIST_MAX * _PCR_BONUS_MAX)


def _compute_confidence(
    direction:         str,
    nifty_snap:        SignalSnapshot,
    bnf_snap:          SignalSnapshot,
    rollover_direction: Optional[str],
    rollover_used:     bool,
) -> float:
    """
    Compute a confidence score in [0.0, 0.95].

    Component table (all deltas are percentage points):

    Component                      | If true | If false
    -------------------------------|---------|--------
    Base                           | +50     | n/a
    W2 PCR strength (NIFTY)        | 0-15    | 0
    W1/W2 alignment (NIFTY)        | +10     | 0
    Both indices agree             | +10     | n/a (gate already passed)
    IV skew aligns with direction  | +10     | 0
    Rollover agrees                | +15     | 0 (if NEUTRAL)
    Rollover disagrees             |  0      | -20
    -------------------------------|---------|--------
    Cap                            |  95     | floor at 0

    NIFTY is used as the primary index for component measurement because it
    is the traded instrument.  BANKNIFTY agreement is already enforced by the
    cross-index gate.
    """
    conf = 0.50

    # W2 PCR distance bonus (NIFTY)
    conf += _pcr_strength_bonus(nifty_snap.w2.pcr)

    # W1/W2 alignment bonus (NIFTY)
    if (
        _window_direction(nifty_snap.w1) == direction
        and _window_direction(nifty_snap.w2) == direction
    ):
        conf += 0.10

    # Cross-index agreement -- gate was passed so this bonus always fires
    conf += 0.10

    # IV skew alignment (NIFTY W2)
    # iv_skew = put_iv - call_iv
    #   < 0 => call IV higher => call demand => aligns with BULLISH
    #   > 0 => put IV higher  => put demand  => aligns with BEARISH
    iv_skew = nifty_snap.w2.iv_skew
    if direction == "BULLISH" and iv_skew < 0:
        conf += 0.10
    elif direction == "BEARISH" and iv_skew > 0:
        conf += 0.10

    # Rollover modifier
    if rollover_used and rollover_direction:
        if rollover_direction == direction:
            conf += 0.15
        elif rollover_direction not in ("NEUTRAL", ""):
            conf -= 0.20

    return max(0.0, min(0.95, conf))


# ---------------------------------------------------------------------------
# Strike recommendation
# ---------------------------------------------------------------------------

def _recommend_strike(direction: str, nifty_snap: SignalSnapshot) -> str:
    """
    Suggest an ATM+1 CE (BULLISH) or ATM-1 PE (BEARISH) entry.

    Uses the NIFTY spot close from W2 (or W1 if W2 is missing) and the
    strike gap from config.  The slightly-OTM choice gives better risk/reward
    for an overnight directional hold than the ATM itself.
    """
    spot = nifty_snap.w2.spot_close or nifty_snap.w1.spot_close
    gap  = config.STRIKE_GAPS.get("NIFTY", 50)

    if spot <= 0:
        return f"NIFTY ATM {'CE' if direction == 'BULLISH' else 'PE'}"

    atm = round(spot / gap) * gap

    if direction == "BULLISH":
        strike = int(atm + gap)   # ATM+1 = slightly OTM CE
        return f"NIFTY {strike} CE"
    else:
        strike = int(atm - gap)   # ATM-1 = slightly OTM PE
        return f"NIFTY {strike} PE"


# ---------------------------------------------------------------------------
# Exit rule
# ---------------------------------------------------------------------------

def _build_exit_rule(direction: str, basis_spot: float) -> dict:
    """
    Construct the standard BTST exit rule dict.

    Primary  : Exit at market open next trading day (no exceptions unless gate fires).
    Stop     : If spot opens > 0.8 % against position, exit immediately at open.
    Target   : If overnight gap runs > 1.5 % in your favour, exit at open and do
               NOT hold -- bank the move.
    """
    if basis_spot <= 0:
        return {
            "primary":        "Exit at market open next trading day",
            "stop_trigger":   f"Spot opens >{_GAP_STOP_PCT*100:.1f}% against position -> exit at open",
            "stop_level":     None,
            "target_trigger": f"Overnight gap >{_GAP_TARGET_PCT*100:.1f}% in your favour -> exit at open",
            "target_level":   None,
            "basis_spot":     basis_spot,
        }

    if direction == "BULLISH":
        stop_level   = round(basis_spot * (1 - _GAP_STOP_PCT),   2)
        target_level = round(basis_spot * (1 + _GAP_TARGET_PCT), 2)
        stop_note    = f"Spot opens below {stop_level:.2f} -> exit immediately at open"
        target_note  = f"Spot opens above {target_level:.2f} -> exit at open, do not hold"
    else:  # BEARISH
        stop_level   = round(basis_spot * (1 + _GAP_STOP_PCT),   2)
        target_level = round(basis_spot * (1 - _GAP_TARGET_PCT), 2)
        stop_note    = f"Spot opens above {stop_level:.2f} -> exit immediately at open"
        target_note  = f"Spot opens below {target_level:.2f} -> exit at open, do not hold"

    return {
        "primary":        "Exit at market open next trading day",
        "stop_trigger":   stop_note,
        "stop_level":     stop_level,
        "target_trigger": target_note,
        "target_level":   target_level,
        "basis_spot":     basis_spot,
    }


# ---------------------------------------------------------------------------
# Main decision function
# ---------------------------------------------------------------------------

def generate_signal(
    nifty_snap: SignalSnapshot,
    bnf_snap:   SignalSnapshot,
    ctx:        ExpiryContext,
    vix:        Optional[float] = None,
) -> BTSTSignal:
    """
    Run the full BTST decision sequence and return a BTSTSignal.

    Parameters
    ----------
    nifty_snap  : SignalSnapshot for NIFTY on the signal date.
    bnf_snap    : SignalSnapshot for BANKNIFTY on the signal date.
    ctx         : ExpiryContext for the signal date.
    vix         : India VIX (optional).  Triggers NO_TRADE if > 18.
    """
    now_ist  = datetime.now(tz=_IST).strftime("%Y-%m-%d %H:%M:%S IST")
    date_str = nifty_snap.date

    # Convenience: pre-populate a NO_TRADE shell for early returns.
    def _no_trade(reason: str, direction: str = "NONE") -> BTSTSignal:
        return BTSTSignal(
            date=date_str,
            signal="NO_TRADE",
            reason=reason,
            confidence=0.0,
            direction=direction,
            trade_expiry=ctx.trade_expiry,
            strike_ce_pe="",
            exit_rule={},
            nifty_snapshot=nifty_snap.to_dict(),
            banknifty_snapshot=bnf_snap.to_dict(),
            rollover_used=ctx.rollover_mode and (nifty_snap.rollover is not None),
            generated_at=now_ist,
        )

    # ── Gate 1: Data quality ──────────────────────────────────────────────────
    if nifty_snap.w2.data_quality == "MISSING":
        return _no_trade("INSUFFICIENT_DATA: NIFTY W2 is missing")
    if bnf_snap.w2.data_quality == "MISSING":
        return _no_trade("INSUFFICIENT_DATA: BANKNIFTY W2 is missing")

    # ── Gate 2: VIX ───────────────────────────────────────────────────────────
    if vix is not None and vix > _VIX_GATE:
        return _no_trade(f"HIGH_VOLATILITY: VIX {vix:.1f} > {_VIX_GATE}")

    # ── Step 3: Per-index direction ───────────────────────────────────────────
    nifty_dir, nifty_note = _index_direction(nifty_snap)
    bnf_dir,   bnf_note   = _index_direction(bnf_snap)

    logger.info("[SIGNAL] NIFTY direction : %s  (%s)", nifty_dir, nifty_note)
    logger.info("[SIGNAL] BNF   direction : %s  (%s)", bnf_dir,   bnf_note)

    # ── Gate 4: Cross-index confirmation ──────────────────────────────────────
    if nifty_dir == "NEUTRAL" and bnf_dir == "NEUTRAL":
        return _no_trade("NO_CLEAR_DIRECTION: both indices are neutral", direction="NEUTRAL")

    if nifty_dir != bnf_dir:
        return _no_trade(
            f"INDEX_DIVERGENCE: NIFTY={nifty_dir} BANKNIFTY={bnf_dir}",
            direction=nifty_dir,
        )

    final_direction = nifty_dir   # both agree at this point

    # ── Step 5: Rollover confidence modifier ──────────────────────────────────
    rollover_used = ctx.rollover_mode and (nifty_snap.rollover is not None)
    rollover_dir: Optional[str] = None

    if rollover_used and nifty_snap.rollover:
        rollover_dir = nifty_snap.rollover.rollover_direction
        logger.info(
            "[SIGNAL] Rollover %s vs signal %s",
            rollover_dir, final_direction,
        )

    # ── Step 6: Confidence score ──────────────────────────────────────────────
    confidence = _compute_confidence(
        direction=final_direction,
        nifty_snap=nifty_snap,
        bnf_snap=bnf_snap,
        rollover_direction=rollover_dir,
        rollover_used=rollover_used,
    )

    logger.info("[SIGNAL] Confidence: %.0f%%", confidence * 100)

    # ── Step 7: Strike and exit ───────────────────────────────────────────────
    strike_str = _recommend_strike(final_direction, nifty_snap)

    basis_spot = nifty_snap.w2.spot_close or nifty_snap.w1.spot_close
    exit_dict  = _build_exit_rule(final_direction, basis_spot)

    # Build reason string with key diagnostics
    reason_parts = [
        f"direction={final_direction}",
        f"NIFTY:{nifty_note.split(' ')[0]}",
        f"BNF:{bnf_note.split(' ')[0]}",
    ]
    if rollover_dir:
        reason_parts.append(f"rollover={rollover_dir}")
    reason_parts.append(f"confidence={confidence:.0%}")

    return BTSTSignal(
        date=date_str,
        signal=final_direction,
        reason=" | ".join(reason_parts),
        confidence=round(confidence, 4),
        direction=final_direction,
        trade_expiry=ctx.trade_expiry,
        strike_ce_pe=strike_str,
        exit_rule=exit_dict,
        nifty_snapshot=nifty_snap.to_dict(),
        banknifty_snapshot=bnf_snap.to_dict(),
        rollover_used=rollover_used,
        generated_at=now_ist,
    )


# ---------------------------------------------------------------------------
# Full-pipeline convenience wrapper
# ---------------------------------------------------------------------------

def run_btst_signal(
    signal_date,
    vix:  Optional[float] = None,
    ctx:  Optional[ExpiryContext] = None,
) -> BTSTSignal:
    """
    End-to-end pipeline: resolve context -> load cache -> generate signal.

    Parameters
    ----------
    signal_date : date object or "YYYY-MM-DD" string.
    vix         : India VIX value (optional).
    ctx         : Pre-resolved ExpiryContext (resolved automatically if None).
    """
    if isinstance(signal_date, str):
        signal_date = _date.fromisoformat(signal_date)

    if ctx is None:
        ctx = resolve_expiry_context(signal_date)

    snapshots = build_snapshots(signal_date, ctx)
    return generate_signal(
        nifty_snap=snapshots["NIFTY"],
        bnf_snap=snapshots["BANKNIFTY"],
        ctx=ctx,
        vix=vix,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt_signal(sig: BTSTSignal) -> None:
    """Print a BTSTSignal to stdout in a human-readable format."""
    sep   = "=" * 64
    hsep  = "-" * 64

    signal_badge = {
        "BULLISH":  "[BULLISH]",
        "BEARISH":  "[BEARISH]",
        "NO_TRADE": "[NO_TRADE]",
    }.get(sig.signal, f"[{sig.signal}]")

    print(f"\n{sep}")
    print(f"  BTST Signal  --  {sig.date}")
    print(f"{sep}")

    print(f"\n  Signal      : {signal_badge}")
    print(f"  Direction   : {sig.direction}")
    print(f"  Confidence  : {sig.confidence:.0%}")
    print(f"  Reason      : {sig.reason}")

    if sig.signal != "NO_TRADE":
        print(f"\n  Trade expiry: {sig.trade_expiry}")
        print(f"  Strike      : {sig.strike_ce_pe}")
        print(f"\n  Exit rule:")
        for k, v in sig.exit_rule.items():
            if v is not None:
                print(f"    {k:<18} {v}")

    print(f"\n  Rollover used : {sig.rollover_used}")

    # Per-index snapshot summary
    for label, snap_dict in [
        ("NIFTY",     sig.nifty_snapshot),
        ("BANKNIFTY", sig.banknifty_snapshot),
    ]:
        w2  = snap_dict.get("w2", {})
        w1  = snap_dict.get("w1", {})
        rol = snap_dict.get("rollover") or {}

        print(f"\n  {hsep}")
        print(f"  {label}")
        print(f"    W1 [{w1.get('data_quality','?')} {w1.get('n_bars',0)} bars]"
              f"  PCR={w1.get('pcr',0):.3f}"
              f"  delta={w1.get('net_delta',0):+.0f}"
              f"  skew={w1.get('iv_skew',0):+.2f}%")
        print(f"    W2 [{w2.get('data_quality','?')} {w2.get('n_bars',0)} bars]"
              f"  PCR={w2.get('pcr',0):.3f}"
              f"  delta={w2.get('net_delta',0):+.0f}"
              f"  skew={w2.get('iv_skew',0):+.2f}%"
              f"  spot_close={w2.get('spot_close',0):.2f}")
        print(f"    OI change  W2: call={w2.get('oi_change_call',0)/1e6:+.3f}M"
              f"  put={w2.get('oi_change_put',0)/1e6:+.3f}M")
        if rol:
            print(f"    Rollover : [{rol.get('rollover_direction','?')}]"
                  f"  E1 PCR={rol.get('e1_pcr',0):.3f}"
                  f"  E2 PCR={rol.get('e2_pcr',0):.3f}"
                  f"  ratio call={rol.get('rollover_ratio_call',0):.3f}"
                  f" put={rol.get('rollover_ratio_put',0):.3f}")

    print(f"\n  Generated at : {sig.generated_at}")
    print(f"{sep}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2:
        print(
            "Usage: python -m btst_engine.btst_signal YYYY-MM-DD [--vix VALUE]",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        signal_date = _date.fromisoformat(sys.argv[1])
    except ValueError:
        print(f"[ERR] Bad date '{sys.argv[1]}' -- expected YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)

    vix_val: Optional[float] = None
    if "--vix" in sys.argv:
        idx = sys.argv.index("--vix")
        if idx + 1 < len(sys.argv):
            try:
                vix_val = float(sys.argv[idx + 1])
            except ValueError:
                print(f"[ERR] Bad VIX value '{sys.argv[idx + 1]}'", file=sys.stderr)
                sys.exit(1)

    sig = run_btst_signal(signal_date, vix=vix_val)
    _fmt_signal(sig)
