"""
DNS Options Scanner - Configuration
Clean, professional options trading system
"""

import json
import os
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# API CREDENTIALS
# ═══════════════════════════════════════════════════════════════

DHAN_ACCESS_TOKEN = os.getenv('DHAN_ACCESS_TOKEN', '')
DHAN_CLIENT_ID = os.getenv('DHAN_CLIENT_ID', '')

# Optional: custom underlyings for rolling option fetch.
# Env format (JSON string):
# {
#   "RELIANCE": {"security_id": 2885, "instrument": "OPTSTK"},
#   "SBIN": {"security_id": 3045, "instrument": "OPTSTK"}
# }
_UNDERLYING_MAP_RAW = os.getenv('DHAN_UNDERLYING_MAP_JSON', '').strip()
try:
    _parsed = json.loads(_UNDERLYING_MAP_RAW) if _UNDERLYING_MAP_RAW else {}
except Exception:
    _parsed = {}

DHAN_UNDERLYING_MAP = {}
if isinstance(_parsed, dict):
    for k, v in _parsed.items():
        if not isinstance(v, dict):
            continue
        sid = v.get("security_id")
        instr = str(v.get("instrument", "OPTSTK")).upper()
        try:
            sid_i = int(sid)
        except Exception:
            continue
        DHAN_UNDERLYING_MAP[str(k).upper()] = {"security_id": sid_i, "instrument": instr}

# Optional market holiday list for trading-day aware routines.
# Env format:
# MARKET_HOLIDAYS_JSON=["2026-03-30","2026-08-15"]
_MARKET_HOLIDAYS_RAW = os.getenv('MARKET_HOLIDAYS_JSON', '').strip()
try:
    _parsed_holidays = json.loads(_MARKET_HOLIDAYS_RAW) if _MARKET_HOLIDAYS_RAW else []
except Exception:
    _parsed_holidays = []

MARKET_HOLIDAYS = set()
if isinstance(_parsed_holidays, list):
    for v in _parsed_holidays:
        try:
            MARKET_HOLIDAYS.add(str(v))
        except Exception:
            continue

# ═══════════════════════════════════════════════════════════════
# MARKET SETTINGS
# ═══════════════════════════════════════════════════════════════

# Run system only for NIFTY.
INDICES = ['NIFTY']

# Strike configuration
STRIKE_GAPS = {
    'NIFTY': 50,
    'BANKNIFTY': 100
}

LOT_SIZES = {
    'NIFTY': 50,
    'BANKNIFTY': 15
}

# How many OTM strikes to scan on each side
OTM_STRIKES_TO_SCAN = 10

# ═══════════════════════════════════════════════════════════════
# TIMING CONFIGURATION
# ═══════════════════════════════════════════════════════════════

BASELINE_TIME_START = "09:15"
BASELINE_TIME_END = "09:30"
ANALYSIS_TIME = "10:00"

# Maximum time to enter position (after this, theta decay reduces edge)
MAX_ENTRY_TIME = "10:30"

# ═══════════════════════════════════════════════════════════════
# SIGNAL GENERATION THRESHOLDS
# ═══════════════════════════════════════════════════════════════

# Strength thresholds for directional signals
BULLISH_THRESHOLD = 8.0    # Strength > 8 = Clear bullish
BEARISH_THRESHOLD = -8.0   # Strength < -8 = Clear bearish
# Between -8 and +8 = SIDEWAYS (sell delta-neutral)

# Minimum confidence to trade
MIN_CONFIDENCE = 0.65

# ═══════════════════════════════════════════════════════════════
# DNS EDGE RATIO THRESHOLDS (Your Risk Management)
# ═══════════════════════════════════════════════════════════════

# Minimum edge ratio to execute trade
# Practical live default: keep as a quality filter without over-blocking.
MIN_EDGE_RATIO = 0.20  # Theta should be at least 20% of modeled total risk

# Risk calculation parameters
DELTA_RISK_MOVE = 0.02      # 2% adverse move for delta risk
GAMMA_RISK_MOVE = 0.02      # 2% move for gamma risk  
VEGA_RISK_IV_CHANGE = 0.01  # 1% IV increase for vega risk

# Risk buffers (conservative)
DELTA_RISK_BUFFER = 1.5
GAMMA_RISK_BUFFER = 2.0
VEGA_RISK_BUFFER = 1.5

# Delta neutrality threshold for straddles/strangles
MAX_DELTA_NEUTRAL = 0.15  # Position delta must be < 0.15 for "neutral"
DELTA_NEUTRAL_DTE_THRESHOLDS = [
    (7, 0.15),    # weekly: keep strict neutrality
    (14, 0.17),   # 2-week: slightly relaxed due to skew/term structure
    (999, 0.20),  # farther expiries
]

# Expiry selection rules for straddle execution
MIN_STRADDLE_PREMIUM_RATIO = 0.01  # ATM straddle must be >= 1% of spot
MAX_EXPIRIES_TO_SCAN = 6

# Live execution policy for sideways regime.
# If True, sideways entries are not hard-blocked by delta/edge filters.
FORCE_SIDEWAYS_ENTRY = True

# Vol-short safety gate (for sideways but unsafe volatility expansion).
ENABLE_VOL_SHORT_SAFETY_GATE = True
# Reject new short-vol entries if ATM average IV jumps too much from baseline.
MAX_ATM_IV_JUMP_POINTS = 2.0
# Reject if ATM straddle premium expands too much from baseline.
MAX_ATM_STRADDLE_EXPANSION_PCT = 0.15

# ═══════════════════════════════════════════════════════════════
# GREEKS CALCULATION
# ═══════════════════════════════════════════════════════════════

RISK_FREE_RATE = 0.07  # 7% India
DIVIDEND_YIELD = 0.02  # 2% typical for indices

# Days to expiry default (weekly options)
DEFAULT_DTE = 7

# NO DEFAULT IV - All IV must come from Dhan API
# System will fail if IV not available (safer than assumptions)

# ═══════════════════════════════════════════════════════════════
# OUTPUT & LOGGING
# ═══════════════════════════════════════════════════════════════

DATA_DIR = './data'
BASELINE_DIR = f'{DATA_DIR}/baselines'
REPORTS_DIR = f'{DATA_DIR}/reports'
TRACKER_DIR = f'{DATA_DIR}/tracker'

ENABLE_EXCEL_LOGGING = True
ENABLE_JSON_LOGGING = True

# Create directories
os.makedirs(BASELINE_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(TRACKER_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# DISPLAY SETTINGS
# ═══════════════════════════════════════════════════════════════

SHOW_DETAILED_GREEKS = True
SHOW_RISK_BREAKDOWN = True
SHOW_VALIDATION_DETAILS = True

# Dhan parser logging
# False = show only summary counts for rejected strikes
# True = print every rejected strike reason
VERBOSE_REJECT_LOGS = False
# API fetch/parsing logs in live run (INFO/WARN noise control).
# True = keep console mostly analysis-only; False = verbose API diagnostics.
QUIET_API_LOGS = True

# -----------------------------------------------------------------
# RULE-BASED ENGINE SETTINGS (mathematically reworked)
# -----------------------------------------------------------------
# Number of strikes each side of ATM used for baseline-relative flow.
FLOW_WINDOW_STRIKES = 6

# Deterministic regime thresholds (IV+delta structural states only).
REGIME_IV_FLAT_TOLERANCE = 0.05
REGIME_DELTA_NEAR_ZERO_THRESHOLD = 5.0
REGIME_GAMMA_STRESS_THRESHOLD = 50000.0
REGIME_IV_SPIKE_THRESHOLD = 0.8

# Raw-rule thresholds (no normalization).
RULE_IV_FLAT_TOLERANCE = 0.05
RULE_DELTA_NEAR_ZERO_THRESHOLD = 5.0
RULE_GAMMA_STRESS_THRESHOLD = 50000.0
RULE_IV_STRONG_EXPANSION_THRESHOLD = 0.6

# Directional filter thresholds (data-calibrated baseline from 2024 train,
# validated on 2025-2026):
# - Higher values reduce false directional calls but also reduce coverage.
RULE_DIRECTIONAL_FLOW_MIN = 500000.0
RULE_DIRECTIONAL_VEGA_MIN = 2000000.0
RULE_DIRECTIONAL_IV_SPREAD_MIN = 0.30

# Intraday continuation override (promote SIDEWAYS -> directional when
# same-day history and baseline-to-now move strongly agree).
INTRADAY_OVERRIDE_MIN_PRIOR_READS = 3
INTRADAY_OVERRIDE_MAX_SIDEWAYS_AGREEMENT = 0.40
INTRADAY_DIRECTIONAL_SPOT_MOVE_PCT = 0.35
INTRADAY_OVERRIDE_FLOW_FLOOR_RATIO = 0.20
INTRADAY_OVERRIDE_VEGA_FLOOR_RATIO = 0.20

# Live-trading hard gates (production safety profile).
LIVE_SIDEWAYS_ONLY = True
LIVE_REQUIRE_CALM_REGIME = True

# -----------------------------------------------------------------
# OVERNIGHT GAP MODULE (separate from core intraday engine)
# -----------------------------------------------------------------
# 2:45-3:00 PM vs 3:20-3:25 PM structural comparison thresholds.
OVERNIGHT_GAP_DELTA_MIN = 500000.0
OVERNIGHT_GAP_VEGA_MIN = 2000000.0
OVERNIGHT_GAP_IV_SPREAD_MIN = 0.20
