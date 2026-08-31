# Project Rules and System Design

## 1) Project Objective
Build a deterministic options analytics system that:
- captures opening structure,
- evaluates intraday structure against baseline,
- prioritizes safe deployment on sideways conditions,
- keeps directional logic available for research and gradual promotion.

## 2) Current Deployment Policy (Locked)
- Live deployment is **sideways-first**.
- Directional days are not primary live-trade days.
- Risk control is hard-gated and non-negotiable.

Live gates (from `config.py`):
- `LIVE_SIDEWAYS_ONLY = True`
- `LIVE_REQUIRE_CALM_REGIME = True`

## 3) Folder Structure
- `live/`: live launchers
  - `live/live_ready.py`
  - `live/routine_wrapper.py`
- `analytics/`: math + regime + rule engine
  - `analytics/greek_flow.py`
  - `analytics/regime.py`
  - `analytics/rules.py`
- `vega_theta_engine.py`: orchestration and signal generation
- `main.py`: execution workflow
- `research/`: replay/backtest/inspection scripts
- `data/backtest/`: records, outputs, rolling cache

## 4) Core Runtime Workflow

### Step A: Baseline capture
Command:
```bash
python live/live_ready.py --mode baseline
```
Purpose:
- store opening chain baseline (09:15-09:30 window operationally).

### Step B: Decision run
Command:
```bash
python live/live_ready.py --mode decision
```
Purpose:
- compare current chain vs baseline,
- generate regime + direction + safety output,
- apply live gates.

### Step C: Optional routine mode
```bash
python live/live_ready.py --mode routine --baseline-time 09:20 --decision-time 10:00
```

## 5) Data and Feature Logic
The engine computes baseline-relative raw structural features:
- `flow_delta_true`
- `flow_vega_true`
- `gamma_shift_true`
- `gamma_skew_shift_true`
- `pos_delta_now`
- `spot_change_pct`
- `total_call_oi`, `total_put_oi`
- `total_call_oi_change`, `total_put_oi_change`
- `call_iv_diff`, `put_iv_diff`, `total_iv_change`
- `oi_imbalance`

No z-score/percentile normalization is used in active logic.

## 6) Regime Logic (Deterministic)
Regime labels:
- `CALM`
- `TRENDING`
- `VOLATILE`

High-level:
- vega contracting/flat environments lean `CALM`.
- vega expansion with directional structure can become `TRENDING`.
- gamma/IV stress or unstable expansion can become `VOLATILE`.

Regime is structural; no persistence manager is used.

## 7) Classifier Logic (Deterministic)
Primary hierarchy:
1. Vega state
2. Delta state
3. IV-side confirmation
4. OI as secondary confidence context
5. Gamma stress as confidence/risk modifier

Important handling:
- Contradiction states are explicitly blocked (avoid/no-trade branches).
- Sideways interpretation uses neutral/range framing.

Directional threshold filter (configurable):
- `RULE_DIRECTIONAL_FLOW_MIN`
- `RULE_DIRECTIONAL_VEGA_MIN`
- `RULE_DIRECTIONAL_IV_SPREAD_MIN`

Current calibrated defaults:
- `RULE_DIRECTIONAL_FLOW_MIN = 500000`
- `RULE_DIRECTIONAL_VEGA_MIN = 2000000`
- `RULE_DIRECTIONAL_IV_SPREAD_MIN = 0.30`

## 8) Live Safety and Execution Rules

### 8.1 Confidence gate
- Decision path uses confidence threshold from config.

### 8.2 Vol-short safety gate
Used on sideways path to reject unsafe short-vol states when IV/straddle expansion is excessive.

### 8.3 Expiry selection rule
- Trade expiry must satisfy:
  - `ATM straddle premium / spot >= 1%`
- If nearest expiry fails, system tries next expiry.

## 9) Intraday Context Rules
Decision mode compares:
- baseline -> now
- all prior same-day readings -> now

For sideways states, intraday labels are:
- `RANGE_TIGHTENING`
- `RANGE_STABLE`
- `RANGE_WIDENING`
- `RANGE_BREAK_RISK`

Directional terms like strengthening/weakening/flip are retained for directional contexts.

## 10) Backtest and Research Rules
- Keep and use existing records:
  - `data/backtest/accuracy_record_sheet.csv`
  - `data/backtest/sideways_safe_mtm.csv`
- Keep rolling cache:
  - `data/backtest/cache/rolling_options/...`
- Research scripts remain under `research/`.

## 11) Sideways MTM Simulation Rules
Safe simulation model (`research/backtest_safe_sideways_mtm.py`):
- trade only `CALM + SIDEWAYS + trade_allowed`,
- short ATM straddle near 10:00,
- hard daily stop: 1% of margin,
- stop-out intraday when hit, else EOD exit.

## 12) Commands Cheat Sheet

Baseline:
```bash
python live/live_ready.py --mode baseline
```

Decision:
```bash
python live/live_ready.py --mode decision
```

Routine:
```bash
python live/live_ready.py --mode routine --baseline-time 09:20 --decision-time 10:00
```

Midday read:
```bash
python live/live_ready.py --mode midday
```

Sideways MTM:
```bash
python research/backtest_safe_sideways_mtm.py --index NIFTY --start-date 2025-01-01 --end-date 2026-12-31 --margin 2000000 --sl-pct 0.01 --lots 10
```

## 13) Operating Principle
The system is intentionally conservative:
- preserve capital by filtering poor environments,
- deploy where structure is explainable and stable,
- keep directional calls on strict filter until sustained validation improves.
