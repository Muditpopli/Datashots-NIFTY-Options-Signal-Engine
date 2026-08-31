# DNS Scanner - Quick Start Guide

## What You Have Now

A **clean, professional options trading system** with:

✅ **Fast Performance** - Single API fetch (no repeated calls)  
✅ **Accurate Greeks** - Black-Scholes implementation  
✅ **True ATM Calculation** - Synthetic futures method  
✅ **Delta Neutrality Validation** - For sideways trades  
✅ **Professional Risk Management** - DNS Edge Ratio  

---

## Files Overview

```
dns_scanner/
├── config.py              # All settings (edit thresholds here)
├── greeks.py              # Black-Scholes Greeks calculator
├── dhan_api.py            # Dhan API wrapper
├── signal_detector.py     # Signal generation engine
├── trade_validator.py     # Edge ratio validation
├── main.py               # Main system (run this)
├── test_system.py        # Validation tests
├── requirements.txt      # Dependencies
├── .env.example         # Template for credentials
├── README.md            # Full documentation
└── data/
    ├── baselines/       # Baseline snapshots (auto-created)
    └── reports/         # Future: reports
```

**Total Code**: ~800 lines (vs 2,500+ in old system)

---

## Setup (5 Minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Credentials
```bash
# Copy template
cp .env.example .env

# Edit .env and add:
DHAN_ACCESS_TOKEN=your_token_here
DHAN_CLIENT_ID=your_client_id_here
```

### 3. Test System
```bash
python test_system.py
```

Should show all tests passing ✅

---

## Daily Usage

### Morning Routine (9:15 AM)

```bash
python main.py baseline
```

**What it does:**
- Fetches opening option chain
- Calculates ATM using synthetic futures
- Saves baseline Greeks
- Takes ~10 seconds

**Output:**
```
✅ Baseline captured for NIFTY
   Spot: 23515.00 | ATM: 23500
   Strikes: 21
💾 Baseline saved: data/baselines/baseline_NIFTY_20260212_091523.json
```

### Analysis Routine (10:00 AM)

```bash
python main.py analyze
```

**What it does:**
- Fetches current option chain
- Compares with baseline
- Generates signal (BULLISH/BEARISH/SIDEWAYS)
- Validates trades with edge ratio
- Recommends best trade

**Output Example 1 - Bullish:**
```
🟢 SIGNAL: BULLISH
   Strength: +12.5
   Confidence: 85%

💰 TRADE RECOMMENDATION
🎯 Direction: BULLISH
   Strategy: Sell PE

✅ BEST STRIKE: 23450 PE
   Premium: ₹185.50 (Total: ₹9,275)
   
💵 DAILY INCOME:
   Theta: ₹825 per day

⚠️ RISKS:
   Delta Risk: ₹3,200
   Gamma Risk: ₹4,500
   Vega Risk: ₹650
   Total Risk: ₹8,350

📊 EDGE RATIO: 0.89x (EXCELLENT)

✅ VERDICT: SAFE TO TRADE
```

**Output Example 2 - Sideways:**
```
🟡 SIGNAL: SIDEWAYS
   Strength: +2.3
   Confidence: 70%

💰 TRADE RECOMMENDATION
🎯 Strategy: DELTA-NEUTRAL STRADDLE

✅ SETUP: Sell ATM 23500 Straddle
   Total Premium: ₹19,185
   Position Delta: 0.08 (✓ Neutral)

💵 DAILY INCOME:
   Theta: ₹1,580 per day

⚠️ RISKS:
   Total Risk: ₹10,900

📊 EDGE RATIO: 0.82x (EXCELLENT)

✅ VERDICT: SAFE TO TRADE
```

---

## Your Trading Rules (Built-In)

### ✅ Automatic Checks

1. **Edge Ratio ≥ 0.75**
   - System only recommends trades with edge ≥ 0.75
   - Earn at least 75¢ per ₹1 risk

2. **Delta Neutrality** (for sideways)
   - Position delta must be < 0.15
   - System validates before recommending

3. **Timing Check**
   - Warns if entering after 10:30 AM
   - Theta decay reduces edge

4. **ATM Accuracy**
   - Uses synthetic futures (not middle of chain)
   - True forward price

5. **Risk Buffers**
   - Delta: 1.5x buffer
   - Gamma: 2.0x buffer
   - Vega: 1.5x buffer

### 🎯 Manual Rules (You Enforce)

1. **Max Risk**: 2% per trade
2. **Stop Loss**: Exit at 1.5× total risk
3. **Square Off**: Close by 3:30 PM (never hold overnight)
4. **No Override**: Follow system exactly

---

## Key Improvements Over Old System

| Aspect | Old System | New System |
|--------|------------|------------|
| **Speed** | Slow (multiple API calls) | Fast (single fetch) |
| **ATM Calc** | Middle of chain ❌ | Synthetic futures ✅ |
| **Delta Neutral** | Not checked ❌ | Validated ✅ |
| **Risk Formula** | Theta/(Gamma+Vega) | Theta/(Delta+Gamma+Vega) ✅ |
| **Timing** | No warning | Warns after 10:30 AM ✅ |
| **Code** | 2,500+ lines | ~800 lines ✅ |
| **Structure** | Complex | Clean ✅ |

---

## Configuration

Edit `config.py` to customize:

```python
# Edge ratio (minimum to trade)
MIN_EDGE_RATIO = 0.75

# Signal strength thresholds
BULLISH_THRESHOLD = 8.0    # Strength > 8 = bullish
BEARISH_THRESHOLD = -8.0   # Strength < -8 = bearish

# Risk buffers (conservative)
DELTA_RISK_BUFFER = 1.5
GAMMA_RISK_BUFFER = 2.0
VEGA_RISK_BUFFER = 1.5

# Delta neutrality threshold
MAX_DELTA_NEUTRAL = 0.15   # |delta| < 0.15 = neutral
```

---

## Troubleshooting

### "Missing Dhan credentials"
→ Create `.env` file with your credentials

### "No baseline captured"
→ Run `python main.py baseline` first (9:15 AM)

### "Failed to fetch option chain"
→ Check API credentials and internet connection

### "Edge ratio too low"
→ Market not favorable - **skip trade, wait for tomorrow**

---

## What to Track

Daily:
- ✅ Did you follow the signal?
- ✅ Was edge ratio > 0.75?
- ✅ What was your P&L?
- ✅ Did you follow all rules?

Weekly:
- Win rate (target: 65%+)
- Average edge ratio (target: 0.75+)
- Total P&L
- Rule violations (should be 0)

---

## Next Steps

1. ✅ **Test** - Run `python test_system.py`
2. ✅ **Paper Trade** - Follow signals for 1 week without real money
3. ✅ **Small Size** - Start with 1 lot only
4. ✅ **Track** - Keep daily log of trades
5. ✅ **Scale** - Increase size only after 10+ winning trades

---

## Critical Notes

⚠️ **The Dhan API Wrapper is a Template**

The `dhan_api.py` file contains a template for fetching option chains. You'll need to:
1. Check Dhan's actual API documentation
2. Implement the `_fetch_dhan_option_chain()` method
3. Ensure it returns the right data structure

**Data structure needed:**
```python
[
    {
        'strike': 23500,
        'call_ltp': 150.5,
        'put_ltp': 145.2,
        'call_iv': 18.5,  # Optional
        'put_iv': 18.2,   # Optional
        # ... other fields
    },
    ...
]
```

⚠️ **Fallback to Black-Scholes**

If Dhan doesn't return Greeks, the system calculates them using Black-Scholes with:
- IV from Dhan (if available)
- Default 18% IV (if not available)
- 7 days to expiry (weekly options)

---

## Philosophy

> **"Better to miss a trade than take a bad one"**

- The system helps you **wait** for quality setups
- It **skips** when edge ratio < 0.75
- It **preserves** capital above all
- It forces **discipline** through automation

Your job is simple:
1. Run baseline at 9:15 AM
2. Run analysis at 10:00 AM  
3. Execute **only** when edge ≥ 0.75
4. Follow risk rules **100%**

That's it. No emotions. No overrides. Pure mechanical execution.

---

**Good luck! 🎯**

Questions? Check `README.md` for full documentation.
