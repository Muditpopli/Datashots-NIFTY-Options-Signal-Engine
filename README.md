# datashots

A quantitative signal engine for NIFTY/BANKNIFTY index options, focused on
BTST (Buy Today Sell Tomorrow) overnight positioning.

## What this is

Options-chain-driven models that decide, once a day, whether to hold an
overnight CE/PE position going into the next session. Signals are generated
from OI, PCR, net delta, and IV-skew features across the current and next
expiry, confirmed across NIFTY and BANKNIFTY, and combined with an ML
classifier layer. All performance numbers referenced in this repo come from
walk-forward out-of-sample backtests, not live trading.

## Repository structure

- `btst_engine/`
  - Core pipeline: expiry handling, signal building, rule-based + ML signal
    generation, backtesting, and the daily orchestrator.
- `intraday_engine/`
  - Exploratory intraday direction model (documented as not production-viable
    based on backtest results; kept for reference).
- `pipelines/data_backed/`
  - Entry points for the data-driven fetch/tag/live flow.
- `live/`
  - Production launcher(s) for the live routine.
- `data/`
  - `ml/` — engineered feature tables and trained model artifacts (gitignored)
  - `backtest/cache/` — raw historical options-chain cache (gitignored, not
    committed — multi-GB)
  - `backtest/outputs/`, `reports/`, `tracker/` — backtest and signal-log
    outputs (gitignored)
  - `sample/` — a small, clearly-labeled sample slice of feature data for
    demonstration and EDA, safe to commit
- `notebooks/`
  - EDA and model-diagnostics notebooks, runnable end-to-end on `data/sample/`.
- `assets/`
  - Figures exported from the notebooks (PNG), referenced from the README.
- `_archive/`
  - Retired experiments and superseded modules, kept for history.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# then edit .env with your own broker credentials:
#   DHAN_ACCESS_TOKEN=...
#   DHAN_CLIENT_ID=...
```

`.env` is gitignored and must never be committed. `.env.example` only ever
contains placeholder values.

## Data

The full options-chain cache and generated signal/report history are not
committed to this repository (too large, and not meant for redistribution).
A small sampled slice of the feature dataset lives in `data/sample/` purely
so the exploratory notebooks in this repo can run end-to-end for anyone
reading the code — it is not the dataset the live models are trained or
evaluated on.

## Exploratory Analysis

See [`notebooks/01_eda_and_validation.ipynb`](notebooks/01_eda_and_validation.ipynb) for the EDA and model-diagnostics walkthrough (feature distributions, correlation structure, the walk-forward validation scheme, and calibration/feature-importance checks), run against the sample in `data/sample/`.

## Important notes

- Signal logic is validated via expanding-window walk-forward
  cross-validation, not single backtests.
- Nothing in this repository should be read as investment advice. Options
  trading carries significant risk of loss.
