# Mudit-Style Overnight Gap System

## Run

```powershell
python overnight_gap_predictor.py --start-date 2023-01-01 --end-date 2026-03-02 --indices NIFTY,BANKNIFTY,SENSEX
```

## Outputs

- `data/backtest/outputs/overnight_gap_backtest_results.csv`
- `data/backtest/outputs/overnight_gap_backtest_results.xlsx`
- `data/backtest/outputs/overnight_gap_backtest_results_summary.json`

## Notes

- Uses rolling cache from `data/backtest/cache/rolling_options/<INDEX>`.
- Current workspace has `NIFTY` and `BANKNIFTY` cache; `SENSEX` is optional and will be reported as missing.
- Time phases used:
  - `open_0915` (9:10-9:30 nearest 9:15)
  - `close_1515` (14:45-15:30 nearest 15:15)
  - `next_open_0925` (9:20-9:35 nearest 9:25)

