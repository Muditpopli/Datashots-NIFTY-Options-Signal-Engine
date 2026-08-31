from __future__ import annotations

import pandas as pd


def main() -> None:
    src = "data/backtest/outputs/eth_straddle_1955_2045_last1y_excluded_rv_gt_iv_at_1955.csv"
    out = "data/backtest/outputs/eth_trades_only_rv_gt_iv_1955_simple.xlsx"

    df = pd.read_csv(src)
    cols = [
        "date",
        "day",
        "entry_straddle",
        "exit_straddle",
        "change_points",
        "iv_pct_1955",
        "rv1d_pct_1955",
    ]
    keep = [c for c in cols if c in df.columns]
    out_df = df[keep].sort_values("date")
    out_df.to_excel(out, index=False)

    print(f"rows {len(out_df)}")
    print(f"file {out}")


if __name__ == "__main__":
    main()
