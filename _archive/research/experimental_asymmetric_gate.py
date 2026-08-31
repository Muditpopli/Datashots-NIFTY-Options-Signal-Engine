import argparse
import json
from pathlib import Path

import pandas as pd


def _metric(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "total_points": 0.0,
        }
    wins = int((df["result"] == "WIN").sum())
    return {
        "trades": int(len(df)),
        "wins": wins,
        "losses": int(len(df) - wins),
        "win_rate": float((df["result"] == "WIN").mean()),
        "expectancy": float(df["points"].mean()),
        "total_points": float(df["points"].sum()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Build an experimental asymmetric confidence gate on top of the locked daily-tag predictions.")
    p.add_argument(
        "--source-csv",
        default="data/backtest/outputs/final_daily_tag_lock_v2_3y_maxexp_predictions.csv",
    )
    p.add_argument("--bull-conf-min", type=float, default=0.0)
    p.add_argument("--bear-conf-min", type=float, default=80.0)
    p.add_argument(
        "--out-csv",
        default="data/backtest/outputs/experimental_asymmetric_gate_predictions.csv",
    )
    p.add_argument(
        "--out-summary",
        default="data/backtest/outputs/experimental_asymmetric_gate_summary.json",
    )
    args = p.parse_args()

    src = Path(args.source_csv)
    df = pd.read_csv(src)

    gate = (
        ((df["tag"] == "BULLISH") & (df["confidence"] >= float(args.bull_conf_min)))
        | ((df["tag"] == "BEARISH") & (df["confidence"] >= float(args.bear_conf_min)))
    )
    out = df.loc[gate].copy()
    out["cumulative_points"] = out["points"].cumsum()

    summary = {
        "rule": "asymmetric_confidence_gate",
        "parameters": {
            "bull_conf_min": float(args.bull_conf_min),
            "bear_conf_min": float(args.bear_conf_min),
        },
        "source_csv": str(src),
        "base": _metric(df),
        "gated": _metric(out),
        "by_tag": {},
        "by_year": {},
    }

    for tag, g in out.groupby("tag"):
        summary["by_tag"][str(tag)] = _metric(g)

    out["year"] = pd.to_datetime(out["signal_date"]).dt.year
    for year, g in out.groupby("year"):
        summary["by_year"][str(year)] = _metric(g)
    out = out.drop(columns=["year"])

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    Path(args.out_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Done | trades={summary['gated']['trades']} "
        f"win_rate={summary['gated']['win_rate']:.2%} "
        f"expectancy={summary['gated']['expectancy']:+.2f} "
        f"total_points={summary['gated']['total_points']:+.2f}"
    )
    print(f"csv={args.out_csv}")
    print(f"summary={args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
