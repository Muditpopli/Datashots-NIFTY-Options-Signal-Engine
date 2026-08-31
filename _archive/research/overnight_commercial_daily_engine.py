from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class DailyPred:
    signal_date: str
    next_open_date: str
    p_up: float
    p_down: float
    direction: str
    conviction: str
    overnight_move_points: float
    signed_points: float
    outcome: str
    model_used: str


def _build_models() -> List[Tuple[str, Pipeline]]:
    return [
        (
            "LR_DAILY_V1",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
                ]
            ),
        ),
        (
            "RF_DAILY_V1",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=500,
                            max_depth=6,
                            min_samples_leaf=4,
                            class_weight="balanced_subsample",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        (
            "GB_DAILY_V1",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", GradientBoostingClassifier(random_state=42)),
                ]
            ),
        ),
    ]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["signal_date"] = pd.to_datetime(x["signal_date"], errors="coerce")
    x = x.dropna(subset=["signal_date"]).sort_values("signal_date").reset_index(drop=True)
    x["iv_spread"] = x["call_iv_diff"] - x["put_iv_diff"]
    x["iv_spread_abs"] = x["iv_spread"].abs()
    x["iv_sum"] = x["call_iv_diff"] + x["put_iv_diff"]
    x["flow_vega_scaled"] = x["flow_vega_true"] / 1_000_000.0
    x["flow_delta_scaled"] = x["flow_delta_true"] / 1_000_000.0
    x["delta_sign"] = np.sign(x["flow_delta_true"])
    x["vega_sign"] = np.sign(x["flow_vega_true"])
    x["target_up"] = (x["overnight_move_points"].astype(float) >= 1.0).astype(int)
    return x


def _features() -> List[str]:
    return [
        "flow_vega_scaled",
        "flow_delta_scaled",
        "call_iv_diff",
        "put_iv_diff",
        "total_iv_change",
        "iv_spread",
        "iv_spread_abs",
        "iv_sum",
        "delta_sign",
        "vega_sign",
    ]


def _conviction(p_up: float) -> str:
    edge = abs(p_up - 0.5)
    if edge >= 0.12:
        return "HIGH_CONVICTION"
    if edge >= 0.07:
        return "MEDIUM_CONVICTION"
    return "LOW_CONVICTION"


def run(
    input_csv: Path,
    out_events_csv: Path,
    out_summary_csv: Path,
    end_date: str,
    lookback_days: int,
    min_train_days: int,
) -> Dict[str, float]:
    raw = pd.read_csv(input_csv)
    df = _prep(raw)
    df = df[df["signal_date"] <= pd.to_datetime(end_date)].copy()
    if len(df) < min_train_days + 40:
        raise RuntimeError("Not enough rows after date filtering for walk-forward.")

    feat_cols = _features()
    preds: List[DailyPred] = []
    years = df["signal_date"].dt.year
    models = _build_models()
    best_name = None
    best_key = None
    best_oos_rows: List[DailyPred] = []

    for name, pipe in models:
        oos_rows: List[DailyPred] = []
        try:
            for vy in sorted(years.unique()):
                train_mask = years < vy
                valid_mask = years == vy
                if train_mask.sum() < min_train_days or valid_mask.sum() < 10:
                    continue
                pipe.fit(df.loc[train_mask, feat_cols], df.loc[train_mask, "target_up"])
                p_up_arr = pipe.predict_proba(df.loc[valid_mask, feat_cols])[:, 1]
                valid_rows = df.loc[valid_mask].copy()
                for idx, (_, row) in enumerate(valid_rows.iterrows()):
                    p_up = float(p_up_arr[idx])
                    direction = "BULLISH" if p_up >= 0.5 else "BEARISH"
                    move = float(row["overnight_move_points"])
                    signed = move if direction == "BULLISH" else -move
                    oos_rows.append(
                        DailyPred(
                            signal_date=row["signal_date"].strftime("%Y-%m-%d"),
                            next_open_date=str(row.get("next_open_date", "")),
                            p_up=p_up,
                            p_down=1.0 - p_up,
                            direction=direction,
                            conviction=_conviction(p_up),
                            overnight_move_points=move,
                            signed_points=signed,
                            outcome="WIN" if signed >= 1.0 else "LOSS",
                            model_used=name,
                        )
                    )
        except Exception:
            continue

        if not oos_rows:
            continue
        signed = np.array([r.signed_points for r in oos_rows], dtype=float)
        pvals = np.array([r.p_up for r in oos_rows], dtype=float)
        ytrue = np.array([1 if r.overnight_move_points >= 1.0 else 0 for r in oos_rows], dtype=int)
        pred = (pvals >= 0.5).astype(int)
        wins = int((signed >= 1.0).sum())
        losses = int((signed <= -1.0).sum())
        wr = float(wins / (wins + losses)) if (wins + losses) else 0.0
        exp = float(np.mean(signed)) if len(signed) else -999.0
        brier = float(np.mean((pvals - ytrue) ** 2))
        acc = float(np.mean(pred == ytrue))
        key = (exp, wr, -brier, acc)
        if best_key is None or key > best_key:
            best_key = key
            best_name = name
            best_oos_rows = oos_rows

    preds = best_oos_rows

    if not preds:
        raise RuntimeError("No walk-forward predictions produced.")

    events = pd.DataFrame([p.__dict__ for p in preds])
    events["signal_date"] = pd.to_datetime(events["signal_date"])
    start_cut = pd.to_datetime(end_date) - pd.Timedelta(days=lookback_days)
    events_lookback = events[events["signal_date"] >= start_cut].copy()
    if events_lookback.empty:
        events_lookback = events.copy()

    events_lookback = events_lookback.sort_values("signal_date")
    out_events_csv.parent.mkdir(parents=True, exist_ok=True)
    events_lookback.to_csv(out_events_csv, index=False, date_format="%Y-%m-%d")

    signed = events_lookback["signed_points"].astype(float).to_numpy()
    wins = signed[signed >= 1.0]
    losses = signed[signed <= -1.0]
    win_rate = float(len(wins) / (len(wins) + len(losses))) if (len(wins) + len(losses)) else 0.0
    avg_win = float(np.mean(wins)) if len(wins) else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) else 0.0
    expectancy = float(np.mean(signed)) if len(signed) else 0.0

    rows = []
    rows.append(
        {
            "bucket": "ALL_DAILY",
            "trades": int(len(events_lookback)),
            "win_rate": round(win_rate, 4),
            "avg_win_points": round(avg_win, 4),
            "avg_loss_points": round(avg_loss, 4),
            "expectancy_points": round(expectancy, 4),
        }
    )
    for b in ["HIGH_CONVICTION", "MEDIUM_CONVICTION", "LOW_CONVICTION"]:
        sub = events_lookback[events_lookback["conviction"] == b]
        if sub.empty:
            rows.append(
                {
                    "bucket": b,
                    "trades": 0,
                    "win_rate": 0.0,
                    "avg_win_points": 0.0,
                    "avg_loss_points": 0.0,
                    "expectancy_points": 0.0,
                }
            )
            continue
        s = sub["signed_points"].astype(float).to_numpy()
        w = s[s >= 1.0]
        l = s[s <= -1.0]
        wr = float(len(w) / (len(w) + len(l))) if (len(w) + len(l)) else 0.0
        rows.append(
            {
                "bucket": b,
                "trades": int(len(sub)),
                "win_rate": round(wr, 4),
                "avg_win_points": round(float(np.mean(w)) if len(w) else 0.0, 4),
                "avg_loss_points": round(float(np.mean(l)) if len(l) else 0.0, 4),
                "expectancy_points": round(float(np.mean(s)) if len(s) else 0.0, 4),
            }
        )

    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "bucket",
                "trades",
                "win_rate",
                "avg_win_points",
                "avg_loss_points",
                "expectancy_points",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return {
        "trades": float(len(events_lookback)),
        "win_rate": win_rate,
        "avg_win_points": avg_win,
        "avg_loss_points": avg_loss,
        "expectancy_points": expectancy,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Commercial-grade daily overnight directional engine (walk-forward).")
    p.add_argument("--input-csv", default="data/backtest/overnight_simple_30min_backtest_3y_details.csv")
    p.add_argument("--end-date", default="2026-03-02")
    p.add_argument("--lookback-days", type=int, default=365)
    p.add_argument("--min-train-days", type=int, default=220)
    p.add_argument(
        "--events-csv",
        default="data/backtest/outputs/overnight_commercial_daily_last1y_to_2026-03-02.csv",
    )
    p.add_argument(
        "--summary-csv",
        default="data/backtest/outputs/overnight_commercial_daily_last1y_to_2026-03-02_summary.csv",
    )
    args = p.parse_args()

    stats = run(
        input_csv=Path(args.input_csv),
        out_events_csv=Path(args.events_csv),
        out_summary_csv=Path(args.summary_csv),
        end_date=args.end_date,
        lookback_days=args.lookback_days,
        min_train_days=args.min_train_days,
    )
    print(f"events_csv={args.events_csv}")
    print(f"summary_csv={args.summary_csv}")
    print(
        "ALL_DAILY "
        f"trades={int(stats['trades'])} "
        f"win_rate={stats['win_rate']:.2%} "
        f"avg_win={stats['avg_win_points']:+.2f} "
        f"avg_loss={stats['avg_loss_points']:+.2f} "
        f"expectancy={stats['expectancy_points']:+.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
