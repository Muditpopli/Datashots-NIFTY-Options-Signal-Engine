from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from feature_engineering import engineer_features
from model_training import train_and_save


def _signal_strength(conf: float) -> str:
    if conf >= 70.0:
        return "HIGH"
    if conf >= 55.0:
        return "MEDIUM"
    return "LOW"


def _calc_streaks(results: np.ndarray) -> Dict[str, float]:
    max_win = 0
    max_loss = 0
    cur_win = 0
    cur_loss = 0
    win_streaks = []
    loss_streaks = []
    for r in results:
        if r == 1:
            cur_win += 1
            if cur_loss > 0:
                loss_streaks.append(cur_loss)
            cur_loss = 0
            max_win = max(max_win, cur_win)
        else:
            cur_loss += 1
            if cur_win > 0:
                win_streaks.append(cur_win)
            cur_win = 0
            max_loss = max(max_loss, cur_loss)
    if cur_win > 0:
        win_streaks.append(cur_win)
    if cur_loss > 0:
        loss_streaks.append(cur_loss)
    return {
        "max_consecutive_wins": float(max_win),
        "max_consecutive_losses": float(max_loss),
        "avg_win_streak": float(np.mean(win_streaks)) if win_streaks else 0.0,
        "avg_loss_streak": float(np.mean(loss_streaks)) if loss_streaks else 0.0,
    }


def _build_trade_log(df_pred: pd.DataFrame) -> pd.DataFrame:
    out = df_pred.copy()
    out["Result"] = np.where(out["signed_points"] >= 1.0, "WIN", "LOSS")
    out["Points_Gained"] = out["signed_points"].round(2)
    out["Cumulative_PnL"] = out["Points_Gained"].cumsum().round(2)
    out["Day"] = pd.to_datetime(out["Date"]).dt.day_name().str.slice(0, 3)
    out["predicted_gap_size"] = out["predicted_gap_size"].round(2)
    out["Actual_Gap"] = out["actual_gap_points"].round(2)
    out = out[
        [
            "Date",
            "Day",
            "Signal",
            "Confidence",
            "Signal_Strength",
            "predicted_gap_size",
            "Actual_Gap",
            "Result",
            "Points_Gained",
            "Cumulative_PnL",
        ]
    ].rename(columns={"predicted_gap_size": "Predicted_Gap"})
    return out


def run_backtest(
    input_csv: Path,
    model_path: Path,
    excel_out: Path,
    min_train_days: int,
) -> Dict[str, float]:
    # Train if missing.
    if not model_path.exists():
        train_and_save(
            input_csv=input_csv,
            out_model=model_path,
            out_summary_json=Path("data/backtest/outputs/nifty_gap_model_train_summary.json"),
        )
    bundle = joblib.load(model_path)
    clf = bundle["classifier"]
    reg = bundle["regressor"]
    feat_cols = bundle["feature_cols"]

    raw = pd.read_csv(input_csv)
    fb = engineer_features(raw)
    df = fb.frame.copy()

    preds: List[Dict] = []
    # Use fixed trained model for full daily coverage (no skip).
    x_all = df[feat_cols]
    p_up_all = clf.predict_proba(x_all)[:, 1]
    pred_gap_all = reg.predict(x_all)

    for i, row in df.reset_index(drop=True).iterrows():
        p_up = float(p_up_all[i])
        p_down = 1.0 - p_up
        signal = "BULLISH" if p_up >= 0.5 else "BEARISH"
        conf = max(p_up, p_down) * 100.0
        move = float(row["target_move_points"])
        signed = move if signal == "BULLISH" else -move
        preds.append(
            {
                "Date": row["signal_date"].strftime("%Y-%m-%d"),
                "Signal": signal,
                "Confidence": round(conf, 2),
                "Signal_Strength": _signal_strength(conf),
                "p_up": p_up,
                "p_down": p_down,
                "predicted_gap_size": float(pred_gap_all[i]),
                "actual_gap_points": move,
                "signed_points": signed,
                "target_up": int(row["target_up"]),
            }
        )

    pred_df = pd.DataFrame(preds)
    trade_log = _build_trade_log(pred_df)

    # Core metrics.
    signed = pred_df["signed_points"].to_numpy(dtype=float)
    wins = signed[signed >= 1.0]
    losses = signed[signed <= -1.0]
    win_rate = float(len(wins) / (len(wins) + len(losses))) if (len(wins) + len(losses)) else 0.0
    avg_win = float(np.mean(wins)) if len(wins) else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) else 0.0
    expectancy = float(np.mean(signed)) if len(signed) else 0.0
    pred_up = (pred_df["p_up"] >= 0.5).astype(int).to_numpy()
    y_true = pred_df["target_up"].to_numpy()
    brier = float(brier_score_loss(y_true, pred_df["p_up"].to_numpy(dtype=float)))
    acc = float(np.mean(pred_up == y_true))

    # Confidence breakdown.
    conf_buckets = []
    for label, lo, hi in [("HIGH", 70.0, 100.0), ("MEDIUM", 55.0, 70.0), ("LOW", 0.0, 55.0)]:
        sub = pred_df[(pred_df["Confidence"] >= lo) & (pred_df["Confidence"] < hi if hi < 100 else pred_df["Confidence"] <= hi)]
        s = sub["signed_points"].to_numpy(dtype=float)
        w = s[s >= 1.0]
        l = s[s <= -1.0]
        wr = float(len(w) / (len(w) + len(l))) if (len(w) + len(l)) else 0.0
        conf_buckets.append(
            {
                "bucket": label,
                "trades": int(len(sub)),
                "win_rate": wr,
                "avg_points_trade": float(np.mean(s)) if len(s) else 0.0,
            }
        )

    # Monthly.
    m = trade_log.copy()
    m["Month"] = pd.to_datetime(m["Date"]).dt.to_period("M").astype(str)
    monthly = (
        m.groupby("Month", as_index=False)
        .agg(
            Trades=("Result", "size"),
            Wins=("Result", lambda s: int((s == "WIN").sum())),
            Total_Points=("Points_Gained", "sum"),
            Avg_Points_Trade=("Points_Gained", "mean"),
        )
    )
    monthly["Win_Rate"] = np.where(monthly["Trades"] > 0, monthly["Wins"] / monthly["Trades"], 0.0)

    # Confidence calibration table.
    cal_rows = []
    bins = [(90, 100), (80, 90), (70, 80), (60, 70), (50, 60), (0, 50)]
    for lo, hi in bins:
        sub = pred_df[(pred_df["Confidence"] >= lo) & (pred_df["Confidence"] < hi if hi < 100 else pred_df["Confidence"] <= hi)]
        if sub.empty:
            cal_rows.append(
                {
                    "Confidence_Range": f"{lo}-{hi}",
                    "Trades": 0,
                    "Actual_Win_Rate": 0.0,
                    "Calibration_Error": 0.0,
                }
            )
            continue
        actual_wr = float((sub["signed_points"] >= 1.0).mean())
        implied = float(sub["Confidence"].mean() / 100.0)
        cal_rows.append(
            {
                "Confidence_Range": f"{lo}-{hi}",
                "Trades": int(len(sub)),
                "Actual_Win_Rate": actual_wr,
                "Calibration_Error": actual_wr - implied,
            }
        )
    calibration = pd.DataFrame(cal_rows)

    # Feature importance.
    clf_base = clf.calibrated_classifiers_[0].estimator
    feat_imp = pd.DataFrame({"Feature": feat_cols, "Importance_Score": 0.0})
    try:
        inner = clf_base.named_steps["clf"] if hasattr(clf_base, "named_steps") else clf_base
        if hasattr(inner, "feature_importances_"):
            feat_imp["Importance_Score"] = inner.feature_importances_
        elif hasattr(inner, "coef_"):
            feat_imp["Importance_Score"] = np.abs(inner.coef_[0])
        feat_imp = feat_imp.sort_values("Importance_Score", ascending=False).reset_index(drop=True)
        feat_imp.insert(0, "Rank", np.arange(1, len(feat_imp) + 1))
    except Exception:
        feat_imp.insert(0, "Rank", np.arange(1, len(feat_imp) + 1))

    # Regime analysis proxy from iv/vega signs.
    reg = pred_df.copy()
    reg["regime"] = np.where(reg["actual_gap_points"].abs() >= 60, "Trending", "Sideways")
    regime_tbl = (
        reg.groupby("regime", as_index=False)
        .agg(
            Trades=("Signal", "size"),
            Win_Rate=("signed_points", lambda s: float((s >= 1.0).mean())),
            Avg_Points=("signed_points", "mean"),
        )
    )

    # Visual data.
    viz = pred_df[["Date", "Confidence", "signed_points"]].copy()
    viz["Cumulative_PnL"] = viz["signed_points"].cumsum()

    # Summary sheet.
    streaks = _calc_streaks((pred_df["signed_points"] >= 1.0).astype(int).to_numpy())
    drawdown = (viz["Cumulative_PnL"] - viz["Cumulative_PnL"].cummax()).min()
    summary_rows = [
        ("Total Trading Days", int(len(pred_df))),
        ("Total Signals", int(len(pred_df))),
        ("Wins", int((pred_df["signed_points"] >= 1.0).sum())),
        ("Losses", int((pred_df["signed_points"] <= -1.0).sum())),
        ("Win Rate", win_rate),
        ("Average Winning Gap", avg_win),
        ("Average Losing Gap", avg_loss),
        ("Best Single Trade", float(pred_df["signed_points"].max())),
        ("Worst Single Trade", float(pred_df["signed_points"].min())),
        ("Total Points Gained", float(pred_df["signed_points"].sum())),
        ("Average Points/Trade", expectancy),
        ("Accuracy", acc),
        ("Brier Score", brier),
        ("Max Consecutive Wins", streaks["max_consecutive_wins"]),
        ("Max Consecutive Losses", streaks["max_consecutive_losses"]),
        ("Average Win Streak", streaks["avg_win_streak"]),
        ("Average Loss Streak", streaks["avg_loss_streak"]),
        ("Max Drawdown (points)", float(drawdown)),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

    excel_out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_out, engine="openpyxl") as w:
        trade_log.to_excel(w, sheet_name="Trade Log", index=False)
        summary_df.to_excel(w, sheet_name="Summary Metrics", index=False)
        monthly.to_excel(w, sheet_name="Monthly Breakdown", index=False)
        calibration.to_excel(w, sheet_name="Confidence Calibration", index=False)
        feat_imp.to_excel(w, sheet_name="Feature Importance", index=False)
        regime_tbl.to_excel(w, sheet_name="Regime Analysis", index=False)
        viz.to_excel(w, sheet_name="Visualization Data", index=False)

    return {
        "trades": float(len(pred_df)),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "accuracy": acc,
        "brier": brier,
        "excel": str(excel_out),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Binary daily overnight NIFTY backtest engine.")
    p.add_argument("--input-csv", default="data/backtest/overnight_simple_30min_backtest_3y_details.csv")
    p.add_argument("--model-path", default="trained_model.pkl")
    p.add_argument("--excel-out", default="nifty_gap_backtest_results.xlsx")
    p.add_argument("--min-train-days", type=int, default=180)
    args = p.parse_args()

    stats = run_backtest(
        input_csv=Path(args.input_csv),
        model_path=Path(args.model_path),
        excel_out=Path(args.excel_out),
        min_train_days=args.min_train_days,
    )
    print(
        "BACKTEST "
        f"trades={int(stats['trades'])} "
        f"win_rate={stats['win_rate']:.2%} "
        f"avg_win={stats['avg_win']:+.2f} "
        f"avg_loss={stats['avg_loss']:+.2f} "
        f"expectancy={stats['expectancy']:+.2f} "
        f"accuracy={stats['accuracy']:.2%} "
        f"brier={stats['brier']:.4f}"
    )
    print(f"excel={stats['excel']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

