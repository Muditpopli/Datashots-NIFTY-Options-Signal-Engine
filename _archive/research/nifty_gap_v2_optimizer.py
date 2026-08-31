from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import engineer_features, split_train_test_by_last_year
from threshold_optimizer import (
    evaluate_threshold_grid,
    optimize_global_threshold,
    optimize_regime_thresholds,
)


REGIMES = ["TRENDING_UP", "TRENDING_DOWN", "SIDEWAYS", "HIGH_VOLATILITY"]


def _build_candidates() -> List[Tuple[str, Pipeline]]:
    return [
        (
            "logistic",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=3500, class_weight="balanced")),
                ]
            ),
        ),
        (
            "random_forest",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=400,
                            max_depth=7,
                            min_samples_leaf=4,
                            class_weight="balanced_subsample",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        (
            "gradient_boosting",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("clf", GradientBoostingClassifier(random_state=42)),
                ]
            ),
        ),
    ]


def add_regime_columns(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x = x.sort_values("signal_date").reset_index(drop=True)
    x["close_ret_1d"] = x["close_spot_1520"].pct_change().fillna(0.0)
    x["ma_20"] = x["close_spot_1520"].rolling(20, min_periods=5).mean()
    x["ma_20_slope"] = x["ma_20"].diff().fillna(0.0)
    x["abs_ret_20"] = x["close_ret_1d"].abs().rolling(20, min_periods=5).mean().fillna(0.0)
    hv_cut = float(x["abs_ret_20"].quantile(0.8))
    x["regime"] = "SIDEWAYS"
    x.loc[x["abs_ret_20"] >= hv_cut, "regime"] = "HIGH_VOLATILITY"
    x.loc[(x["ma_20_slope"] > 0) & (x["abs_ret_20"] < hv_cut), "regime"] = "TRENDING_UP"
    x.loc[(x["ma_20_slope"] < 0) & (x["abs_ret_20"] < hv_cut), "regime"] = "TRENDING_DOWN"
    return x


def _pick_best_model(train_df: pd.DataFrame, feat_cols: List[str], target_col: str) -> Tuple[str, CalibratedClassifierCV]:
    split_n = max(30, int(len(train_df) * 0.2))
    fit_df = train_df.iloc[:-split_n].copy()
    calib_df = train_df.iloc[-split_n:].copy()
    if len(fit_df) < 40:
        fit_df = train_df.copy()
        calib_df = train_df.copy()

    best_name = None
    best_cal = None
    best_key = None
    for name, pipe in _build_candidates():
        try:
            pipe.fit(fit_df[feat_cols], fit_df[target_col].astype(int).to_numpy())
            cal = CalibratedClassifierCV(pipe, method="sigmoid", cv="prefit")
            cal.fit(calib_df[feat_cols], calib_df[target_col].astype(int).to_numpy())
            p = cal.predict_proba(calib_df[feat_cols])[:, 1]
            pred = (p >= 0.5).astype(int)
            y = calib_df[target_col].astype(int).to_numpy()
            moves = calib_df["target_move_points"].astype(float).to_numpy()
            signed = np.where(pred == 1, moves, -moves)
            wins = (signed >= 1.0).sum()
            losses = (signed <= -1.0).sum()
            wr = float(wins / (wins + losses)) if (wins + losses) else 0.0
            exp = float(np.mean(signed)) if len(signed) else 0.0
            brier = float(np.mean((p - y) ** 2))
            key = (exp, wr, -brier)
            if best_key is None or key > best_key:
                best_key = key
                best_name = name
                best_cal = cal
        except Exception:
            continue

    if best_cal is None or best_name is None:
        # Hard fallback
        name, pipe = _build_candidates()[0]
        pipe.fit(train_df[feat_cols], train_df[target_col].astype(int).to_numpy())
        best_cal = CalibratedClassifierCV(pipe, method="sigmoid", cv="prefit")
        best_cal.fit(train_df[feat_cols], train_df[target_col].astype(int).to_numpy())
        best_name = name
    return best_name, best_cal


def _signal_strength(conf: float) -> str:
    if conf >= 70:
        return "HIGH"
    if conf >= 55:
        return "MEDIUM"
    return "LOW"


def _signed_points(direction: str, actual_move: float) -> float:
    return actual_move if direction == "BULLISH" else -actual_move


def _summarize(pred_df: pd.DataFrame) -> Dict[str, float]:
    s = pred_df["signed_points"].to_numpy(dtype=float)
    wins = s[s >= 1.0]
    losses = s[s <= -1.0]
    wr = float(len(wins) / (len(wins) + len(losses))) if (len(wins) + len(losses)) else 0.0
    return {
        "trades": float(len(pred_df)),
        "win_rate": wr,
        "avg_win": float(np.mean(wins)) if len(wins) else 0.0,
        "avg_loss": float(np.mean(losses)) if len(losses) else 0.0,
        "expectancy": float(np.mean(s)) if len(s) else 0.0,
        "total_points": float(np.sum(s)),
    }


def run_v2(input_csv: Path, out_excel: Path, out_model: Path) -> Dict[str, float]:
    raw = pd.read_csv(input_csv)
    fb = engineer_features(raw)
    df = add_regime_columns(fb.frame)
    train_df, test_df = split_train_test_by_last_year(df)

    feat_cols = fb.feature_cols + ["close_ret_1d", "ma_20_slope", "abs_ret_20"]
    target_col = fb.target_col

    # Main train split: keep tune slice for threshold learning only.
    tune_n = max(50, int(len(train_df) * 0.2))
    model_train_df = train_df.iloc[:-tune_n].copy()
    tune_df = train_df.iloc[-tune_n:].copy()

    # Global model.
    global_model_name, global_model = _pick_best_model(model_train_df, feat_cols, target_col)

    # Regime models.
    regime_models: Dict[str, CalibratedClassifierCV] = {}
    regime_model_names: Dict[str, str] = {}
    for rg in REGIMES:
        sub = model_train_df[model_train_df["regime"] == rg].copy()
        if len(sub) < 60:
            regime_models[rg] = global_model
            regime_model_names[rg] = f"{global_model_name}(fallback)"
            continue
        nm, mdl = _pick_best_model(sub, feat_cols, target_col)
        regime_models[rg] = mdl
        regime_model_names[rg] = nm

    # Optional gap-size model.
    reg = Pipeline([("imp", SimpleImputer(strategy="median")), ("reg", GradientBoostingRegressor(random_state=42))])
    reg.fit(model_train_df[feat_cols], model_train_df["target_move_points"].astype(float).to_numpy())

    # Predict on tune set for threshold optimization.
    tune_preds = []
    for _, row in tune_df.iterrows():
        rg = str(row["regime"])
        mdl = regime_models.get(rg, global_model)
        p_up = float(mdl.predict_proba(pd.DataFrame([row[feat_cols].to_dict()]))[0][1])
        p_down = 1.0 - p_up
        direction = "BULLISH" if p_up >= 0.5 else "BEARISH"
        conf = max(p_up, p_down) * 100.0
        move = float(row["target_move_points"])
        tune_preds.append(
            {
                "date": row["signal_date"].strftime("%Y-%m-%d"),
                "regime": rg,
                "direction": direction,
                "confidence": conf,
                "signed_points": _signed_points(direction, move),
            }
        )
    tune_pred_df = pd.DataFrame(tune_preds)
    global_thr = optimize_global_threshold(tune_pred_df, min_trades=25, target_win_rate=0.55)
    regime_thrs = optimize_regime_thresholds(tune_pred_df, REGIMES, min_trades_per_regime=8)

    # Predict on test (strict OOS).
    test_rows = []
    for _, row in test_df.iterrows():
        rg = str(row["regime"])
        mdl = regime_models.get(rg, global_model)
        p_up = float(mdl.predict_proba(pd.DataFrame([row[feat_cols].to_dict()]))[0][1])
        p_down = 1.0 - p_up
        direction = "BULLISH" if p_up >= 0.5 else "BEARISH"
        conf = max(p_up, p_down) * 100.0
        pred_gap = float(reg.predict(pd.DataFrame([row[feat_cols].to_dict()]))[0])
        move = float(row["target_move_points"])
        signed = _signed_points(direction, move)
        thr_global = global_thr.min_confidence
        thr_regime = regime_thrs.get(rg, global_thr).min_confidence
        trade_global = conf >= thr_global
        trade_regime = conf >= thr_regime
        test_rows.append(
            {
                "Date": row["signal_date"].strftime("%Y-%m-%d"),
                "Day": row["signal_date"].day_name()[:3],
                "Regime": rg,
                "Signal": direction,
                "Confidence": round(conf, 2),
                "Signal_Strength": _signal_strength(conf),
                "Predicted_Gap": round(pred_gap, 2),
                "Actual_Gap": round(move, 2),
                "signed_points_all": signed,
                "Trade_GlobalThreshold": bool(trade_global),
                "Trade_RegimeThreshold": bool(trade_regime),
                "Global_Threshold": float(thr_global),
                "Regime_Threshold": float(thr_regime),
                "Model_Used": regime_model_names.get(rg, global_model_name),
            }
        )

    test_pred = pd.DataFrame(test_rows)
    test_pred["Result_All"] = np.where(test_pred["signed_points_all"] >= 1.0, "WIN", "LOSS")

    # V1 baseline on test = all trades.
    v1_df = test_pred.copy()
    v1_df["signed_points"] = v1_df["signed_points_all"]
    v1_stats = _summarize(v1_df)

    # V2 filtered = regime thresholds.
    v2_df = test_pred[test_pred["Trade_RegimeThreshold"]].copy()
    if v2_df.empty:
        v2_df = test_pred.copy()
    v2_df["signed_points"] = v2_df["signed_points_all"]
    v2_stats = _summarize(v2_df)

    # Trade log (V2 executed set).
    trade_log = v2_df.copy()
    trade_log["Result"] = np.where(trade_log["signed_points"] >= 1.0, "WIN", "LOSS")
    trade_log["Points_Gained"] = trade_log["signed_points"].round(2)
    trade_log["Cumulative_PnL"] = trade_log["Points_Gained"].cumsum().round(2)
    trade_log = trade_log[
        [
            "Date",
            "Day",
            "Signal",
            "Confidence",
            "Signal_Strength",
            "Predicted_Gap",
            "Actual_Gap",
            "Result",
            "Points_Gained",
            "Cumulative_PnL",
            "Regime",
            "Model_Used",
        ]
    ]

    # Summary metrics (V2 focus).
    summary_rows = [
        ("Total Trading Days (Test)", int(len(test_pred))),
        ("Traded Days (V2 Filtered)", int(len(v2_df))),
        ("Skipped Days (V2)", int(len(test_pred) - len(v2_df))),
        ("Win Rate (V2)", v2_stats["win_rate"]),
        ("Average Winning Gap (V2)", v2_stats["avg_win"]),
        ("Average Losing Gap (V2)", v2_stats["avg_loss"]),
        ("Expectancy (V2)", v2_stats["expectancy"]),
        ("Total Points (V2)", v2_stats["total_points"]),
        ("Win Rate (V1 all days)", v1_stats["win_rate"]),
        ("Expectancy (V1 all days)", v1_stats["expectancy"]),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

    # Monthly breakdown (V2).
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

    # Confidence calibration (V2 traded days).
    cal_rows = []
    bins = [(90, 100), (80, 90), (70, 80), (60, 70), (50, 60), (0, 50)]
    for lo, hi in bins:
        sub = v2_df[(v2_df["Confidence"] >= lo) & (v2_df["Confidence"] < hi if hi < 100 else v2_df["Confidence"] <= hi)]
        if sub.empty:
            cal_rows.append({"Confidence_Range": f"{lo}-{hi}", "Trades": 0, "Actual_Win_Rate": 0.0, "Calibration_Error": 0.0})
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

    # Feature importance from global model.
    fi = pd.DataFrame({"Feature": feat_cols, "Importance_Score": 0.0})
    try:
        est = global_model.calibrated_classifiers_[0].estimator
        inner = est.named_steps["clf"] if hasattr(est, "named_steps") else est
        if hasattr(inner, "feature_importances_"):
            fi["Importance_Score"] = inner.feature_importances_
        elif hasattr(inner, "coef_"):
            fi["Importance_Score"] = np.abs(inner.coef_[0])
        fi = fi.sort_values("Importance_Score", ascending=False).reset_index(drop=True)
        fi.insert(0, "Rank", np.arange(1, len(fi) + 1))
    except Exception:
        fi.insert(0, "Rank", np.arange(1, len(fi) + 1))

    # Regime analysis table.
    reg_rows = []
    for rg in REGIMES:
        all_rg = test_pred[test_pred["Regime"] == rg]
        trd_rg = v2_df[v2_df["Regime"] == rg]
        s = trd_rg["signed_points"].to_numpy(dtype=float) if not trd_rg.empty else np.array([])
        wr = float((s >= 1.0).mean()) if len(s) else 0.0
        exp = float(np.mean(s)) if len(s) else 0.0
        reg_rows.append(
            {
                "Regime": rg,
                "Days": int(len(all_rg)),
                "Traded": int(len(trd_rg)),
                "Skipped": int(len(all_rg) - len(trd_rg)),
                "Win_Rate": wr,
                "Expectancy": exp,
            }
        )
    regime_tbl = pd.DataFrame(reg_rows)

    # Visualization data.
    viz = v2_df[["Date", "Confidence", "signed_points"]].copy()
    viz["Cumulative_PnL"] = viz["signed_points"].cumsum()

    # Sheet 9 threshold impact from tune set.
    threshold_impact = evaluate_threshold_grid(tune_pred_df, thresholds=[0, 60, 65, 70, 75, 80])

    # Sheet 10 V1 vs V2.
    cmp = pd.DataFrame(
        [
            {"Metric": "Win Rate", "V1 (Current)": v1_stats["win_rate"], "V2 (Target)": v2_stats["win_rate"], "Change": v2_stats["win_rate"] - v1_stats["win_rate"]},
            {"Metric": "Expectancy", "V1 (Current)": v1_stats["expectancy"], "V2 (Target)": v2_stats["expectancy"], "Change": v2_stats["expectancy"] - v1_stats["expectancy"]},
            {"Metric": "Trades/Year", "V1 (Current)": v1_stats["trades"], "V2 (Target)": v2_stats["trades"], "Change": v2_stats["trades"] - v1_stats["trades"]},
            {"Metric": "Avg Win", "V1 (Current)": v1_stats["avg_win"], "V2 (Target)": v2_stats["avg_win"], "Change": v2_stats["avg_win"] - v1_stats["avg_win"]},
            {"Metric": "Avg Loss", "V1 (Current)": v1_stats["avg_loss"], "V2 (Target)": v2_stats["avg_loss"], "Change": v2_stats["avg_loss"] - v1_stats["avg_loss"]},
        ]
    )

    # Save model bundle V2.
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": "NIFTY_GAP_V2",
            "feature_cols": feat_cols,
            "global_model": global_model,
            "global_model_name": global_model_name,
            "regime_models": regime_models,
            "regime_model_names": regime_model_names,
            "global_threshold": float(global_thr.min_confidence),
            "regime_thresholds": {k: float(v.min_confidence) for k, v in regime_thrs.items()},
            "gap_regressor": reg,
        },
        out_model,
    )

    out_excel.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_excel, engine="openpyxl") as w:
        trade_log.to_excel(w, sheet_name="Trade Log", index=False)
        summary_df.to_excel(w, sheet_name="Summary Metrics", index=False)
        monthly.to_excel(w, sheet_name="Monthly Breakdown", index=False)
        calibration.to_excel(w, sheet_name="Confidence Calibration", index=False)
        fi.to_excel(w, sheet_name="Feature Importance", index=False)
        regime_tbl.to_excel(w, sheet_name="Regime Analysis", index=False)
        viz.to_excel(w, sheet_name="Visualization Data", index=False)
        regime_tbl.to_excel(w, sheet_name="Sheet8_Regime_Analysis", index=False)
        threshold_impact.to_excel(w, sheet_name="Sheet9_Threshold_Impact", index=False)
        cmp.to_excel(w, sheet_name="Sheet10_V1_vs_V2", index=False)

    # Recommendation.
    ready = (
        (v2_stats["expectancy"] >= 10.0)
        and (v2_stats["win_rate"] >= 0.55)
        and (v2_stats["avg_win"] >= abs(v2_stats["avg_loss"]))
    )
    rec = "Ready to commercialize" if ready else "Needs more work"

    return {
        "v1_win_rate": v1_stats["win_rate"],
        "v1_expectancy": v1_stats["expectancy"],
        "v2_win_rate": v2_stats["win_rate"],
        "v2_expectancy": v2_stats["expectancy"],
        "v2_trades": v2_stats["trades"],
        "recommendation": rec,
        "global_threshold": float(global_thr.min_confidence),
        "excel": str(out_excel),
        "model": str(out_model),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="NIFTY overnight V2 optimizer (regime + threshold).")
    p.add_argument("--input-csv", default="data/backtest/overnight_simple_30min_backtest_3y_details.csv")
    p.add_argument("--out-excel", default="nifty_gap_backtest_results_v2.xlsx")
    p.add_argument("--out-model", default="trained_model_v2.pkl")
    args = p.parse_args()

    stats = run_v2(Path(args.input_csv), Path(args.out_excel), Path(args.out_model))
    print(
        f"V1 win_rate={stats['v1_win_rate']:.2%} expectancy={stats['v1_expectancy']:+.2f} | "
        f"V2 win_rate={stats['v2_win_rate']:.2%} expectancy={stats['v2_expectancy']:+.2f} "
        f"trades={int(stats['v2_trades'])} threshold={stats['global_threshold']:.1f}"
    )
    print(f"recommendation={stats['recommendation']}")
    print(f"excel={stats['excel']}")
    print(f"model={stats['model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
