import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _signed_points(tag: str, gap: float) -> float:
    if tag == "BULLISH":
        return gap if gap >= 1.0 else -abs(gap)
    return abs(gap) if gap <= -1.0 else -abs(gap)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    base_feats = [
        "n_spot_day",
        "n_spot_l30",
        "n_oi_day",
        "n_oi_l30",
        "n_pcr_day",
        "n_pcr_l30",
        "b_spot_day",
        "b_spot_l30",
        "b_oi_day",
        "b_oi_l30",
        "b_pcr_day",
        "b_pcr_l30",
    ]
    x = df[base_feats].copy()
    x["spot_align_l30"] = np.sign(df["n_spot_l30"]) * np.sign(df["b_spot_l30"])
    x["oi_align_l30"] = np.sign(df["n_oi_l30"]) * np.sign(df["b_oi_l30"])
    x["pcr_align_l30"] = np.sign(df["n_pcr_l30"]) * np.sign(df["b_pcr_l30"])
    x["spot_div_l30"] = df["n_spot_l30"] - df["b_spot_l30"]
    x["oi_div_l30"] = df["n_oi_l30"] - df["b_oi_l30"]
    x["pcr_div_l30"] = df["n_pcr_l30"] - df["b_pcr_l30"]
    x["spot_sum_day"] = df["n_spot_day"] + df["b_spot_day"]
    x["spot_sum_l30"] = df["n_spot_l30"] + df["b_spot_l30"]
    x["oi_sum_day"] = df["n_oi_day"] + df["b_oi_day"]
    x["oi_sum_l30"] = df["n_oi_l30"] + df["b_oi_l30"]
    x["pcr_sum_day"] = df["n_pcr_day"] + df["b_pcr_day"]
    x["pcr_sum_l30"] = df["n_pcr_l30"] + df["b_pcr_l30"]
    x["bn_minus_n_day"] = df["b_spot_day"] - df["n_spot_day"]
    x["bn_minus_n_l30"] = df["b_spot_l30"] - df["n_spot_l30"]
    for c in [
        "n_spot_day",
        "n_spot_l30",
        "b_spot_day",
        "b_spot_l30",
        "n_pcr_day",
        "n_pcr_l30",
        "b_pcr_day",
        "b_pcr_l30",
    ]:
        x[f"{c}_sq"] = x[c] ** 2
    return x


def _evaluate_probs(probs: np.ndarray, gaps: np.ndarray, threshold: float) -> dict:
    points = []
    for p, g in zip(probs, gaps):
        tag = "BULLISH" if p >= threshold else "BEARISH"
        points.append(_signed_points(tag, float(g)))
    pts = np.array(points, dtype=float)
    wins = int((pts >= 1.0).sum())
    return {
        "trades": int(len(pts)),
        "wins": wins,
        "losses": int(len(pts) - wins),
        "win_rate": float(wins / len(pts)) if len(pts) else 0.0,
        "expectancy": float(pts.mean()) if len(pts) else 0.0,
        "total_points": float(pts.sum()) if len(pts) else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Train walk-forward experimental BTST classifiers on the saved active feature table.")
    p.add_argument(
        "--source-csv",
        default="data/backtest/outputs/final_daily_tag_lock_v2_3y_maxexp_predictions_to_2026-03-13.csv",
    )
    p.add_argument("--start-train-size", type=int, default=220)
    p.add_argument(
        "--out-summary",
        default="data/backtest/outputs/experimental_btst_model_search_summary.json",
    )
    args = p.parse_args()

    df = pd.read_csv(args.source_csv)
    x = _build_features(df)
    y = (df["actual_gap"] >= 1.0).astype(int)
    gaps = df["actual_gap"].to_numpy(dtype=float)

    base = {
        "trades": int(len(df)),
        "wins": int((df["result"] == "WIN").sum()),
        "losses": int((df["result"] != "WIN").sum()),
        "win_rate": float((df["result"] == "WIN").mean()),
        "expectancy": float(df["points"].mean()),
        "total_points": float(df["points"].sum()),
    }

    models = {
        "logreg": Pipeline(
            [
                ("sc", StandardScaler()),
                ("m", LogisticRegression(max_iter=3000, class_weight="balanced")),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=250,
            max_depth=5,
            min_samples_leaf=8,
            random_state=42,
        ),
        "et": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=6,
            random_state=42,
        ),
    }

    results = {
        "source_csv": args.source_csv,
        "base": base,
        "start_train_size": int(args.start_train_size),
        "models": {},
    }

    for name, model in models.items():
        probs = []
        for i in range(int(args.start_train_size), len(x)):
            m = clone(model)
            m.fit(x.iloc[:i], y.iloc[:i])
            p_up = float(m.predict_proba(x.iloc[[i]])[0, 1])
            probs.append(p_up)
        probs_arr = np.array(probs, dtype=float)
        gaps_eval = gaps[int(args.start_train_size) :]

        best = None
        for threshold in np.arange(0.30, 0.71, 0.02):
            metrics = _evaluate_probs(probs_arr, gaps_eval, float(threshold))
            candidate = {"threshold": float(round(threshold, 2)), **metrics}
            if best is None:
                best = candidate
            elif (candidate["expectancy"] > best["expectancy"]) and (candidate["win_rate"] >= best["win_rate"]):
                best = candidate
            elif (candidate["expectancy"] + candidate["win_rate"]) > (best["expectancy"] + best["win_rate"]):
                best = candidate

        results["models"][name] = best

    Path(args.out_summary).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
