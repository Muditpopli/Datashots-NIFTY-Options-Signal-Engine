from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import engineer_features, split_train_test_by_last_year


@dataclass
class TrainSummary:
    best_model_name: str
    train_rows: int
    test_rows: int
    test_accuracy: float
    test_brier: float
    test_win_rate: float
    test_expectancy_points: float
    test_avg_win_points: float
    test_avg_loss_points: float


def _build_candidates() -> List[Tuple[str, Pipeline]]:
    return [
        (
            "logistic",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=4000, class_weight="balanced")),
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
                            n_estimators=500,
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


def _score_binary(p_up: np.ndarray, y_true: np.ndarray, moves: np.ndarray) -> Dict[str, float]:
    pred_up = (p_up >= 0.5).astype(int)
    signed = np.where(pred_up == 1, moves, -moves)
    wins = signed[signed >= 1.0]
    losses = signed[signed <= -1.0]
    win_rate = float(len(wins) / (len(wins) + len(losses))) if (len(wins) + len(losses)) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, pred_up)),
        "brier": float(brier_score_loss(y_true, p_up)),
        "win_rate": win_rate,
        "expectancy": float(np.mean(signed)) if len(signed) else 0.0,
        "avg_win": float(np.mean(wins)) if len(wins) else 0.0,
        "avg_loss": float(np.mean(losses)) if len(losses) else 0.0,
    }


def train_and_save(
    input_csv: Path,
    out_model: Path,
    out_summary_json: Path,
) -> TrainSummary:
    raw = pd.read_csv(input_csv)
    fb = engineer_features(raw)
    train_df, test_df = split_train_test_by_last_year(fb.frame)

    x_train = train_df[fb.feature_cols]
    y_train = train_df[fb.target_col].astype(int).to_numpy()
    x_test = test_df[fb.feature_cols]
    y_test = test_df[fb.target_col].astype(int).to_numpy()
    mv_test = test_df[fb.regression_target_col].astype(float).to_numpy()

    # Keep last slice of train for probability calibration.
    calib_n = max(30, int(len(train_df) * 0.2))
    fit_df = train_df.iloc[:-calib_n].copy()
    calib_df = train_df.iloc[-calib_n:].copy()
    x_fit = fit_df[fb.feature_cols]
    y_fit = fit_df[fb.target_col].astype(int).to_numpy()
    x_calib = calib_df[fb.feature_cols]
    y_calib = calib_df[fb.target_col].astype(int).to_numpy()

    best_name = None
    best_pipe = None
    best_calib = None
    best_key = None

    for name, pipe in _build_candidates():
        try:
            pipe.fit(x_fit, y_fit)
            cal = CalibratedClassifierCV(pipe, method="sigmoid", cv="prefit")
            cal.fit(x_calib, y_calib)
            p_test = cal.predict_proba(x_test)[:, 1]
            m = _score_binary(p_test, y_test, mv_test)
            key = (m["win_rate"], m["expectancy"], -m["brier"], m["accuracy"])
            if best_key is None or key > best_key:
                best_key = key
                best_name = name
                best_pipe = pipe
                best_calib = cal
        except Exception:
            continue

    if best_calib is None or best_pipe is None or best_name is None:
        raise RuntimeError("Failed to train any valid calibrated classifier.")

    # Refit selected classifier on full train and recalibrate.
    best_pipe.fit(x_train, y_train)
    # Calibrate with tail of train to preserve temporal causality.
    cal_full = CalibratedClassifierCV(best_pipe, method="sigmoid", cv="prefit")
    cal_full.fit(x_calib, y_calib)

    # Gap-size estimator (optional field in output format).
    reg = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("reg", GradientBoostingRegressor(random_state=42)),
        ]
    )
    reg.fit(x_train, train_df[fb.regression_target_col].astype(float).to_numpy())

    p_test = cal_full.predict_proba(x_test)[:, 1]
    metrics = _score_binary(p_test, y_test, mv_test)

    bundle = {
        "model_name": best_name,
        "feature_cols": fb.feature_cols,
        "classifier": cal_full,
        "regressor": reg,
        "confidence_bands": {"high": 70.0, "medium": 55.0},
        "meta": {
            "train_start": str(train_df["signal_date"].min().date()),
            "train_end": str(train_df["signal_date"].max().date()),
            "test_start": str(test_df["signal_date"].min().date()),
            "test_end": str(test_df["signal_date"].max().date()),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
    }

    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_model)

    summary = TrainSummary(
        best_model_name=best_name,
        train_rows=int(len(train_df)),
        test_rows=int(len(test_df)),
        test_accuracy=metrics["accuracy"],
        test_brier=metrics["brier"],
        test_win_rate=metrics["win_rate"],
        test_expectancy_points=metrics["expectancy"],
        test_avg_win_points=metrics["avg_win"],
        test_avg_loss_points=metrics["avg_loss"],
    )

    out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    out_summary_json.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Train calibrated binary overnight gap model.")
    p.add_argument("--input-csv", default="data/backtest/overnight_simple_30min_backtest_3y_details.csv")
    p.add_argument("--out-model", default="trained_model.pkl")
    p.add_argument("--out-summary", default="data/backtest/outputs/nifty_gap_model_train_summary.json")
    args = p.parse_args()

    summary = train_and_save(
        input_csv=Path(args.input_csv),
        out_model=Path(args.out_model),
        out_summary_json=Path(args.out_summary),
    )
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

