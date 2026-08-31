"""
Overnight gap ML baseline trainer (time-split, leakage-safe).

Data source:
  data/backtest/overnight_simple_30min_backtest_2y_details.csv

Goal:
  Predict next-open gap direction from late-session structure features only.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLASSES = ["BEARISH", "SIDEWAYS", "BULLISH"]


@dataclass
class Dataset:
    x: pd.DataFrame
    y: np.ndarray
    dates: List[str]


def _to_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _label_from_move(move_pct: float, band_pct: float) -> str:
    if move_pct > band_pct:
        return "BULLISH"
    if move_pct < -band_pct:
        return "BEARISH"
    return "SIDEWAYS"


def load_dataset(path: Path, band_pct: float) -> pd.DataFrame:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    out: List[Dict[str, object]] = []
    for r in rows:
        date_txt = str(r.get("signal_date", "")).strip()
        if not date_txt:
            continue
        try:
            _ = datetime.strptime(date_txt, "%Y-%m-%d")
        except Exception:
            continue

        move_pct = _to_float(r.get("overnight_move_pct", 0.0))
        call_iv = _to_float(r.get("call_iv_diff", 0.0))
        put_iv = _to_float(r.get("put_iv_diff", 0.0))
        flow_delta = _to_float(r.get("flow_delta_true", 0.0))
        flow_vega = _to_float(r.get("flow_vega_true", 0.0))
        total_iv = _to_float(r.get("total_iv_change", 0.0))

        out.append(
            {
                "signal_date": date_txt,
                "flow_delta_true": flow_delta,
                "flow_vega_true": flow_vega,
                "call_iv_diff": call_iv,
                "put_iv_diff": put_iv,
                "total_iv_change": total_iv,
                "iv_spread_abs": abs(call_iv - put_iv),
                "iv_sum": call_iv + put_iv,
                "delta_sign": 1.0 if flow_delta > 0 else -1.0 if flow_delta < 0 else 0.0,
                "vega_sign": 1.0 if flow_vega > 0 else -1.0 if flow_vega < 0 else 0.0,
                "target": _label_from_move(move_pct, band_pct=band_pct),
                "overnight_move_pct": move_pct,
            }
        )
    return pd.DataFrame(out)


def split_by_year(df: pd.DataFrame, train_end_year: int) -> tuple[Dataset, Dataset]:
    years = df["signal_date"].str.slice(0, 4).astype(int)
    train_df = df[years <= train_end_year].copy()
    valid_df = df[years > train_end_year].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError("Train/validation split is empty. Check date range.")

    feature_cols = [
        "flow_delta_true",
        "flow_vega_true",
        "call_iv_diff",
        "put_iv_diff",
        "total_iv_change",
        "iv_spread_abs",
        "iv_sum",
        "delta_sign",
        "vega_sign",
    ]
    train = Dataset(
        x=train_df[feature_cols],
        y=train_df["target"].to_numpy(),
        dates=train_df["signal_date"].tolist(),
    )
    valid = Dataset(
        x=valid_df[feature_cols],
        y=valid_df["target"].to_numpy(),
        dates=valid_df["signal_date"].tolist(),
    )
    return train, valid


def train_and_eval(
    in_csv: Path,
    out_summary_csv: Path,
    out_predictions_csv: Path,
    band_pct: float,
    train_end_year: int,
) -> None:
    df = load_dataset(in_csv, band_pct=band_pct)
    train, valid = split_by_year(df, train_end_year=train_end_year)

    model = Pipeline(
        steps=[
            (
                "pre",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            list(train.x.columns),
                        )
                    ]
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=6,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(train.x, train.y)
    pred = model.predict(valid.x)
    proba = model.predict_proba(valid.x)
    class_order = list(model.named_steps["rf"].classes_)

    acc = float(accuracy_score(valid.y, pred))
    cm = confusion_matrix(valid.y, pred, labels=CLASSES)
    rep = classification_report(valid.y, pred, labels=CLASSES, output_dict=True, zero_division=0)

    out_predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_predictions_csv.open("w", newline="", encoding="utf-8") as f:
        headers = [
            "signal_date",
            "y_true",
            "y_pred",
            "p_bearish",
            "p_sideways",
            "p_bullish",
            "flow_delta_true",
            "flow_vega_true",
            "call_iv_diff",
            "put_iv_diff",
            "total_iv_change",
            "iv_spread_abs",
        ]
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for i, d in enumerate(valid.dates):
            p = {c: 0.0 for c in CLASSES}
            for j, c in enumerate(class_order):
                p[c] = float(proba[i, j])
            row_x = valid.x.iloc[i]
            w.writerow(
                {
                    "signal_date": d,
                    "y_true": valid.y[i],
                    "y_pred": pred[i],
                    "p_bearish": round(p["BEARISH"], 6),
                    "p_sideways": round(p["SIDEWAYS"], 6),
                    "p_bullish": round(p["BULLISH"], 6),
                    "flow_delta_true": row_x["flow_delta_true"],
                    "flow_vega_true": row_x["flow_vega_true"],
                    "call_iv_diff": row_x["call_iv_diff"],
                    "put_iv_diff": row_x["put_iv_diff"],
                    "total_iv_change": row_x["total_iv_change"],
                    "iv_spread_abs": row_x["iv_spread_abs"],
                }
            )

    with out_summary_csv.open("w", newline="", encoding="utf-8") as f:
        headers = [
            "section",
            "value",
        ]
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerow({"section": "band_pct", "value": band_pct})
        w.writerow({"section": "train_rows", "value": len(train.y)})
        w.writerow({"section": "valid_rows", "value": len(valid.y)})
        w.writerow({"section": "valid_accuracy", "value": round(acc, 6)})
        for c in CLASSES:
            w.writerow({"section": f"precision_{c.lower()}", "value": round(rep[c]["precision"], 6)})
            w.writerow({"section": f"recall_{c.lower()}", "value": round(rep[c]["recall"], 6)})
            w.writerow({"section": f"f1_{c.lower()}", "value": round(rep[c]["f1-score"], 6)})

        # Confusion matrix flattened
        for i, actual in enumerate(CLASSES):
            for j, predc in enumerate(CLASSES):
                w.writerow({"section": f"cm_{actual}_as_{predc}", "value": int(cm[i, j])})

    print("Overnight ML baseline complete")
    print(f"train_rows={len(train.y)} valid_rows={len(valid.y)} valid_accuracy={acc:.4f}")
    print(f"summary_csv={out_summary_csv}")
    print(f"predictions_csv={out_predictions_csv}")


def main() -> int:
    p = argparse.ArgumentParser(description="Train simple ML baseline for overnight gap prediction")
    p.add_argument(
        "--input-csv",
        default="data/backtest/overnight_simple_30min_backtest_2y_details.csv",
    )
    p.add_argument(
        "--summary-csv",
        default="data/backtest/overnight_ml_summary.csv",
    )
    p.add_argument(
        "--predictions-csv",
        default="data/backtest/overnight_ml_predictions.csv",
    )
    p.add_argument(
        "--band-pct",
        type=float,
        default=0.25,
        help="Absolute move %% threshold for SIDEWAYS class.",
    )
    p.add_argument(
        "--train-end-year",
        type=int,
        default=2024,
        help="Train on years <= this, validate on years > this.",
    )
    args = p.parse_args()

    train_and_eval(
        in_csv=Path(args.input_csv),
        out_summary_csv=Path(args.summary_csv),
        out_predictions_csv=Path(args.predictions_csv),
        band_pct=float(args.band_pct),
        train_end_year=int(args.train_end_year),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

