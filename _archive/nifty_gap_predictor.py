from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from feature_engineering import engineer_features


@dataclass
class PredictionOutput:
    date: str
    direction: str
    confidence: float
    predicted_gap_size: float
    signal_strength: str


def _signal_strength(confidence: float) -> str:
    if confidence >= 70.0:
        return "HIGH"
    if confidence >= 55.0:
        return "MEDIUM"
    return "LOW"


def _load_model(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    bundle = joblib.load(path)
    required = ["classifier", "regressor", "feature_cols"]
    miss = [k for k in required if k not in bundle]
    if miss:
        raise ValueError(f"Invalid model bundle, missing keys: {miss}")
    return bundle


def predict_overnight_gap(date: str, row: Dict[str, Any], model_path: Path = Path("trained_model.pkl")) -> Dict[str, Any]:
    """
    Args:
        date: Trading date YYYY-MM-DD.
        row: Dict containing one-day 3:15 PM feature context with schema compatible
             with overnight_simple_30min_backtest_3y_details.csv.
        model_path: Saved trained model path.
    Returns:
        {
            'date': '2024-01-15',
            'direction': 'BULLISH' | 'BEARISH',
            'confidence': 78.5,
            'predicted_gap_size': 42.3,
            'signal_strength': 'HIGH' | 'MEDIUM' | 'LOW'
        }
    """
    bundle = _load_model(model_path)
    clf = bundle["classifier"]
    reg = bundle["regressor"]
    feat_cols = bundle["feature_cols"]

    one = pd.DataFrame([row])
    fb = engineer_features(one)
    x = fb.frame[feat_cols]

    p_up = float(clf.predict_proba(x)[:, 1][0])
    p_down = 1.0 - p_up
    direction = "BULLISH" if p_up >= 0.5 else "BEARISH"
    confidence = max(p_up, p_down) * 100.0
    pred_gap = float(reg.predict(x)[0])

    out = PredictionOutput(
        date=date,
        direction=direction,
        confidence=round(confidence, 2),
        predicted_gap_size=round(pred_gap, 2),
        signal_strength=_signal_strength(confidence),
    )
    return out.__dict__


def _infer_regime_for_row(df_one: pd.DataFrame) -> str:
    # Minimal single-row fallback regime: derive from signs of structural fields.
    delta = float(df_one.iloc[0].get("flow_delta_true", 0.0))
    vega = float(df_one.iloc[0].get("flow_vega_true", 0.0))
    ivc = float(df_one.iloc[0].get("total_iv_change", 0.0))
    if abs(vega) > 80_000_000 or abs(ivc) > 1.8:
        return "HIGH_VOLATILITY"
    if delta > 0:
        return "TRENDING_UP"
    if delta < 0:
        return "TRENDING_DOWN"
    return "SIDEWAYS"


def predict_overnight_gap_v2(date: str, row: Dict[str, Any], model_path: Path = Path("trained_model_v2.pkl")) -> Dict[str, Any]:
    """
    V2 prediction with regime-conditioned model selection and threshold metadata.
    Direction is always binary (BULLISH/BEARISH); execution filtering is advisory.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    bundle = joblib.load(model_path)
    if "regime_models" not in bundle:
        # Fallback to v1 output format if user passes v1 model.
        return predict_overnight_gap(date, row, model_path=model_path)

    feat_cols = bundle["feature_cols"]
    regressor = bundle["gap_regressor"]
    regime_models = bundle["regime_models"]
    regime_thresholds = bundle.get("regime_thresholds", {})
    global_thr = float(bundle.get("global_threshold", 0.0))

    one = pd.DataFrame([row])
    fb = engineer_features(one)
    x = fb.frame.copy()
    if "close_ret_1d" not in x.columns:
        x["close_ret_1d"] = 0.0
    if "ma_20_slope" not in x.columns:
        x["ma_20_slope"] = 0.0
    if "abs_ret_20" not in x.columns:
        x["abs_ret_20"] = 0.0
    # Ensure all feature columns exist.
    for c in feat_cols:
        if c not in x.columns:
            x[c] = 0.0
    x = x[feat_cols]

    regime = _infer_regime_for_row(fb.frame)
    mdl = regime_models.get(regime, bundle["global_model"])
    p_up = float(mdl.predict_proba(x)[:, 1][0])
    p_down = 1.0 - p_up
    direction = "BULLISH" if p_up >= 0.5 else "BEARISH"
    confidence = max(p_up, p_down) * 100.0
    pred_gap = float(regressor.predict(x)[0])
    thr = float(regime_thresholds.get(regime, global_thr))
    execute = bool(confidence >= thr)

    return {
        "date": date,
        "direction": direction,
        "confidence": round(confidence, 2),
        "predicted_gap_size": round(pred_gap, 2),
        "signal_strength": _signal_strength(confidence),
        "regime": regime,
        "min_confidence_to_execute": round(thr, 2),
        "execute_trade": execute,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="NIFTY overnight binary gap predictor.")
    p.add_argument("--date", required=True)
    p.add_argument(
        "--input-row-json",
        required=True,
        help="Path to JSON file containing one row dict with required fields.",
    )
    p.add_argument("--model-path", default="trained_model.pkl")
    args = p.parse_args()

    row = json.loads(Path(args.input_row_json).read_text(encoding="utf-8"))
    out = predict_overnight_gap(args.date, row, Path(args.model_path))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
