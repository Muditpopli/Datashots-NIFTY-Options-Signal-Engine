from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from feature_engineering import engineer_features, split_train_test_by_last_year


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
                    ("clf", RandomForestClassifier(n_estimators=450, max_depth=7, min_samples_leaf=4, class_weight="balanced_subsample", random_state=42)),
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
    x = df.copy().sort_values("signal_date").reset_index(drop=True)
    x["close_ret_1d"] = x["close_spot_1520"].pct_change().fillna(0.0)
    x["ma_20"] = x["close_spot_1520"].rolling(20, min_periods=5).mean()
    x["ma_20_slope"] = x["ma_20"].diff().fillna(0.0)
    x["abs_ret_20"] = x["close_ret_1d"].abs().rolling(20, min_periods=5).mean().fillna(0.0)
    hv_cut = float(x["abs_ret_20"].quantile(0.8))
    x["regime"] = "SIDEWAYS"
    x.loc[x["abs_ret_20"] >= hv_cut, "regime"] = "HIGH_VOLATILITY"
    x.loc[(x["ma_20_slope"] > 0) & (x["abs_ret_20"] < hv_cut), "regime"] = "TRENDING_UP"
    x.loc[(x["ma_20_slope"] < 0) & (x["abs_ret_20"] < hv_cut), "regime"] = "TRENDING_DOWN"
    # Regime strength [0..1]
    slope_abs = x["ma_20_slope"].abs()
    q = float(slope_abs.quantile(0.9)) if len(slope_abs) else 1.0
    if q <= 1e-9:
        x["regime_strength"] = 0.5
    else:
        x["regime_strength"] = (slope_abs / q).clip(0, 1)
    return x


def _pick_best_model(train_df: pd.DataFrame, feat_cols: List[str], target_col: str) -> Tuple[str, CalibratedClassifierCV]:
    split_n = max(30, int(len(train_df) * 0.2))
    fit_df = train_df.iloc[:-split_n].copy()
    calib_df = train_df.iloc[-split_n:].copy()
    if len(fit_df) < 40:
        fit_df, calib_df = train_df.copy(), train_df.copy()

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
            wins = int((signed >= 1.0).sum())
            losses = int((signed <= -1.0).sum())
            wr = wins / (wins + losses) if (wins + losses) else 0.0
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


def _summ(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"trades": 0.0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "expectancy": 0.0, "total_points": 0.0}
    s = df["signed_points"].to_numpy(dtype=float)
    w = s[s >= 1.0]
    l = s[s <= -1.0]
    wr = float(len(w) / (len(w) + len(l))) if (len(w) + len(l)) else 0.0
    return {
        "trades": float(len(df)),
        "win_rate": wr,
        "avg_win": float(np.mean(w)) if len(w) else 0.0,
        "avg_loss": float(np.mean(l)) if len(l) else 0.0,
        "expectancy": float(np.mean(s)) if len(s) else 0.0,
        "total_points": float(np.sum(s)),
    }


def _direction_vote(row: pd.Series) -> int:
    # +1 bullish, -1 bearish
    votes = 0
    votes += 1 if row["flow_delta_true"] > 0 else -1 if row["flow_delta_true"] < 0 else 0
    iv_spread = float(row["call_iv_diff"] - row["put_iv_diff"])
    votes += 1 if iv_spread > 0 else -1 if iv_spread < 0 else 0
    votes += 1 if row["total_iv_change"] > 0 else -1 if row["total_iv_change"] < 0 else 0
    return 1 if votes > 0 else -1 if votes < 0 else 0


def _optimize_asym_thresholds(tune: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for rg in REGIMES:
        for dr in ["BULLISH", "BEARISH"]:
            sub = tune[(tune["regime"] == rg) & (tune["direction"] == dr)].copy()
            if len(sub) < 10:
                out[(rg, dr)] = 65.0
                continue
            best_t = 65.0
            best_key = None
            for t in range(55, 86):
                s = sub[sub["confidence"] >= t]
                if len(s) < max(6, int(len(sub) * 0.2)):
                    continue
                m = _summ(s.rename(columns={"signed_points": "signed_points"}))
                key = (m["expectancy"], m["win_rate"], -abs(t - 70))
                if best_key is None or key > best_key:
                    best_key = key
                    best_t = float(t)
            out[(rg, dr)] = best_t
    return out


def run_v3(input_csv: Path, out_excel: Path, out_model: Path) -> Dict[str, float]:
    raw = pd.read_csv(input_csv)
    fb = engineer_features(raw)
    df = add_regime_columns(fb.frame)
    train_df, test_df = split_train_test_by_last_year(df)

    feat_cols = fb.feature_cols + ["close_ret_1d", "ma_20_slope", "abs_ret_20", "regime_strength"]
    target_col = fb.target_col

    # Split train -> model_train + tune
    tune_n = max(60, int(len(train_df) * 0.2))
    model_train = train_df.iloc[:-tune_n].copy()
    tune = train_df.iloc[-tune_n:].copy()

    # Global + regime models
    global_name, global_model = _pick_best_model(model_train, feat_cols, target_col)
    regime_models: Dict[str, CalibratedClassifierCV] = {}
    regime_model_names: Dict[str, str] = {}
    for rg in REGIMES:
        sub = model_train[model_train["regime"] == rg]
        if len(sub) < 60:
            regime_models[rg] = global_model
            regime_model_names[rg] = f"{global_name}(fallback)"
        else:
            nm, mdl = _pick_best_model(sub, feat_cols, target_col)
            regime_models[rg] = mdl
            regime_model_names[rg] = nm

    # Gap size regressor
    reg = Pipeline([("imp", SimpleImputer(strategy="median")), ("reg", GradientBoostingRegressor(random_state=42))])
    reg.fit(model_train[feat_cols], model_train["target_move_points"].astype(float).to_numpy())

    # Prepare tune predictions for threshold/meta learning.
    tune_rows = []
    for _, r in tune.iterrows():
        rg = str(r["regime"])
        mdl = regime_models.get(rg, global_model)
        p_up = float(mdl.predict_proba(pd.DataFrame([r[feat_cols].to_dict()]))[0][1])
        p_dn = 1.0 - p_up
        dr = "BULLISH" if p_up >= 0.5 else "BEARISH"
        conf = max(p_up, p_dn) * 100.0
        sp = _signed_points(dr, float(r["target_move_points"]))
        vote_dir = _direction_vote(r)
        dir_sign = 1 if dr == "BULLISH" else -1
        agreement = 1.0 if vote_dir == dir_sign else 0.5 if vote_dir == 0 else 0.0
        tune_rows.append(
            {
                "date": r["signal_date"].strftime("%Y-%m-%d"),
                "regime": rg,
                "direction": dr,
                "confidence": conf,
                "signed_points": sp,
                "regime_strength": float(r["regime_strength"]),
                "agreement": agreement,
                "pred_gap": float(reg.predict(pd.DataFrame([r[feat_cols].to_dict()]))[0]),
            }
        )
    tune_pred = pd.DataFrame(tune_rows)

    # Historical regime wr (from model_train base predictions)
    model_train_rows = []
    for _, r in model_train.iterrows():
        rg = str(r["regime"])
        mdl = regime_models.get(rg, global_model)
        p_up = float(mdl.predict_proba(pd.DataFrame([r[feat_cols].to_dict()]))[0][1])
        dr = "BULLISH" if p_up >= 0.5 else "BEARISH"
        sp = _signed_points(dr, float(r["target_move_points"]))
        model_train_rows.append({"regime": rg, "signed_points": sp})
    mtr = pd.DataFrame(model_train_rows)
    regime_wr = mtr.groupby("regime")["signed_points"].apply(lambda s: float((s >= 1.0).mean())).to_dict()

    # L1 thresholds (asymmetric by regime+direction)
    asym_th = _optimize_asym_thresholds(tune_pred)

    # L2 quality threshold tuned on tune
    tune_pred["quality_score"] = (
        (tune_pred["confidence"] / 100.0) * 40.0
        + tune_pred["regime_strength"] * 20.0
        + tune_pred["agreement"] * 20.0
        + tune_pred["regime"].map(lambda x: regime_wr.get(x, 0.5)) * 20.0
    )
    q_best = 75.0
    q_key = None
    for q in range(65, 91):
        s = tune_pred[tune_pred["quality_score"] >= q]
        if len(s) < 20:
            continue
        m = _summ(s)
        key = (m["expectancy"], m["win_rate"], -abs(q - 75))
        if q_key is None or key > q_key:
            q_key = key
            q_best = float(q)

    # L5 similarity model from model_train
    sim_cols = ["flow_delta_true", "flow_vega_true", "call_iv_diff", "put_iv_diff", "total_iv_change", "close_spot_1520", "regime_strength"]
    sim_train = model_train[sim_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sim_scaler = StandardScaler()
    sim_x = sim_scaler.fit_transform(sim_train.to_numpy())
    nn = NearestNeighbors(n_neighbors=min(25, len(model_train)), metric="euclidean")
    nn.fit(sim_x)

    # Direction-specific historical loss for L6.
    dir_loss_abs = {
        "BULLISH": abs(float(np.mean(tune_pred.loc[tune_pred["signed_points"] <= -1.0, "signed_points"]))) if (tune_pred["signed_points"] <= -1.0).any() else 80.0,
        "BEARISH": abs(float(np.mean(tune_pred.loc[tune_pred["signed_points"] <= -1.0, "signed_points"]))) if (tune_pred["signed_points"] <= -1.0).any() else 80.0,
    }
    rr_thresh = 1.2

    # Build meta-model on tune (L7)
    # Create layer flags first on tune to train meta.
    tune_meta_rows = []
    # regime history for L3 from combined historical sequence
    hist_seq = train_df[["signal_date", "regime"]].copy().sort_values("signal_date")
    regime_map = {d.strftime("%Y-%m-%d"): r for d, r in zip(hist_seq["signal_date"], hist_seq["regime"])}
    dates_sorted = sorted(regime_map.keys())
    date_to_idx = {d: i for i, d in enumerate(dates_sorted)}

    for i, r in tune_pred.iterrows():
        rg = r["regime"]
        dr = r["direction"]
        date = r["date"]
        # L1
        l1 = bool(r["confidence"] >= asym_th.get((rg, dr), 65.0))
        # L2
        l2 = bool(r["quality_score"] >= q_best)
        # L3 regime stability
        idx = date_to_idx.get(date, None)
        if idx is None or idx < 5:
            l3 = True
        else:
            last5 = [regime_map[dates_sorted[j]] for j in range(idx - 5, idx)]
            chg = sum(last5[k] != last5[k - 1] for k in range(1, len(last5)))
            if chg >= 3:
                l3 = False
            elif chg == 2 and r["confidence"] < 80:
                l3 = False
            else:
                l3 = True
        # L4 consensus
        l4 = bool(r["agreement"] >= 0.5)
        # L5 similarity
        orig_row = tune.iloc[i]
        qx = sim_scaler.transform(np.array([[orig_row[c] for c in sim_cols]], dtype=float))
        _, idxs = nn.kneighbors(qx, return_distance=True)
        neigh = model_train.iloc[idxs[0]]
        # historical correctness for this predicted direction
        hist_signed = np.where(dr == "BULLISH", neigh["target_move_points"].astype(float), -neigh["target_move_points"].astype(float))
        hist_wr = float((hist_signed >= 1.0).mean()) if len(hist_signed) else 0.5
        l5 = bool(hist_wr >= 0.55)
        # L6 risk reward
        rr = abs(float(r["pred_gap"])) / max(dir_loss_abs.get(dr, 80.0), 1.0)
        l6 = bool(rr >= rr_thresh)
        correctness = 1 if float(r["signed_points"]) >= 1.0 else 0
        tune_meta_rows.append(
            {
                "confidence": float(r["confidence"]),
                "quality": float(r["quality_score"]),
                "agreement": float(r["agreement"]),
                "regime_strength": float(r["regime_strength"]),
                "l1": int(l1),
                "l2": int(l2),
                "l3": int(l3),
                "l4": int(l4),
                "l5": int(l5),
                "l6": int(l6),
                "dir_bull": 1 if dr == "BULLISH" else 0,
                "reg_tr_up": 1 if rg == "TRENDING_UP" else 0,
                "reg_tr_dn": 1 if rg == "TRENDING_DOWN" else 0,
                "reg_hv": 1 if rg == "HIGH_VOLATILITY" else 0,
                "y": correctness,
            }
        )
    meta_df = pd.DataFrame(tune_meta_rows)
    meta_feats = [c for c in meta_df.columns if c != "y"]
    meta_model = Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    meta_model.fit(meta_df[meta_feats], meta_df["y"].astype(int))

    # Tune V3 layer cutoffs to avoid over-restrictive funnel collapsing to V1.
    tune_eval_rows = []
    for i, r in tune_pred.iterrows():
        rg = str(r["regime"])
        dr = str(r["direction"])
        date = str(r["date"])
        l1 = bool(r["confidence"] >= asym_th.get((rg, dr), 65.0))
        l2 = bool(r["quality_score"] >= q_best)
        idx = date_to_idx.get(date, None)
        if idx is None or idx < 5:
            l3 = True
        else:
            last5 = [regime_map[dates_sorted[j]] for j in range(idx - 5, idx)]
            chg = sum(last5[k] != last5[k - 1] for k in range(1, len(last5)))
            if chg >= 3:
                l3 = False
            elif chg == 2 and r["confidence"] < 80:
                l3 = False
            else:
                l3 = True

        orig_row = tune.iloc[i]
        qx = sim_scaler.transform(np.array([[orig_row[c] for c in sim_cols]], dtype=float))
        _, idxs = nn.kneighbors(qx, return_distance=True)
        neigh = model_train.iloc[idxs[0]]
        hist_signed = np.where(dr == "BULLISH", neigh["target_move_points"].astype(float), -neigh["target_move_points"].astype(float))
        hist_wr = float((hist_signed >= 1.0).mean()) if len(hist_signed) else 0.5
        rr = abs(float(r["pred_gap"])) / max(dir_loss_abs.get(dr, 80.0), 1.0)

        mrow = pd.DataFrame(
            [
                {
                    "confidence": float(r["confidence"]),
                    "quality": float(r["quality_score"]),
                    "agreement": float(r["agreement"]),
                    "regime_strength": float(r["regime_strength"]),
                    "l1": int(l1),
                    "l2": int(l2),
                    "l3": int(l3),
                    "l4": int(float(r["agreement"]) >= 0.5),
                    "l5": int(hist_wr >= 0.55),
                    "l6": int(rr >= rr_thresh),
                    "dir_bull": 1 if dr == "BULLISH" else 0,
                    "reg_tr_up": 1 if rg == "TRENDING_UP" else 0,
                    "reg_tr_dn": 1 if rg == "TRENDING_DOWN" else 0,
                    "reg_hv": 1 if rg == "HIGH_VOLATILITY" else 0,
                }
            ]
        )
        meta_p = float(meta_model.predict_proba(mrow[meta_feats])[:, 1][0])
        tune_eval_rows.append(
            {
                "l1": int(l1),
                "l2": int(l2),
                "l3": int(l3),
                "agreement": float(r["agreement"]),
                "hist_wr": hist_wr,
                "rr": rr,
                "meta_p": meta_p,
                "signed_points": float(r["signed_points"]),
            }
        )

    tune_eval = pd.DataFrame(tune_eval_rows)
    best_cfg = {"agree_min": 0.5, "hist_wr_min": 0.55, "rr_min": 1.2, "meta_min": 0.60}
    best_key = None
    min_trades_tune = max(12, int(len(tune_eval) * 0.15))
    for agree_min in [0.0, 0.5, 1.0]:
        for hist_wr_min in [0.50, 0.55, 0.60]:
            for rr_min in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
                for meta_min in [0.50, 0.55, 0.60, 0.65]:
                    msk = (
                        (tune_eval["l1"] == 1)
                        & (tune_eval["l2"] == 1)
                        & (tune_eval["l3"] == 1)
                        & (tune_eval["agreement"] >= agree_min)
                        & (tune_eval["hist_wr"] >= hist_wr_min)
                        & (tune_eval["rr"] >= rr_min)
                        & (tune_eval["meta_p"] >= meta_min)
                    )
                    sub = tune_eval.loc[msk, ["signed_points"]]
                    if len(sub) < min_trades_tune:
                        continue
                    sm = _summ(sub)
                    meet_wr = int(sm["win_rate"] >= 0.55)
                    meet_exp = int(sm["expectancy"] >= 10.0)
                    key = (meet_wr, meet_exp, sm["win_rate"], sm["expectancy"], sm["trades"])
                    if best_key is None or key > best_key:
                        best_key = key
                        best_cfg = {
                            "agree_min": float(agree_min),
                            "hist_wr_min": float(hist_wr_min),
                            "rr_min": float(rr_min),
                            "meta_min": float(meta_min),
                        }

    # Choose active funnel depth on tune data (avoid forcing all 7 layers if over-restrictive).
    tune_eval = tune_eval.copy()
    tune_eval["l4"] = (tune_eval["agreement"] >= best_cfg["agree_min"]).astype(int)
    tune_eval["l5"] = (tune_eval["hist_wr"] >= best_cfg["hist_wr_min"]).astype(int)
    tune_eval["l6"] = (tune_eval["rr"] >= best_cfg["rr_min"]).astype(int)
    tune_eval["l7"] = (tune_eval["meta_p"] >= best_cfg["meta_min"]).astype(int)
    layer_cols = ["l1", "l2", "l3", "l4", "l5", "l6", "l7"]
    active_depth = 7
    best_depth_key = None
    for depth in range(1, 8):
        msk = np.ones(len(tune_eval), dtype=bool)
        for c in layer_cols[:depth]:
            msk &= tune_eval[c].to_numpy() == 1
        sub = tune_eval.loc[msk, ["signed_points"]]
        if len(sub) < min_trades_tune:
            continue
        sm = _summ(sub)
        meet_wr = int(sm["win_rate"] >= 0.55)
        meet_exp = int(sm["expectancy"] >= 10.0)
        key = (meet_wr, meet_exp, sm["win_rate"], sm["expectancy"], sm["trades"], depth)
        if best_depth_key is None or key > best_depth_key:
            best_depth_key = key
            active_depth = depth

    # Test prediction with 7-layer funnel.
    test = test_df.sort_values("signal_date").reset_index(drop=True)
    # regime history for L3 includes train + test evolving.
    reg_hist = pd.concat([train_df[["signal_date", "regime"]], test_df[["signal_date", "regime"]]], axis=0).sort_values("signal_date").reset_index(drop=True)
    reg_dates = [d.strftime("%Y-%m-%d") for d in reg_hist["signal_date"]]
    reg_vals = list(reg_hist["regime"])
    reg_idx = {d: i for i, d in enumerate(reg_dates)}

    rows = []
    for _, r in test.iterrows():
        rg = str(r["regime"])
        mdl = regime_models.get(rg, global_model)
        p_up = float(mdl.predict_proba(pd.DataFrame([r[feat_cols].to_dict()]))[0][1])
        p_dn = 1.0 - p_up
        dr = "BULLISH" if p_up >= 0.5 else "BEARISH"
        conf = max(p_up, p_dn) * 100.0
        move = float(r["target_move_points"])
        signed = _signed_points(dr, move)
        pred_gap = float(reg.predict(pd.DataFrame([r[feat_cols].to_dict()]))[0])
        vote_dir = _direction_vote(r)
        dir_sign = 1 if dr == "BULLISH" else -1
        agreement = 1.0 if vote_dir == dir_sign else 0.5 if vote_dir == 0 else 0.0
        quality = (conf / 100.0) * 40 + float(r["regime_strength"]) * 20 + agreement * 20 + regime_wr.get(rg, 0.5) * 20

        # L1
        l1 = conf >= asym_th.get((rg, dr), 65.0)
        # L2
        l2 = quality >= q_best
        # L3
        dtxt = r["signal_date"].strftime("%Y-%m-%d")
        i = reg_idx.get(dtxt, -1)
        if i < 5:
            l3 = True
        else:
            last5 = reg_vals[i - 5 : i]
            chg = sum(last5[k] != last5[k - 1] for k in range(1, len(last5)))
            if chg >= 3:
                l3 = False
            elif chg == 2 and conf < 80:
                l3 = False
            else:
                l3 = True
        # L4
        l4 = agreement >= best_cfg["agree_min"]
        # L5
        qx = sim_scaler.transform(np.array([[r[c] for c in sim_cols]], dtype=float))
        _, idxs = nn.kneighbors(qx, return_distance=True)
        neigh = model_train.iloc[idxs[0]]
        hist_signed = np.where(dr == "BULLISH", neigh["target_move_points"].astype(float), -neigh["target_move_points"].astype(float))
        hist_wr = float((hist_signed >= 1.0).mean()) if len(hist_signed) else 0.5
        l5 = hist_wr >= best_cfg["hist_wr_min"]
        # L6
        rr = abs(pred_gap) / max(dir_loss_abs.get(dr, 80.0), 1.0)
        l6 = rr >= best_cfg["rr_min"]
        # L7 meta
        mrow = pd.DataFrame(
            [
                {
                    "confidence": conf,
                    "quality": quality,
                    "agreement": agreement,
                    "regime_strength": float(r["regime_strength"]),
                    "l1": int(l1),
                    "l2": int(l2),
                    "l3": int(l3),
                    "l4": int(l4),
                    "l5": int(l5),
                    "l6": int(l6),
                    "dir_bull": 1 if dr == "BULLISH" else 0,
                    "reg_tr_up": 1 if rg == "TRENDING_UP" else 0,
                    "reg_tr_dn": 1 if rg == "TRENDING_DOWN" else 0,
                    "reg_hv": 1 if rg == "HIGH_VOLATILITY" else 0,
                }
            ]
        )
        meta_p = float(meta_model.predict_proba(mrow[meta_feats])[:, 1][0])
        l7 = meta_p >= best_cfg["meta_min"]
        layer_flags = [bool(l1), bool(l2), bool(l3), bool(l4), bool(l5), bool(l6), bool(l7)]
        trade = bool(all(layer_flags[:active_depth]))

        rows.append(
            {
                "Date": dtxt,
                "Day": r["signal_date"].day_name()[:3],
                "Regime": rg,
                "Signal": dr,
                "Confidence": round(conf, 2),
                "Signal_Strength": _signal_strength(conf),
                "Predicted_Gap": round(pred_gap, 2),
                "Actual_Gap": round(move, 2),
                "signed_points": signed,
                "L1_Confidence": int(l1),
                "L2_Quality": int(l2),
                "L3_Stability": int(l3),
                "L4_Consensus": int(l4),
                "L5_Historical": int(l5),
                "L6_RiskReward": int(l6),
                "L7_MetaModel": int(l7),
                "Meta_Prob": round(meta_p, 4),
                "Trade_V3": trade,
                "quality_score": round(quality, 2),
                "rr_ratio": round(rr, 3),
                "hist_similarity_wr": round(hist_wr, 4),
                "asym_threshold": asym_th.get((rg, dr), 65.0),
                "cfg_agree_min": best_cfg["agree_min"],
                "cfg_hist_wr_min": best_cfg["hist_wr_min"],
                "cfg_rr_min": best_cfg["rr_min"],
                "cfg_meta_min": best_cfg["meta_min"],
                "model_used": regime_model_names.get(rg, global_name),
            }
        )
    pred = pd.DataFrame(rows)

    # baseline V1 (all days)
    v1 = pred.copy()
    v1_stats = _summ(v1)
    # V3 traded
    v3 = pred[pred["Trade_V3"]].copy()
    if v3.empty:
        v3 = pred.copy()
    v3_stats = _summ(v3)

    # Trade log
    trade_log = v3.copy()
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
            "model_used",
        ]
    ]

    # summary
    summary = pd.DataFrame(
        [
            ("Total Trading Days (Test)", int(len(pred))),
            ("Traded Days (V3)", int(len(v3))),
            ("Skipped Days (V3)", int(len(pred) - len(v3))),
            ("Win Rate (V3)", v3_stats["win_rate"]),
            ("Expectancy (V3)", v3_stats["expectancy"]),
            ("Avg Win (V3)", v3_stats["avg_win"]),
            ("Avg Loss (V3)", v3_stats["avg_loss"]),
            ("Win Rate (V1 all)", v1_stats["win_rate"]),
            ("Expectancy (V1 all)", v1_stats["expectancy"]),
        ],
        columns=["Metric", "Value"],
    )

    # monthly
    m = trade_log.copy()
    m["Month"] = pd.to_datetime(m["Date"]).dt.to_period("M").astype(str)
    monthly = m.groupby("Month", as_index=False).agg(
        Trades=("Result", "size"),
        Wins=("Result", lambda s: int((s == "WIN").sum())),
        Total_Points=("Points_Gained", "sum"),
        Avg_Points_Trade=("Points_Gained", "mean"),
    )
    monthly["Win_Rate"] = np.where(monthly["Trades"] > 0, monthly["Wins"] / monthly["Trades"], 0.0)

    # calibration
    cal_rows = []
    for lo, hi in [(90, 100), (80, 90), (70, 80), (60, 70), (50, 60), (0, 50)]:
        sub = v3[(v3["Confidence"] >= lo) & (v3["Confidence"] < hi if hi < 100 else v3["Confidence"] <= hi)]
        if sub.empty:
            cal_rows.append({"Confidence_Range": f"{lo}-{hi}", "Trades": 0, "Actual_Win_Rate": 0.0, "Calibration_Error": 0.0})
        else:
            wr = float((sub["signed_points"] >= 1.0).mean())
            implied = float(sub["Confidence"].mean() / 100.0)
            cal_rows.append({"Confidence_Range": f"{lo}-{hi}", "Trades": int(len(sub)), "Actual_Win_Rate": wr, "Calibration_Error": wr - implied})
    cal = pd.DataFrame(cal_rows)

    # feature importance (global)
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

    # regime analysis
    reg_rows = []
    for rg in REGIMES:
        all_rg = pred[pred["Regime"] == rg]
        trd_rg = v3[v3["Regime"] == rg]
        ms = _summ(trd_rg)
        reg_rows.append({"Regime": rg, "Days": int(len(all_rg)), "Traded": int(len(trd_rg)), "Skipped": int(len(all_rg) - len(trd_rg)), "Win_Rate": ms["win_rate"], "Expectancy": ms["expectancy"]})
    regime_tbl = pd.DataFrame(reg_rows)

    # viz
    viz = v3[["Date", "Confidence", "signed_points"]].copy()
    viz["Cumulative_PnL"] = viz["signed_points"].cumsum()

    # threshold impact from tune (for sheet9)
    impact_rows = []
    for t in [0, 60, 65, 70, 75, 80]:
        s = tune_pred[tune_pred["confidence"] >= t] if t > 0 else tune_pred
        mtr = _summ(s.rename(columns={"signed_points": "signed_points"}))
        impact_rows.append({"Min_Confidence": t, "Trades": int(len(s)), "Win_Rate": mtr["win_rate"], "Expectancy": mtr["expectancy"], "Annual_Points": mtr["total_points"]})
    thr_impact = pd.DataFrame(impact_rows)

    # V1 vs V3
    cmp = pd.DataFrame(
        [
            {"Metric": "Win Rate", "V1 (Current)": v1_stats["win_rate"], "V3 (Target)": v3_stats["win_rate"], "Change": v3_stats["win_rate"] - v1_stats["win_rate"]},
            {"Metric": "Expectancy", "V1 (Current)": v1_stats["expectancy"], "V3 (Target)": v3_stats["expectancy"], "Change": v3_stats["expectancy"] - v1_stats["expectancy"]},
            {"Metric": "Trades/Year", "V1 (Current)": v1_stats["trades"], "V3 (Target)": v3_stats["trades"], "Change": v3_stats["trades"] - v1_stats["trades"]},
            {"Metric": "Avg Win", "V1 (Current)": v1_stats["avg_win"], "V3 (Target)": v3_stats["avg_win"], "Change": v3_stats["avg_win"] - v1_stats["avg_win"]},
            {"Metric": "Avg Loss", "V1 (Current)": v1_stats["avg_loss"], "V3 (Target)": v3_stats["avg_loss"], "Change": v3_stats["avg_loss"] - v1_stats["avg_loss"]},
        ]
    )

    # Sheet11 filter funnel
    funnel = []
    cumul = pred.copy()
    layers = [
        ("L0_Base", None),
        ("L1_Confidence", "L1_Confidence"),
        ("L2_Quality", "L2_Quality"),
        ("L3_Stability", "L3_Stability"),
        ("L4_Consensus", "L4_Consensus"),
        ("L5_Historical", "L5_Historical"),
        ("L6_RiskReward", "L6_RiskReward"),
        ("L7_MetaModel", "L7_MetaModel"),
    ]
    reached = len(pred)
    for name, col in layers:
        if col is None:
            passed_df = cumul
        else:
            passed_df = cumul[cumul[col] == 1]
            cumul = passed_df
        sm = _summ(passed_df[["signed_points"]]) if len(passed_df) else {"win_rate": 0.0, "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
        wr = float(sm["win_rate"])
        funnel.append(
            {
                "Filter_Layer": name,
                "Days_Reached": int(reached),
                "Days_Passed": int(len(passed_df)),
                "Pass_Rate": float(len(passed_df) / reached) if reached else 0.0,
                "Win_Rate_After": wr,
                "Expectancy_After": float(sm["expectancy"]),
                "Avg_Win_After": float(sm["avg_win"]),
                "Avg_Loss_After": float(sm["avg_loss"]),
            }
        )
        reached = len(passed_df)
    funnel_df = pd.DataFrame(funnel)

    # save model bundle
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": "NIFTY_GAP_V3",
            "feature_cols": feat_cols,
            "global_model": global_model,
            "regime_models": regime_models,
            "regime_model_names": regime_model_names,
            "asymmetric_thresholds": {f"{k[0]}::{k[1]}": v for k, v in asym_th.items()},
            "quality_threshold": q_best,
            "v3_filter_config": best_cfg,
            "active_depth": int(active_depth),
            "meta_model": meta_model,
            "meta_features": meta_feats,
            "gap_regressor": reg,
        },
        out_model,
    )

    # write excel
    out_excel.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_excel, engine="openpyxl") as w:
        trade_log.to_excel(w, sheet_name="Trade Log", index=False)
        summary.to_excel(w, sheet_name="Summary Metrics", index=False)
        monthly.to_excel(w, sheet_name="Monthly Breakdown", index=False)
        cal.to_excel(w, sheet_name="Confidence Calibration", index=False)
        fi.to_excel(w, sheet_name="Feature Importance", index=False)
        regime_tbl.to_excel(w, sheet_name="Regime Analysis", index=False)
        viz.to_excel(w, sheet_name="Visualization Data", index=False)
        regime_tbl.to_excel(w, sheet_name="Sheet8_Regime_Analysis", index=False)
        thr_impact.to_excel(w, sheet_name="Sheet9_Threshold_Impact", index=False)
        cmp.to_excel(w, sheet_name="Sheet10_V1_vs_V3", index=False)
        funnel_df.to_excel(w, sheet_name="Sheet11_Filter_Funnel", index=False)

    ready = (v3_stats["win_rate"] >= 0.55) and (v3_stats["expectancy"] >= 10.0) and (v3_stats["avg_win"] >= abs(v3_stats["avg_loss"]))
    rec = "Ready to commercialize" if ready else "Needs more work"
    return {
        "v1_win_rate": v1_stats["win_rate"],
        "v1_expectancy": v1_stats["expectancy"],
        "v3_win_rate": v3_stats["win_rate"],
        "v3_expectancy": v3_stats["expectancy"],
        "v3_trades": v3_stats["trades"],
        "recommendation": rec,
        "excel": str(out_excel),
        "model": str(out_model),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="NIFTY overnight V3 optimizer (7-layer filter funnel).")
    p.add_argument("--input-csv", default="data/backtest/overnight_simple_30min_backtest_3y_details.csv")
    p.add_argument("--out-excel", default="nifty_gap_backtest_results_v3.xlsx")
    p.add_argument("--out-model", default="trained_model_v3.pkl")
    args = p.parse_args()

    stats = run_v3(Path(args.input_csv), Path(args.out_excel), Path(args.out_model))
    print(
        f"V1 win_rate={stats['v1_win_rate']:.2%} expectancy={stats['v1_expectancy']:+.2f} | "
        f"V3 win_rate={stats['v3_win_rate']:.2%} expectancy={stats['v3_expectancy']:+.2f} "
        f"trades={int(stats['v3_trades'])}"
    )
    print(f"recommendation={stats['recommendation']}")
    print(f"excel={stats['excel']}")
    print(f"model={stats['model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
