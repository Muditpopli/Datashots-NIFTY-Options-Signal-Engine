from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datashots_gap.data_loader import RollingOptionsDataLoader
from datashots_gap.greek_calculator import calculate_greek_changes


@dataclass
class SplitResult:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def _next_day_map(days: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i in range(len(days) - 1):
        out[days[i]] = days[i + 1]
    return out


def _pcr(chain: Dict) -> float:
    rows = chain.get("strikes", [])
    if not rows:
        return 0.0
    p = 0.0
    c = 0.0
    for r in rows:
        p += float(r["pe"].get("oi", 0.0) or 0.0)
        c += float(r["ce"].get("oi", 0.0) or 0.0)
    if c <= 0:
        return 0.0
    return p / c


def _day_features(loader: RollingOptionsDataLoader, day: str, index: str) -> Optional[Dict[str, float]]:
    exp = loader.choose_expiry(day, phase="close_1515")
    if not exp:
        return None
    flag, code = exp
    o = loader.build_chain(day, index, flag, code, "open_0915")
    p = loader.build_chain(day, index, flag, code, "preclose_1445")
    c = loader.build_chain(day, index, flag, code, "close_1515")
    if not o or not p or not c:
        return None

    day_ch = calculate_greek_changes(o, c)
    l30_ch = calculate_greek_changes(p, c)

    d = {
        "flag": flag,
        "code": code,
        "cmp": float(c["spot"]),
        "spot_day": float(c["spot"]) - float(o["spot"]),
        "spot_l30": float(c["spot"]) - float(p["spot"]),
        "oi_day": float(day_ch["put_oi"] - day_ch["call_oi"]),
        "oi_l30": float(l30_ch["put_oi"] - l30_ch["call_oi"]),
        "pcr_day": float(_pcr(c) - _pcr(o)),
        "pcr_l30": float(_pcr(c) - _pcr(p)),
    }
    return d


def _signed_points(tag: str, gap: float) -> float:
    if tag == "BULLISH":
        return gap if gap >= 1.0 else -abs(gap)
    return abs(gap) if gap <= -1.0 else -abs(gap)


def _metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "expectancy": 0.0}
    pts = df["points"].astype(float).to_numpy()
    wins = int((pts >= 1.0).sum())
    losses = int((pts < 1.0).sum())
    return {
        "trades": int(len(df)),
        "wins": wins,
        "losses": losses,
        "win_rate": float(wins / len(df)),
        "expectancy": float(np.mean(pts)),
    }


def _split(df: pd.DataFrame) -> SplitResult:
    n = len(df)
    a = int(n * 0.6)
    b = int(n * 0.8)
    return SplitResult(train=df.iloc[:a].copy(), valid=df.iloc[a:b].copy(), test=df.iloc[b:].copy())


def _build_dataset(
    cache_root: Path,
    start_date: str,
    end_date: str,
    primary_index: str,
    confirm_index: str,
) -> pd.DataFrame:
    feat_by_idx: Dict[str, Dict[str, Dict[str, float]]] = {primary_index: {}, confirm_index: {}}
    next_open_by_idx: Dict[str, Dict[str, float]] = {primary_index: {}, confirm_index: {}}
    all_days: Optional[List[str]] = None

    for idx in [primary_index, confirm_index]:
        ld = RollingOptionsDataLoader(cache_root=cache_root)
        ld.build_samples(index=idx, start_date=start_date, end_date=end_date)
        days = ld.trading_days()
        if all_days is None:
            all_days = days
        nxt = _next_day_map(days)
        for d in days:
            f = _day_features(ld, d, idx)
            if not f:
                continue
            feat_by_idx[idx][d] = f
            nd = nxt.get(d)
            if nd:
                no = ld.spot_at(nd, idx, "next_open_0925", flag=f["flag"], code=f["code"])
                if no is not None:
                    next_open_by_idx[idx][d] = float(no)

    if not all_days:
        return pd.DataFrame()

    rows: List[Dict] = []
    for d in all_days:
        nf = feat_by_idx[primary_index].get(d)
        bf = feat_by_idx[confirm_index].get(d)
        no = next_open_by_idx[primary_index].get(d)
        if not nf or not bf or no is None:
            continue
        gap = float(no) - float(nf["cmp"])
        rows.append(
            {
                "signal_date": d,
                "cmp": float(nf["cmp"]),
                "next_open": float(no),
                "actual_gap": float(gap),
                # primary features
                "n_spot_day": float(nf["spot_day"]),
                "n_spot_l30": float(nf["spot_l30"]),
                "n_oi_day": float(nf["oi_day"]),
                "n_oi_l30": float(nf["oi_l30"]),
                "n_pcr_day": float(nf["pcr_day"]),
                "n_pcr_l30": float(nf["pcr_l30"]),
                # confirm features
                "b_spot_day": float(bf["spot_day"]),
                "b_spot_l30": float(bf["spot_l30"]),
                "b_oi_day": float(bf["oi_day"]),
                "b_oi_l30": float(bf["oi_l30"]),
                "b_pcr_day": float(bf["pcr_day"]),
                "b_pcr_l30": float(bf["pcr_l30"]),
            }
        )
    out = pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)
    return out


def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feat_cols = [
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
    x = df[feat_cols].copy()
    # Divergence informative interactions.
    x["spot_align_l30"] = np.sign(df["n_spot_l30"]) * np.sign(df["b_spot_l30"])
    x["oi_align_l30"] = np.sign(df["n_oi_l30"]) * np.sign(df["b_oi_l30"])
    x["pcr_align_l30"] = np.sign(df["n_pcr_l30"]) * np.sign(df["b_pcr_l30"])
    x["spot_div_l30"] = df["n_spot_l30"] - df["b_spot_l30"]
    x["oi_div_l30"] = df["n_oi_l30"] - df["b_oi_l30"]
    x["pcr_div_l30"] = df["n_pcr_l30"] - df["b_pcr_l30"]
    return x, list(x.columns)


def _standardize_fit(x: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0).to_numpy(dtype=float)
    sd = x.std(axis=0).replace(0.0, 1.0).to_numpy(dtype=float)
    return mu, sd


def _score(x: pd.DataFrame, mu: np.ndarray, sd: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    z = (x.to_numpy(dtype=float) - mu) / sd
    return z @ w + b


def _evaluate(df: pd.DataFrame, scores: np.ndarray) -> Dict[str, float]:
    tag = np.where(scores >= 0.0, "BULLISH", "BEARISH")
    tag = _apply_squeeze_override(tag, df)
    pts = np.array([_signed_points(t, g) for t, g in zip(tag, df["actual_gap"].to_numpy(dtype=float))], dtype=float)
    wins = (pts >= 1.0).sum()
    wr = float(wins / len(df)) if len(df) else 0.0
    exp = float(np.mean(pts)) if len(df) else 0.0
    return {"win_rate": wr, "expectancy": exp, "trades": int(len(df))}


def _objective(m: Dict[str, float]) -> float:
    # Strong push for expectancy, soft floor on win-rate.
    wr_penalty = max(0.0, 0.55 - m["win_rate"]) * 200.0
    return m["expectancy"] - wr_penalty


def _apply_squeeze_override(tags: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    out = np.array(tags, copy=True)
    # Bull squeeze override: strong price continuation + positive OI build on both indices.
    bull = (
        (df["n_spot_day"] > 200.0)
        & (df["n_spot_l30"] > 100.0)
        & (df["b_spot_l30"] > 150.0)
        & (df["n_oi_day"] > 0.0)
        & (df["b_oi_day"] > 0.0)
        & (df["n_oi_l30"] > 0.0)
    )
    # Symmetric bear squeeze override.
    bear = (
        (df["n_spot_day"] < -200.0)
        & (df["n_spot_l30"] < -100.0)
        & (df["b_spot_l30"] < -150.0)
        & (df["n_oi_day"] < 0.0)
        & (df["b_oi_day"] < 0.0)
        & (df["n_oi_l30"] < 0.0)
    )
    out[bull.to_numpy()] = "BULLISH"
    out[bear.to_numpy()] = "BEARISH"
    return out


def optimize_and_lock(
    df: pd.DataFrame,
    seed: int = 42,
    iters: int = 8000,
) -> Dict:
    sp = _split(df)
    x_all, cols = _prepare_features(df)
    x_train, _ = _prepare_features(sp.train)
    x_valid, _ = _prepare_features(sp.valid)
    x_test, _ = _prepare_features(sp.test)

    mu, sd = _standardize_fit(x_train)
    rng = np.random.default_rng(seed)
    dim = len(cols)

    # Prior: give more initial importance to NIFTY l30 spot + oi.
    prior = np.zeros(dim, dtype=float)
    col_idx = {c: i for i, c in enumerate(cols)}
    prior[col_idx["n_spot_l30"]] = 1.2
    prior[col_idx["n_oi_l30"]] = 0.8
    prior[col_idx["spot_align_l30"]] = 0.4
    prior[col_idx["oi_align_l30"]] = 0.3

    best = None
    for _ in range(iters):
        w = prior + rng.normal(0.0, 1.0, size=dim)
        b = float(rng.normal(0.0, 0.25))
        tr = _evaluate(sp.train, _score(x_train, mu, sd, w, b))
        va = _evaluate(sp.valid, _score(x_valid, mu, sd, w, b))
        score = 0.35 * _objective(tr) + 0.65 * _objective(va)
        if best is None or score > best["opt_score"]:
            best = {"w": w, "b": b, "train": tr, "valid": va, "opt_score": score}

    assert best is not None

    te = _evaluate(sp.test, _score(x_test, mu, sd, best["w"], best["b"]))
    full = _evaluate(df, _score(x_all, mu, sd, best["w"], best["b"]))

    lock = {
        "features": cols,
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "weights": best["w"].tolist(),
        "bias": float(best["b"]),
        "train_metrics": best["train"],
        "valid_metrics": best["valid"],
        "test_metrics": te,
        "full_metrics": full,
    }
    return lock


def run_locked(df: pd.DataFrame, lock: Dict) -> pd.DataFrame:
    x, _ = _prepare_features(df)
    mu = np.array(lock["mu"], dtype=float)
    sd = np.array(lock["sd"], dtype=float)
    w = np.array(lock["weights"], dtype=float)
    b = float(lock["bias"])
    scores = _score(x, mu, sd, w, b)
    raw_tags = np.where(scores >= 0.0, "BULLISH", "BEARISH")
    tags = _apply_squeeze_override(raw_tags, df)
    pts = np.array([_signed_points(t, g) for t, g in zip(tags, df["actual_gap"].to_numpy(dtype=float))], dtype=float)

    out = df.copy()
    out["score"] = scores
    out["raw_tag"] = raw_tags
    out["tag"] = tags
    out["override_applied"] = out["raw_tag"] != out["tag"]
    out["points"] = np.round(pts, 2)
    out["result"] = np.where(pts >= 1.0, "WIN", "LOSS")
    out["trade"] = True
    out["confidence"] = np.clip(np.abs(scores) * 10.0, 0.0, 100.0)
    out["cumulative_points"] = out["points"].cumsum()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Final attempt: daily bull/bear lock with BN-aware features.")
    p.add_argument("--cache-root", default="data/backtest/cache/rolling_options")
    p.add_argument("--start-date", default="2025-01-01")
    p.add_argument("--end-date", default="2026-02-28")
    p.add_argument("--primary-index", default="NIFTY")
    p.add_argument("--confirm-index", default="BANKNIFTY")
    p.add_argument("--iters", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-lock", default="data/backtest/outputs/final_daily_tag_lock.json")
    p.add_argument("--out-csv", default="data/backtest/outputs/final_daily_tag_lock_predictions.csv")
    p.add_argument("--out-summary", default="data/backtest/outputs/final_daily_tag_lock_summary.json")
    args = p.parse_args()

    cache_root = Path(args.cache_root)
    df = _build_dataset(
        cache_root=cache_root,
        start_date=args.start_date,
        end_date=args.end_date,
        primary_index=args.primary_index.upper(),
        confirm_index=args.confirm_index.upper(),
    )
    if df.empty:
        raise RuntimeError("No dataset rows built. Check cache coverage.")

    lock = optimize_and_lock(df, seed=int(args.seed), iters=int(args.iters))
    pred = run_locked(df, lock)

    out_lock = Path(args.out_lock)
    out_lock.parent.mkdir(parents=True, exist_ok=True)
    out_lock.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    out_csv = Path(args.out_csv)
    pred.to_csv(out_csv, index=False)

    full_m = _metrics(pred)
    by_year = {}
    pred["year"] = pd.to_datetime(pred["signal_date"]).dt.year
    for y, g in pred.groupby("year"):
        by_year[str(y)] = _metrics(g)

    summary = {
        "full": full_m,
        "by_year": by_year,
        "train": lock["train_metrics"],
        "valid": lock["valid_metrics"],
        "test": lock["test_metrics"],
        "meta": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "primary_index": args.primary_index.upper(),
            "confirm_index": args.confirm_index.upper(),
            "iters": int(args.iters),
            "seed": int(args.seed),
            "rows": int(len(pred)),
            "daily_tagging": True,
            "skip_allowed": False,
            "squeeze_override_enabled": True,
            "override_count": int(pred["override_applied"].sum()),
        },
    }
    out_summary = Path(args.out_summary)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Done | rows={len(pred)} win_rate={full_m['win_rate']:.2%} "
        f"expectancy={full_m['expectancy']:+.2f}"
    )
    print(f"lock={out_lock}")
    print(f"csv={out_csv}")
    print(f"summary={out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
