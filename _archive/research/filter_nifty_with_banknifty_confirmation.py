from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _signed_points(predicted: str, actual_gap: float) -> float:
    if predicted == "BULLISH":
        return float(actual_gap) if actual_gap >= 1.0 else -abs(float(actual_gap))
    if predicted == "BEARISH":
        return abs(float(actual_gap)) if actual_gap <= -1.0 else -abs(float(actual_gap))
    return 0.0


def _tilt_from_strength(strength: float, threshold: float) -> str:
    if strength > threshold:
        return "BULLISH"
    if strength < -threshold:
        return "BEARISH"
    return "SIDEWAYS"


def _flow_growing(tilt: str, d_pos: float, d_strength: float, flow_threshold: float) -> bool:
    if tilt == "BULLISH":
        return d_pos > flow_threshold and d_strength > flow_threshold
    if tilt == "BEARISH":
        return d_pos < -flow_threshold and d_strength < -flow_threshold
    return False


def _metric(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
        }
    pts = df["points"].astype(float).to_numpy()
    wins = pts[pts >= 1.0]
    losses = np.abs(pts[pts < 1.0])
    return {
        "trades": int(len(pts)),
        "wins": int((pts >= 1.0).sum()),
        "losses": int((pts < 1.0).sum()),
        "win_rate": float((pts >= 1.0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "expectancy": float(pts.mean()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Trade NIFTY only when BANKNIFTY confirms same direction.")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--primary-index", default="NIFTY")
    p.add_argument("--confirm-index", default="BANKNIFTY")
    p.add_argument(
        "--signal-mode",
        choices=["sentiment", "strength_tilt"],
        default="strength_tilt",
        help="Use model sentiment or strength-sign tilt for alignment.",
    )
    p.add_argument(
        "--strength-threshold",
        type=float,
        default=0.0,
        help="Minimum abs(strength) to treat as bullish/bearish in strength_tilt mode.",
    )
    p.add_argument(
        "--require-flow-growth",
        action="store_true",
        help="Require d_pos and d_strength to grow in the same direction as tilt for both primary and confirm index.",
    )
    p.add_argument(
        "--flow-threshold",
        type=float,
        default=0.0,
        help="Minimum signed change threshold for d_pos and d_strength when --require-flow-growth is used.",
    )
    p.add_argument(
        "--flow-on",
        choices=["primary", "both"],
        default="both",
        help="Apply flow-growth gate on primary index only, or both primary and confirm index.",
    )
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-summary", required=True)
    args = p.parse_args()

    src = Path(args.input_csv)
    if not src.exists():
        raise FileNotFoundError(f"Input CSV not found: {src}")

    df = pd.read_csv(src)
    req_cols = {"signal_date", "index", "actual_gap", "cmp", "next_open", "conviction", "score"}
    if args.signal_mode == "sentiment":
        req_cols = req_cols | {"sentiment"}
    if args.signal_mode == "strength_tilt":
        req_cols = req_cols | {"strength"}
    if args.require_flow_growth:
        req_cols = req_cols | {"d_pos", "d_strength"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows: List[Dict] = []
    for day, g in df.groupby("signal_date"):
        p_row = g[g["index"] == args.primary_index]
        c_row = g[g["index"] == args.confirm_index]

        if p_row.empty:
            continue

        p0 = p_row.iloc[0]
        if args.signal_mode == "strength_tilt":
            p_sent = _tilt_from_strength(float(p0["strength"]), float(args.strength_threshold))
        else:
            p_sent = str(p0["sentiment"]).upper()

        # Only directional primary views are considered.
        if p_sent not in {"BULLISH", "BEARISH"}:
            rows.append(
                {
                    "signal_date": day,
                    "trade": False,
                    "reason": "primary_sideways",
                    "primary_sentiment": p_sent,
                    "confirm_sentiment": None,
                    "points": 0.0,
                    "result": "NO_TRADE",
                    "actual_gap": float(p0["actual_gap"]),
                    "cmp": float(p0["cmp"]),
                    "next_open": float(p0["next_open"]),
                    "conviction": str(p0["conviction"]),
                    "score": float(p0["score"]),
                }
            )
            continue

        if c_row.empty:
            rows.append(
                {
                    "signal_date": day,
                    "trade": False,
                    "reason": "confirm_missing",
                    "primary_sentiment": p_sent,
                    "confirm_sentiment": None,
                    "points": 0.0,
                    "result": "NO_TRADE",
                    "actual_gap": float(p0["actual_gap"]),
                    "cmp": float(p0["cmp"]),
                    "next_open": float(p0["next_open"]),
                    "conviction": str(p0["conviction"]),
                    "score": float(p0["score"]),
                }
            )
            continue

        c0 = c_row.iloc[0]
        if args.signal_mode == "strength_tilt":
            c_sent = _tilt_from_strength(float(c0["strength"]), float(args.strength_threshold))
        else:
            c_sent = str(c0["sentiment"]).upper()
        aligned = c_sent == p_sent
        if not aligned:
            rows.append(
                {
                    "signal_date": day,
                    "trade": False,
                    "reason": "misaligned",
                    "primary_sentiment": p_sent,
                    "confirm_sentiment": c_sent,
                    "points": 0.0,
                    "result": "NO_TRADE",
                    "actual_gap": float(p0["actual_gap"]),
                    "cmp": float(p0["cmp"]),
                    "next_open": float(p0["next_open"]),
                    "conviction": str(p0["conviction"]),
                    "score": float(p0["score"]),
                }
            )
            continue

        if args.require_flow_growth:
            p_flow = _flow_growing(
                p_sent,
                float(p0["d_pos"]),
                float(p0["d_strength"]),
                float(args.flow_threshold),
            )
            if args.flow_on == "both":
                c_flow = _flow_growing(
                    c_sent,
                    float(c0["d_pos"]),
                    float(c0["d_strength"]),
                    float(args.flow_threshold),
                )
            else:
                c_flow = True

            if not (p_flow and c_flow):
                rows.append(
                    {
                        "signal_date": day,
                        "trade": False,
                        "reason": "flow_not_growing",
                        "primary_sentiment": p_sent,
                        "confirm_sentiment": c_sent,
                        "points": 0.0,
                        "result": "NO_TRADE",
                        "actual_gap": float(p0["actual_gap"]),
                        "cmp": float(p0["cmp"]),
                        "next_open": float(p0["next_open"]),
                        "conviction": str(p0["conviction"]),
                        "score": float(p0["score"]),
                        "primary_strength": float(p0["strength"]) if "strength" in p0 else None,
                        "confirm_strength": float(c0["strength"]) if "strength" in c0 else None,
                    }
                )
                continue

        pts = _signed_points(p_sent, float(p0["actual_gap"]))
        rows.append(
            {
                "signal_date": day,
                "trade": True,
                "reason": "aligned",
                "primary_sentiment": p_sent,
                "confirm_sentiment": c_sent,
                "points": pts,
                "result": "WIN" if pts >= 1.0 else "LOSS",
                "actual_gap": float(p0["actual_gap"]),
                "cmp": float(p0["cmp"]),
                "next_open": float(p0["next_open"]),
                "conviction": str(p0["conviction"]),
                "score": float(p0["score"]),
                "primary_strength": float(p0["strength"]) if "strength" in p0 else None,
                "confirm_strength": float(c0["strength"]) if "strength" in c0 else None,
            }
        )

    out = pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    traded = out[out["trade"] == True].copy()
    summary = {
        "overall": _metric(traded),
        "meta": {
            "input_csv": str(src),
            "primary_index": args.primary_index,
            "confirm_index": args.confirm_index,
            "signal_mode": args.signal_mode,
            "strength_threshold": float(args.strength_threshold),
            "require_flow_growth": bool(args.require_flow_growth),
            "flow_threshold": float(args.flow_threshold),
            "flow_on": str(args.flow_on),
            "total_days": int(len(out)),
            "traded_days": int(len(traded)),
            "skipped_days": int((out["trade"] == False).sum()),
            "skip_reason_counts": out[out["trade"] == False]["reason"].value_counts().to_dict(),
        },
    }
    Path(args.out_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    o = summary["overall"]
    print(
        f"Done | trades={o['trades']} wins={o['wins']} losses={o['losses']} "
        f"win_rate={o['win_rate']:.2%} expectancy={o['expectancy']:+.2f}"
    )
    print(f"csv={out_path}")
    print(f"summary={args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
