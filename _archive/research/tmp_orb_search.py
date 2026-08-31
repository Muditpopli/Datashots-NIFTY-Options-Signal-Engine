import pandas as pd
import numpy as np
from datetime import timedelta

p = "data/backtest/eth_ny_1y_rebuild.csv"
df = pd.read_csv(p)
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
df = df.dropna(subset=["timestamp_utc", "spot", "entry_date_ist"]).copy()
df["ts_ist"] = df["timestamp_utc"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
df["day"] = pd.to_datetime(df["entry_date_ist"]).dt.date

frames = {}
for d, g in df.groupby("day", sort=True):
    g = g.sort_values("ts_ist").copy()
    if len(g) < 40:
        continue
    sess_open = g["ts_ist"].min() + timedelta(minutes=10)
    g["m"] = (g["ts_ist"] - sess_open).dt.total_seconds() / 60.0
    frames[d] = g

results = []

range_ends = [10, 15, 20]
entry_end_list = [45, 60]
buffers = [0.0, 0.03, 0.06, 0.10]
stop_atr_mults = [0.6, 0.8, 1.0, 1.2]
rrs = [1.2, 1.5, 1.8, 2.0]
timeouts = [60, 75, 85]
pre_move_lows = [0.0, 0.02, 0.05]
pre_move_highs = [0.18, 0.25, 0.35]

for re in range_ends:
    for entry_end in entry_end_list:
        for buf in buffers:
            for sm in stop_atr_mults:
                for rr in rrs:
                    for to in timeouts:
                        for pmin in pre_move_lows:
                            for pmax in pre_move_highs:
                                trades = []
                                for _, g in frames.items():
                                    pre = g[(g["m"] >= -10) & (g["m"] <= -5)]
                                    if len(pre) < 2:
                                        continue
                                    pre_move = abs((float(pre.iloc[-1]["spot"]) / float(pre.iloc[0]["spot"]) - 1) * 100)
                                    if not (pmin <= pre_move <= pmax):
                                        continue

                                    rg = g[(g["m"] >= 0) & (g["m"] < re)]
                                    if len(rg) < 5:
                                        continue
                                    hi = float(rg["spot"].max())
                                    lo = float(rg["spot"].min())

                                    run = g[(g["m"] >= re) & (g["m"] <= entry_end)]
                                    if len(run) < 2:
                                        continue
                                    entry_row = None
                                    side = 0
                                    up_level = hi * (1 + buf / 100)
                                    dn_level = lo * (1 - buf / 100)
                                    for _, row in run.iterrows():
                                        px = float(row["spot"])
                                        if px > up_level:
                                            entry_row = row
                                            side = 1
                                            break
                                        if px < dn_level:
                                            entry_row = row
                                            side = -1
                                            break
                                    if entry_row is None:
                                        continue

                                    ent_t = entry_row["ts_ist"]
                                    ent_px = float(entry_row["spot"])
                                    rg_ret = np.log(rg["spot"] / rg["spot"].shift(1)).dropna()
                                    if len(rg_ret) < 3:
                                        continue
                                    vol = float(rg_ret.std())
                                    atr_pts = max(ent_px * vol * 2.0, 0.5)
                                    stop_dist = max(atr_pts * sm, 0.5)
                                    if side == 1:
                                        sl = ent_px - stop_dist
                                        tp = ent_px + stop_dist * rr
                                    else:
                                        sl = ent_px + stop_dist
                                        tp = ent_px - stop_dist * rr

                                    path = g[(g["ts_ist"] > ent_t) & (g["m"] <= to)]
                                    if len(path) == 0:
                                        continue

                                    exit_px = float(path.iloc[-1]["spot"])
                                    for _, r in path.iterrows():
                                        px = float(r["spot"])
                                        if side == 1:
                                            if px <= sl:
                                                exit_px = sl
                                                break
                                            if px >= tp:
                                                exit_px = tp
                                                break
                                        else:
                                            if px >= sl:
                                                exit_px = sl
                                                break
                                            if px <= tp:
                                                exit_px = tp
                                                break

                                    pnl = (exit_px - ent_px) if side == 1 else (ent_px - exit_px)
                                    trades.append(pnl)

                                if len(trades) < 40:
                                    continue
                                arr = np.array(trades, dtype=float)
                                wins = arr[arr > 0]
                                losses = arr[arr < 0]
                                if len(wins) == 0 or len(losses) == 0:
                                    continue
                                wr = (arr > 0).mean()
                                avg_win = wins.mean()
                                avg_loss = abs(losses.mean())
                                pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan
                                exp = arr.mean()
                                if avg_win > avg_loss and wr >= 0.50 and exp > 0:
                                    results.append(
                                        {
                                            "range_end": re,
                                            "entry_end": entry_end,
                                            "buffer_pct": buf,
                                            "stop_mult": sm,
                                            "rr": rr,
                                            "timeout": to,
                                            "pre_move_min": pmin,
                                            "pre_move_max": pmax,
                                            "trades": len(arr),
                                            "win_rate": wr,
                                            "avg_win": avg_win,
                                            "avg_loss_abs": avg_loss,
                                            "pf": pf,
                                            "expectancy_pts": exp,
                                        }
                                    )

out = pd.DataFrame(results)
if out.empty:
    print("No config met strict criteria.")
else:
    out = out.sort_values(["expectancy_pts", "win_rate", "pf"], ascending=False)
    print("Found", len(out), "configs meeting criteria")
    print(out.head(20).round(4).to_string(index=False))
    out.to_csv("data/backtest/outputs/eth_perp_orb_search_good_configs.csv", index=False)
    print("saved data/backtest/outputs/eth_perp_orb_search_good_configs.csv")
