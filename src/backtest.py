"""Backtest the simulator against real undercut situations.

Event definition (per race):
  * Driver A pits on lap k, running directly behind rival B (gap <= max_gap s,
    measured at the end of lap k-1).
  * B pits 1..max_delay laps later (so we know the "cover delay").
  * Outcome: is A still ahead of B two laps after B's stop?
  * Model: P(A ahead) from the Monte Carlo simulator, using parameters fitted
    on that race. Events near Safety Car / red flag laps are excluded.

Metrics: accuracy, Brier score, Brier skill vs always predicting the base rate.
CAVEAT: parameters are fitted on the same race (in-sample, track-specific).
"""
import sqlite3
from dataclasses import replace

import numpy as np
import pandas as pd

from ingest import DB_PATH
from simulator import simulate_undercut
from tyre_model import RACE_ID, build_params, load_clean, fit_degradation


def load_laps(con, race_id=None) -> pd.DataFrame:
    extra = f"WHERE {RACE_ID} = ?" if race_id else ""
    args = (race_id,) if race_id else ()
    return pd.read_sql(f"""
        SELECT {RACE_ID} AS race_id, driver, lap_number, compound, tyre_life,
               is_pit_in, track_status, position, lap_end_s
        FROM laps {extra}""", con, params=args)


def find_events(L: pd.DataFrame, max_gap=4.0, max_delay=3) -> pd.DataFrame:
    L = L.copy()
    L["track_status"] = L["track_status"].fillna("1").astype(str)
    neutral = set(L.loc[L["track_status"].str.contains("[4567]"), "lap_number"])
    pos = L.pivot(index="lap_number", columns="driver", values="position")
    end = L.pivot(index="lap_number", columns="driver", values="lap_end_s")
    age = L.pivot(index="lap_number", columns="driver", values="tyre_life")
    comp = L.pivot(index="lap_number", columns="driver", values="compound")
    pits = L[L["is_pit_in"] == 1].groupby("driver")["lap_number"].apply(sorted).to_dict()

    events = []
    for a, a_pits in pits.items():
        for k in a_pits:
            if k < 4 or (k - 1) in a_pits or (k - 1) not in pos.index:
                continue
            pa = pos.at[k - 1, a]
            if pd.isna(pa):
                continue
            ahead = pos.loc[k - 1][pos.loc[k - 1] == pa - 1].index
            if len(ahead) != 1:
                continue
            b = ahead[0]
            if (k - 1) in pits.get(b, []):
                continue
            gap = end.at[k - 1, a] - end.at[k - 1, b]
            if not (0 < gap <= max_gap):
                continue
            kb = [x for x in pits.get(b, []) if k < x <= k + max_delay]
            if not kb:
                continue
            kb = kb[0]
            d, e = kb - k, kb + 2
            if e not in pos.index or any(l in neutral for l in range(k - 2, e + 1)):
                continue
            if pd.isna(pos.at[e, a]) or pd.isna(pos.at[e, b]):
                continue
            events.append(dict(
                race_id=L["race_id"].iloc[0], a=a, b=b, lap=k, gap=gap, delay=d,
                age_a=age.at[k - 1, a], age_b=age.at[k - 1, b],
                comp_old=comp.at[k - 1, a],
                comp_new=comp.at[k + 1, a] if (k + 1) in comp.index else None,
                outcome=int(pos.at[e, a] < pos.at[e, b])))
    return pd.DataFrame(events)


def predict(ev, base, deg, n_sims=4000, seed=0) -> float:
    p = replace(
        base, gap_ahead=float(ev["gap"]), laps_after=2,
        start_age=int(round(ev["age_a"])), rival_start_age=int(round(ev["age_b"])),
        rival_cover_delay=int(ev["delay"]),
        deg_old=max(deg.get(ev["comp_old"], {}).get("deg", base.deg_old), 0.01),
        deg_new=max(deg.get(ev["comp_new"], {}).get("deg", base.deg_new), 0.01))
    return simulate_undercut(p, pit_lap=1, n=n_sims, seed=seed)["p_gain"]


def summarize(ev: pd.DataFrame):
    if ev.empty:
        return None
    y, p = ev["outcome"].to_numpy(), ev["p_model"].to_numpy()
    base = y.mean()
    brier, brier_base = float(((p - y) ** 2).mean()), float(((base - y) ** 2).mean())
    return {"n_events": int(len(ev)), "base_rate": float(base),
            "accuracy": float(((p > 0.5) == y).mean()),
            "brier": brier, "brier_baseline": brier_base,
            "brier_skill": float(1 - brier / brier_base) if brier_base > 0 else float("nan")}


def run(con, race_ids=None, n_sims=4000):
    if race_ids is None:
        race_ids = [r[0] for r in con.execute(f"SELECT DISTINCT {RACE_ID} FROM laps ORDER BY 1")]
    frames = []
    for rid in race_ids:
        ev = find_events(load_laps(con, rid))
        if ev.empty:
            continue
        base, info = build_params(con, rid)
        deg = info["degradation"]
        ev["p_model"] = [predict(r, base, deg, n_sims, seed=i) for i, r in ev.iterrows()]
        frames.append(ev)
    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return events, summarize(events)


if __name__ == "__main__":
    with sqlite3.connect(DB_PATH) as con:
        events, summary = run(con)
    print(events.head(10).to_string() if len(events) else "No events found")
    print(summary)
