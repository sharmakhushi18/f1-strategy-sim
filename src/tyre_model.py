"""Calibrate simulator parameters from lap data in the database.

Fits: tyre degradation (per compound, fuel-corrected), pit-stop time loss,
and Safety Car probability per lap.
"""
import sqlite3

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ingest import DB_PATH
from simulator import Params

RACE_ID = "year || '_' || race"


def _where(race_id):
    return (f"AND {RACE_ID} = ?", (race_id,)) if race_id else ("", ())


def load_clean(con, race_id=None) -> pd.DataFrame:
    """Representative racing laps only: no lap 1, in/out laps, SC/VSC/yellow."""
    extra, args = _where(race_id)
    df = pd.read_sql(f"""
        SELECT {RACE_ID} AS race_id, driver, lap_number, lap_time_s, compound, tyre_life
        FROM laps
        WHERE is_pit_in=0 AND is_pit_out=0 AND track_status='1' AND lap_number>1
          AND lap_time_s IS NOT NULL AND tyre_life IS NOT NULL
          AND compound IS NOT NULL {extra}""", con, params=args)
    df["rd"] = df["race_id"] + "_" + df["driver"]
    med = df.groupby("rd")["lap_time_s"].transform("median")
    return df[df["lap_time_s"] <= 1.07 * med].reset_index(drop=True)   # drop outliers


def fit_degradation(df: pd.DataFrame, min_laps=40) -> dict:
    """One joint regression over all compounds:

        lap_time ~ compound + compound x tyre_life + lap_number + driver effect

    Within a single stint lap_number and tyre_life move together, so fitting
    compounds separately cannot separate fuel burn from tyre wear. Pooling
    compounds (drivers pit on different laps) identifies both.
    Returns {compound: {deg (s/lap), se, n}, "_fuel": s/lap}.
    """
    comps = [c for c, g in df.groupby("compound")
             if len(g) >= min_laps and g["tyre_life"].nunique() >= 5]
    if not comps:
        return {}
    d = df[df["compound"].isin(comps)].copy()
    terms = []
    for c in comps:
        d[f"age_{c}"] = np.where(d["compound"] == c, d["tyre_life"], 0)
        terms.append(f"age_{c}")
    m = smf.ols("lap_time_s ~ C(compound) + lap_number + C(rd) + " + " + ".join(terms), d).fit()
    out = {c: {"deg": float(m.params[f"age_{c}"]), "se": float(m.bse[f"age_{c}"]),
               "n": int((d["compound"] == c).sum())} for c in comps}
    out["_fuel"] = float(m.params["lap_number"])
    return out


def estimate_pit_loss(con, race_id=None):
    """(median, robust std) of pit loss = in-lap + out-lap minus 2x the driver's clean median.

Median and MAD (scaled by 1.4826) are used instead of mean/std so that a few
odd stops (red flags, timing glitches) don't blow up the spread. Slow stops are
modelled separately in the simulator (slow_stop_prob).
"""
    extra, args = _where(race_id)
    df = pd.read_sql(f"""
        SELECT {RACE_ID} AS race_id, driver, lap_number, lap_time_s,
               is_pit_in, is_pit_out, track_status
        FROM laps WHERE 1=1 {extra}""", con, params=args)
    df = df.sort_values(["race_id", "driver", "lap_number"])
    g = df.groupby(["race_id", "driver"])
    df["next_t"] = g["lap_time_s"].shift(-1)
    df["next_out"] = g["is_pit_out"].shift(-1)
    df["next_status"] = g["track_status"].shift(-1)
    clean = df[(df.is_pit_in == 0) & (df.is_pit_out == 0) & (df.track_status == "1")]
    med = clean.groupby(["race_id", "driver"])["lap_time_s"].median().rename("med")
    stops = df[(df.is_pit_in == 1) & (df.next_out == 1) &
               (df.track_status == "1") & (df.next_status == "1")]
    stops = stops.join(med, on=["race_id", "driver"]).dropna(subset=["med", "next_t", "lap_time_s"])
    loss = stops["lap_time_s"] + stops["next_t"] - 2 * stops["med"]
    loss = loss[(loss > 8) & (loss < 45)]
    if len(loss) < 5:
        return None
    med = float(loss.median())
    mad = float((loss - med).abs().median())
    return med, 1.4826 * mad


def estimate_sc_prob(con) -> float:
    """Safety Car periods starting per lap, pooled over all races in the DB."""
    df = pd.read_sql(f"SELECT {RACE_ID} AS race_id, lap_number, track_status FROM laps", con)
    df["sc"] = df["track_status"].fillna("").str.contains("4")
    lap_sc = df.groupby(["race_id", "lap_number"])["sc"].any().reset_index()
    starts = laps = 0
    for _, g in lap_sc.groupby("race_id"):
        s = g.sort_values("lap_number")["sc"].to_numpy()
        starts += int((s[1:] & ~s[:-1]).sum()) + int(s[0])
        laps += len(s)
    return max(starts / max(laps, 1), 0.005)


def build_params(con, race_id=None, old="MEDIUM", new="HARD"):
    """Returns (Params, info dict) calibrated from the DB."""
    clean = load_clean(con, race_id)
    deg = fit_degradation(clean)
    p = Params()
    fitted = [v["deg"] for k, v in deg.items() if k != "_fuel"]
    fallback = float(np.mean(fitted)) if fitted else None   # e.g. only 'UNKNOWN'
    if old in deg:
        p.deg_old = max(deg[old]["deg"], 0.01)
    elif fallback is not None:
        p.deg_old = max(fallback, 0.01)
    if new in deg:
        p.deg_new = max(deg[new]["deg"], 0.01)
    elif fallback is not None:
        p.deg_new = max(fallback, 0.01)
    if not clean.empty:
        p.base_lap = float((clean["lap_time_s"] - p.deg_old * clean["tyre_life"]).median())
        p.start_age = int(clean["tyre_life"].median())
    pl = estimate_pit_loss(con, race_id)
    if pl:
        p.pit_loss_mean, p.pit_loss_std = pl[0], max(pl[1], 0.3)
    p.sc_prob_per_lap = estimate_sc_prob(con)
    return p, {"degradation": deg, "pit_loss": pl, "n_clean_laps": len(clean)}


if __name__ == "__main__":
    import sys
    rid = sys.argv[1] if len(sys.argv) > 1 else None
    with sqlite3.connect(DB_PATH) as con:
        params, info = build_params(con, rid)
    print(info)
    print(params)
