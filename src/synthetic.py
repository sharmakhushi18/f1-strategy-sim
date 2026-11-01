"""Synthetic race generator (same schema as real data).

PURPOSE: test the pipeline end-to-end without internet, and power the demo.
Results on this data validate the CODE, not the model's real-world accuracy.
"""
import numpy as np
import pandas as pd

from ingest import COLUMNS, save

TRUE_DEG = {"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.04}   # s/lap (ground truth)


def make_race(seed=0, n_cars=20, n_laps=57, base=90.0, fuel=-0.04,
              pit_loss=21.0, year=2099, race=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    race = race or f"SYNTH-{seed}"
    drivers = [f"D{i:02d}" for i in range(n_cars)]
    pace = rng.normal(0, 0.35, n_cars)
    start_off = np.arange(n_cars) * 0.4
    sc = set()
    if rng.random() < 0.4:
        s = int(rng.integers(8, n_laps - 8))
        sc = set(range(s, s + 3))

    rows = []
    for i, d in enumerate(drivers):
        s1 = int(np.clip(rng.normal(22, 4), 10, 30))
        stops = [s1]
        if rng.random() < 0.4:
            stops.append(int(np.clip(rng.normal(40, 4), s1 + 9, n_laps - 6)))
        comps = [rng.choice(["SOFT", "MEDIUM"])] + \
                [rng.choice(["MEDIUM", "HARD"]) for _ in stops]
        stint, age, t = 1, 0, start_off[i]
        for lap in range(1, n_laps + 1):
            age += 1
            comp = comps[stint - 1]
            is_in = lap in stops
            is_out = (lap - 1) in stops
            under_sc = lap in sc
            lt = base + pace[i] + TRUE_DEG[comp] * age + fuel * lap
            lt += rng.normal(0, 0.25)
            loss = max(rng.normal(pit_loss, 0.8), 15) * (0.5 if under_sc else 1)
            if is_in:
                lt += loss * 0.4
            if is_out:
                lt += loss * 0.6 + 1.2
            if under_sc:
                lt = base * 1.4 + rng.normal(0, 0.2)
            t += lt
            rows.append(dict(
                year=year, race=race, driver=d, lap_number=lap, lap_time_s=lt,
                stint=stint, compound=comp, tyre_life=age, fresh_tyre=1,
                is_pit_in=int(is_in), is_pit_out=int(is_out),
                track_status="4" if under_sc else "1",
                track_temp=35.0, air_temp=25.0, position=0, lap_end_s=t))
            if is_in:
                stint += 1
                age = 0
    df = pd.DataFrame(rows)
    df["position"] = df.groupby("lap_number")["lap_end_s"].rank(method="first").astype(int)
    return df[COLUMNS]


def generate(n_races=6, db_path=None):
    """Create n synthetic races and save them to the database."""
    from ingest import DB_PATH
    for k in range(n_races):
        save(make_race(seed=k), db_path or DB_PATH)
