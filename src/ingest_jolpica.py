"""Free data source: Jolpica-F1 (successor of Ergast). No login needed.

Usage (from project root):
    python src/ingest_jolpica.py --list 2023      # show round numbers
    python src/ingest_jolpica.py 2023 15          # load round 15 of 2023

What you get:  lap times, positions, pit stops.
What you DON'T get: tyre compound, track status, weather. So:
  * compound is stored as 'UNKNOWN' (one pooled degradation curve is fitted)
  * tyre_life = laps since the driver's last pit stop
  * Safety Car laps are DETECTED heuristically: a lap where the field's median
    lap time is >20% slower than the race median is flagged as neutralised.
Limits: 4 requests/second, 500 requests/hour (one race needs ~25 requests).
"""
import sys
import time

import numpy as np
import pandas as pd
import requests

from ingest import COLUMNS, save

BASE = "https://api.jolpi.ca/ergast/f1"
PAGE = 100                 # API maximum records per request
SC_SLOWDOWN = 1.20         # field-median lap > 1.2x race median => neutralised


# ---------- fetching (needs internet) ----------
def _get(url, params, retries=4):
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:               # rate limited: wait and retry
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()["MRData"]
    raise RuntimeError(f"Failed or rate limited: {url}")


def fetch_paged(url):
    """All pages of an endpoint; returns a list of 'Race' dicts (one per page)."""
    pages, offset, total = [], 0, None
    while total is None or offset < total:
        m = _get(url, {"limit": PAGE, "offset": offset})
        total = int(m["total"])
        races = m["RaceTable"]["Races"]
        if races:
            pages.append(races[0])
        offset += PAGE
        time.sleep(0.3)                        # stay under 4 requests/second
    return pages


def list_rounds(year):
    m = _get(f"{BASE}/{year}.json", {"limit": PAGE})
    return [(int(r["round"]), r["raceName"]) for r in m["RaceTable"]["Races"]]


# ---------- parsing (pure functions, unit-tested offline) ----------
def parse_time(s):
    """'1:35.404' -> 95.404 seconds."""
    if not s:
        return np.nan
    mins, _, secs = s.rpartition(":")
    return float(secs) + (60 * int(mins) if mins else 0)


def build_dataframe(lap_pages, pit_pages, year, race_name) -> pd.DataFrame:
    rows = []
    for page in lap_pages:
        for lap in page.get("Laps", []):
            for t in lap["Timings"]:
                rows.append(dict(driver=t["driverId"], lap_number=int(lap["number"]),
                                 position=int(t["position"]),
                                 lap_time_s=parse_time(t.get("time"))))
    df = (pd.DataFrame(rows).drop_duplicates(["driver", "lap_number"])
            .sort_values(["driver", "lap_number"]).reset_index(drop=True))

    stops = {(s["driverId"], int(s["lap"]))
             for page in pit_pages for s in page.get("PitStops", [])}
    key = list(zip(df["driver"], df["lap_number"]))
    df["is_pit_in"] = [int(k in stops) for k in key]
    df["is_pit_out"] = [int((d, l - 1) in stops) for d, l in key]

    g = df.groupby("driver")
    df["stint"] = g["is_pit_in"].cumsum() - df["is_pit_in"] + 1
    df["tyre_life"] = df.groupby(["driver", "stint"]).cumcount() + 1

    lap_filled = df["lap_time_s"].fillna(df["lap_time_s"].median())
    df["lap_end_s"] = lap_filled.groupby(df["driver"]).cumsum()

    # Heuristic Safety Car / VSC detection from the field's median lap time.
    lap_med = df.groupby("lap_number")["lap_time_s"].median()
    neutral = lap_med > SC_SLOWDOWN * lap_med.median()
    neutral.loc[1] = False                      # lap 1 is slow because of the start
    df["track_status"] = np.where(df["lap_number"].map(neutral), "4", "1")

    df["year"], df["race"] = year, race_name
    df["compound"], df["fresh_tyre"] = "UNKNOWN", None
    df["track_temp"], df["air_temp"] = np.nan, np.nan
    return df[COLUMNS]


def load_race(year: int, rnd: int) -> pd.DataFrame:
    lap_pages = fetch_paged(f"{BASE}/{year}/{rnd}/laps.json")
    pit_pages = fetch_paged(f"{BASE}/{year}/{rnd}/pitstops.json")
    if not lap_pages:
        raise SystemExit(f"No lap data for {year} round {rnd}.")
    return build_dataframe(lap_pages, pit_pages, year, lap_pages[0]["raceName"])


if __name__ == "__main__":
    if sys.argv[1] == "--list":
        for rnd, name in list_rounds(int(sys.argv[2])):
            print(rnd, name)
    else:
        data = load_race(int(sys.argv[1]), int(sys.argv[2]))
        save(data)
        print(f"Saved {len(data)} laps: {data['race'].iloc[0]} {sys.argv[1]}")
