"""Load race laps from FastF1 into SQLite.

Usage (from project root):  python src/ingest.py 2023 Monza
NOTE: written without live FastF1 access -> run once and fix any column-name
differences for your FastF1 version.
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "f1.db"
CACHE = ROOT / "data" / "cache"

SCHEMA = """
CREATE TABLE IF NOT EXISTS laps (
    year INTEGER, race TEXT, driver TEXT, lap_number INTEGER,
    lap_time_s REAL, stint INTEGER, compound TEXT, tyre_life INTEGER,
    fresh_tyre INTEGER, is_pit_in INTEGER, is_pit_out INTEGER,
    track_status TEXT, track_temp REAL, air_temp REAL,
    position INTEGER, lap_end_s REAL,
    PRIMARY KEY (year, race, driver, lap_number)
);
"""
COLUMNS = ["year", "race", "driver", "lap_number", "lap_time_s", "stint",
           "compound", "tyre_life", "fresh_tyre", "is_pit_in", "is_pit_out",
           "track_status", "track_temp", "air_temp", "position", "lap_end_s"]


def load_race(year: int, race: str) -> pd.DataFrame:
    import fastf1  # imported lazily so the rest of the project works without it
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))
    session = fastf1.get_session(year, race, "R")
    session.load(telemetry=False, weather=True, messages=False)

    laps = session.laps.reset_index(drop=True)
    weather = laps.get_weather_data().reset_index(drop=True)

    return pd.DataFrame({
        "year": year,
        "race": race,
        "driver": laps["Driver"],
        "lap_number": laps["LapNumber"].astype(int),
        "lap_time_s": laps["LapTime"].dt.total_seconds(),
        "stint": laps["Stint"],
        "compound": laps["Compound"],
        "tyre_life": laps["TyreLife"],
        "fresh_tyre": laps["FreshTyre"].astype("Int64"),
        "is_pit_in": laps["PitInTime"].notna().astype(int),
        "is_pit_out": laps["PitOutTime"].notna().astype(int),
        "track_status": laps["TrackStatus"],
        "track_temp": weather["TrackTemp"],
        "air_temp": weather["AirTemp"],
        "position": laps["Position"],
        "lap_end_s": laps["Time"].dt.total_seconds(),   # session time at lap end
    })[COLUMNS]


def save(df: pd.DataFrame, db_path=DB_PATH) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(SCHEMA)
        con.execute("DELETE FROM laps WHERE year=? AND race=?",
                    (int(df["year"].iloc[0]), df["race"].iloc[0]))
        df[COLUMNS].to_sql("laps", con, if_exists="append", index=False)


if __name__ == "__main__":
    year, race = int(sys.argv[1]), sys.argv[2]
    data = load_race(year, race)
    save(data)
    print(f"Saved {len(data)} laps for {year} {race} -> {DB_PATH}")
