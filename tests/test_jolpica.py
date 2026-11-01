"""Offline tests: build Ergast-style JSON from a synthetic race, parse it back."""
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import backtest, ingest_jolpica as ij, synthetic, tyre_model   # noqa: E402
from ingest import SCHEMA                                      # noqa: E402


def fmt(t):
    return f"{int(t // 60)}:{t % 60:06.3f}"


def to_ergast(df):
    laps = [{"number": str(n), "Timings": [
        {"driverId": r.driver, "position": str(r.position), "time": fmt(r.lap_time_s)}
        for r in g.itertuples()]} for n, g in df.groupby("lap_number")]
    stops = [{"driverId": r.driver, "lap": str(r.lap_number)}
             for r in df[df.is_pit_in == 1].itertuples()]
    return [{"raceName": "Test GP", "Laps": laps}], [{"PitStops": stops}]


def test_parse_time():
    assert abs(ij.parse_time("1:35.404") - 95.404) < 1e-9
    assert abs(ij.parse_time("59.9") - 59.9) < 1e-9


def test_roundtrip_matches_original():
    orig = synthetic.make_race(seed=3)
    lap_pages, pit_pages = to_ergast(orig)
    new = ij.build_dataframe(lap_pages, pit_pages, 2099, "Test GP")
    m = orig.merge(new, on=["driver", "lap_number"], suffixes=("_o", "_n"))
    assert len(m) == len(orig)
    assert (m.is_pit_in_o == m.is_pit_in_n).all()
    assert (m.is_pit_out_o == m.is_pit_out_n).all()
    assert (m.tyre_life_o == m.tyre_life_n).all()
    assert (m.position_o == m.position_n).all()


def test_full_pipeline_without_compound_data():
    c = sqlite3.connect(":memory:")
    c.executescript(SCHEMA)
    for k in range(6):
        lp, pp = to_ergast(synthetic.make_race(seed=k, race=f"R{k}"))
        ij.build_dataframe(lp, pp, 2099, f"R{k}").to_sql("laps", c, if_exists="append", index=False)
    params, info = tyre_model.build_params(c)
    assert 0.03 < params.deg_old < 0.12            # true curves: 0.04 / 0.07 / 0.12
    assert 17 < info["pit_loss"][0] < 26
    events, summary = backtest.run(c, n_sims=1000)
    assert summary and summary["n_events"] > 10
