import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import backtest, synthetic, tyre_model          # noqa: E402
from ingest import SCHEMA                       # noqa: E402
from simulator import Params, simulate_undercut  # noqa: E402


def test_probability_bounds_and_reproducible():
    a = simulate_undercut(Params(), 3, n=2000, seed=1)
    b = simulate_undercut(Params(), 3, n=2000, seed=1)
    assert 0 <= a["p_gain"] <= 1 and a["p_gain"] == b["p_gain"]


def test_bigger_gap_means_lower_success():
    lo = simulate_undercut(Params(gap_ahead=0.5), 4, n=4000)["p_gain"]
    hi = simulate_undercut(Params(gap_ahead=6.0), 4, n=4000)["p_gain"]
    assert lo > hi


def test_worn_tyres_help_the_undercut():
    fresh = simulate_undercut(Params(start_age=3), 2, n=4000)["p_gain"]
    worn = simulate_undercut(Params(start_age=25), 2, n=4000)["p_gain"]
    assert worn > fresh


@pytest.fixture(scope="module")
def con():
    c = sqlite3.connect(":memory:")
    c.executescript(SCHEMA)
    for k in range(6):
        synthetic.make_race(seed=k).to_sql("laps", c, if_exists="append", index=False)
    return c


def test_tyre_model_recovers_true_degradation(con):
    deg = tyre_model.fit_degradation(tyre_model.load_clean(con))
    for comp, true in synthetic.TRUE_DEG.items():
        assert abs(deg[comp]["deg"] - true) < 0.03, (comp, deg[comp])


def test_pit_loss_is_plausible(con):
    mean, std = tyre_model.estimate_pit_loss(con)
    assert 17 < mean < 26


def test_backtest_runs_and_beats_baseline(con):
    events, summary = backtest.run(con, n_sims=1500)
    assert summary and summary["n_events"] >= 20
    assert summary["brier"] <= summary["brier_baseline"]
