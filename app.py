"""Streamlit front-end:  streamlit run app.py"""
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import backtest
import synthetic
import tyre_model
from ingest import DB_PATH
from simulator import Params, grid, simulate_undercut

st.set_page_config(page_title="F1 Pit-Window Risk Simulator", layout="wide")
st.title("F1 Pit-Window Risk Simulator")
st.caption("Monte Carlo undercut model calibrated on lap data. "
           "Parameters come from the database (FastF1) or manual sliders.")


def race_list():
    if not DB_PATH.exists():
        return []
    with sqlite3.connect(DB_PATH) as con:
        return [r[0] for r in con.execute(
            "SELECT DISTINCT year || '_' || race FROM laps ORDER BY 1")]


races = race_list()
with st.sidebar:
    st.header("Data")
    if not races:
        st.info("No race data yet. Run `python src/ingest.py 2023 Monza` "
                "or generate synthetic demo data.")
    if st.button("Generate synthetic demo data"):
        synthetic.generate(6)
        st.rerun()
    race = st.selectbox("Calibrate on race", ["(defaults)"] + races)
    if any(r.startswith("2099_SYNTH") for r in races):
        st.warning("Synthetic data in DB: results demonstrate the code, "
                   "not real-world accuracy.")


@st.cache_data(show_spinner="Calibrating from data...")
def calibrate(rid):
    with sqlite3.connect(DB_PATH) as con:
        return tyre_model.build_params(con, rid)


if race == "(defaults)":
    base, info = Params(), None
else:
    base, info = calibrate(race)

tab_sim, tab_tyre, tab_bt = st.tabs(["Undercut simulator", "Tyre model", "Backtest"])

with tab_sim:
    c1, c2, c3 = st.columns(3)
    pit_lap = c1.slider("Pit in N laps", 1, 12, 3)
    gap = c1.slider("Gap to car ahead (s)", 0.0, 5.0, float(base.gap_ahead), 0.1)
    delay = c1.slider("Rival cover delay (laps)", 1, 4, base.rival_cover_delay)
    age = c2.slider("Tyre age now (laps)", 1, 40, int(base.start_age))
    deg_old = c2.slider("Deg, current tyre (s/lap)", 0.01, 0.30, float(round(base.deg_old, 3)), 0.005)
    deg_new = c2.slider("Deg, new tyre (s/lap)", 0.01, 0.30, float(round(base.deg_new, 3)), 0.005)
    pit_loss = c3.slider("Pit loss (s)", 15.0, 30.0, float(round(base.pit_loss_mean, 1)), 0.1)
    sc = c3.slider("Safety Car prob / lap", 0.0, 0.10, float(round(base.sc_prob_per_lap, 3)), 0.005)
    traffic = c3.slider("Traffic probability", 0.0, 1.0, base.traffic_prob, 0.05)

    p = Params(**{**asdict(base), "gap_ahead": gap, "rival_cover_delay": delay,
                  "start_age": age, "deg_old": deg_old, "deg_new": deg_new,
                  "pit_loss_mean": pit_loss, "sc_prob_per_lap": sc,
                  "traffic_prob": traffic})
    r = simulate_undercut(p, pit_lap, n=10_000)
    m1, m2, m3 = st.columns(3)
    m1.metric("P(end up ahead)", f"{r['p_gain']:.0%}")
    m2.metric("Mean margin", f"{r['mean_delta']:+.2f} s")
    m3.metric("90% range", f"{r['p5']:+.1f} to {r['p95']:+.1f} s")

    left, right = st.columns(2)
    fig = px.histogram(x=r["delta"], nbins=60,
                       labels={"x": "Margin vs rival after both stops (s)"},
                       title="Outcome distribution (>0 = ahead)")
    fig.add_vline(x=0, line_dash="dash")
    left.plotly_chart(fig, width="stretch")

    laps_axis = list(range(1, 11))
    gaps_axis = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    z = grid(p, laps_axis, gaps_axis, n=1500)
    heat = px.imshow(z, x=laps_axis, y=gaps_axis, origin="lower", zmin=0, zmax=1,
                     color_continuous_scale="RdYlGn", aspect="auto",
                     labels={"x": "Pit in N laps", "y": "Gap (s)", "color": "P(ahead)"},
                     title="Undercut success probability")
    right.plotly_chart(heat, width="stretch")

with tab_tyre:
    if race == "(defaults)":
        st.info("Select a race in the sidebar to see fitted tyre data.")
    else:
        deg = info["degradation"]
        st.write("**Fitted degradation (s/lap)**")
        st.dataframe(pd.DataFrame(
            {c: v for c, v in deg.items() if c != "_fuel"}).T.rename(
            columns={"deg": "deg s/lap", "se": "std err", "n": "laps"}))
        if "_fuel" in deg:
            st.write(f"Fuel-burn effect: **{deg['_fuel']:+.3f} s/lap**")
        if info["pit_loss"]:
            st.write(f"Pit loss: **{info['pit_loss'][0]:.1f} ± {info['pit_loss'][1]:.1f} s**")
        with sqlite3.connect(DB_PATH) as con:
            clean = tyre_model.load_clean(con, race)
        avg = clean.groupby(["compound", "tyre_life"], as_index=False)["lap_time_s"].mean()
        st.plotly_chart(px.line(avg, x="tyre_life", y="lap_time_s", color="compound",
                                markers=True, title="Mean clean-lap time vs tyre age"),
                        width="stretch")

with tab_bt:
    st.write("Replays real undercut situations (driver behind pits, rival covers "
             "within 3 laps) and compares the model's probability with what happened.")
    if not races:
        st.info("Load race data first.")
    elif st.button("Run backtest on all loaded races"):
        with sqlite3.connect(DB_PATH) as con:
            events, summary = backtest.run(con, n_sims=2000)
        if not summary:
            st.warning("No qualifying undercut events found.")
        else:
            a, b, c, d = st.columns(4)
            a.metric("Events", summary["n_events"])
            b.metric("Accuracy", f"{summary['accuracy']:.0%}")
            c.metric("Brier score", f"{summary['brier']:.3f}",
                     help="Lower is better. Baseline = always predict the base rate.")
            d.metric("Skill vs baseline", f"{summary['brier_skill']:+.0%}")
            st.dataframe(events.round(2))
