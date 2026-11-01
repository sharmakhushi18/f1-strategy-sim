"""Monte Carlo undercut simulator.

Question answered: "If I pit on lap `pit_lap` (from now), what is the
probability I end up ahead of the car in front after it has also pitted?"

All parameters below are PLACEHOLDERS. Calibrate them from real race data
(tyre_model.py / SQL) before showing results anywhere.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Params:
    base_lap: float = 90.0        # clean-air pace on fresh tyres (s)
    deg_old: float = 0.09         # s/lap degradation, current compound
    deg_new: float = 0.06         # s/lap degradation, new compound
    start_age: int = 14           # current tyre age of both cars
    gap_ahead: float = 1.5        # I am this many seconds behind rival
    pit_loss_mean: float = 21.0   # in-lap + out-lap + stationary (s)
    pit_loss_std: float = 0.7
    slow_stop_prob: float = 0.05  # chance of a slow stop (+3 to 6 s)
    outlap_penalty: float = 1.2   # cold tyres on out-lap (s)
    sc_prob_per_lap: float = 0.02 # Safety Car probability per lap
    sc_pit_discount: float = 0.5  # pit loss multiplier if pitting under SC
    traffic_prob: float = 0.35    # chance of rejoining behind slower cars
    traffic_loss_per_lap: float = 0.6
    traffic_max_laps: int = 3
    rival_cover_delay: int = 2    # rival reacts this many laps after me
    laps_after: int = 6           # laps simulated after rival's stop
    rival_start_age: int | None = None  # None -> same age as me


def _pit_loss(p, rng, n, sc_active):
    loss = rng.normal(p.pit_loss_mean, p.pit_loss_std, n)
    slow = rng.random(n) < p.slow_stop_prob
    loss += slow * rng.uniform(3, 6, n)
    return np.where(sc_active, loss * p.sc_pit_discount, loss)


def simulate_undercut(p: Params, pit_lap: int, n: int = 10_000, seed: int = 0):
    """Returns dict with P(position gain) and delta distribution (s).

    delta > 0 means I am AHEAD of the rival at the end of the horizon.
    """
    rng = np.random.default_rng(seed)
    rival_pit = pit_lap + p.rival_cover_delay
    horizon = rival_pit + p.laps_after

    # Safety Car: first lap on which it appears (inf if none in horizon)
    sc_lap = rng.geometric(p.sc_prob_per_lap, n)
    sc_lap = np.where(sc_lap <= horizon, sc_lap, 10**6)

    # Traffic after my stop
    in_traffic = rng.random(n) < p.traffic_prob
    traffic_laps = np.where(in_traffic, rng.integers(1, p.traffic_max_laps + 1, n), 0)

    me_pit_loss = _pit_loss(p, rng, n, sc_lap == pit_lap)
    rv_pit_loss = _pit_loss(p, rng, n, sc_lap == rival_pit)

    me_t = np.zeros(n)
    rv_t = np.zeros(n)
    me_age = np.full(n, float(p.start_age))
    rv_age = np.full(n, float(p.rival_start_age if p.rival_start_age is not None else p.start_age))
    me_new = rv_new = False

    for lap in range(1, horizon + 1):
        under_sc = sc_lap == lap
        # lap times (SC laps are slow and equal for both -> no gain either way)
        me_deg = p.deg_new if me_new else p.deg_old
        rv_deg = p.deg_new if rv_new else p.deg_old
        me_lap = p.base_lap + me_deg * me_age
        rv_lap = p.base_lap + rv_deg * rv_age
        sc_lap_time = p.base_lap * 1.4
        me_lap = np.where(under_sc, sc_lap_time, me_lap)
        rv_lap = np.where(under_sc, sc_lap_time, rv_lap)

        if lap == pit_lap:
            me_lap = me_lap + me_pit_loss
            me_age[:] = 0
            me_new = True
        elif lap == pit_lap + 1:
            me_lap = me_lap + p.outlap_penalty
        if pit_lap < lap <= pit_lap + p.traffic_max_laps:
            me_lap = me_lap + np.where(lap - pit_lap <= traffic_laps,
                                       p.traffic_loss_per_lap, 0.0)

        if lap == rival_pit:
            rv_lap = rv_lap + rv_pit_loss
            rv_age[:] = 0
            rv_new = True
        elif lap == rival_pit + 1:
            rv_lap = rv_lap + p.outlap_penalty

        me_t += me_lap
        rv_t += rv_lap
        me_age += 1
        rv_age += 1

    delta = rv_t - me_t - p.gap_ahead  # >0 => I'm ahead
    return {
        "p_gain": float((delta > 0).mean()),
        "mean_delta": float(delta.mean()),
        "p5": float(np.percentile(delta, 5)),
        "p95": float(np.percentile(delta, 95)),
        "delta": delta,
    }


def grid(p: Params, pit_laps, gaps, n=5_000):
    """P(gain) matrix for a heatmap: rows = gaps, cols = pit laps."""
    out = np.zeros((len(gaps), len(pit_laps)))
    for i, g in enumerate(gaps):
        q = Params(**{**p.__dict__, "gap_ahead": g})
        for j, L in enumerate(pit_laps):
            out[i, j] = simulate_undercut(q, L, n)["p_gain"]
    return out


if __name__ == "__main__":
    p = Params()
    for L in (1, 3, 5, 8):
        r = simulate_undercut(p, L)
        print(f"pit in {L} laps: P(gain)={r['p_gain']:.0%}, "
              f"mean delta={r['mean_delta']:+.2f}s "
              f"[{r['p5']:+.1f}, {r['p95']:+.1f}]")
