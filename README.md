# F1 Pit-Window Risk Simulator

A Monte Carlo model that answers one strategy question:

> *"If I pit in N laps, what is the probability I come out ahead of the car in front,
> given tyre wear, pit-stop variance, Safety Car risk and traffic?"*

Python + SQL pipeline: **FastF1 → SQLite → calibration (regression) → Monte Carlo → Streamlit**.

## How it works

| Stage | File | What it does |
|---|---|---|
| Ingest | `src/ingest.py` | Loads laps, tyre life, compound, pit flags, track status, weather, position from FastF1 into SQLite |
| Analysis | `sql/*.sql` | Clean-lap view, degradation curves, pit stops (window functions), gap to car ahead (self-join) |
| Calibration | `src/tyre_model.py` | Fuel-corrected tyre degradation per compound (joint OLS with driver fixed effects), pit loss, Safety Car rate |
| Simulation | `src/simulator.py` | Vectorised Monte Carlo: pit-time variance, slow stops, Safety Car, traffic re-entry, rival cover delay |
| Validation | `src/backtest.py` | Replays real undercut situations; compares model probability with outcome (accuracy, Brier score vs base-rate baseline) |
| UI | `app.py` | Streamlit: live probability, outcome distribution, success heatmap, tyre model, backtest |

### Modelling choices (and why)
- **Linear regression for degradation**, not deep learning: explainable, robust on small data.
- **Joint fit across compounds**: inside one stint, lap number and tyre age move together, so fuel burn and tyre wear can't be separated per compound. Pooling compounds (drivers pit on different laps) identifies both.
- **Clean laps only**: lap 1, in/out laps, and SC/VSC/yellow laps removed; laps >107% of the driver's median dropped.
- **Backtest events** exclude laps near Safety Car / red flags, where undercut logic doesn't apply.

## Run it

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Option A (free, no login): Jolpica-F1 -> laps, positions, pit stops (no tyre compound)
python src/ingest_jolpica.py --list 2023             # find round numbers
python src/ingest_jolpica.py 2023 15                 # repeat for more rounds
# Option B (needs an F1TV Access account since Oct 2026): FastF1 -> adds compound, track status, weather
python src/ingest.py 2023 Monza
python src/tyre_model.py 2023_Monza                  # fitted parameters
python src/backtest.py                               # validation
streamlit run app.py
pytest                                               # tests
```

No internet / no FastF1 yet? Click **Generate synthetic demo data** in the app sidebar.

## Data sources
| Source | Login | Gives | Missing |
|---|---|---|---|
| Jolpica-F1 (`ingest_jolpica.py`) | none | lap times, positions, pit stops | compound, track status (SC laps are *detected heuristically*), weather |
| FastF1 (`ingest.py`) | F1TV Access account | all of the above + compound, weather | - |

With Jolpica data a single pooled degradation curve is fitted (compound = `UNKNOWN`).

## Validation status

**Setup:** 11 races of the 2023 season loaded from Jolpica-F1 (rounds 1-8, 10, 11, 14).
Parameters are fitted per race; the backtest replays real undercut situations
(driver behind pits, car directly ahead covers within 3 laps, no Safety Car / red flag nearby)
and compares the simulated probability of "still ahead 2 laps after the rival's stop" with what happened.

| Undercut events | Accuracy | Brier score | Baseline Brier* | Brier skill |
|---|---|---|---|---|
| 39 | 61.5% (24/39) | 0.194 | 0.246 | +21% |

\* Baseline = always predict the observed base rate (43.6%).

**How to read this:** encouraging, not conclusive. With 39 events the 95% interval on accuracy
is roughly 46%-75%, which still includes a coin flip. The probabilities are better than the
base-rate baseline, and they move in the right direction (larger gap -> lower probability),
but more races are needed before claiming real predictive skill.

**Development notes (kept honest):**
- First backtest on 5 races: skill +14%; on 11 races: +17%.
- Found that pit-loss mean/std was distorted by a few odd stops (British GP std 7.3 s). Switched to
  median / MAD (std 2.0 s, typical loss 23.6 -> 19.9 s). Skill after the fix: +21%.
  The fix was motivated by the diagnosed outlier problem, and no further tuning was done on this sample.
- Code is also verified on synthetic races with known ground truth (`pytest`).

## Limitations (be upfront about these)
- Parameters are fitted on the same race they are tested on (in-sample, track-specific).
- Jolpica data has no tyre compound: one pooled degradation curve is used, and Safety Car laps are detected heuristically from field lap times.
- No DRS, dirty air, or driver-specific pace modelling; traffic is a probability, not a field model.
- Safety Car probability is pooled from only the races you load; more races = better estimate.
- Few events per race, so backtest metrics are noisy; pool several races.

## Next steps
- Leave-one-race-out validation across seasons at the same circuit.
- Driver-specific pace and gap-dependent traffic model.
- Move storage to PostgreSQL (Docker) and expose the simulator via FastAPI.
