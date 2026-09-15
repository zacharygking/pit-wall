# Pit Wall

Play strategist in a real Grand Prix. Pick a race and a car, call the pit stops lap by lap, and a
simulator fitted to that race's actual lap times plays it out against the real field. At the flag
you see three lines on one chart: your race, the real team's race, and the race an agent ran for
the same car with the same tools.

Data is OpenF1's free history from 2023 on. Nothing is live, and nothing here is affiliated with
Formula 1.

**Status: evening 1 of 5.** The data layer, the lap-time model, the simulator and the calibration
work and are tested on one race. No agent yet, no web page yet.

## How well does the simulator reproduce a real race?

Replaying every driver's real strategy through the simulator, 2024 Bahrain Grand Prix:

| Metric | Value |
|---|---|
| Kendall tau, simulated vs real finishing order | 0.92 |
| Spearman rho | 0.98 |
| Mean absolute position error | 0.60 |
| Exact positions | 14 of 20 |
| Finishing-time RMSE, finishers with a recorded total, after removing the mean offset | 3.3 s |
| Mean offset, simulated minus real | −0.8 s |
| Per-lap gap-to-leader RMSE, all drivers and laps | 19.7 s |
| Per-lap gap-to-leader RMSE, median driver | about 5 s |

The per-lap gap error is dominated by three drivers whose races contained something no lap-time
model can know: Sargeant lost more than a minute in the first ten laps, Bottas had a 50-second pit
stop where the model charges the race's median of 24 s, and Hülkenberg's early stop at the end of
lap 1 still costs him in the real data. The other seventeen drivers sit within a few seconds of
their real gap to the leader all race. One race is not a calibration; the season-wide table is
evening 2.

Fitted on 757 clean-air laps, residual 0.43 s: pit loss 24.3 s, fuel 0.068 s per lap of fuel
remaining, hard-tyre degradation 0.074 s per lap, soft 0.121 s per lap, soft 0.30 s quicker new.

### Sensitivity to the traffic rule

The one hand-set parameter is the pace advantage a car needs to pass the car ahead. It was chosen
on this race and is provisional until the season table exists.

| Overtake threshold | tau | exact | finishing-time RMSE |
|---|---|---|---|
| none, cars pass freely | 0.91 | 12 | 2.1 s |
| 0.3 s per lap, the default | 0.92 | 14 | 3.3 s |
| 0.5 s per lap | 0.87 | 8 | 8.0 s |
| 0.8 s per lap | 0.84 | 6 | 8.8 s |

## How it works

**Data.** `pit_wall/openf1.py` fetches a session's drivers, laps, stints, pit stops, positions,
race-control messages and results, and writes them as JSONL under `data/<session_key>/`, committed
so every number is reproducible offline.

**Lap-time model.** `pit_wall/model.py` fits, by least squares on clean laps,

```
lap_time = base[driver] + offset[compound] + deg1[compound] * tyre_age + deg2[compound] * tyre_age^2
           + fuel * laps_remaining
```

Clean laps exclude lap 1, in-laps, out-laps, laps under a safety car, virtual safety car or red
flag, and per-driver outliers. Laps run within 1.5 s of the car ahead are dirty-air laps and are
also excluded, unless a driver has fewer than eight clean-air laps. Pit loss is the median over
stops of in-lap plus out-lap minus two predicted clean laps, split between the two laps as the data
shows. Safety-car pace is the median real lap under the safety car when there was one.

**Simulator.** `pit_wall/sim.py` starts from the real end-of-lap-1 state, so the standing start is
taken from the data, then runs every lap under three rules: a car cannot finish a lap ahead of the
car in front unless its free-air pace advantage that lap is at least the overtake threshold, a
safety car bunches the field at its real pace, and a virtual safety car slows everyone by the same
factor and makes a stop cheaper. Retirements happen on the lap they really happened.

**Calibration.** `pit_wall/calibrate.py` replays the real strategies and compares the finishing
order, the finishing times and the per-lap gaps to the leader with the real ones.

## Run

```
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/python -m pit_wall fetch 9472          # cache one session (2024 Bahrain)
.venv/bin/python -m pit_wall fit 9472            # print the fitted model
.venv/bin/python -m pit_wall replay 9472         # replay real strategies, print the calibration
.venv/bin/python -m pit_wall fetch-season 2024   # cache every race of a season
.venv/bin/python -m pit_wall season 2024         # the calibration table
```

## Plan

1. Data, model, simulator, calibration on one race. Done.
2. Calibration across every race since 2023: the table, the sensitivity of the traffic threshold,
   and the exogenous-event list per race, so the game can say how much to trust each replay.
3. The agent: a hosted model with tools for gaps, tyre age, pit loss and the undercut window,
   returning a decision and a one-line reason each lap. Reasons are checked for claims against the
   data. Its races are precomputed for the top ten cars of every race in a GitHub Action, so the
   page needs no server.
4. The web page. A race picker over every Grand Prix since 2023. Each circuit drawn from OpenF1's
   real car positions, so every track has its own map without hand-made art. A timing tower, cars
   moving around the map, a gap chart, and the strategy calls you make as the race runs, resolved
   by a TypeScript port of this simulator in the browser. Your line, the real line and the agent's
   line on one chart at the flag.
5. The "why" panel with the agent's reasons and their receipts, and the README's two tables and one
   figure.

Going deeper, after that: a hierarchical Bayesian tyre model with intervals, a PPO pit policy
against the agent and the real teams, and the agent distilled into a small model that runs in the
browser too.

## Data source and licence

Timing data from [OpenF1](https://openf1.org), used under its terms for non-commercial projects.
Code is MIT.
