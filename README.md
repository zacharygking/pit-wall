# Pit Wall

Play strategist in a real Grand Prix. Pick a race and a car, call the pit stops lap by lap, and a
simulator fitted to that race's actual lap times plays it out against the real field. At the flag
you see your race, the real team's race, the race an agent ran for the same car with the same
tools, and the best strategy the simulator could find, on one chart.

Data is OpenF1's free history from 2023 on. Nothing is live, and nothing here is affiliated with
Formula 1.

**Status: step 1 of 8.** The data layer, the lap-time model, the simulator and the calibration
work and are tested on one race. No agent yet, no web page yet. The plan is at the end.

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
next.

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
2. Calibration across every race since 2023, with two additions that make the table honest:
   anomaly flags for the laps and stops no model can explain (a 50-second pit stop, a minute lost
   to damage), so each race carries a trust score; and a learned overtaking model in place of the
   hand-set threshold, a logistic model of lap-to-lap position changes on pace delta, gap and
   circuit, scored by log-loss on held-out races.
3. The pre-race model. Fitting tyre degradation on the race being played lets the game peek at
   its own future. The fair version fits from what a strategist has on Sunday morning: the
   weekend's practice long runs and every earlier race at the circuit, pooled hierarchically
   across seasons, compounds and teams. The calibration table then has two rows per race,
   hindsight and pre-race, and the game plays on the pre-race one.
4. Strategy search. Dynamic programming over pit laps and compounds against the deterministic
   simulator gives the best strategy in hindsight and the best pre-race strategy. That is the
   game's score, your gap to the optimum, and the baseline every agent has to beat.
5. The agent: a hosted model with tools for gaps, tyre age, pit loss, the undercut window, a
   Monte Carlo forecast from the model's uncertainty and the circuit's safety-car hazard, and
   retrieval of precedents, the most similar situations from every race since 2023 with what the
   team did and what it earned. Reasons are checked for claims against the data; a reason that
   rests on a rule cites the FIA sporting regulation it retrieved. Precomputed for the top ten
   cars of every race in a GitHub Action.
6. The web page. A race picker over every Grand Prix since 2023, each circuit drawn from OpenF1's
   real car positions, a timing tower, cars moving round the map, a gap chart, your calls resolved
   by a TypeScript port of the simulator, and at the flag your line, the real line, the agent's
   line and the optimum on one chart, with the outcome spread from the Monte Carlo.
7. The "why" panel with the agent's reasons, their receipts and their rule citations, and the
   grading tool that validates the judge behind it. Grading is assisted, not led: for each reason
   the tool shows the race state, a fact sheet computed by the simulator (undercut and overcut
   windows, pit loss here, laps to the tyre cliff, the second-compound obligation, the safety-car
   hazard, the optimizer's call at that lap, the Monte Carlo spread for pit-now against stay-out),
   a mechanism glossary, and a short model-written brief of the situation generated from the
   state alone, never from the reason under review. The judge's verdict stays hidden until the
   label is committed. A second pass then reveals it and allows a revision, recorded separately.
   The headline kappa is the blind pass; the shift after reveal is reported as its own number.
8. Two more strategists, scored by the same harness against the optimizer, the hosted agent and
   the real teams. A PPO pit policy over the stochastic simulator, wrapped as a Gymnasium
   environment and trained on the CPU in minutes; it reacts to safety cars and cannot explain
   itself. And a small language model, 360M to 0.5B, taken up the post-training ladder with the
   simulated finishing position as a verifiable reward: supervised fine-tuning on the optimizer's
   plans, rejection-sampling fine-tuning and DPO on plans the simulator scored, then a short GRPO
   polish. Actions are plan-level, a full strategy at the start and a re-plan at each safety car,
   so a decision is a few tokens. Everything runs in MLX on the Mac, overnight at most, and the
   best rung ships to the browser with a table of what each rung bought.

## Modelling roadmap

What each technique is for, what it is measured by, and what it costs on a laptop.

| Technique | Why it is here | Measured by | Cost |
|---|---|---|---|
| Anomaly flags on laps and stops | Separates model error from events no model can know; gives each race a trust score | Count of flagged events per race, shown on the page | Hours |
| Learned overtaking model | Replaces the one hand-set parameter with a fitted probability of passing per lap, per circuit | Log-loss and calibration on held-out races; tau of the replay | An evening |
| Pre-race model from practice and history | Stops the game from peeking at the race it simulates | Calibration rows for hindsight against pre-race; lap-time RMSE on held-out races | Two evenings |
| Hierarchical Bayesian tyre model | Partial pooling across seasons, compounds and teams, with posterior intervals | Interval coverage of real lap times and finishing positions | Two evenings, numpyro on CPU |
| Safety-car hazard model | Per-circuit probability of a neutralisation per lap, so a strategy can hedge | Brier score of per-race safety-car probability | An evening |
| Monte Carlo outcomes | Turns one simulated result into a distribution the page can show | Coverage of the 80 percent interval on real finishing positions | An evening |
| Strategy search by dynamic programming | The optimum in hindsight and pre-race: the game's score and every agent's baseline | Gap to optimum for the real teams, per race | An evening |
| Precedent retrieval for the agent | Retrieval-augmented decisions: the nearest past situations at this circuit and what they earned | Agent finishing position with and against without retrieval, over every race | Two evenings |
| Regulation retrieval for the reasons | A rule-based reason cites the article; the two-compound rule is enforced with its text | Recall at 5 on 60 hand-labelled rule questions; grounding rate of citations | An evening, plus P's retriever |
| PPO pit policy | A learned policy that reacts to safety cars, against the DP optimum, the agent and the real teams | Expected finishing position under Monte Carlo | Two evenings, CPU |
| Strategist SFT, rejection sampling and DPO | A small language model learns to plan from the optimizer, then from its own simulator-scored samples | Finishing position per rung against the optimum | One to two hours per round, MLX |
| Strategist GRPO polish | Online RL with the simulated finishing position as the reward, and the reward-hacking ablation | Finishing position per rung, plus the ablation | Overnight, MLX |
| Regulation question set | Synthetic questions generated from the regulation text, plus twenty written by hand | Recall at 5 on both sets, reported separately | Two hours |
| Judge validation | Sixty reasons graded blind with computed context, a second rater on twenty | Kappa on the blind pass, and the shift after the judge is revealed | Two hours |

## Data source and licence

Timing data from [OpenF1](https://openf1.org), used under its terms for non-commercial projects.
Code is MIT.
