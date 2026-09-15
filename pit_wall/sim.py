"""Deterministic race simulator over a fitted LapModel.

The simulation starts from the real end-of-lap-1 state, so the standing start and first-corner
chaos are taken from the data rather than modelled. From lap 2 on, every car runs the model's
clean lap time for its tyre and fuel, plus a pit loss on the lap it pits, under three rules:

* traffic: a car cannot finish a lap ahead of the car in front of it unless its free-air pace
  advantage on that lap is at least ``overtake_threshold`` seconds (0.3 by default); otherwise it is held to
  ``min_gap`` behind. A car in the pit lane is not an obstacle.
* safety car: every car runs the safety-car lap time and the field bunches to ``bunch_gap``.
* virtual safety car: every car's lap time is multiplied by the model's VSC factor and a pit
  stop costs ``vsc_pit_factor`` of the normal loss.

Retirements are exogenous: a car retires on the lap it really retired, whatever its strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import LapModel
from .race import Race


@dataclass(frozen=True)
class Strategy:
    """Pit at the end of ``lap`` and fit ``compound``. Laps must be increasing."""

    pits: tuple[tuple[int, str], ...]

    @classmethod
    def from_list(cls, items: list[tuple[int, str]]) -> "Strategy":
        return cls(tuple(sorted(items)))

    def compound_after(self, lap: int) -> str | None:
        fitted = None
        for p_lap, comp in self.pits:
            if p_lap <= lap:
                fitted = comp
        return fitted


@dataclass
class SimResult:
    order: list[int]                                 # final classification, driver numbers
    finish_time: dict[int, float]                    # cumulative time for finishers
    laps_done: dict[int, int]
    lap_times: dict[int, list[float]] = field(default_factory=dict)      # per driver, index 0 = lap 1
    positions: dict[int, list[int]] = field(default_factory=dict)        # per driver, position after each lap
    cumulative: dict[int, list[float]] = field(default_factory=dict)     # per driver, time after each lap

    def position_of(self, driver: int) -> int:
        return self.order.index(driver) + 1


def lap1_state(race: Race) -> dict[int, float]:
    """Cumulative time after lap 1 for every driver who completed it, from the real lap 1."""
    out = {}
    for d in race.driver_numbers():
        lap = race.laps.get((d, 1))
        if lap is not None and lap.duration is not None and race.laps_completed(d) >= 1:
            out[d] = float(lap.duration)
    if not out:
        return out
    # a driver with a missing lap-1 duration but more laps completed is slotted at the back
    filler = max(out.values()) + 1.0
    for d in race.driver_numbers():
        if d not in out and race.laps_completed(d) >= 1:
            out[d] = filler
            filler += 1.0
    return out


def simulate(
    race: Race,
    model: LapModel,
    strategies: dict[int, Strategy],
    *,
    overtake_threshold: float = 0.3,  # provisional: chosen on 2024 Bahrain, see README
    min_gap: float = 0.7,
    bunch_gap: float = 0.9,
    vsc_pit_factor: float = 0.6,
    noise: float = 0.0,
    seed: int | None = None,
) -> SimResult:
    rng = np.random.default_rng(seed)
    start = lap1_state(race)
    active = [d for d in race.driver_numbers() if d in start]
    time = {d: start[d] for d in active}
    compound: dict[int, str] = {}
    age: dict[int, int] = {}
    outlap_due: dict[int, float] = {}  # driver -> pit loss still owed on the out-lap
    for d in active:
        ca = race.compound_on_lap(d, 1)
        comp, a = (ca if ca else (model.reference_compound, 0))
        compound[d], age[d] = comp, a + 1  # age at the start of lap 2
        lap1_pit = strategies.get(d, Strategy(())).compound_after(1)
        if lap1_pit is not None and any(p_lap == 1 for p_lap, _ in strategies[d].pits):
            compound[d], age[d] = lap1_pit, 0
            outlap_due[d] = model.pit_loss * (1.0 - model.pit_in_share)
    laps_done = {d: 1 for d in active}
    retired: dict[int, int] = {}
    lap_times = {d: [start[d]] for d in active}
    cumulative = {d: [start[d]] for d in active}
    positions = {d: [] for d in active}
    first_order = sorted(active, key=lambda d: time[d])
    for pos, d in enumerate(first_order, 1):
        positions[d].append(pos)

    running = list(first_order)
    for lap in range(2, race.total_laps + 1):
        # retirements happen at the start of the lap after their last completed lap
        for d in list(running):
            if race.laps_completed(d) < lap:
                retired[d] = laps_done[d]
                running.remove(d)
        if not running:
            break
        sc, vsc = lap in race.sc_laps, lap in race.vsc_laps
        free: dict[int, float] = {}
        pitting: set[int] = set()
        for d in running:
            if sc:
                t = model.sc_lap_time
            else:
                t = model.clean_lap_time(d, compound[d], age[d], race.total_laps - lap)
                if vsc:
                    t *= model.vsc_factor
            if noise > 0:
                t += float(rng.normal(0.0, noise))
            new_comp = None
            for p_lap, comp in strategies.get(d, Strategy(())).pits:
                if p_lap == lap:
                    new_comp = comp
            if d in outlap_due:
                t += outlap_due.pop(d)
            if new_comp is not None:
                loss = model.pit_loss * (vsc_pit_factor if vsc else 1.0)
                t += loss * model.pit_in_share
                outlap_due[d] = loss * (1.0 - model.pit_in_share)
                pitting.add(d)
            free[d] = t
        # traffic rule, processed in the running order at the start of the lap
        new_time: dict[int, float] = {}
        ahead: int | None = None
        for d in running:
            candidate = time[d] + free[d]
            if ahead is not None and ahead not in pitting and d not in pitting:
                if candidate < new_time[ahead] + min_gap:
                    pace_gain = free[ahead] - free[d]
                    if pace_gain < overtake_threshold:
                        candidate = new_time[ahead] + min_gap
            new_time[d] = candidate
            ahead = d
        running.sort(key=lambda d: new_time[d])
        if sc:
            leader = new_time[running[0]]
            for k, d in enumerate(running):
                new_time[d] = leader + k * bunch_gap
        for d in running:
            lap_times[d].append(new_time[d] - time[d])
            time[d] = new_time[d]
            cumulative[d].append(time[d])
            laps_done[d] = lap
            if d in pitting:
                compound[d] = strategies[d].compound_after(lap) or compound[d]
                age[d] = 0  # the out-lap is age 0, as OpenF1 counts tyre_age_at_start
            else:
                age[d] += 1
        for pos, d in enumerate(running, 1):
            positions[d].append(pos)

    finishers = sorted(running, key=lambda d: time[d])
    dnfs = sorted(retired, key=lambda d: (-retired[d], time[d]))
    order = finishers + dnfs
    return SimResult(
        order=order,
        finish_time={d: time[d] for d in finishers},
        laps_done=laps_done,
        lap_times=lap_times,
        positions=positions,
        cumulative=cumulative,
    )


def actual_strategies(race: Race) -> dict[int, Strategy]:
    return {d: Strategy.from_list(race.actual_strategy(d)) for d in race.driver_numbers()}
