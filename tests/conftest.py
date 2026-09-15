"""A small synthetic race generated from a known LapModel, so tests can check recovery."""

from __future__ import annotations

import numpy as np
import pytest

from pit_wall.model import LapModel
from pit_wall.race import Lap, Race, Stint
from pit_wall.sim import Strategy, simulate


def make_model(drivers: list[int]) -> LapModel:
    base = {d: 92.0 + 0.15 * i for i, d in enumerate(drivers)}
    return LapModel(
        session_key=1,
        drivers=drivers,
        compounds=["HARD", "MEDIUM", "SOFT"],
        reference_compound="MEDIUM",
        base=base,
        offset={"SOFT": -0.6, "MEDIUM": 0.0, "HARD": 0.5},
        deg1={"SOFT": 0.09, "MEDIUM": 0.05, "HARD": 0.03},
        deg2={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
        fuel=0.035,
        sigma=0.0,
        pit_loss=21.0,
        sc_lap_time=125.0,
        vsc_factor=1.3,
        n_laps_fit=0,
    )


def build_race(model: LapModel, strategies: dict[int, Strategy], total_laps: int = 40, noise: float = 0.0, seed: int = 0) -> Race:
    """Generate a race whose laps come from the model itself (no traffic), so fits are checkable."""
    rng = np.random.default_rng(seed)
    drivers = {d: {"acronym": f"D{d}", "name": f"Driver {d}", "team": "T"} for d in model.drivers}
    laps, stints, pits = {}, {}, {}
    finish = {}
    for d in model.drivers:
        comp, age, total = "MEDIUM", 0, 0.0
        st, st_start = [], 1
        for n in range(1, total_laps + 1):
            t = model.clean_lap_time(d, comp, age, total_laps - n) + (5.0 if n == 1 else 0.0)
            if noise:
                t += float(rng.normal(0, noise))
            pit_here = next((c for (pl, c) in strategies[d].pits if pl == n), None)
            if pit_here:
                t += model.pit_loss
            total += t
            laps[(d, n)] = Lap(d, n, t, pit_out=(n > 1 and any(pl == n - 1 for pl, _ in strategies[d].pits)), date_start=None)
            if pit_here:
                st.append(Stint(len(st) + 1, comp, st_start, n, 0))
                comp, age, st_start = pit_here, 0, n + 1
                pits.setdefault(d, []).append((n, 24.0))
            else:
                age += 1
        st.append(Stint(len(st) + 1, comp, st_start, total_laps, 0))
        stints[d] = st
        finish[d] = total
    order = sorted(model.drivers, key=lambda d: finish[d])
    result = [{"position": i + 1, "driver_number": d, "number_of_laps": total_laps, "dnf": False, "duration": finish[d], "gap_to_leader": finish[d] - finish[order[0]]} for i, d in enumerate(order)]
    return Race(session_key=1, name="synthetic", year=2024, total_laps=total_laps, drivers=drivers, laps=laps, stints=stints, pit_laps=pits, result=result)


@pytest.fixture
def synthetic():
    drivers = [1, 4, 16, 44, 63]
    model = make_model(drivers)
    strategies = {
        1: Strategy.from_list([(14, "HARD"), (28, "SOFT")]),
        4: Strategy.from_list([(12, "HARD"), (26, "HARD")]),
        16: Strategy.from_list([(18, "HARD")]),
        44: Strategy.from_list([(10, "SOFT"), (24, "HARD")]),
        63: Strategy.from_list([(16, "HARD")]),
    }
    return model, strategies, build_race(model, strategies)
