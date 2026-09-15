"""Replay the real strategies through the simulator and score the result against what happened."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
from scipy.stats import kendalltau, spearmanr

from .model import LapModel, fit
from .race import Race
from .sim import SimResult, actual_strategies, simulate


@dataclass
class Calibration:
    session_key: int
    race: str
    n_drivers: int
    n_finishers: int
    kendall_tau: float
    spearman_rho: float
    mean_abs_position_error: float
    exact_positions: int
    finish_time_rmse: float          # seconds, finishers only, after removing the mean offset
    finish_time_bias: float          # seconds, mean of (sim - actual) over finishers
    gap_to_leader_rmse: float        # seconds, per lap per driver, against real cumulative times
    pit_loss: float
    sigma: float
    n_laps_fit: int
    notes: list[str]

    def row(self) -> str:
        return (
            f"{self.race:<34} tau={self.kendall_tau:5.2f} rho={self.spearman_rho:5.2f} "
            f"|dpos|={self.mean_abs_position_error:4.2f} exact={self.exact_positions:2d}/{self.n_drivers:2d} "
            f"gapRMSE={self.gap_to_leader_rmse:5.1f}s  finRMSE={self.finish_time_rmse:5.1f}s  pit={self.pit_loss:4.1f}s"
        )


def compare(race: Race, sim: SimResult) -> dict:
    actual_order = race.classified_order()
    common = [d for d in actual_order if d in sim.order]
    a_rank = {d: i for i, d in enumerate(common)}
    s_rank = {d: i for i, d in enumerate([d for d in sim.order if d in a_rank])}
    a = np.array([a_rank[d] for d in common])
    s = np.array([s_rank[d] for d in common])
    tau = float(kendalltau(a, s).statistic) if len(common) > 2 else float("nan")
    rho = float(spearmanr(a, s).statistic) if len(common) > 2 else float("nan")
    mae = float(np.mean(np.abs(a - s))) if len(common) else float("nan")
    exact = int(np.sum(a == s))
    # finishing times: real classified finishers with a duration
    finishers = [r for r in race.result if not r.get("dnf") and r.get("duration") and r["driver_number"] in sim.finish_time]
    diffs = np.array([sim.finish_time[r["driver_number"]] - float(r["duration"]) for r in finishers])
    bias = float(diffs.mean()) if len(diffs) else float("nan")
    rmse = float(np.sqrt(np.mean((diffs - bias) ** 2))) if len(diffs) else float("nan")
    # per-lap gap to leader against the real cumulative times
    real_cum = {d: race.cumulative_times(d) for d in race.driver_numbers()}
    errs = []
    for lap in range(2, race.total_laps + 1):
        real_leader = min((t[lap] for t in real_cum.values() if lap in t), default=None)
        sim_leader = min((c[lap - 1] for c in sim.cumulative.values() if len(c) >= lap), default=None)
        if real_leader is None or sim_leader is None:
            continue
        for d, c in sim.cumulative.items():
            if len(c) >= lap and lap in real_cum.get(d, {}):
                errs.append((c[lap - 1] - sim_leader) - (real_cum[d][lap] - real_leader))
    gap_rmse = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
    return {
        "n_drivers": len(common),
        "n_finishers": len(finishers),
        "kendall_tau": tau,
        "spearman_rho": rho,
        "mean_abs_position_error": mae,
        "exact_positions": exact,
        "finish_time_rmse": rmse,
        "finish_time_bias": bias,
        "gap_to_leader_rmse": gap_rmse,
    }


def calibrate(race: Race, model: LapModel | None = None, **sim_kwargs) -> tuple[Calibration, LapModel, SimResult]:
    model = model or fit(race)
    sim = simulate(race, model, actual_strategies(race), **sim_kwargs)
    c = compare(race, sim)
    cal = Calibration(
        session_key=race.session_key,
        race=race.name,
        pit_loss=model.pit_loss,
        sigma=model.sigma,
        n_laps_fit=model.n_laps_fit,
        notes=list(model.notes),
        **c,
    )
    return cal, model, sim


def to_json(cal: Calibration) -> str:
    return json.dumps(asdict(cal), indent=2)
