"""A lap-time model fitted to one race's clean laps.

lap_time = base[driver] + offset[compound] + deg1[compound] * age + deg2[compound] * age^2
           + fuel * laps_remaining

Clean laps exclude lap 1, in-laps, out-laps, laps under a safety car, virtual safety car or red
flag, laps without a duration, and per-driver outliers. Everything else about the race, pit loss,
safety-car pace, is estimated from the same data and stored on the model so the simulator has one
object to read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import median

import numpy as np

from .race import Race


@dataclass
class LapModel:
    session_key: int
    drivers: list[int]
    compounds: list[str]
    reference_compound: str
    base: dict[int, float]
    offset: dict[str, float]
    deg1: dict[str, float]
    deg2: dict[str, float]
    fuel: float
    sigma: float
    pit_loss: float
    sc_lap_time: float
    vsc_factor: float
    n_laps_fit: int
    pit_in_share: float = 0.5
    notes: list[str] = field(default_factory=list)

    def clean_lap_time(self, driver: int, compound: str, age: int, laps_remaining: int) -> float:
        comp = compound if compound in self.offset else self.reference_compound
        return (
            self.base[driver]
            + self.offset.get(comp, 0.0)
            + self.deg1.get(comp, 0.0) * age
            + self.deg2.get(comp, 0.0) * age * age
            + self.fuel * laps_remaining
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["base"] = {str(k): v for k, v in self.base.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LapModel":
        d = dict(d)
        d["base"] = {int(k): v for k, v in d["base"].items()}
        return cls(**d)


def gap_ahead(race: Race) -> dict[tuple[int, int], float]:
    """Gap in seconds to the car directly ahead at the end of each lap, from real cumulative times."""
    cum = {d: race.cumulative_times(d) for d in race.driver_numbers()}
    out: dict[tuple[int, int], float] = {}
    for n in range(1, race.total_laps + 1):
        order = sorted((d for d in cum if n in cum[d]), key=lambda d: cum[d][n])
        for ahead, d in zip(order, order[1:]):
            out[(d, n)] = cum[d][n] - cum[ahead][n]
        if order:
            out[(order[0], n)] = float("inf")
    return out


def clean_laps(
    race: Race, outlier_margin: float = 5.0, traffic_gap: float = 1.5, min_free_laps: int = 8
) -> list[tuple[int, int, float, str, int]]:
    """(driver, lap_number, duration, compound, age) for every lap the model may learn from.

    Laps run within ``traffic_gap`` seconds of the car ahead are dirty-air laps and are excluded,
    unless that leaves a driver with fewer than ``min_free_laps``, in which case all of that
    driver's laps are kept so a car stuck in a train all race still gets a pace estimate.
    """
    gaps = gap_ahead(race)
    rows: list[tuple[int, int, float, str, int]] = []
    per_driver: dict[int, list[float]] = {}
    free_count: dict[int, int] = {}
    for (driver, n), lap in race.laps.items():
        if n <= 1 or lap.duration is None or lap.pit_out:
            continue
        if n in race.sc_laps or n in race.vsc_laps or n in race.red_flag_laps:
            continue
        if n in race.in_lap_set(driver):
            continue
        ca = race.compound_on_lap(driver, n)
        if ca is None or ca[0] in ("INTERMEDIATE", "WET", "UNKNOWN"):
            continue
        rows.append((driver, n, float(lap.duration), ca[0], ca[1]))
        per_driver.setdefault(driver, []).append(float(lap.duration))
        if gaps.get((driver, n), float("inf")) >= traffic_gap:
            free_count[driver] = free_count.get(driver, 0) + 1
    med = {d: median(v) for d, v in per_driver.items()}
    keep = []
    for r in rows:
        driver, n, dur = r[0], r[1], r[2]
        if not (med[driver] - outlier_margin <= dur <= med[driver] + outlier_margin):
            continue
        in_traffic = gaps.get((driver, n), float("inf")) < traffic_gap
        if in_traffic and free_count.get(driver, 0) >= min_free_laps:
            continue
        keep.append(r)
    return keep


def fit(race: Race, ridge: float = 1e-3) -> LapModel:
    rows = clean_laps(race)
    if len(rows) < 40:
        raise ValueError(f"only {len(rows)} clean laps in {race.name}; not enough to fit")
    drivers = sorted({r[0] for r in rows})
    compounds = sorted({r[3] for r in rows})
    counts = {c: sum(1 for r in rows if r[3] == c) for c in compounds}
    ref = max(counts, key=counts.get)
    others = [c for c in compounds if c != ref]
    d_idx = {d: i for i, d in enumerate(drivers)}
    o_idx = {c: i for i, c in enumerate(others)}
    c_idx = {c: i for i, c in enumerate(compounds)}
    n_cols = len(drivers) + len(others) + 2 * len(compounds) + 1
    X = np.zeros((len(rows), n_cols))
    y = np.zeros(len(rows))
    col_off = len(drivers)
    col_deg1 = col_off + len(others)
    col_deg2 = col_deg1 + len(compounds)
    col_fuel = col_deg2 + len(compounds)
    for i, (driver, n, dur, comp, age) in enumerate(rows):
        X[i, d_idx[driver]] = 1.0
        if comp in o_idx:
            X[i, col_off + o_idx[comp]] = 1.0
        X[i, col_deg1 + c_idx[comp]] = age
        X[i, col_deg2 + c_idx[comp]] = age * age
        X[i, col_fuel] = race.total_laps - n
        y[i] = dur
    # tiny ridge on everything but the driver intercepts keeps sparse compounds sane
    penalty = np.zeros(n_cols)
    penalty[len(drivers):] = ridge
    A = X.T @ X + np.diag(penalty)
    beta = np.linalg.solve(A, X.T @ y)
    resid = y - X @ beta
    sigma = float(np.sqrt(np.mean(resid**2)))
    model = LapModel(
        session_key=race.session_key,
        drivers=drivers,
        compounds=compounds,
        reference_compound=ref,
        base={d: float(beta[d_idx[d]]) for d in drivers},
        offset={c: float(beta[col_off + o_idx[c]]) for c in others} | {ref: 0.0},
        deg1={c: float(beta[col_deg1 + c_idx[c]]) for c in compounds},
        deg2={c: float(beta[col_deg2 + c_idx[c]]) for c in compounds},
        fuel=float(beta[col_fuel]),
        sigma=sigma,
        pit_loss=0.0,
        sc_lap_time=0.0,
        vsc_factor=1.3,
        n_laps_fit=len(rows),
    )
    # drivers with no clean laps (early retirements) get the field's median base pace
    missing = [d for d in race.driver_numbers() if d not in model.base]
    if missing:
        fill = float(median(model.base.values()))
        for d in missing:
            model.base[d] = fill
        model.drivers = sorted(model.base)
        model.notes.append(f"base pace filled with the field median for {missing}")
    model.pit_loss, model.pit_in_share = _estimate_pit_loss(race, model)
    model.sc_lap_time, model.vsc_factor = _estimate_neutralised_pace(race, model)
    return model


def _predict_lap(race: Race, model: LapModel, driver: int, n: int) -> float | None:
    ca = race.compound_on_lap(driver, n)
    if ca is None:
        return None
    return model.clean_lap_time(driver, ca[0], ca[1], race.total_laps - n)


def _estimate_pit_loss(race: Race, model: LapModel) -> tuple[float, float]:
    """Median over stops of (in-lap + out-lap) minus two predicted clean laps, and the share of
    that loss that lands on the in-lap (the rest lands on the out-lap)."""
    losses, shares = [], []
    for driver, stops in race.pit_laps.items():
        for lap_n, _lane in stops:
            if lap_n in race.sc_laps or lap_n in race.vsc_laps or (lap_n + 1) in race.sc_laps or (lap_n + 1) in race.vsc_laps:
                continue
            a, b = race.laps.get((driver, lap_n)), race.laps.get((driver, lap_n + 1))
            if a is None or b is None or a.duration is None or b.duration is None:
                continue
            pa, pb = _predict_lap(race, model, driver, lap_n), _predict_lap(race, model, driver, lap_n + 1)
            if pa is None or pb is None:
                continue
            loss = a.duration + b.duration - pa - pb
            if loss <= 0:
                continue
            losses.append(loss)
            shares.append(min(0.95, max(0.05, (a.duration - pa) / loss)))
    if not losses:
        model.notes.append("no clean pit stops; pit loss defaulted to 21 s, half on each lap")
        return 21.0, 0.5
    return float(median(losses)), float(median(shares))


def _estimate_neutralised_pace(race: Race, model: LapModel) -> tuple[float, float]:
    sc_times, vsc_ratios = [], []
    for (driver, n), lap in race.laps.items():
        if lap.duration is None or lap.pit_out or n in race.in_lap_set(driver) or n <= 1:
            continue
        if n in race.sc_laps:
            sc_times.append(lap.duration)
        elif n in race.vsc_laps:
            p = _predict_lap(race, model, driver, n)
            if p:
                vsc_ratios.append(lap.duration / p)
    clean_med = median(r[2] for r in clean_laps(race))
    sc = float(median(sc_times)) if sc_times else clean_med * 1.35
    vsc = float(median(vsc_ratios)) if vsc_ratios else 1.3
    if not sc_times:
        model.notes.append("no safety car laps; SC pace defaulted to 1.35x clean")
    if not vsc_ratios:
        model.notes.append("no VSC laps; VSC factor defaulted to 1.3")
    return sc, vsc
