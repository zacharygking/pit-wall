"""A race, loaded from the cached OpenF1 JSONL into plain structures the model and simulator use."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .openf1 import DATA_DIR, load_jsonl

COMPOUNDS = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")


@dataclass(frozen=True)
class Stint:
    number: int
    compound: str
    lap_start: int
    lap_end: int
    age_at_start: int


@dataclass(frozen=True)
class Lap:
    driver: int
    number: int
    duration: float | None
    pit_out: bool
    date_start: str | None


@dataclass
class Race:
    session_key: int
    name: str
    year: int
    total_laps: int
    drivers: dict[int, dict]                      # driver_number -> {acronym, name, team}
    laps: dict[tuple[int, int], Lap]              # (driver, lap_number) -> Lap
    stints: dict[int, list[Stint]]                # driver -> stints in order
    pit_laps: dict[int, list[tuple[int, float]]]  # driver -> [(lap_number, lane_duration)]
    result: list[dict]                            # session_result rows sorted by position
    sc_laps: set[int] = field(default_factory=set)   # full safety car laps
    vsc_laps: set[int] = field(default_factory=set)  # virtual safety car laps
    red_flag_laps: set[int] = field(default_factory=set)

    # ---- convenience -------------------------------------------------------------------
    def driver_numbers(self) -> list[int]:
        return sorted(self.drivers)

    def laps_completed(self, driver: int) -> int:
        row = next((r for r in self.result if r["driver_number"] == driver), None)
        return int(row["number_of_laps"]) if row and row.get("number_of_laps") is not None else 0

    def classified_order(self) -> list[int]:
        """Finishing order by classification, DNFs after finishers in the order OpenF1 gives."""
        return [r["driver_number"] for r in self.result]

    def compound_on_lap(self, driver: int, lap: int) -> tuple[str, int] | None:
        """(compound, tyre age at the start of this lap) or None if the driver has no stint there."""
        for st in self.stints.get(driver, []):
            if st.lap_start <= lap <= st.lap_end:
                return st.compound, st.age_at_start + (lap - st.lap_start)
        return None

    def actual_strategy(self, driver: int) -> list[tuple[int, str]]:
        """[(lap to pit at the end of, compound fitted)] for every stint after the first."""
        out = []
        stints = self.stints.get(driver, [])
        for prev, nxt in zip(stints, stints[1:]):
            out.append((prev.lap_end, nxt.compound))
        return out

    def in_lap_set(self, driver: int) -> set[int]:
        return {lap for lap, _ in self.pit_laps.get(driver, [])}

    def cumulative_times(self, driver: int) -> dict[int, float]:
        """Cumulative race time at the end of each lap from the lap durations, where all are known."""
        out, total = {}, 0.0
        for n in range(1, self.total_laps + 1):
            lap = self.laps.get((driver, n))
            if lap is None or lap.duration is None:
                break
            total += lap.duration
            out[n] = total
        return out


_SC_ON = re.compile(r"^SAFETY CAR DEPLOYED", re.I)
_SC_OFF = re.compile(r"^SAFETY CAR IN THIS LAP", re.I)
_VSC_ON = re.compile(r"^VIRTUAL SAFETY CAR DEPLOYED", re.I)
_VSC_OFF = re.compile(r"^VIRTUAL SAFETY CAR ENDING", re.I)
_RED = re.compile(r"^RED FLAG", re.I)


def neutralised_laps(messages: list[dict], total_laps: int) -> tuple[set[int], set[int], set[int]]:
    """Laps run under a full safety car, a virtual safety car, and a red flag, from race control.

    A deployment on lap L neutralises lap L through the lap of the 'in this lap' / 'ending'
    message, inclusive. A window never closed is held to the end of the race.
    """
    sc, vsc, red = set(), set(), set()
    sc_open = vsc_open = None
    for m in sorted(messages, key=lambda r: r["date"]):
        text = (m.get("message") or "").strip()
        lap = m.get("lap_number")
        if lap is None:
            continue
        if _SC_ON.match(text) and sc_open is None:
            sc_open = lap
        elif _SC_OFF.match(text) and sc_open is not None:
            sc.update(range(sc_open, lap + 1))
            sc_open = None
        elif _VSC_ON.match(text) and vsc_open is None:
            vsc_open = lap
        elif _VSC_OFF.match(text) and vsc_open is not None:
            vsc.update(range(vsc_open, lap + 1))
            vsc_open = None
        elif _RED.match(text):
            red.add(lap)
    if sc_open is not None:
        sc.update(range(sc_open, total_laps + 1))
    if vsc_open is not None:
        vsc.update(range(vsc_open, total_laps + 1))
    return sc, vsc - sc, red


def load_race(session_key: int, data_dir: Path = DATA_DIR) -> Race:
    d = data_dir / str(session_key)
    session = load_jsonl(d / "session.jsonl")[0]
    drivers = {
        r["driver_number"]: {"acronym": r["name_acronym"], "name": r["full_name"], "team": r["team_name"]}
        for r in load_jsonl(d / "drivers.jsonl")
    }
    laps: dict[tuple[int, int], Lap] = {}
    for r in load_jsonl(d / "laps.jsonl"):
        laps[(r["driver_number"], r["lap_number"])] = Lap(
            driver=r["driver_number"],
            number=r["lap_number"],
            duration=r.get("lap_duration"),
            pit_out=bool(r.get("is_pit_out_lap")),
            date_start=r.get("date_start"),
        )
    stints: dict[int, list[Stint]] = {}
    for r in load_jsonl(d / "stints.jsonl"):
        if r.get("lap_start") is None or r.get("lap_end") is None:
            continue
        stints.setdefault(r["driver_number"], []).append(
            Stint(r["stint_number"], (r.get("compound") or "UNKNOWN").upper(), r["lap_start"], r["lap_end"], int(r.get("tyre_age_at_start") or 0))
        )
    for v in stints.values():
        v.sort(key=lambda s: s.number)
    pits: dict[int, list[tuple[int, float]]] = {}
    for r in load_jsonl(d / "pit.jsonl"):
        pits.setdefault(r["driver_number"], []).append((r["lap_number"], float(r.get("lane_duration") or r.get("pit_duration") or 0.0)))
    for v in pits.values():
        v.sort()
    result = sorted(load_jsonl(d / "session_result.jsonl"), key=lambda r: (r.get("position") is None, r.get("position") or 99))
    total_laps = max(int(r.get("number_of_laps") or 0) for r in result)
    sc, vsc, red = neutralised_laps(load_jsonl(d / "race_control.jsonl"), total_laps)
    return Race(
        session_key=session_key,
        name=f'{session["year"]} {session["country_name"]} ({session["circuit_short_name"]})',
        year=session["year"],
        total_laps=total_laps,
        drivers=drivers,
        laps=laps,
        stints=stints,
        pit_laps=pits,
        result=result,
        sc_laps=sc,
        vsc_laps=vsc,
        red_flag_laps=red,
    )
