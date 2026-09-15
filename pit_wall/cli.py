"""Command line.

  python -m pit_wall fetch 9472            cache one session
  python -m pit_wall fetch-season 2024     cache every race of a season
  python -m pit_wall fit 9472              print the fitted lap-time model
  python -m pit_wall replay 9472           replay the real strategies, print the calibration
  python -m pit_wall season 2024           calibration table for every cached race of a season
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .calibrate import calibrate, to_json
from .model import fit
from .openf1 import DATA_DIR, cache_session, race_sessions
from .race import load_race


def _fmt_model(model) -> str:
    lines = [f"session {model.session_key}: {model.n_laps_fit} clean laps, sigma {model.sigma:.3f} s"]
    lines.append(f"  fuel {model.fuel:+.4f} s per lap of fuel remaining; pit loss {model.pit_loss:.1f} s ({model.pit_in_share:.0%} on the in-lap); "
                 f"SC lap {model.sc_lap_time:.1f} s; VSC x{model.vsc_factor:.2f}")
    for c in model.compounds:
        lines.append(f"  {c:<7} offset {model.offset[c]:+.3f}  deg {model.deg1[c]:+.4f} s/lap  deg2 {model.deg2[c]:+.5f}")
    for d in sorted(model.base, key=model.base.get):
        lines.append(f"  #{d:<3} base {model.base[d]:.3f}")
    for n in model.notes:
        lines.append(f"  note: {n}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pit_wall")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fetch"); s.add_argument("session_key", type=int); s.add_argument("--force", action="store_true")
    s = sub.add_parser("fetch-season"); s.add_argument("year", type=int)
    s = sub.add_parser("fit"); s.add_argument("session_key", type=int)
    s = sub.add_parser("replay"); s.add_argument("session_key", type=int); s.add_argument("--json", action="store_true")
    s = sub.add_parser("season"); s.add_argument("year", type=int); s.add_argument("--out", type=Path)
    a = p.parse_args(argv)

    if a.cmd == "fetch":
        out = cache_session(a.session_key, force=a.force)
        print(f"cached {a.session_key} -> {out}")
    elif a.cmd == "fetch-season":
        for row in race_sessions(a.year):
            out = cache_session(row["session_key"])
            print(f"cached {row['session_key']} {row['country_name']} -> {out}")
    elif a.cmd == "fit":
        print(_fmt_model(fit(load_race(a.session_key))))
    elif a.cmd == "replay":
        cal, model, sim = calibrate(load_race(a.session_key))
        if a.json:
            print(to_json(cal))
        else:
            race = load_race(a.session_key)
            print(_fmt_model(model))
            print()
            print(f"{'pos':>3} {'actual':<8} {'sim':<8} {'d':>3}")
            actual = race.classified_order()
            for i, d in enumerate(actual, 1):
                sp = sim.position_of(d) if d in sim.order else None
                print(f"{i:>3} {race.drivers[d]['acronym']:<8} {race.drivers.get(sim.order[i-1], {}).get('acronym','-') if i-1 < len(sim.order) else '-':<8} {('' if sp is None else f'{sp - i:+d}'):>3}")
            print()
            print(cal.row())
    elif a.cmd == "season":
        rows = []
        for d in sorted(DATA_DIR.iterdir()):
            if not d.is_dir():
                continue
            try:
                race = load_race(int(d.name))
            except Exception:
                continue
            if race.year != a.year:
                continue
            try:
                cal, _, _ = calibrate(race)
            except ValueError as exc:
                print(f"skip {race.name}: {exc}", file=sys.stderr)
                continue
            rows.append(cal)
            print(cal.row())
        if rows:
            import statistics
            print(f"\n{len(rows)} races: median tau {statistics.median(r.kendall_tau for r in rows):.2f}, "
                  f"median |dpos| {statistics.median(r.mean_abs_position_error for r in rows):.2f}, "
                  f"median gap RMSE {statistics.median(r.gap_to_leader_rmse for r in rows):.1f}s")
        if a.out:
            a.out.write_text(json.dumps([json.loads(to_json(r)) for r in rows], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
