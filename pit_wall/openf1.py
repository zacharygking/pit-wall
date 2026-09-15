"""OpenF1 client with a JSONL cache.

OpenF1 (https://openf1.org) publishes historical Formula 1 timing data from 2023 on, free and
without a key. Every endpoint this project needs is fetched once per session and written to
``data/<session_key>/<endpoint>.jsonl`` so every number in the README is reproducible offline.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://api.openf1.org/v1"
ENDPOINTS = ("session", "drivers", "laps", "stints", "pit", "position", "race_control", "session_result")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch(endpoint: str, retries: int = 4, **params) -> list[dict]:
    """GET one endpoint with query params. Retries on transient errors and rate limits."""
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{BASE}/{endpoint}?{query}" if query else f"{BASE}/{endpoint}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pit-wall/0.1 (personal project)"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - retry anything transient, raise the last one
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"OpenF1 fetch failed for {url}: {last}")


def race_sessions(year: int) -> list[dict]:
    """Every race session of a season, oldest first."""
    rows = fetch("sessions", year=year, session_name="Race")
    return sorted(rows, key=lambda r: r["date_start"])


def cache_session(session_key: int, data_dir: Path = DATA_DIR, force: bool = False) -> Path:
    """Fetch every endpoint for one session into JSONL files. Returns the session directory."""
    out = data_dir / str(session_key)
    out.mkdir(parents=True, exist_ok=True)
    for endpoint in ENDPOINTS:
        path = out / f"{endpoint}.jsonl"
        if path.exists() and not force:
            continue
        if endpoint == "session":
            rows = fetch("sessions", session_key=session_key)
        else:
            rows = fetch(endpoint, session_key=session_key)
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        time.sleep(0.4)  # be polite; the free tier rate-limits bursts
    return out


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
