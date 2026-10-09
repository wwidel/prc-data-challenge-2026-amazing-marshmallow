"""Pull stand (parking-position) names and coordinates from OpenStreetMap via Overpass.

Open data (ODbL). The Scenery Gateway packs leave LTFM at 0% stand coverage and EDDM/LSZH
partial; OSM maps the same aprons with `aeroway=parking_position` (and `aeroway=gate`)
carrying `ref` names. One query per airport against the aerodrome area keyed on its `icao`
tag; way/relation members come back with their centroid via `out center`.

    uv run python scripts/pull_osm_stands.py [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import polars as pl

from taxiout import paths

AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
#: The main Overpass instance first, then a mirror of the same OSM data for when it is overloaded.
OVERPASS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

QUERY = """
[out:json][timeout:90];
area["icao"="{icao}"]->.a;
(
  node(area.a)["aeroway"="parking_position"];
  way(area.a)["aeroway"="parking_position"];
  node(area.a)["aeroway"="gate"];
);
out center tags;
"""


def fetch(icao: str, endpoint: str) -> dict:
    data = urllib.parse.urlencode({"data": QUERY.format(icao=icao)}).encode()
    req = urllib.request.Request(endpoint, data=data, headers={"User-Agent": "taxiout-research"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = paths.DATA / "external" / "osm"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for icao in AIRPORTS:
        raw = out_dir / f"{icao}.json"
        if args.force or not raw.exists():
            attempts = 8
            for attempt in range(attempts):
                endpoint = OVERPASS[attempt % len(OVERPASS)]
                try:
                    raw.write_text(json.dumps(fetch(icao, endpoint)))
                    break
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                    code = getattr(e, "code", None)
                    # overloaded or refusing (5xx, 406, 429, no answer): try the next endpoint
                    busy = code is None or code in (406, 429) or code >= 500
                    if not busy or attempt == attempts - 1:
                        raise
                    wait = 30 * (attempt + 1)
                    print(f"{icao}: {code or type(e).__name__} from {endpoint}, retry in {wait}s")
                    time.sleep(wait)
            time.sleep(15)
        doc = json.loads(raw.read_text())
        n = 0
        for el in doc.get("elements", []):
            tags = el.get("tags", {})
            ref = tags.get("ref")
            lat = el.get("lat") or el.get("center", {}).get("lat")
            lon = el.get("lon") or el.get("center", {}).get("lon")
            if ref and lat is not None:
                rows.append(
                    {
                        "airport": icao,
                        "stand": str(ref),
                        "lat": float(lat),
                        "lon": float(lon),
                        "kind": tags.get("aeroway", ""),
                    }
                )
                n += 1
        print(f"{icao}: {n} named positions")
    table = pl.DataFrame(rows).unique(["airport", "stand"], keep="first")
    table.write_parquet(out_dir / "osm_stands.parquet")
    print(f"combined: {table.height:,} -> {out_dir / 'osm_stands.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
