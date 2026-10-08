"""Pull the ten airports' recommended scenery packs from the X-Plane Scenery Gateway and
extract their stand tables (apt.dat row code 1300: lat, lon, heading, type, aircraft, name).

The stand *names* in these packs match our `stand` column (verified at LIRF), which makes
them the join key between an adsb.lol surface trace and a specific departure record. Raw
packs are archived under data/external/gateway/ before parsing; the parsed table lands in
data/external/gateway/stands.parquet.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import urllib.request
import zipfile

import polars as pl

from taxiout import paths

AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
API = "https://gateway.x-plane.com/apiv1"


def fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def parse_stands(scenery: dict, icao: str) -> list[dict]:
    blob = zipfile.ZipFile(io.BytesIO(base64.b64decode(scenery["masterZipBlob"])))
    dat = next(n for n in blob.namelist() if n.upper().endswith(f"{icao}.DAT".upper()))
    rows = []
    for line in blob.read(dat).decode("utf8", errors="replace").splitlines():
        parts = line.split()
        if parts[:1] == ["1300"]:
            rows.append(
                {
                    "airport": icao,
                    "lat": float(parts[1]),
                    "lon": float(parts[2]),
                    "heading": float(parts[3]),
                    "kind": parts[4],
                    "stand": " ".join(parts[6:]),
                }
            )
    return rows


def parse_runways(scenery: dict, icao: str) -> list[dict]:
    """apt.dat row code 100: land runway with both ends' idents and coordinates."""
    blob = zipfile.ZipFile(io.BytesIO(base64.b64decode(scenery["masterZipBlob"])))
    dat = next(n for n in blob.namelist() if n.upper().endswith(f"{icao}.DAT".upper()))
    rows = []
    for line in blob.read(dat).decode("utf8", errors="replace").splitlines():
        parts = line.split()
        if parts[:1] == ["100"] and len(parts) >= 20:
            for ident, lat, lon in (
                (parts[8], parts[9], parts[10]),
                (parts[17], parts[18], parts[19]),
            ):
                rows.append(
                    {
                        "airport": icao,
                        "runway": ident,
                        "thr_lat": float(lat),
                        "thr_lon": float(lon),
                    }
                )
    return rows


def build_runways() -> int:
    """Extract runway-threshold coordinates from the cached scenery packs."""
    out_dir = paths.DATA / "external" / "gateway"
    rows = []
    for icao in AIRPORTS:
        scenery = json.loads((out_dir / f"{icao}_scenery.json").read_text())["scenery"]
        got = parse_runways(scenery, icao)
        rows.extend(got)
        print(f"{icao}: {len(got)} runway ends")
    pl.DataFrame(rows).write_parquet(out_dir / "runways.parquet")
    print(f"-> {out_dir / 'runways.parquet'}")
    return 0


def list_scenery(icao: str) -> int:
    """Print every scenery submission for one airport, newest first."""
    packs = fetch(f"{API}/airport/{icao}")["airport"]["scenery"]
    for pk in sorted(packs, key=lambda x: x.get("DateAccepted") or "", reverse=True):
        print(
            f"id={pk['sceneryId']}  accepted={pk.get('DateAccepted')}  "
            f"status={pk.get('Status')}  artist={pk.get('userName')}"
        )
    return 0


def try_scenery(icao: str, scenery_id: int) -> int:
    """Extract one pack's stands to data/external/gateway/stands_<icao>_<id>.parquet."""
    scenery = fetch(f"{API}/scenery/{scenery_id}")["scenery"]
    rows = parse_stands(scenery, icao)
    out = paths.DATA / "external" / "gateway" / f"stands_{icao}_{scenery_id}.parquet"
    pl.DataFrame(rows).write_parquet(out)
    names = sorted({r["stand"] for r in rows})
    print(f"{icao} scenery {scenery_id}: {len(rows)} stands -> {out}")
    print("name sample:", names[:15])
    return 0


def main() -> int:
    out_dir = paths.DATA / "external" / "gateway"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for icao in AIRPORTS:
        raw_path = out_dir / f"{icao}_scenery.json"
        if not raw_path.exists():
            airport = fetch(f"{API}/airport/{icao}")["airport"]
            scenery = fetch(f"{API}/scenery/{airport['recommendedSceneryId']}")
            raw_path.write_text(json.dumps(scenery))
        scenery = json.loads(raw_path.read_text())["scenery"]
        blob = zipfile.ZipFile(io.BytesIO(base64.b64decode(scenery["masterZipBlob"])))
        dat = next(n for n in blob.namelist() if n.upper().endswith(f"{icao}.DAT".upper()))
        for line in blob.read(dat).decode("utf8", errors="replace").splitlines():
            parts = line.split()
            if parts[:1] == ["1300"]:
                rows.append(
                    {
                        "airport": icao,
                        "lat": float(parts[1]),
                        "lon": float(parts[2]),
                        "heading": float(parts[3]),
                        "kind": parts[4],
                        "stand": " ".join(parts[6:]),
                    }
                )
        print(f"{icao}: {sum(1 for r in rows if r['airport'] == icao)} stands")
    table = pl.DataFrame(rows)
    table.write_parquet(out_dir / "stands.parquet")
    print(f"wrote {out_dir / 'stands.parquet'} ({table.height} stands)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--runways":
        sys.exit(build_runways())
    if len(sys.argv) >= 3 and sys.argv[1] == "--list-scenery":
        sys.exit(list_scenery(sys.argv[2]))
    if len(sys.argv) >= 4 and sys.argv[1] == "--scenery-id":
        sys.exit(try_scenery(sys.argv[2], int(sys.argv[3])))
    sys.exit(main())
