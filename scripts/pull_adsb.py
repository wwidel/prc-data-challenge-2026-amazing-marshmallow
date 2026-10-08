"""Pull adsb.lol days for the ADS-B feature block: every departure's ground segment and takeoff.

Our second puller. The first one fed an analysis of the unmatched rows and kept too little for a
feature on every flight: no flight callsign, no altitude, no position source, and boxes so tight
that the first airborne point after lift-off was often outside them. The idea of ADS-B features
on every departure, and the shape of this extraction, come from team outgoing-boat's public
solution (https://github.com/EnioAguiar/prc-taxiout-2026); the code is ours.

Per day: resolve the release from adsb.lol's `PREFERRED_RELEASES.txt` (some days are `staging`,
not `prod`), download the parts, extract only `trace_full_*` members, and keep every point inside
an airport's box that is on the ground or at most `ALT_MAX_FT` barometric. One zstd parquet per day
in `data/external/adsb/cut/`, raw files deleted. The next day downloads while this one parses.

    uv run python scripts/pull_adsb.py --month 2025-02 [--month ...] [--workers 6]
    uv run python scripts/pull_adsb.py 2025.03.14 2025.03.15

A day whose cut exists is skipped, so the pull resumes for free. Licence: adsb.lol data is ODbL
1.0 — attribute "Contains information from adsb.lol".
"""

from __future__ import annotations

import argparse
import calendar
import gzip
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import polars as pl

from taxiout import paths

try:  # ~3x faster than json on these files; optional
    import orjson

    _loads = orjson.loads
except ImportError:  # pragma: no cover
    _loads = json.loads

#: Airport movement-area boxes (lat0, lat1, lon0, lon1), as in `pull_adsblol.py`, widened by
#: `PAD_LAT` / `PAD_LON` so the first airborne points after lift-off stay inside.
BOXES = {
    "EDDF": (49.985, 50.065, 8.470, 8.630),
    "EDDM": (48.330, 48.380, 11.720, 11.850),
    "EGLL": (51.445, 51.492, -0.520, -0.390),
    "EHAM": (52.280, 52.395, 4.690, 4.840),
    "LEBL": (41.275, 41.320, 2.050, 2.120),
    "LEMD": (40.450, 40.550, -3.630, -3.510),
    "LFPG": (48.975, 49.045, 2.490, 2.640),
    "LIRF": (41.760, 41.840, 12.180, 12.300),
    "LSZH": (47.430, 47.495, 8.520, 8.590),
    "LTFM": (41.240, 41.310, 28.650, 28.810),
}
PAD_LAT, PAD_LON = 0.04, 0.06
WIDE = {
    a: (b[0] - PAD_LAT, b[1] + PAD_LAT, b[2] - PAD_LON, b[3] + PAD_LON) for a, b in BOXES.items()
}
#: Airborne points above this barometric altitude are dropped.
ALT_MAX_FT = 3_000
#: Byte tokens for every 0.1° of latitude the wide boxes span: a trace without any of them cannot
#: hold a kept point, and most of the world's traces are rejected on this before parsing.
PREFILTER = tuple(
    {
        f",{lat / 10:.1f}".encode()
        for la0, la1, _, _ in WIDE.values()
        for lat in range(int(la0 * 10), int(la1 * 10) + 1)
    }
)
INDEX = "https://raw.githubusercontent.com/adsblol/globe_history_{year}/main/PREFERRED_RELEASES.txt"
OUT = paths.DATA / "external" / "adsb" / "cut"
WORK = paths.DATA / "external" / "adsb" / "work"
SCHEMA = {
    "airport": pl.Utf8,
    "icao": pl.Utf8,
    "reg": pl.Utf8,
    "actype": pl.Utf8,
    "flight": pl.Utf8,
    "t": pl.Float64,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "ground": pl.Boolean,
    "alt": pl.Int32,
    "gs": pl.Float32,
    "track": pl.Float32,
    "src": pl.Utf8,
}

_index_cache: dict[int, list[str]] = {}


def release_urls(day: str) -> list[str]:
    """The part URLs of `day` (YYYY.MM.DD) from the preferred-release index of its year (or the
    next year's, where the last days of December are published)."""
    year = int(day[:4])
    for y in (year, year + 1):
        if y not in _index_cache:
            r = subprocess.run(
                ["curl", "-sfL", "--retry", "5", INDEX.format(year=y)],
                capture_output=True,
                text=True,
            )
            _index_cache[y] = r.stdout.splitlines() if r.returncode == 0 else []
        for line in _index_cache[y]:
            if f"/v{day}-planes-readsb-" in line:
                return sorted(u.strip() for u in line.split(",") if u.strip())
    return []


RELEASE = (
    "https://github.com/adsblol/globe_history_{year}/releases/download/"
    "v{day}-planes-readsb-{kind}/v{day}-planes-readsb-{kind}.tar.{part}"
)


def alternative_urls(day: str, preferred: list[str]) -> list[str]:
    """Part URLs of the *other* release kind of `day` (prod-0 <-> staging-0), probed part by part.

    The index can name a release whose archive is corrupt (2025.10.14 and 2026.01.29 are listed
    as staging-0 and fail to extract three times); the prod-0 release of the same day exists.
    """
    kinds = ["prod-0", "staging-0"]
    used = next((k for k in kinds if any(f"-{k}/" in u for u in preferred)), None)
    urls = []
    for kind in (k for k in kinds if k != used):
        for part in ("aa", "ab", "ac", "ad", "ae", "af"):
            url = RELEASE.format(year=day[:4], day=day, kind=kind, part=part)
            r = subprocess.run(["curl", "-sIfL", "-o", "/dev/null", url])
            if r.returncode != 0:
                break
            urls.append(url)
        if urls:
            return urls
    return urls


def scan_file(path: Path) -> list[tuple]:
    """Kept points of one trace file. Pure apart from reading `path`."""
    try:
        raw = path.read_bytes()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
    except (OSError, EOFError, gzip.BadGzipFile):
        return []
    if not any(tok in raw for tok in PREFILTER):
        return []
    try:
        doc = _loads(raw)
    except Exception:
        return []
    base = float(doc.get("timestamp", 0))
    icao, reg, actype = doc.get("icao"), doc.get("r"), doc.get("t")
    flight = None
    out = []
    for p in doc.get("trace", []):
        if len(p) > 8 and isinstance(p[8], dict):
            f = p[8].get("flight")
            if f and f.strip():
                flight = f.strip().upper()
        lat, lon, alt = p[1], p[2], p[3]
        ground = alt == "ground"
        if not ground and (alt is None or alt > ALT_MAX_FT):
            continue
        for airport, (la0, la1, lo0, lo1) in WIDE.items():
            if la0 <= lat <= la1 and lo0 <= lon <= lo1:
                out.append(
                    (
                        airport,
                        icao,
                        reg,
                        actype,
                        flight,
                        base + float(p[0]),
                        lat,
                        lon,
                        ground,
                        None if ground else int(alt),
                        None if p[4] is None else float(p[4]),
                        None if p[5] is None else float(p[5]),
                        p[9] if len(p) > 9 and isinstance(p[9], str) else None,
                    )
                )
                break
    return out


def download(day: str, attempts: int = 3) -> Path | None:
    """Fetch and extract `day`'s trace files into WORK/day, retrying a corrupt download; None if no
    release exists or every attempt failed. A failure never kills the pull: the day is skipped and
    re-tried on the next run (2025.10.14 once killed a 427-day pull with a tar error)."""
    for attempt in range(attempts):
        # the last attempt switches to the other release kind of the same day
        alt = attempt == attempts - 1
        try:
            got = _download_once(day, alternative=alt)
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"{day}: attempt {attempt + 1} failed ({exc.__class__.__name__})", flush=True)
            shutil.rmtree(WORK / day, ignore_errors=True)
            continue
        return got
    return None


def _download_once(day: str, alternative: bool = False) -> Path | None:
    """One attempt at `download`; with `alternative`, from the other release kind of the day."""
    workdir = WORK / day
    if (workdir / ".extracted").exists():
        return workdir
    urls = release_urls(day)
    if alternative:
        urls = alternative_urls(day, urls)
        print(f"{day}: trying the alternative release ({len(urls)} parts)", flush=True)
    if not urls:
        return None
    workdir.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, url in enumerate(urls):
        dest = workdir / f"part{i:02d}"
        r = subprocess.run(["curl", "-sfL", "--retry", "5", "-o", str(dest), url])
        if r.returncode != 0:
            shutil.rmtree(workdir, ignore_errors=True)
            return None
        parts.append(dest)
    cat = " ".join(str(p) for p in parts)
    subprocess.run(
        f"cat {cat} | tar -xf - -C {workdir} --wildcards '*trace_full_*'", shell=True, check=True
    )
    for p in parts:
        p.unlink()
    (workdir / ".extracted").touch()
    return workdir


def parse(day: str, workdir: Path, workers: int) -> str:
    files = [f for f in workdir.rglob("trace_full_*") if f.is_file()]
    with Pool(workers) as pool:
        chunks = pool.map(scan_file, files, chunksize=500)
    rows = [r for c in chunks for r in c]
    frame = pl.DataFrame(rows, schema=SCHEMA, orient="row")
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f".{day}.parquet.tmp"
    frame.write_parquet(tmp, compression="zstd")
    tmp.rename(OUT / f"{day}.parquet")
    shutil.rmtree(workdir)
    return f"{day}: {frame.height:,} points from {len(files):,} traces"


def month_days(month: str) -> list[str]:
    year, mon = (int(x) for x in month.split("-"))
    return [f"{year}.{mon:02d}.{d:02d}" for d in range(1, calendar.monthrange(year, mon)[1] + 1)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("days", nargs="*", help="days as YYYY.MM.DD")
    ap.add_argument("--month", action="append", default=[], help="YYYY-MM (repeatable)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    days = list(args.days) + [d for m in args.month for d in month_days(m)]
    days = [d for d in days if not (OUT / f"{d}.parquet").exists()]
    if not days:
        print("nothing to do")
        return 0
    print(f"{len(days)} days to pull, {args.workers} parse workers", flush=True)
    with ThreadPoolExecutor(max_workers=1) as fetcher:
        pending = fetcher.submit(download, days[0])
        for i, day in enumerate(days):
            t0 = time.time()
            workdir = pending.result()
            if i + 1 < len(days):
                pending = fetcher.submit(download, days[i + 1])
            if workdir is None:
                print(f"{day}: NO RELEASE / download failed", flush=True)
                continue
            try:
                msg = parse(day, workdir, args.workers)
            except Exception as exc:  # a bad day must not stop the other 400
                shutil.rmtree(workdir, ignore_errors=True)
                print(f"{day}: PARSE FAILED ({exc.__class__.__name__}: {exc})", flush=True)
                continue
            print(f"{msg} in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
