"""Pull METAR observations for the ten airports from the Iowa Environmental Mesonet archive.

Open data (IEM ASOS/METAR download service), UTC timestamps, routine reports and specials.
The window starts mid-December 2024 so backward-rolling features have history before the
first training month, and ends after July 2026. Raw CSVs are archived per airport under
data/external/metar/; the combined, unit-converted table lands in metar.parquet.

    uv run python scripts/pull_metar.py [--force]
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import polars as pl

from taxiout import paths

AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
FIELDS = ["tmpf", "dwpf", "relh", "drct", "sknt", "gust", "p01i", "vsby", "wxcodes"]


def url_for(icao: str) -> str:
    params = [
        ("station", icao),
        *[("data", f) for f in FIELDS],
        ("year1", "2024"),
        ("month1", "12"),
        ("day1", "15"),
        ("year2", "2026"),
        ("month2", "8"),
        ("day2", "1"),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("elev", "no"),
        ("missing", "empty"),
        ("trace", "0.0001"),
        ("report_type", "3"),
        ("report_type", "4"),
    ]
    return BASE + "?" + urllib.parse.urlencode(params)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = paths.DATA / "external" / "metar"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for icao in AIRPORTS:
        raw = out_dir / f"{icao}.csv"
        if args.force or not raw.exists():
            req = urllib.request.Request(url_for(icao), headers={"User-Agent": "taxiout-research"})
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(req, timeout=180) as r:
                        raw.write_bytes(r.read())
                    break
                except urllib.error.HTTPError as e:
                    if e.code != 429 or attempt == 4:
                        raise
                    wait = 60 * (attempt + 1)
                    print(f"{icao}: 429, backing off {wait}s")
                    time.sleep(wait)
            time.sleep(5)
        d = pl.read_csv(raw, null_values=[""], infer_schema_length=10000, glob=False)
        frames.append(
            d.select(
                pl.lit(icao).alias("airport"),
                pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC").alias("time"),
                ((pl.col("tmpf") - 32) * 5 / 9).alias("temp_c"),
                ((pl.col("dwpf") - 32) * 5 / 9).alias("dewpoint_c"),
                pl.col("relh").alias("rh"),
                pl.col("drct").alias("wind_dir"),
                pl.col("sknt").alias("wind_kt"),
                pl.col("gust").alias("gust_kt"),
                (pl.col("p01i") * 25.4).alias("precip_mm"),
                (pl.col("vsby") * 1.609344).alias("vis_km"),
                pl.col("wxcodes").cast(pl.Utf8).alias("wx"),
            )
        )
        print(
            f"{icao}: {frames[-1].height:,} obs "
            f"({frames[-1]['time'].min()} .. {frames[-1]['time'].max()})"
        )
    table = pl.concat(frames).sort("airport", "time")
    table.write_parquet(out_dir / "metar.parquet")
    print(f"combined: {table.height:,} rows -> {out_dir / 'metar.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
