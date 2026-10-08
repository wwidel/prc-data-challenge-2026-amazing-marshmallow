"""Download EUROCONTROL's daily per-airport series for the `eurocontrol_daily` block.

Public EUROCONTROL performance data (free use with attribution: "Source: EUROCONTROL Performance
Review Unit, https://ansperformance.eu/csv/"). Three daily series, 2025 and 2026, ~4 MB each
(plus the delay-by-cause files and the monthly taxi-out indicator, below):
``atfm_slot_adherence``, ``all_pre_departure_delays``, ``atc_pre_departure_delays``. The idea comes
from team outgoing-boat's public solution (https://github.com/EnioAguiar/prc-taxiout-2026).

    uv run python scripts/pull_eurocontrol.py [NAME ...]

With names, only those series are fetched — so a new file can be added without replacing the
three CSVs ``eurocontrol_daily`` was built from (its cache key does not see them).
"""

from __future__ import annotations

import subprocess
import sys

from taxiout import paths

URL = "https://www.eurocontrol.int/performance/data/download/csv/{name}_{year}.csv"
SERIES = ("atfm_slot_adherence", "all_pre_departure_delays", "atc_pre_departure_delays")
#: Published bz2-compressed; decompressed on download. ``apt_dly``: arrival ATFM delay per
#: airport-day BY CAUSE (weather, de-icing, aerodrome capacity, ATC capacity/staffing, industrial
#: action, special event, accident, equipment, environment, ...) — the structured record of
#: disruptions. Tested in scripts/analysis/a118_apt_dly.py and not used by the best submission.
#: ``ert_dly_ansp``: en-route ATFM delay per ANSP-day by the same causes; likewise.
SERIES_BZ2 = ("apt_dly", "ert_dly_ansp")
#: Monthly per airport, plain CSV: the APDF taxi-out indicator (flights, valid flights, flights
#: without a reference, reference and additional minutes). Its 2026 rows are a published level to
#: check our predictions against; the organisers confirmed it may be used (a119, a120).
MONTHLY = ("taxi_out_additional_time",)
YEARS = (2025, 2026)
OUT = paths.DATA / "external" / "eurocontrol"


def main(names: list[str]) -> int:
    known = SERIES + MONTHLY + SERIES_BZ2
    unknown = [n for n in names if n not in known]
    if unknown:
        print(f"unknown series {unknown}; known: {known}")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    for name in SERIES + MONTHLY:
        if names and name not in names:
            continue
        for year in YEARS:
            dest = OUT / f"{name}_{year}.csv"
            r = subprocess.run(
                ["curl", "-sfL", "--retry", "5", "-o", str(dest), URL.format(name=name, year=year)]
            )
            size = dest.stat().st_size if dest.exists() else 0
            print(f"{dest.name}: {'ok' if r.returncode == 0 else 'FAILED'} {size:,} bytes")
    for name in SERIES_BZ2:
        if names and name not in names:
            continue
        for year in YEARS:
            dest = OUT / f"{name}_{year}.csv"
            url = URL.format(name=name, year=year) + ".bz2"
            r = subprocess.run(f"curl -sfL --retry 5 '{url}' | bzcat > '{dest}'", shell=True)
            size = dest.stat().st_size if dest.exists() else 0
            print(f"{dest.name}: {'ok' if r.returncode == 0 and size else 'FAILED'} {size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
