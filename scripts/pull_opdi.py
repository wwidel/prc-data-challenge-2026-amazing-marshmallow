"""Download the OPDI v002 flight lists for the `opdi` block (same-aircraft ground time).

OPDI — EUROCONTROL PRC and the OpenSky Network's Open Performance Data Initiative, open data
the organiser accepted (CC BY 4.0 per the organiser, as recorded in kind-mango's README). One
parquet per month (20–55 MB); 2025 for training, and January, June (for 1 July's previous legs)
and July 2026 for the ranking months.

    uv run python scripts/pull_opdi.py [YYYYMM ...]
"""

from __future__ import annotations

import subprocess
import sys

from taxiout import paths

URL = "https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_list/flight_list_{m}.parquet"
MONTHS = tuple(f"2025{m:02d}" for m in range(1, 13)) + ("202601", "202606", "202607")
OUT = paths.DATA / "external" / "opdi"


def main(months: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bad = 0
    for m in months or MONTHS:
        dest = OUT / f"flight_list_{m}.parquet"
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"{dest.name}: present {dest.stat().st_size:,} bytes")
            continue
        r = subprocess.run(["curl", "-sfL", "--retry", "5", "-o", str(dest), URL.format(m=m)])
        size = dest.stat().st_size if dest.exists() else 0
        ok = r.returncode == 0 and size > 1_000_000
        bad += not ok
        print(f"{dest.name}: {'ok' if ok else 'FAILED'} {size:,} bytes")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
