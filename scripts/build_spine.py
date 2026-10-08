"""Build the canonical spine. Idempotent unless --overwrite is passed."""

from __future__ import annotations

import argparse

from taxiout import paths, spine


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--months", nargs="*", default=list(paths.ALL_MONTHS))
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="rebuild existing months (invalidates every cached feature)",
    )
    args = ap.parse_args()

    built = spine.build(tuple(args.months), overwrite=args.overwrite)
    total = sum(built.values())
    for month, n in sorted(built.items()):
        print(f"  {month}  {n:>9,}")
    print(f"  {'total':<7} {total:>9,}")


if __name__ == "__main__":
    main()
