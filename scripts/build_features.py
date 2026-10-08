"""Build feature blocks into the store. Only rebuilds what changed."""

from __future__ import annotations

import argparse

from taxiout import store
from taxiout.features import REGISTRY, registry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("blocks", nargs="*", help="block names; default is every cacheable block")
    ap.add_argument("--overwrite", action="store_true", help="rebuild even if the hash matches")
    ap.add_argument("--prune", action="store_true", help="delete cached output under stale hashes")
    ap.add_argument("--list", action="store_true", help="list registered blocks and exit")
    args = ap.parse_args()

    if args.list:
        for name, block in sorted(REGISTRY.items()):
            state = "cached" if block.kind.cacheable else "per-fold"
            print(f"  {name:<14} {block.kind:<17} {state:<9} {len(block.outputs):>2} cols")
            print(f"  {'':14} {block.doc.splitlines()[0] if block.doc else ''}")
        return

    names = args.blocks or [n for n, b in REGISTRY.items() if b.kind.cacheable]
    for name in names:
        block = registry.get(name)
        result = store.build(block, overwrite=args.overwrite)
        state = "built" if result.rebuilt else "cached"
        print(f"  {name:<14} {result.hash}  {state:<7} {len(block.outputs):>2} cols")
        if args.prune:
            for stale in store.prune(block):
                print(f"  {'':14} pruned stale hash {stale}")


if __name__ == "__main__":
    main()
