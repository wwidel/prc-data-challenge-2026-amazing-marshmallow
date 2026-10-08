"""The light tier's fit and rules, on a synthetic frame. Pure logic."""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
import pytest

from taxiout import paths

sys.path.insert(0, str(paths.ROOT / "scripts"))
import lite  # noqa: E402


def frame(n: int = 400, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    pa = rng.uniform(300, 1500, n)
    pe = pa + rng.uniform(-200, 600, n)
    return pl.DataFrame(
        {
            "mvt_id": np.arange(n),
            "airport": ["EDDF"] * (n // 2) + ["LIRF"] * (n - n // 2),
            "stand": rng.choice(["A1", "B2"], n),
            "pa": pa,
            "pe": pe,
            "y": 100.0 + 0.5 * pa + 0.25 * pe,
            "unm": [False] * n,
            "gap": rng.uniform(0, 3000, n),
        }
    )


def test_fit_recovers_per_airport_coefficients():
    coefs, _, _ = lite.fit(frame())
    for row in coefs.iter_rows(named=True):
        assert row["b0"] == pytest.approx(100.0, abs=1e-6)
        assert row["b_pa"] == pytest.approx(0.5, abs=1e-9)
        assert row["b_pe"] == pytest.approx(0.25, abs=1e-9)


def test_rules_fire_only_on_their_rows_and_missing_stamps_fall_back():
    train = frame()
    test = pl.DataFrame(
        {
            "mvt_id": [1, 2, 3, 4],
            "airport": ["LIRF", "LIRF", "EDDF", "EDDF"],
            "stand": ["A1", "A1", "A1", "Z9"],
            "pa": [None, 20_000.0, 600.0, None],
            "pe": [None, 20_100.0, 700.0, None],
            "y": [None, None, None, None],
            "unm": [True, False, False, True],
            "gap": [50_000.0, 19_000.0, 100.0, 100.0],
        },
        schema_overrides={"y": pl.Float64, "pa": pl.Float64, "pe": pl.Float64},
    )
    got = lite.predict(test, lite.fit(train))["prediction"].to_list()
    assert got[0] == 50_000.0, "unmatched LIRF, gap > 12 h: the gap"
    # matched, takeoff - aobt_3 > 4 h: that value, then the gap bound (gap + 1800 = 20,800)
    assert got[1] == 20_000.0
    assert got[2] == pytest.approx(100.0 + 0.5 * 600 + 0.25 * 700)
    # no stamps and an unknown stand: the airport median, capped by the gap bound (100 + 1800)
    ap_med = float(train.filter(pl.col("airport") == "EDDF")["y"].median())
    assert got[3] == pytest.approx(min(ap_med, 1_900.0))
