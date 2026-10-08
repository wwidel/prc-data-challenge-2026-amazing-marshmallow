"""Unit tests for `eurocontrol_daily`'s table: pure logic, no files."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from taxiout.features.eurocontrol_daily import OUTPUTS, daily

D = date(2026, 1, 5)


def test_ratios_and_full_join():
    slot = pl.DataFrame(
        {
            "airport": ["X"],
            "day": [D],
            "FLT_DEP_1": [400.0],
            "FLT_DEP_REG_1": [100.0],
            "FLT_DEP_OUT_LATE_1": [10.0],
            "FLT_DEP_OUT_EARLY_1": [5.0],
        }
    )
    pre = pl.DataFrame(
        {
            "airport": ["X", "Y"],
            "day": [D, D],
            "FLT_DEP_IFR_2": [380.0, 0.0],
            "DLY_ALL_PRE_2": [3800.0, 5.0],
        }
    )
    atc = pl.DataFrame(
        {"airport": ["X"], "day": [D], "FLT_DEP_3": [200.0], "DLY_ATC_PRE_3": [100.0]}
    )
    t = {r["airport"]: r for r in daily(slot, pre, atc).to_dicts()}
    assert set(t["X"]) == {"airport", "day", *OUTPUTS}
    assert t["X"]["ec_reg_frac"] == pytest.approx(0.25)
    assert t["X"]["ec_late_frac"] == pytest.approx(0.10)
    assert t["X"]["ec_early_frac"] == pytest.approx(0.05)
    assert t["X"]["ec_pre_min"] == pytest.approx(10.0)
    assert t["X"]["ec_atc_min"] == pytest.approx(0.5)
    # Y appears in one series only; a zero denominator gives null, not inf
    assert t["Y"]["ec_pre_min"] is None and t["Y"]["ec_dep_n"] is None
