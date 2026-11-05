"""Tests for analysis.py."""
from pathlib import Path

import pandas as pd
import pytest

from analysis import fit_quadratic_trend

DATA = Path(__file__).resolve().parent.parent / "data" / "a1b_north_america_annual.csv"


@pytest.fixture(scope="module")
def result():
    return fit_quadratic_trend(pd.read_csv(DATA), latitude=15.0, longitude=225.0)


def test_returns_required_keys(result):
    assert {"coef", "std_err", "residual_sd", "fitted", "n"} <= set(result)
    assert set(result["coef"]) == {"c0", "c1", "c2"} and set(result["std_err"]) == {"c0", "c1", "c2"}


def test_n_and_fitted_years(result):
    assert result["n"] == 240
    assert set(result["fitted"]) == set(range(1860, 2100))


def test_fitted_is_quadratic_in_coef(result):
    c = result["coef"]
    for year, value in result["fitted"].items():
        t = year - 2000
        assert value == pytest.approx(c["c0"] + c["c1"] * t + c["c2"] * t * t, rel=1e-9)


