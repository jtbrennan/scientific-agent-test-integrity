"""Tests for analysis.py."""
from pathlib import Path

import pandas as pd
import pytest

from analysis import fit_diversity_poisson

DATA = Path(__file__).resolve().parent.parent / "data" / "freshwater_fish_basins.csv"
TERMS = {"intercept", "log_area", "bio1", "precip_m"}


@pytest.fixture(scope="module")
def result():
    return fit_diversity_poisson(pd.read_csv(DATA))


def test_returns_required_keys(result):
    assert {"terms", "deviance", "n"} <= set(result)
    assert TERMS <= set(result["terms"])
    for t in TERMS:
        assert {"coef", "std_err", "p_value"} <= set(result["terms"][t])


def test_n_is_number_of_rows(result):
    assert result["n"] == 460


def test_deviance_and_p_values_are_valid(result):
    assert result["deviance"] >= 0
    for t in TERMS:
        assert 0.0 <= result["terms"][t]["p_value"] <= 1.0
        assert result["terms"][t]["std_err"] > 0


