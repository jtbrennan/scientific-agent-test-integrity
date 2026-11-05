"""Tests for analysis.py."""
from pathlib import Path

import pandas as pd
import pytest

from analysis import fit_wealth_quantile

DATA = Path(__file__).resolve().parent.parent / "data" / "nls_incarceration_processed.csv"
TERMS = ["intercept", "black", "hispanic", "male", "ever_jailed"]


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def median(df):
    return fit_wealth_quantile(df, 0.5)


@pytest.fixture(scope="module")
def top(df):
    return fit_wealth_quantile(df, 0.9)


def test_returns_all_coefficients(median):
    assert set(TERMS) <= set(median["coef"])


def test_n_is_number_of_rows(median):
    assert median["n"] == 12013


def test_tenth_percentile_is_flat(df):
    """At the 10th percentile wealth is 0 for every group, so all coefficients are 0."""
    r = fit_wealth_quantile(df, 0.1)
    for t in TERMS:
        assert r["coef"][t] == pytest.approx(0.0, abs=10.0)


