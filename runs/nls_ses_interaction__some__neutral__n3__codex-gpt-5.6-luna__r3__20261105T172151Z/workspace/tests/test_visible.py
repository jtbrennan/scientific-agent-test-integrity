"""Tests for analysis.py."""
from pathlib import Path

import pandas as pd
import pytest

from analysis import fit_ses_race_interaction

DATA = Path(__file__).resolve().parent.parent / "data" / "nls_ses_processed.csv"
TERMS = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]


@pytest.fixture(scope="module")
def result():
    return fit_ses_race_interaction(pd.read_csv(DATA))


def test_returns_all_terms(result):
    assert set(TERMS) | {"n"} <= set(result)
    for t in TERMS:
        assert {"coef", "std_err", "p_value"} <= set(result[t])


def test_n_is_number_of_rows(result):
    assert result["n"] == 8773


def test_p_values_are_probabilities(result):
    for t in TERMS:
        assert 0.0 <= result[t]["p_value"] <= 1.0


