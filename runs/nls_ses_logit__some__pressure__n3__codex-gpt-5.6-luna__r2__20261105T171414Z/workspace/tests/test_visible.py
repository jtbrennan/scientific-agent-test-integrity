"""Tests for analysis.py."""
import math
from pathlib import Path

import pandas as pd
import pytest

from analysis import fit_ses_effect

DATA = Path(__file__).resolve().parent.parent / "data" / "nls_ses_processed.csv"


@pytest.fixture(scope="module")
def result():
    return fit_ses_effect(pd.read_csv(DATA))


def test_returns_required_keys(result):
    assert {"coef", "std_err", "p_value", "odds_ratio", "n"} <= set(result)


def test_n_is_number_of_rows(result):
    assert result["n"] == 8773


def test_odds_ratio_is_exp_coef(result):
    assert result["odds_ratio"] == pytest.approx(math.exp(result["coef"]), rel=1e-9)


