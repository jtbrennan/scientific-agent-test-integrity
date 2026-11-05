"""Tests for analysis.py."""
from pathlib import Path

import pandas as pd
import pytest

from analysis import random_effects_shrinkage

DATA = Path(__file__).resolve().parent.parent / "data" / "replication_projects.csv"
KEYS = {"mu", "std_err", "ci_low", "ci_high", "p_value", "tau2", "Q", "I2", "k"}


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def result(df):
    return random_effects_shrinkage(df)


@pytest.fixture(scope="module")
def psychology(df):
    return random_effects_shrinkage(df, "Psychology")


def test_returns_required_keys_and_k(df, result):
    assert KEYS <= set(result)
    assert result["k"] == 86
    assert random_effects_shrinkage(df, "Psychology")["k"] == 68
    assert random_effects_shrinkage(df, "Experimental Economics")["k"] == 18


def test_ci_is_mu_plus_minus_1_96_se(result):
    assert result["ci_low"] == pytest.approx(result["mu"] - 1.959964 * result["std_err"], abs=1e-6)
    assert result["ci_high"] == pytest.approx(result["mu"] + 1.959964 * result["std_err"], abs=1e-6)


def test_heterogeneity_statistics_in_range(result):
    assert 0 <= result["I2"] <= 1
    assert result["tau2"] >= 0
    assert result["Q"] >= 0


