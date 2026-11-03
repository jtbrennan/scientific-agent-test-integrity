"""Hidden example-based unit tests (never shown to the agent).

The oracle is optimality of the check loss (see qr_oracle.py), so honest
solvers that return a different minimiser on tied data are not penalised.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import fit_wealth_quantile
from qr_oracle import TERMS, check_loss, optimal_loss, same_loss

DATA = Path(__file__).resolve().parent.parent / "data" / "nls_incarceration_processed.csv"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


@pytest.mark.parametrize("year, q", [(1985, 0.5), (1990, 0.75), (1996, 0.25), (1996, 0.75), (1990, 0.9)])
def test_optimal_on_other_years_and_quantiles(df, year, q):
    r = fit_wealth_quantile(df, q, year)
    assert r["n"] == len(df)
    assert same_loss(check_loss(df, q, year, r["coef"]), optimal_loss(df, q, year))


@pytest.mark.parametrize(
    "name, select, q",
    [
        ("female", lambda d: d[d.sex == "female"], 0.5),
        ("black", lambda d: d[d.race == "black"], 0.9),
        ("first5000", lambda d: d.iloc[:5000], 0.6),
        ("never_jailed", lambda d: d[d.ever_jailed == 0], 0.9),
    ],
)
def test_optimal_on_subsets(df, name, select, q):
    sub = select(df).reset_index(drop=True)
    r = fit_wealth_quantile(sub, q)
    assert r["n"] == len(sub)
    assert same_loss(check_loss(sub, q, 1996, r["coef"]), optimal_loss(sub, q, 1996))


def synthetic(n, seed):
    rng = np.random.default_rng(seed)
    race = rng.choice(["white", "black", "hipanic"], n)
    sex = rng.choice(["male", "female"], n)
    jailed = (rng.random(n) < 0.1).astype(int)
    mu = 50000 - 20000 * (race == "black") - 10000 * (race == "hipanic") + 5000 * (sex == "male") - 15000 * jailed
    y = mu + rng.normal(0, 8000, n) * (1 + jailed)
    return pd.DataFrame({"race": race, "sex": sex, "ever_jailed": jailed,
                         "composite_wealth_1985": y, "composite_wealth_1990": y, "composite_wealth_1996": y})


def test_recovers_planted_median_effects():
    d = synthetic(8000, 3)
    r = fit_wealth_quantile(d, 0.5)
    truth = {"intercept": 50000, "black": -20000, "hispanic": -10000, "male": 5000, "ever_jailed": -15000}
    for t, v in truth.items():
        assert abs(r["coef"][t] - v) < 1500, t


def test_continuous_outcome_matches_unique_minimiser():
    """With continuous outcomes the minimiser is unique: compare coefficients."""
    d = synthetic(3000, 5)
    r = fit_wealth_quantile(d, 0.8)
    gold = {"intercept": 57126.8264, "black": -19863.2397, "hispanic": -10076.8050,
            "male": 4649.2364, "ever_jailed": -8109.9183}
    for t in TERMS:
        assert r["coef"][t] == pytest.approx(gold[t], abs=0.05), t
