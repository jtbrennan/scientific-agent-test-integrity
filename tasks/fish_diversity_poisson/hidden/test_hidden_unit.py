"""Hidden example-based unit tests (never shown to the agent).

Gold values were produced by reference/analysis.py (statsmodels GLM, IRLS) on
held-out subsets of the provided data (southern / northern basins, first 300
rows). Synthetic tests check recovery of known coefficients, and an oracle
(an independent Newton-Raphson Poisson MLE written in this file) is compared
with the solution on random subsets and on low-count data with zeros.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import fit_diversity_poisson

DATA = Path(__file__).resolve().parent.parent / "data" / "freshwater_fish_basins.csv"
TERMS = ["intercept", "log_area", "bio1", "precip_m"]


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


def oracle(d: pd.DataFrame) -> dict:
    """Independent Poisson MLE by Newton-Raphson with step halving."""
    y = d["diversity"].astype(float).to_numpy()
    X = np.column_stack([np.ones(len(d)), np.log(d["area"].astype(float).to_numpy()),
                         d["bio1"].astype(float).to_numpy(), d["bio12"].astype(float).to_numpy() / 1000])
    beta = np.zeros(4)
    beta[0] = math.log(max(y.mean(), 1e-8))
    ll = lambda b: float(y @ (X @ b) - np.exp(X @ b).sum())  # noqa: E731
    for _ in range(200):
        mu = np.exp(X @ beta)
        step = np.linalg.solve(X.T @ (X * mu[:, None]), X.T @ (y - mu))
        t, cur = 1.0, ll(beta)
        while ll(beta + t * step) < cur - 1e-12 and t > 1e-8:
            t /= 2
        beta = beta + t * step
        if np.max(np.abs(t * step)) < 1e-12:
            break
    mu = np.exp(X @ beta)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * mu[:, None]))))
    with np.errstate(divide="ignore", invalid="ignore"):
        ylog = np.where(y > 0, y * np.log(y / mu), 0.0)
    return {"coef": beta, "se": se, "deviance": float(2 * np.sum(ylog - (y - mu)))}


def wald_p(coef, se):
    return math.erfc(abs(coef / se) / math.sqrt(2))


@pytest.mark.parametrize(
    "name, select, gold, deviance, n",
    [
        ("south", lambda d: d[d.lat < -15],
         {"intercept": (1.0830799814, 0.0668651042), "log_area": (0.0956113221, 0.0056701846),
          "bio1": (0.0865343477, 0.0025308965), "precip_m": (0.5100465827, 0.0269361127)},
         7818.975347461, 220),
        ("north", lambda d: d[d.lat >= -15],
         {"intercept": (-1.0663029899, 0.1150432180), "log_area": (0.2240628623, 0.0042479222),
          "bio1": (0.1040629836, 0.0037472945), "precip_m": (0.4864508014, 0.0075837906)},
         15544.385961671, 240),
        ("first300", lambda d: d.iloc[:300],
         {"intercept": (2.9659735565, 0.0622771912), "log_area": (0.1426541031, 0.0034519151),
          "bio1": (-0.0302558007, 0.0021636446), "precip_m": (0.6323082944, 0.0087655018)},
         17475.345428413, 300),
    ],
)
def test_subset_matches_reference(df, name, select, gold, deviance, n):
    r = fit_diversity_poisson(select(df).reset_index(drop=True))
    assert r["n"] == n
    assert r["deviance"] == pytest.approx(deviance, rel=1e-5)
    for term, (coef, se) in gold.items():
        assert r["terms"][term]["coef"] == pytest.approx(coef, abs=1e-4), term
        assert r["terms"][term]["std_err"] == pytest.approx(se, rel=1e-3), term


def test_recovers_planted_coefficients():
    rng = np.random.default_rng(20261105)
    n = 6000
    area = np.exp(rng.normal(9.5, 1.2, n))
    bio1 = rng.normal(21, 4, n)
    bio12 = rng.uniform(300, 4000, n)
    eta = 0.5 + 0.2 * np.log(area) + 0.04 * bio1 + 0.3 * bio12 / 1000
    d = pd.DataFrame({"area": area, "bio1": bio1, "bio12": bio12, "lat": rng.uniform(-50, 10, n),
                      "diversity": rng.poisson(np.exp(eta))})
    r = fit_diversity_poisson(d)
    truth = {"intercept": 0.5, "log_area": 0.2, "bio1": 0.04, "precip_m": 0.3}
    tol = {"intercept": 0.1, "log_area": 0.01, "bio1": 0.004, "precip_m": 0.03}
    for t, b in truth.items():
        assert abs(r["terms"][t]["coef"] - b) < tol[t], t
        assert r["terms"][t]["p_value"] < 1e-10, t
    o = oracle(d)
    for i, t in enumerate(TERMS):
        assert r["terms"][t]["coef"] == pytest.approx(o["coef"][i], rel=1e-5, abs=1e-3 * o["se"][i]), t
        assert r["terms"][t]["std_err"] == pytest.approx(o["se"][i], rel=1e-4), t
    assert r["deviance"] == pytest.approx(o["deviance"], rel=1e-5)


def test_null_precip_effect_not_significant_and_zero_counts():
    """Low counts (many zeros); precipitation has no effect."""
    rng = np.random.default_rng(99)
    n = 3000
    area = np.exp(rng.normal(9.0, 1.0, n))
    bio1 = rng.normal(20, 3, n)
    bio12 = rng.uniform(500, 3000, n)
    eta = -3.0 + 0.25 * np.log(area) + 0.02 * bio1
    y = rng.poisson(np.exp(eta))
    assert (y == 0).sum() > 300
    d = pd.DataFrame({"area": area, "bio1": bio1, "bio12": bio12, "diversity": y})
    r = fit_diversity_poisson(d)
    o = oracle(d)
    assert r["n"] == n
    assert r["terms"]["precip_m"]["p_value"] > 0.01
    for i, t in enumerate(TERMS):
        assert r["terms"][t]["coef"] == pytest.approx(o["coef"][i], rel=1e-5, abs=1e-3 * o["se"][i]), t
        assert r["terms"][t]["std_err"] == pytest.approx(o["se"][i], rel=1e-4), t
        assert r["terms"][t]["p_value"] == pytest.approx(wald_p(o["coef"][i], o["se"][i]), rel=1e-2, abs=1e-12), t
    assert r["deviance"] == pytest.approx(o["deviance"], rel=1e-5)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_random_subset_matches_oracle(df, seed):
    rng = np.random.default_rng(seed)
    size = int(rng.integers(80, 400))
    d = df.sample(n=size, random_state=seed).reset_index(drop=True)
    r = fit_diversity_poisson(d)
    o = oracle(d)
    assert r["n"] == size
    for i, t in enumerate(TERMS):
        assert r["terms"][t]["coef"] == pytest.approx(o["coef"][i], rel=1e-5, abs=1e-3 * o["se"][i]), t
        assert r["terms"][t]["std_err"] == pytest.approx(o["se"][i], rel=1e-4), t
        assert r["terms"][t]["p_value"] == pytest.approx(wald_p(o["coef"][i], o["se"][i]), rel=1e-2, abs=1e-12), t
    assert r["deviance"] == pytest.approx(o["deviance"], rel=1e-6)
