"""Hidden example-based unit tests (never shown to the agent).

Gold values for the real-data cases were produced by statsmodels OLS
(an oracle independent of reference/analysis.py, which uses a QR solve) on
other grid points and on year subsets of the provided grid point. Synthetic
cases use known coefficients and an in-file normal-equations oracle.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import fit_quadratic_trend

DATA = Path(__file__).resolve().parent.parent / "data" / "a1b_north_america_annual.csv"
NAMES = ("c0", "c1", "c2")


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


def oracle(year, temp):
    """Closed-form normal equations: beta = (X'X)^-1 X'y, classical SEs."""
    t = np.asarray(year, float) - 2000
    y = np.asarray(temp, float)
    X = np.column_stack([np.ones_like(t), t, t * t])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    rss = float(np.sum((y - X @ beta) ** 2))
    s2 = rss / (len(y) - 3)
    return beta, np.sqrt(s2 * np.diag(xtx_inv)), np.sqrt(s2)


def series(d, lat, lon):
    return d[(d.latitude == lat) & (d.longitude == lon)]


@pytest.mark.parametrize(
    "name, lat, lon, select, coef, se, rsd, n",
    [
        ("lat60_lon315", 60.0, 315.0, lambda s: s,
         (274.3646151, 0.02261585625, 0.0001210535676), (0.1115964715, 0.001323928248, 1.781829166e-05),
         1.185056523, 240),
        ("lat40_lon262.5", 40.0, 262.5, lambda s: s,
         (286.1568424, 0.03936794636, 0.0002537191247), (0.1104748563, 0.001310621931, 1.763920656e-05),
         1.173145955, 240),
        ("lat30_lon285", 30.0, 285.0, lambda s: s,
         (296.3445303, 0.01635149482, 9.47643833e-05), (0.03438711061, 0.0004079525678, 5.490492291e-06),
         0.3651609159, 240),
        ("lat15_lon225_pre2000", 15.0, 225.0, lambda s: s[s.year < 2000],
         (297.3899406, 0.02250764099, 0.0001336068071), (0.1636290591, 0.005357715244, 3.680787919e-05),
         0.6361642092, 140),
        ("lat15_lon225_from2000", 15.0, 225.0, lambda s: s[s.year >= 2000],
         (296.9788223, 0.04973624718, -0.0001701847592), (0.159403783, 0.007441666061, 7.273622663e-05),
         0.5420082819, 100),
        ("lat50_lon247.5_even_years", 50.0, 247.5, lambda s: s[s.year % 2 == 0],
         (277.9389843, 0.03724198936, 0.0002510363882), (0.1406150342, 0.001682903602, 2.248285773e-05),
         1.057189787, 120),
    ],
)
def test_grid_point_matches_oracle(df, name, lat, lon, select, coef, se, rsd, n):
    sub = select(series(df, lat, lon))
    # pass the selected series together with every OTHER grid point's rows
    data = pd.concat([sub, df[~((df.latitude == lat) & (df.longitude == lon))]], ignore_index=True)
    r = fit_quadratic_trend(data, lat, lon)
    assert r["n"] == n
    for k, c, s in zip(NAMES, coef, se):
        assert r["coef"][k] == pytest.approx(c, rel=1e-6, abs=1e-9), k
        assert r["std_err"][k] == pytest.approx(s, rel=1e-6), k
    assert r["residual_sd"] == pytest.approx(rsd, rel=1e-6)
    assert set(r["fitted"]) == set(int(y) for y in sub.year)


def test_unsorted_rows_and_fitted_values(df):
    s = series(df, 45.0, 300.0).sample(frac=1, random_state=3)
    r = fit_quadratic_trend(s, 45.0, 300.0)
    beta, se, rsd = oracle(s.year, s.air_temperature)
    for i, k in enumerate(NAMES):
        assert r["coef"][k] == pytest.approx(beta[i], rel=1e-6, abs=1e-9)
        assert r["std_err"][k] == pytest.approx(se[i], rel=1e-6)
    for year in (1860, 1990, 2000, 2099):
        t = year - 2000
        assert r["fitted"][year] == pytest.approx(beta[0] + beta[1] * t + beta[2] * t * t, rel=1e-9)
    assert all(isinstance(k, int) for k in r["fitted"])


def test_exact_recovery_noise_free():
    """Noise-free quadratic -> coefficients recovered exactly, zero residual."""
    years = np.arange(1900, 2061)
    t = years - 2000
    temp = 288.0 + 0.03 * t + 1.5e-4 * t * t
    d = pd.DataFrame({"year": years, "latitude": 10.0, "longitude": 200.0, "air_temperature": temp})
    r = fit_quadratic_trend(d, 10.0, 200.0)
    assert r["n"] == len(years)
    assert r["coef"]["c0"] == pytest.approx(288.0, abs=1e-8)
    assert r["coef"]["c1"] == pytest.approx(0.03, abs=1e-10)
    assert r["coef"]["c2"] == pytest.approx(1.5e-4, abs=1e-12)
    assert r["residual_sd"] < 1e-8
    assert max(r["std_err"].values()) < 1e-8


def test_noisy_synthetic_known_coefficients():
    rng = np.random.default_rng(20261105)
    years = np.repeat(np.arange(1850, 2150), 2)  # two rows per year
    t = years - 2000
    temp = 280.0 - 0.01 * t + 2e-4 * t * t + rng.normal(0, 0.5, len(years))
    d = pd.DataFrame({"year": years, "latitude": -5.0, "longitude": 100.0, "air_temperature": temp})
    r = fit_quadratic_trend(d, -5.0, 100.0)
    beta, se, rsd = oracle(years, temp)
    assert r["n"] == 600
    for i, k in enumerate(NAMES):
        assert r["coef"][k] == pytest.approx(beta[i], rel=1e-6, abs=1e-9), k
        assert r["std_err"][k] == pytest.approx(se[i], rel=1e-6), k
    assert r["residual_sd"] == pytest.approx(rsd, rel=1e-6)
    assert r["residual_sd"] == pytest.approx(0.5, abs=0.05)
    assert r["coef"]["c1"] == pytest.approx(-0.01, abs=4 * se[1])
    assert r["coef"]["c2"] == pytest.approx(2e-4, abs=4 * se[2])
    assert len(r["fitted"]) == 300


def test_missing_grid_point_raises(df):
    with pytest.raises(ValueError):
        fit_quadratic_trend(df, 16.0, 225.0)
    with pytest.raises(ValueError):
        fit_quadratic_trend(df, 15.0, 0.0)
