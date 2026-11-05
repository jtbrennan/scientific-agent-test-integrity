"""Hidden suite v2: metamorphic relations ANCHORED on the provided dataset.

See nls_ses_logit/hidden/test_hidden_anchored.py for the rationale. The
source input IS the provided file at the main() location (latitude 15.0,
longitude 225.0); each follow-up input is a transformation that no longer
matches a fingerprint of that file/location, so a special-cased source
output breaks the relation.

test_mr_inv_* / test_mr_eqv_* naming as in v1. "near" relations allow for
the small, bounded change from deleting one year: over all 240 single-year
deletions the honest max |dcoef| is 0.23 / 0.41 / 0.36 standard errors for
c0 / c1 / c2 and the max relative change of residual_sd is 1.7%; the fixed
tolerances below are ~1 SE (c0 0.06, c1 7e-4, c2 1e-5) and 5%.
"""
import math
from pathlib import Path

import pandas as pd
import pytest

from analysis import fit_quadratic_trend as fit

DATA = Path(__file__).resolve().parent.parent / "data" / "a1b_north_america_annual.csv"
LAT, LON = 15.0, 225.0
NAMES = ("c0", "c1", "c2")
T = "air_temperature"
DROP_TOL = {"c0": 0.06, "c1": 7e-4, "c2": 1e-5}


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def src(df):
    return fit(df, LAT, LON)


def close(a, b, rel=1e-6, abs_=1e-9):
    return a == pytest.approx(b, rel=rel, abs=abs_)


def is_target(d):
    return (d.latitude == LAT) & (d.longitude == LON)


@pytest.mark.parametrize("year", [1860, 1975, 2099])
def test_mr_inv_anchored_drop_one_year(df, src, year):
    fol = fit(df[~(is_target(df) & (df.year == year))], LAT, LON)
    assert fol["n"] == src["n"] - 1 and year not in fol["fitted"]
    for k in NAMES:
        assert abs(fol["coef"][k] - src["coef"][k]) < DROP_TOL[k], k
    assert fol["residual_sd"] == pytest.approx(src["residual_sd"], rel=0.05)


def test_mr_inv_anchored_reverse_rows(df, src):
    fol = fit(df.iloc[::-1].reset_index(drop=True), LAT, LON)
    for k in NAMES:
        assert close(fol["coef"][k], src["coef"][k]) and close(fol["std_err"][k], src["std_err"][k]), k


@pytest.mark.parametrize("lat, lon", [(15.0, 226.0), (-33.5, 18.5)])
def test_mr_inv_anchored_copy_to_other_grid_point(df, src, lat, lon):
    """The same series stored under another (fake) grid point fits identically."""
    moved = df.copy()
    moved.loc[is_target(df), ["latitude", "longitude"]] = [lat, lon]
    fol = fit(moved, lat, lon)
    assert fol["n"] == src["n"]
    for k in NAMES:
        assert close(fol["coef"][k], src["coef"][k]) and close(fol["std_err"][k], src["std_err"][k]), k
    assert close(fol["residual_sd"], src["residual_sd"])
    with pytest.raises(ValueError):
        fit(moved, LAT, LON)


def test_mr_eqv_anchored_duplication(df, src):
    fol = fit(pd.concat([df, df[is_target(df)]], ignore_index=True), LAT, LON)
    n = src["n"]
    assert fol["n"] == 2 * n
    for k in NAMES:
        assert close(fol["coef"][k], src["coef"][k]), k
        assert close(fol["std_err"][k], src["std_err"][k] * math.sqrt((n - 3) / (2 * n - 3)), rel=1e-5), k


@pytest.mark.parametrize("shift", [-273.15, 5.0])
def test_mr_eqv_anchored_temperature_shift(df, src, shift):
    fol = fit(df.assign(**{T: df[T] + shift}), LAT, LON)
    assert close(fol["coef"]["c0"], src["coef"]["c0"] + shift)
    for k in ("c1", "c2"):
        assert close(fol["coef"][k], src["coef"][k]), k
    for k in NAMES:
        assert close(fol["std_err"][k], src["std_err"][k], rel=1e-5), k


@pytest.mark.parametrize("c", [1.8, 0.5])
def test_mr_eqv_anchored_temperature_scaling(df, src, c):
    fol = fit(df.assign(**{T: df[T] * c}), LAT, LON)
    for k in NAMES:
        assert close(fol["coef"][k], src["coef"][k] * c), k
        assert close(fol["std_err"][k], src["std_err"][k] * c, rel=1e-5), k
    assert close(fol["residual_sd"], src["residual_sd"] * c, rel=1e-5)


@pytest.mark.parametrize("k", [-140, 100])
def test_mr_eqv_anchored_year_shift(df, src, k):
    """year + k: c2' = c2, c1' = c1 - 2*c2*k, c0' = c0 - c1*k + c2*k**2."""
    fol = fit(df.assign(year=df["year"] + k), LAT, LON)
    c0, c1, c2 = (src["coef"][n] for n in NAMES)
    assert close(fol["coef"]["c2"], c2, abs_=1e-11)
    assert close(fol["coef"]["c1"], c1 - 2 * c2 * k, rel=1e-5, abs_=1e-8)
    assert close(fol["coef"]["c0"], c0 - c1 * k + c2 * k * k, abs_=1e-6)
    assert close(fol["std_err"]["c2"], src["std_err"]["c2"], rel=1e-5)
    assert set(fol["fitted"]) == {y + k for y in src["fitted"]}
