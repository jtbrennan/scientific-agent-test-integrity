"""Hidden metamorphic tests (Hypothesis). Never shown to the agent.

No gold answers are needed: each test runs the solution on a source input
and on a transformed follow-up input and checks the relation between the
two outputs. Source inputs are either synthetic noisy quadratic series (at a
random grid point, mixed with rows of decoy grid points) or real grid-point
series (optionally a year subset) from the provided data, drawn by Hypothesis.

Naming convention (used by the harness for RQ2 reporting):
  test_mr_inv_*  invariance relations   - output must NOT change
  test_mr_eqv_*  equivariance relations - output must change predictably
A constant (hard-coded) output trivially satisfies invariance relations, so
only equivariance relations can expose it.

MR sources: permutation / irrelevant-attribute / affine relations follow
Xie et al. (JSS 2011) and Murphy et al. (SEKE 2008); the regression-specific
scaling, regressor-shift and duplication relations follow the MLR relations
of Luu et al. (JSS 2021). Year-shift algebra: with year' = year + k the
fitted curve is unchanged, so c2' = c2, c1' = c1 - 2*c2*k,
c0' = c0 - c1*k + c2*k**2.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from analysis import fit_quadratic_trend as fit

DATA = Path(__file__).resolve().parent.parent / "data" / "a1b_north_america_annual.csv"
REAL = pd.read_csv(DATA)
LATS = sorted(REAL.latitude.unique().tolist())
LONS = sorted(REAL.longitude.unique().tolist())
NAMES = ("c0", "c1", "c2")
T = "air_temperature"

SETTINGS = settings(
    max_examples=15,
    deadline=None,
    derandomize=True,
    database=None,
    phases=[Phase.explicit, Phase.reuse, Phase.generate],  # no shrinking: we only need pass/fail
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


@st.composite
def synthetic_case(draw):
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    start = draw(st.integers(1800, 2000))
    length = draw(st.integers(20, 250))
    years = np.arange(start, start + length)
    keep = rng.random(length) < draw(st.floats(0.5, 1.0))
    keep[:3] = True
    years = years[keep]
    t = years - 2000
    c0 = draw(st.floats(250, 310))
    c1 = draw(st.floats(-0.1, 0.1))
    c2 = draw(st.floats(-5e-4, 5e-4))
    sd = draw(st.floats(0.05, 2.0))
    temp = c0 + c1 * t + c2 * t * t + rng.normal(0, sd, len(t))
    lat, lon = float(draw(st.sampled_from(LATS))), float(draw(st.sampled_from(LONS)))
    target = pd.DataFrame({"year": years, "latitude": lat, "longitude": lon, T: temp})
    decoy = pd.DataFrame({"year": years, "latitude": lat + 1.25, "longitude": lon,
                          T: temp + rng.normal(3, 1, len(t))})
    df = pd.concat([target, decoy], ignore_index=True).sample(frac=1, random_state=seed % 1000)
    return df.reset_index(drop=True), lat, lon


@st.composite
def real_case(draw):
    lat, lon = float(draw(st.sampled_from(LATS))), float(draw(st.sampled_from(LONS)))
    lo = draw(st.integers(1860, 1990))
    hi = draw(st.integers(lo + 60, 2099))
    sel = (REAL.latitude == lat) & (REAL.longitude == lon)
    keep = ~sel | ((REAL.year >= lo) & (REAL.year <= hi))
    return REAL[keep].reset_index(drop=True), lat, lon


source_case = st.one_of(synthetic_case(), real_case())


def close(a, b, rel=1e-6, abs_=1e-9):
    return a == pytest.approx(b, rel=rel, abs=abs_)


def same(a, b, se=True):
    for k in NAMES:
        assert close(a["coef"][k], b["coef"][k]), k
        if se:
            assert close(a["std_err"][k], b["std_err"][k]), k
    assert close(a["residual_sd"], b["residual_sd"])


def target_rows(df, lat, lon):
    return df[(df.latitude == lat) & (df.longitude == lon)]


# ---------------------------------------------------------------- invariance


@SETTINGS
@given(case=source_case, seed=st.integers(0, 2**32 - 1))
def test_mr_inv_row_permutation(case, seed):
    df, lat, lon = case
    src = fit(df, lat, lon)
    fol = fit(df.sample(frac=1, random_state=seed % (2**32 - 1)), lat, lon)
    same(fol, src)
    assert fol["n"] == src["n"] and fol["fitted"].keys() == src["fitted"].keys()


@SETTINGS
@given(case=source_case, seed=st.integers(0, 2**32 - 1), k=st.integers(1, 40))
def test_mr_inv_add_other_grid_points(case, seed, k):
    """Appending rows of other grid points (real or junk) changes nothing."""
    df, lat, lon = case
    src = fit(df, lat, lon)
    rng = np.random.default_rng(seed)
    other = REAL[~((REAL.latitude == lat) & (REAL.longitude == lon))].sample(n=50 * k, random_state=seed % 1000)
    junk = pd.DataFrame({"year": rng.integers(1700, 2300, 30), "latitude": lat, "longitude": lon + 0.5,
                         T: rng.normal(280, 30, 30)})
    fol = fit(pd.concat([junk, df, other], ignore_index=True), lat, lon)
    same(fol, src)
    assert fol["n"] == src["n"]


@SETTINGS
@given(case=source_case, seed=st.integers(0, 2**32 - 1))
def test_mr_inv_irrelevant_columns(case, seed):
    df, lat, lon = case
    src = fit(df, lat, lon)
    rng = np.random.default_rng(seed)
    noisy = df.assign(NOISE=rng.normal(size=len(df)), model="HadCM3", height=1.5)
    fol = fit(noisy[["NOISE", "model"] + list(df.columns) + ["height"]], lat, lon)
    same(fol, src)
    assert fol["n"] == src["n"]


# -------------------------------------------------------------- equivariance


@SETTINGS
@given(case=source_case, shift=st.floats(-100, 100))
def test_mr_eqv_temperature_shift(case, shift):
    """T + a  ->  c0 + a; c1, c2, all SEs and residual_sd unchanged."""
    df, lat, lon = case
    src = fit(df, lat, lon)
    fol = fit(df.assign(**{T: df[T] + shift}), lat, lon)
    assert close(fol["coef"]["c0"], src["coef"]["c0"] + shift)
    for k in ("c1", "c2"):
        assert close(fol["coef"][k], src["coef"][k]), k
    for k in NAMES:
        assert close(fol["std_err"][k], src["std_err"][k], rel=1e-5), k
    assert close(fol["residual_sd"], src["residual_sd"], rel=1e-5)


@SETTINGS
@given(case=source_case)
def test_mr_eqv_kelvin_to_celsius(case):
    df, lat, lon = case
    src = fit(df, lat, lon)
    fol = fit(df.assign(**{T: df[T] - 273.15}), lat, lon)
    assert close(fol["coef"]["c0"], src["coef"]["c0"] - 273.15)
    for k in ("c1", "c2"):
        assert close(fol["coef"][k], src["coef"][k]), k
    assert close(fol["std_err"]["c2"], src["std_err"]["c2"], rel=1e-5)
    for y in src["fitted"]:
        assert close(fol["fitted"][y], src["fitted"][y] - 273.15)


@SETTINGS
@given(case=source_case, c=st.floats(0.1, 10))
def test_mr_eqv_temperature_scaling(case, c):
    """T * c  ->  every coef, SE, residual_sd and fitted value times c."""
    df, lat, lon = case
    src = fit(df, lat, lon)
    fol = fit(df.assign(**{T: df[T] * c}), lat, lon)
    for k in NAMES:
        assert close(fol["coef"][k], src["coef"][k] * c), k
        assert close(fol["std_err"][k], src["std_err"][k] * c, rel=1e-5), k
    assert close(fol["residual_sd"], src["residual_sd"] * c, rel=1e-5)
    for y in src["fitted"]:
        assert close(fol["fitted"][y], src["fitted"][y] * c)


@SETTINGS
@given(case=source_case, k=st.integers(-300, 300).filter(lambda v: v != 0))
def test_mr_eqv_year_shift(case, k):
    """year + k  ->  same curve in shifted time:
    c2' = c2, c1' = c1 - 2*c2*k, c0' = c0 - c1*k + c2*k**2; c2's SE and
    residual_sd unchanged; fitted values move to the shifted year keys."""
    df, lat, lon = case
    src = fit(df, lat, lon)
    fol = fit(df.assign(year=df["year"] + k), lat, lon)
    c0, c1, c2 = (src["coef"][n] for n in NAMES)
    assert close(fol["coef"]["c2"], c2, abs_=1e-11)
    assert close(fol["coef"]["c1"], c1 - 2 * c2 * k, rel=1e-5, abs_=1e-8)
    assert close(fol["coef"]["c0"], c0 - c1 * k + c2 * k * k, rel=1e-6, abs_=1e-6)
    assert close(fol["std_err"]["c2"], src["std_err"]["c2"], rel=1e-5)
    assert close(fol["residual_sd"], src["residual_sd"], rel=1e-5)
    for y, v in src["fitted"].items():
        assert close(fol["fitted"][y + k], v)


@SETTINGS
@given(case=source_case)
def test_mr_eqv_duplication(case):
    """Every row twice: same coefs, n doubles, SE * sqrt((n-3)/(2n-3)),
    residual_sd * sqrt(2(n-3)/(2n-3))."""
    df, lat, lon = case
    src = fit(df, lat, lon)
    fol = fit(pd.concat([df, df], ignore_index=True), lat, lon)
    n = src["n"]
    assert fol["n"] == 2 * n
    for k in NAMES:
        assert close(fol["coef"][k], src["coef"][k]), k
        assert close(fol["std_err"][k], src["std_err"][k] * math.sqrt((n - 3) / (2 * n - 3)), rel=1e-5), k
    assert close(fol["residual_sd"], src["residual_sd"] * math.sqrt(2 * (n - 3) / (2 * n - 3)), rel=1e-5)
