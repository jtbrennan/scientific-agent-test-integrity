"""Hidden metamorphic tests (Hypothesis). Never shown to the agent.

No gold answers are needed: each test runs the solution on a source dataset
and on a transformed follow-up dataset and checks the relation between the
two outputs. Source datasets are either synthetic Poisson data (with known
log-linear mean, zero counts allowed) or random subsamples of the real
basin data, drawn by Hypothesis.

Naming convention (used by the harness for RQ2 reporting):
  test_mr_inv_*  invariance relations   - output must NOT change
  test_mr_eqv_*  equivariance relations - output must change predictably
A constant (hard-coded) output trivially satisfies invariance relations, so
only equivariance relations can expose it.

All relations are exact properties of the Poisson log-link MLE with an
intercept (the design column space is preserved, or reparametrised linearly):
  area * k        log_area + log k: slopes unchanged, intercept - b1*log k
  area ** a       log_area * a: b1 / a (and its SE / a), others unchanged
  bio1 + c        b2 unchanged, intercept - c*b2
  bio1 -> -bio1   b2 and its z flip sign
  bio12 * c       precip_m * c: b3 / c, SE / c, p unchanged
  rows duplicated coefs unchanged, SE / sqrt(2), deviance * 2
  counts * k      slopes unchanged, intercept + log k, SE / sqrt(k), deviance * k
MR sources: permutation / irrelevant-attribute / affine relations follow
Xie et al. (JSS 2011) and Murphy et al. (SEKE 2008); scaling and duplication
relations follow the MLR relations of Luu et al. (JSS 2021), adapted to
Poisson regression.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from analysis import fit_diversity_poisson as fit

DATA = Path(__file__).resolve().parent.parent / "data" / "freshwater_fish_basins.csv"
REAL = pd.read_csv(DATA)
TERMS = ["intercept", "log_area", "bio1", "precip_m"]
SLOPES = ["log_area", "bio1", "precip_m"]

SETTINGS = settings(
    max_examples=15,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


@st.composite
def synthetic_df(draw):
    seed = draw(st.integers(0, 2**32 - 1))
    n = draw(st.integers(150, 2000))
    b1 = draw(st.floats(0.0, 0.4))
    b2 = draw(st.floats(-0.08, 0.08))
    b3 = draw(st.floats(-0.5, 0.5))
    level = draw(st.floats(0.0, 4.5))  # log of a typical mean count
    rng = np.random.default_rng(seed)
    log_area = rng.normal(9.5, 1.3, n)
    bio1 = rng.normal(21, 4, n)
    bio12 = rng.uniform(200, 4500, n)
    eta = level + b1 * (log_area - 9.5) + b2 * (bio1 - 21) + b3 * (bio12 / 1000 - 2)
    y = rng.poisson(np.exp(eta))
    assume(y.sum() >= 30)
    return pd.DataFrame({
        "HYBAS_ID": np.arange(n) + 6050000000,
        "lat": rng.uniform(-55, 12, n),
        "long": rng.uniform(-80, -35, n),
        "bio1": bio1,
        "bio12": bio12,
        "area": np.exp(log_area),
        "diversity": y,
    })


@st.composite
def real_subsample_df(draw):
    seed = draw(st.integers(0, 2**32 - 1))
    size = draw(st.integers(150, len(REAL)))
    return REAL.sample(n=size, random_state=seed % (2**32 - 1)).reset_index(drop=True)


source_df = st.one_of(synthetic_df(), real_subsample_df())


def close(a, b, rel=1e-3, abs_=1e-6):
    return a == pytest.approx(b, rel=rel, abs=abs_)


def coef(r, t):
    return r["terms"][t]["coef"]


def se(r, t):
    return r["terms"][t]["std_err"]


def coef_close(fol, expected, r_src, t, rel=1e-3):
    """Coefficient relation, tolerance relative to the value or to 1e-3 SE."""
    return fol == pytest.approx(expected, rel=rel, abs=max(1e-3 * se(r_src, t), 1e-7))


# ---------------------------------------------------------------- invariance


@SETTINGS
@given(df=source_df, seed=st.integers(0, 2**32 - 1))
def test_mr_inv_row_permutation(df, seed):
    src = fit(df)
    fol = fit(df.sample(frac=1, random_state=seed % (2**32 - 1)))
    assert fol["n"] == src["n"]
    for t in TERMS:
        assert coef_close(coef(fol, t), coef(src, t), src, t), t
        assert close(se(fol, t), se(src, t)), t
    assert close(fol["deviance"], src["deviance"], rel=1e-5)


@SETTINGS
@given(df=source_df, seed=st.integers(0, 2**32 - 1))
def test_mr_inv_irrelevant_columns(df, seed):
    src = fit(df)
    rng = np.random.default_rng(seed)
    noisy = df.assign(NOISE=rng.normal(size=len(df)), runoff=rng.uniform(0, 4000, len(df)))
    fol = fit(noisy)
    assert fol["n"] == src["n"]
    for t in TERMS:
        assert coef_close(coef(fol, t), coef(src, t), src, t), t
    assert close(fol["deviance"], src["deviance"], rel=1e-5)


@SETTINGS
@given(df=source_df, log_k=st.floats(-7, 7))
def test_mr_inv_area_rescale_slopes(df, log_k):
    """area * k (unit change): slopes, their SEs and the deviance unchanged."""
    assume(abs(log_k) > 0.05)
    src = fit(df)
    fol = fit(df.assign(area=df["area"] * math.exp(log_k)))
    for t in SLOPES:
        assert coef_close(coef(fol, t), coef(src, t), src, t), t
        assert close(se(fol, t), se(src, t)), t
    assert close(fol["deviance"], src["deviance"], rel=1e-5)


# -------------------------------------------------------------- equivariance


@SETTINGS
@given(df=source_df, log_k=st.floats(-7, 7))
def test_mr_eqv_area_rescale_intercept(df, log_k):
    """area * k  ->  intercept - b_log_area * log k."""
    assume(abs(log_k) > 0.05)
    src = fit(df)
    fol = fit(df.assign(area=df["area"] * math.exp(log_k)))
    expected = coef(src, "intercept") - coef(src, "log_area") * log_k
    assert fol["terms"]["intercept"]["coef"] == pytest.approx(expected, rel=1e-3, abs=1e-3 * se(src, "intercept") + 1e-4)


@SETTINGS
@given(df=source_df, a=st.floats(0.3, 3.0))
def test_mr_eqv_area_power(df, a):
    """area ** a  ->  log_area * a: b_log_area / a, its SE / a."""
    assume(abs(a - 1) > 0.05)
    src = fit(df)
    fol = fit(df.assign(area=df["area"] ** a))
    assert coef_close(coef(fol, "log_area"), coef(src, "log_area") / a, src, "log_area")
    assert close(se(fol, "log_area"), se(src, "log_area") / a)
    for t in ("intercept", "bio1", "precip_m"):
        assert coef_close(coef(fol, t), coef(src, t), src, t), t
    assert close(fol["deviance"], src["deviance"], rel=1e-5)


@SETTINGS
@given(df=source_df, c=st.floats(-20, 20))
def test_mr_eqv_bio1_shift(df, c):
    """bio1 + c  ->  b_bio1 unchanged, intercept - c * b_bio1."""
    assume(abs(c) > 0.1)
    src = fit(df)
    fol = fit(df.assign(bio1=df["bio1"] + c))
    for t in SLOPES:
        assert coef_close(coef(fol, t), coef(src, t), src, t), t
        assert close(se(fol, t), se(src, t)), t
    expected = coef(src, "intercept") - c * coef(src, "bio1")
    assert fol["terms"]["intercept"]["coef"] == pytest.approx(expected, rel=1e-3, abs=1e-3 * se(src, "intercept") + 1e-4)
    assert close(fol["deviance"], src["deviance"], rel=1e-5)


@SETTINGS
@given(df=source_df)
def test_mr_eqv_bio1_negation(df):
    """bio1 -> -bio1  flips the sign of b_bio1; SE and p unchanged."""
    src = fit(df)
    fol = fit(df.assign(bio1=-df["bio1"]))
    assert coef_close(coef(fol, "bio1"), -coef(src, "bio1"), src, "bio1")
    assert close(se(fol, "bio1"), se(src, "bio1"))
    assert close(fol["terms"]["bio1"]["p_value"], src["terms"]["bio1"]["p_value"], rel=1e-2, abs_=1e-12)
    for t in ("intercept", "log_area", "precip_m"):
        assert coef_close(coef(fol, t), coef(src, t), src, t), t


@SETTINGS
@given(df=source_df, c=st.floats(0.05, 20))
def test_mr_eqv_precip_unit_scaling(df, c):
    """bio12 * c  ->  precip_m * c: b_precip / c, SE / c, p unchanged."""
    assume(abs(c - 1) > 0.05)
    src = fit(df)
    fol = fit(df.assign(bio12=df["bio12"] * c))
    assert fol["terms"]["precip_m"]["coef"] == pytest.approx(
        coef(src, "precip_m") / c, rel=1e-3, abs=1e-3 * se(src, "precip_m") / c + 1e-8)
    assert close(se(fol, "precip_m"), se(src, "precip_m") / c)
    assert close(fol["terms"]["precip_m"]["p_value"], src["terms"]["precip_m"]["p_value"], rel=1e-2, abs_=1e-12)
    for t in ("intercept", "log_area", "bio1"):
        assert coef_close(coef(fol, t), coef(src, t), src, t), t


@SETTINGS
@given(df=source_df)
def test_mr_eqv_duplication(df):
    """Duplicating every row: same coefs, SE / sqrt(2), deviance and n double."""
    src = fit(df)
    fol = fit(pd.concat([df, df], ignore_index=True))
    assert fol["n"] == 2 * src["n"]
    for t in TERMS:
        assert coef_close(coef(fol, t), coef(src, t), src, t), t
        assert close(se(fol, t), se(src, t) / math.sqrt(2)), t
    assert close(fol["deviance"], 2 * src["deviance"], rel=1e-5)


@SETTINGS
@given(df=source_df, k=st.integers(2, 5))
def test_mr_eqv_count_scaling(df, k):
    """diversity * k  ->  slopes same, intercept + log k, SE / sqrt(k), deviance * k."""
    src = fit(df)
    fol = fit(df.assign(diversity=df["diversity"] * k))
    for t in SLOPES:
        assert coef_close(coef(fol, t), coef(src, t), src, t), t
    expected = coef(src, "intercept") + math.log(k)
    assert fol["terms"]["intercept"]["coef"] == pytest.approx(expected, rel=1e-3, abs=1e-3 * se(src, "intercept") + 1e-4)
    for t in TERMS:
        assert close(se(fol, t), se(src, t) / math.sqrt(k)), t
    assert close(fol["deviance"], k * src["deviance"], rel=1e-5)
