"""Hidden metamorphic tests (Hypothesis) for linear quantile regression.

Because the minimiser need not be unique, every relation is checked on the
objective: the follow-up solution and the source solution mapped through the
transformation must attain the same check loss on the follow-up data (both
are then optimal). A constant output satisfies the invariance relations
(test_mr_inv_*) but not the equivariance relations (test_mr_eqv_*).
Relations: Koenker (2005, Thm 2.3: scale, shift and reparametrisation
equivariance) and Xie et al. (JSS 2011: permutation, irrelevant attribute).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from analysis import fit_wealth_quantile as fit
from qr_oracle import check_loss, same_loss

DATA = Path(__file__).resolve().parent.parent / "data" / "nls_incarceration_processed.csv"
REAL = pd.read_csv(DATA)
YEARS = [1985, 1990, 1996]
SETTINGS = settings(max_examples=10, deadline=None, derandomize=True, database=None,
                    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])


@st.composite
def synthetic_df(draw):
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    n = draw(st.integers(300, 2500))
    race = rng.choice(["white", "black", "hipanic"], n)
    sex = rng.choice(["male", "female"], n)
    jailed = (rng.random(n) < 0.15).astype(int)
    b = [draw(st.floats(-3e4, 3e4)) for _ in range(4)]
    mu = 4e4 + b[0] * (race == "black") + b[1] * (race == "hipanic") + b[2] * (sex == "male") + b[3] * jailed
    cols = {f"composite_wealth_{y}": np.round(mu + rng.standard_t(3, n) * 1e4, -2) for y in YEARS}
    return pd.DataFrame({"race": race, "sex": sex, "ever_jailed": jailed, **cols})


@st.composite
def real_subsample_df(draw):
    seed = draw(st.integers(0, 2**32 - 2))
    return REAL.sample(n=draw(st.integers(1000, 4000)), random_state=seed).reset_index(drop=True)


source = st.tuples(st.one_of(synthetic_df(), real_subsample_df()),
                   st.sampled_from([0.25, 0.5, 0.75, 0.9]), st.sampled_from(YEARS))


def equally_optimal(fol_df, q, year, fol_coef, mapped_coef):
    return same_loss(check_loss(fol_df, q, year, fol_coef), check_loss(fol_df, q, year, mapped_coef))


def wcol(year):
    return f"composite_wealth_{year}"


# ---------------------------------------------------------------- invariance


@SETTINGS
@given(src=source, seed=st.integers(0, 2**32 - 2))
def test_mr_inv_row_permutation(src, seed):
    df, q, year = src
    s = fit(df, q, year)["coef"]
    perm = df.sample(frac=1, random_state=seed)
    assert equally_optimal(perm, q, year, fit(perm, q, year)["coef"], s)


@SETTINGS
@given(src=source, seed=st.integers(0, 2**32 - 1))
def test_mr_inv_irrelevant_column(src, seed):
    df, q, year = src
    s = fit(df, q, year)
    noisy = df.assign(NOISE=np.random.default_rng(seed).normal(size=len(df)))
    f = fit(noisy, q, year)
    assert f["n"] == s["n"] and equally_optimal(noisy, q, year, f["coef"], s["coef"])


@SETTINGS
@given(src=source)
def test_mr_inv_other_years_ignored(src):
    """Changing the wealth of a year that is not analysed changes nothing."""
    df, q, year = src
    other = wcol(next(y for y in YEARS if y != year))
    s = fit(df, q, year)["coef"]
    fol = df.assign(**{other: df[other] * 3 + 1})
    assert equally_optimal(fol, q, year, fit(fol, q, year)["coef"], s)


# -------------------------------------------------------------- equivariance


@SETTINGS
@given(src=source, c=st.floats(0.01, 100))
def test_mr_eqv_scale(src, c):
    """y * c (c > 0)  ->  b * c."""
    df, q, year = src
    assume(abs(c - 1) > 0.05)
    s = fit(df, q, year)["coef"]
    fol = df.assign(**{wcol(year): df[wcol(year)] * c})
    assert equally_optimal(fol, q, year, fit(fol, q, year)["coef"], {k: v * c for k, v in s.items()})


@SETTINGS
@given(src=source)
def test_mr_eqv_negate(src):
    """y -> -y at quantile 1 - q  ->  -b."""
    df, q, year = src
    s = fit(df, q, year)["coef"]
    fol = df.assign(**{wcol(year): -df[wcol(year)]})
    assert equally_optimal(fol, 1 - q, year, fit(fol, 1 - q, year)["coef"], {k: -v for k, v in s.items()})


@SETTINGS
@given(src=source, a=st.floats(-1e5, 1e5), g=st.floats(-1e5, 1e5))
def test_mr_eqv_regression_shift(src, a, g):
    """y + a + g * ever_jailed  ->  intercept + a, ever_jailed + g."""
    df, q, year = src
    assume(abs(a) > 100 or abs(g) > 100)
    s = fit(df, q, year)["coef"]
    fol = df.assign(**{wcol(year): df[wcol(year)] + a + g * df["ever_jailed"]})
    mapped = {**s, "intercept": s["intercept"] + a, "ever_jailed": s["ever_jailed"] + g}
    assert equally_optimal(fol, q, year, fit(fol, q, year)["coef"], mapped)


@SETTINGS
@given(src=source)
def test_mr_eqv_swap_sex_reference(src):
    """Swapping male/female labels: male -> -male, intercept -> intercept + male."""
    df, q, year = src
    s = fit(df, q, year)["coef"]
    fol = df.assign(sex=df["sex"].map({"male": "female", "female": "male"}))
    mapped = {**s, "intercept": s["intercept"] + s["male"], "male": -s["male"]}
    assert equally_optimal(fol, q, year, fit(fol, q, year)["coef"], mapped)


@SETTINGS
@given(src=source)
def test_mr_eqv_duplication(src):
    """Duplicating every row: same coefficients, n doubles."""
    df, q, year = src
    s = fit(df, q, year)
    fol = pd.concat([df, df], ignore_index=True)
    f = fit(fol, q, year)
    assert f["n"] == 2 * s["n"] and equally_optimal(fol, q, year, f["coef"], s["coef"])
