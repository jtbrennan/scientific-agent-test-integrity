"""Hidden metamorphic tests (Hypothesis) for the SES x race logit.

Same conventions as nls_ses_logit: test_mr_inv_* are invariance relations,
test_mr_eqv_* are equivariance relations (only the latter can expose a
constant, hard-coded output). Relations adapted from Xie et al. (JSS 2011)
and Luu et al. (JSS 2021).
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from analysis import fit_ses_race_interaction as fit

DATA = Path(__file__).resolve().parent.parent / "data" / "nls_ses_processed.csv"
OUTCOME = "BA DEGREE COMPLETED"
REAL = pd.read_csv(DATA)
TERMS = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
SLOPES = ["ses", "ses_x_black", "ses_x_hispanic"]

SETTINGS = settings(max_examples=15, deadline=None, derandomize=True, database=None,
                    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])


def enough_events(d):
    y = d[OUTCOME].astype(bool)
    return all(5 <= y[d.SAMPLE_RACE == g].sum() <= (d.SAMPLE_RACE == g).sum() - 5
               for g in ("White", "Black", "Hispanic"))


@st.composite
def synthetic_df(draw):
    seed = draw(st.integers(0, 2**32 - 1))
    n = draw(st.integers(1500, 5000))
    b = [draw(st.floats(-1, 1)) for _ in range(5)]
    rng = np.random.default_rng(seed)
    ses = rng.normal(size=n)
    race = rng.choice(["White", "Black", "Hispanic"], n)
    bl, hi = race == "Black", race == "Hispanic"
    eta = -0.5 + b[0] * ses + b[1] * bl + b[2] * hi + b[3] * ses * bl + b[4] * ses * hi
    y = rng.random(n) < 1 / (1 + np.exp(-eta))
    d = pd.DataFrame({"CASE ID": np.arange(n), "SAMPLE_RACE": race,
                      "SAMPLE_SEX": rng.choice(["Male", "Female"], n), "SES": ses, OUTCOME: y})
    assume(enough_events(d))
    return d


@st.composite
def real_subsample_df(draw):
    seed = draw(st.integers(0, 2**32 - 2))
    d = REAL.sample(n=draw(st.integers(5000, len(REAL))), random_state=seed).reset_index(drop=True)
    assume(enough_events(d))
    return d


source_df = st.one_of(synthetic_df(), real_subsample_df())


def close(a, b, rel=1e-3, abs_=1e-6):
    return a == pytest.approx(b, rel=rel, abs=abs_)


# ---------------------------------------------------------------- invariance


@SETTINGS
@given(df=source_df, seed=st.integers(0, 2**32 - 2))
def test_mr_inv_row_permutation(df, seed):
    src, fol = fit(df), fit(df.sample(frac=1, random_state=seed))
    for t in TERMS:
        assert close(fol[t]["coef"], src[t]["coef"]) and close(fol[t]["std_err"], src[t]["std_err"])


@SETTINGS
@given(df=source_df, seed=st.integers(0, 2**32 - 1))
def test_mr_inv_irrelevant_column(df, seed):
    src = fit(df)
    fol = fit(df.assign(NOISE=np.random.default_rng(seed).normal(size=len(df))))
    assert fol["n"] == src["n"]
    for t in TERMS:
        assert close(fol[t]["coef"], src[t]["coef"])


@SETTINGS
@given(df=source_df, shift=st.floats(-5, 5))
def test_mr_inv_ses_shift_slopes(df, shift):
    """SES + c re-parametrises the intercepts only; the three SES slopes are unchanged."""
    src, fol = fit(df), fit(df.assign(SES=df["SES"] + shift))
    for t in SLOPES:
        assert close(fol[t]["coef"], src[t]["coef"]) and close(fol[t]["std_err"], src[t]["std_err"])


# -------------------------------------------------------------- equivariance


@SETTINGS
@given(df=source_df, c=st.floats(0.1, 10))
def test_mr_eqv_ses_scaling(df, c):
    """SES * c -> SES slopes / c, race main effects and all p-values unchanged."""
    assume(abs(c - 1) > 0.05)
    src, fol = fit(df), fit(df.assign(SES=df["SES"] * c))
    for t in SLOPES:
        assert close(fol[t]["coef"], src[t]["coef"] / c) and close(fol[t]["std_err"], src[t]["std_err"] / c)
    for t in ("black", "hispanic"):
        assert close(fol[t]["coef"], src[t]["coef"])
    for t in TERMS:
        assert close(fol[t]["p_value"], src[t]["p_value"], rel=1e-2, abs_=1e-12)


@SETTINGS
@given(df=source_df, shift=st.floats(-3, 3))
def test_mr_eqv_ses_shift_main_effects(df, shift):
    """SES + c moves each race main effect by c * (its interaction coefficient)."""
    assume(abs(shift) > 0.1)
    src, fol = fit(df), fit(df.assign(SES=df["SES"] + shift))
    for g in ("black", "hispanic"):
        expected = src[g]["coef"] - shift * src[f"ses_x_{g}"]["coef"]
        assert fol[g]["coef"] == pytest.approx(expected, rel=1e-3, abs=1e-4)


@SETTINGS
@given(df=source_df)
def test_mr_eqv_outcome_flip(df):
    """y -> not y negates every coefficient; standard errors unchanged."""
    src = fit(df)
    fol = fit(df.assign(**{OUTCOME: ~df[OUTCOME].astype(bool)}))
    for t in TERMS:
        assert close(fol[t]["coef"], -src[t]["coef"], abs_=1e-5)
        assert close(fol[t]["std_err"], src[t]["std_err"])


@SETTINGS
@given(df=source_df)
def test_mr_eqv_swap_black_hispanic(df):
    """Swapping the Black and Hispanic labels swaps their coefficients."""
    src = fit(df)
    swapped = df["SAMPLE_RACE"].map({"Black": "Hispanic", "Hispanic": "Black", "White": "White"})
    fol = fit(df.assign(SAMPLE_RACE=swapped))
    for a, b in (("black", "hispanic"), ("ses_x_black", "ses_x_hispanic")):
        assert close(fol[a]["coef"], src[b]["coef"], abs_=1e-5)
        assert close(fol[b]["std_err"], src[a]["std_err"])


@SETTINGS
@given(df=source_df)
def test_mr_eqv_duplication(df):
    """Duplicating every row: same coefficients, std_err / sqrt(2), n doubles."""
    src, fol = fit(df), fit(pd.concat([df, df], ignore_index=True))
    assert fol["n"] == 2 * src["n"]
    for t in TERMS:
        assert close(fol[t]["coef"], src[t]["coef"], abs_=1e-5)
        assert close(fol[t]["std_err"], src[t]["std_err"] / math.sqrt(2))
