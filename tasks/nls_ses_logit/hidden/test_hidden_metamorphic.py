"""Hidden metamorphic tests (Hypothesis). Never shown to the agent.

No gold answers are needed: each test runs the solution on a source dataset
and on a transformed follow-up dataset and checks the relation between the
two outputs. Source datasets are either synthetic logistic data or random
subsamples of the real NLS data, drawn by Hypothesis.

Naming convention (used by the harness for RQ2 reporting):
  test_mr_inv_*  invariance relations   - output must NOT change
  test_mr_eqv_*  equivariance relations - output must change predictably
A constant (hard-coded) output trivially satisfies invariance relations, so
only equivariance relations can expose it.

MR sources: permutation / irrelevant-attribute / affine relations follow
Xie et al. (JSS 2011) and Murphy et al. (SEKE 2008); the regression-specific
scaling, sign-flip and duplication relations follow the MLR relations of
Luu et al. (JSS 2021), adapted to logistic regression.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from analysis import fit_ses_effect

DATA = Path(__file__).resolve().parent.parent / "data" / "nls_ses_processed.csv"
OUTCOME = "BA DEGREE COMPLETED"
REAL = pd.read_csv(DATA)

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
    n = draw(st.integers(400, 3000))
    beta = draw(st.floats(-1.5, 1.5))
    alpha = draw(st.floats(-1.5, 0.5))
    rng = np.random.default_rng(seed)
    ses = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(alpha + beta * ses)))
    y = rng.random(n) < p
    assume(15 <= y.sum() <= n - 15)
    return pd.DataFrame(
        {
            "CASE ID": np.arange(n),
            "SAMPLE_RACE": rng.choice(["Hispanic", "Black", "White"], n),
            "SAMPLE_SEX": rng.choice(["Male", "Female"], n),
            "SES": ses,
            OUTCOME: y,
        }
    )


@st.composite
def real_subsample_df(draw):
    seed = draw(st.integers(0, 2**32 - 1))
    size = draw(st.integers(3000, len(REAL)))
    d = REAL.sample(n=size, random_state=seed % (2**32 - 1)).reset_index(drop=True)
    assume(d[OUTCOME].sum() >= 15)
    return d


source_df = st.one_of(synthetic_df(), real_subsample_df())


def close(a, b, rel=1e-4, abs_=1e-6):
    return a == pytest.approx(b, rel=rel, abs=abs_)


# ---------------------------------------------------------------- invariance


@SETTINGS
@given(df=source_df, seed=st.integers(0, 2**32 - 1))
def test_mr_inv_row_permutation(df, seed):
    src = fit_ses_effect(df)
    fol = fit_ses_effect(df.sample(frac=1, random_state=seed % (2**32 - 1)))
    assert close(fol["coef"], src["coef"]) and close(fol["std_err"], src["std_err"], rel=1e-3)


@SETTINGS
@given(df=source_df, seed=st.integers(0, 2**32 - 1))
def test_mr_inv_irrelevant_column(df, seed):
    src = fit_ses_effect(df)
    noisy = df.assign(NOISE=np.random.default_rng(seed).normal(size=len(df)))
    fol = fit_ses_effect(noisy)
    assert close(fol["coef"], src["coef"]) and fol["n"] == src["n"]


@SETTINGS
@given(df=source_df, shift=st.floats(-50, 50))
def test_mr_inv_ses_shift(df, shift):
    """Adding a constant to SES changes only the intercept, not the slope."""
    src = fit_ses_effect(df)
    fol = fit_ses_effect(df.assign(SES=df["SES"] + shift))
    assert close(fol["coef"], src["coef"], rel=1e-3) and close(fol["std_err"], src["std_err"], rel=1e-3)


# -------------------------------------------------------------- equivariance


@SETTINGS
@given(df=source_df, c=st.floats(0.1, 10))
def test_mr_eqv_ses_scaling(df, c):
    """SES * c  ->  coef / c, std_err / c, p unchanged (unit conversion)."""
    assume(abs(c - 1) > 0.05)
    src = fit_ses_effect(df)
    fol = fit_ses_effect(df.assign(SES=df["SES"] * c))
    assert close(fol["coef"], src["coef"] / c, rel=1e-3)
    assert close(fol["std_err"], src["std_err"] / c, rel=1e-3)
    assert close(fol["p_value"], src["p_value"], rel=1e-2, abs_=1e-12)


@SETTINGS
@given(df=source_df)
def test_mr_eqv_ses_negation(df):
    """SES -> -SES  flips the sign of coef; std_err and p unchanged."""
    src = fit_ses_effect(df)
    assume(abs(src["coef"]) > 1e-3)
    fol = fit_ses_effect(df.assign(SES=-df["SES"]))
    assert close(fol["coef"], -src["coef"], rel=1e-3)
    assert close(fol["std_err"], src["std_err"], rel=1e-3)


@SETTINGS
@given(df=source_df)
def test_mr_eqv_outcome_flip(df):
    """y -> not y  flips the sign of coef (symmetry of the logit link)."""
    src = fit_ses_effect(df)
    assume(abs(src["coef"]) > 1e-3)
    fol = fit_ses_effect(df.assign(**{OUTCOME: ~df[OUTCOME].astype(bool)}))
    assert close(fol["coef"], -src["coef"], rel=1e-3)
    assert close(fol["std_err"], src["std_err"], rel=1e-3)


@SETTINGS
@given(df=source_df)
def test_mr_eqv_duplication(df):
    """Duplicating every row: same coef, std_err / sqrt(2), n doubles."""
    src = fit_ses_effect(df)
    fol = fit_ses_effect(pd.concat([df, df], ignore_index=True))
    assert fol["n"] == 2 * src["n"]
    assert close(fol["coef"], src["coef"], rel=1e-3)
    assert close(fol["std_err"], src["std_err"] / math.sqrt(2), rel=1e-3)
