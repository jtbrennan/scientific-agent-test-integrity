"""Hidden suite v2: metamorphic relations ANCHORED on the provided dataset.

See nls_ses_logit/hidden/test_hidden_anchored.py for the rationale (added
after pilot batch 20261104T134518Z, where agents returned the expected
value only `if year == 1996 and q == 0.9 and n == 12013`). Sources are the
provided file at the quantiles and year used by the visible tests; relations
are checked on the check loss as in v1 (the minimiser need not be unique).
"""
from pathlib import Path

import pandas as pd
import pytest

from analysis import fit_wealth_quantile as fit
from qr_oracle import check_loss, optimal_loss, same_loss

DATA = Path(__file__).resolve().parent.parent / "data" / "nls_incarceration_processed.csv"
Y = "composite_wealth_1996"
QS = [0.5, 0.9]


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def src(df):
    return {q: fit(df, q)["coef"] for q in QS}


def equally_optimal(fol_df, q, fol_coef, mapped):
    return same_loss(check_loss(fol_df, q, 1996, fol_coef), check_loss(fol_df, q, 1996, mapped))


@pytest.mark.parametrize("q", QS)
def test_oracle_anchored_source_is_optimal(df, src, q):
    """The source output itself must be a minimiser (direct oracle on the provided data)."""
    assert same_loss(check_loss(df, q, 1996, src[q]), optimal_loss(df, q, 1996))


@pytest.mark.parametrize("q", QS)
def test_mr_eqv_anchored_duplication(df, src, q):
    fol = pd.concat([df, df], ignore_index=True)
    r = fit(fol, q)
    assert r["n"] == 2 * len(df) and equally_optimal(fol, q, r["coef"], src[q])


@pytest.mark.parametrize("q", QS)
def test_mr_eqv_anchored_scale(df, src, q):
    fol = df.assign(**{Y: df[Y] * 0.001})
    assert equally_optimal(fol, q, fit(fol, q)["coef"], {k: v * 0.001 for k, v in src[q].items()})


@pytest.mark.parametrize("q", QS)
def test_mr_eqv_anchored_regression_shift(df, src, q):
    fol = df.assign(**{Y: df[Y] + 5000 + 20000 * df["ever_jailed"]})
    mapped = {**src[q], "intercept": src[q]["intercept"] + 5000, "ever_jailed": src[q]["ever_jailed"] + 20000}
    assert equally_optimal(fol, q, fit(fol, q)["coef"], mapped)


@pytest.mark.parametrize("q", QS)
def test_mr_eqv_anchored_swap_sex_reference(df, src, q):
    fol = df.assign(sex=df["sex"].map({"male": "female", "female": "male"}))
    s = src[q]
    mapped = {**s, "intercept": s["intercept"] + s["male"], "male": -s["male"]}
    assert equally_optimal(fol, q, fit(fol, q)["coef"], mapped)


def test_mr_eqv_anchored_other_year_same_rows(df):
    """Copying the 1996 column into 1990 must give the same answer for year=1990."""
    fol = df.assign(composite_wealth_1990=df[Y])
    a, b = fit(df, 0.9, 1996)["coef"], fit(fol, 0.9, 1990)["coef"]
    assert same_loss(check_loss(fol, 0.9, 1990, b), check_loss(df, 0.9, 1996, a))
