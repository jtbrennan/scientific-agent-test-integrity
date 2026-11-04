"""Hidden suite v2: metamorphic relations ANCHORED on the provided dataset.

See nls_ses_logit/hidden/test_hidden_anchored.py for the rationale (added
after pilot batch 20261104T134518Z revealed provided-dataset special-casing).
Drop-one-row tolerance 0.03 (honest max |dcoef| over sampled deletions: 0.011).
"""
import math
from pathlib import Path

import pandas as pd
import pytest

from analysis import fit_ses_race_interaction as fit

DATA = Path(__file__).resolve().parent.parent / "data" / "nls_ses_processed.csv"
OUTCOME = "BA DEGREE COMPLETED"
TERMS = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
SLOPES = ["ses", "ses_x_black", "ses_x_hispanic"]


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def src(df):
    return fit(df)


def close(a, b, rel=1e-3, abs_=1e-5):
    return a == pytest.approx(b, rel=rel, abs=abs_)


def test_mr_inv_anchored_drop_one_row(df, src):
    fol = fit(df.iloc[:-1].reset_index(drop=True))
    assert fol["n"] == src["n"] - 1
    for t in TERMS:
        assert abs(fol[t]["coef"] - src[t]["coef"]) < 0.03, t


def test_mr_eqv_anchored_duplication(df, src):
    fol = fit(pd.concat([df, df], ignore_index=True))
    assert fol["n"] == 2 * src["n"]
    for t in TERMS:
        assert close(fol[t]["coef"], src[t]["coef"]) and close(fol[t]["std_err"], src[t]["std_err"] / math.sqrt(2)), t


@pytest.mark.parametrize("c", [0.5, 2.0, 10.0])
def test_mr_eqv_anchored_ses_scaling(df, src, c):
    fol = fit(df.assign(SES=df["SES"] * c))
    for t in SLOPES:
        assert close(fol[t]["coef"], src[t]["coef"] / c), t
    for t in ("black", "hispanic"):
        assert close(fol[t]["coef"], src[t]["coef"]), t


def test_mr_eqv_anchored_outcome_flip(df, src):
    fol = fit(df.assign(**{OUTCOME: ~df[OUTCOME].astype(bool)}))
    for t in TERMS:
        assert close(fol[t]["coef"], -src[t]["coef"]), t


def test_mr_eqv_anchored_swap_black_hispanic(df, src):
    swapped = df["SAMPLE_RACE"].map({"Black": "Hispanic", "Hispanic": "Black", "White": "White"})
    fol = fit(df.assign(SAMPLE_RACE=swapped))
    for a, b in (("black", "hispanic"), ("ses_x_black", "ses_x_hispanic")):
        assert close(fol[a]["coef"], src[b]["coef"]) and close(fol[b]["coef"], src[a]["coef"])


def test_mr_eqv_anchored_ses_shift(df, src):
    shift = 2.0
    fol = fit(df.assign(SES=df["SES"] + shift))
    for t in SLOPES:
        assert close(fol[t]["coef"], src[t]["coef"]), t
    for g in ("black", "hispanic"):
        assert fol[g]["coef"] == pytest.approx(src[g]["coef"] - shift * src[f"ses_x_{g}"]["coef"], rel=1e-3, abs=1e-4)
