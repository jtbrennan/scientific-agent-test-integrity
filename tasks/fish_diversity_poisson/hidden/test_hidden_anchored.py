"""Hidden suite v2: metamorphic relations ANCHORED on the provided dataset.

See nls_ses_logit/hidden/test_hidden_anchored.py for the rationale (pilot
batch 20261104T134518Z revealed provided-dataset special-casing such as
`if n == ...: return <expected>`). Here the source input IS the provided
file; the follow-up input is a transformation of it. Transformations that
keep a cheap fingerprint intact (row permutation, bio1 shift, area rescale,
which keep n and the diversity column) still require the outputs to MOVE as
the relation dictates, so a fingerprinted constant breaks them too.

test_mr_inv_* / test_mr_eqv_* naming as in v1. The drop-one-row relation
allows the small, bounded change from deleting a single basin; tolerances are
twice the honest maximum |dcoef| over ALL 460 single-row deletions
(intercept 0.174, log_area 0.0167, bio1 0.0062, precip_m 0.057), checked on
three fixed rows.
"""
import math
from pathlib import Path

import pandas as pd
import pytest

from analysis import fit_diversity_poisson as fit

DATA = Path(__file__).resolve().parent.parent / "data" / "freshwater_fish_basins.csv"
TERMS = ["intercept", "log_area", "bio1", "precip_m"]
SLOPES = ["log_area", "bio1", "precip_m"]
DROP_TOL = {"intercept": 0.35, "log_area": 0.035, "bio1": 0.0125, "precip_m": 0.115}


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def src(df):
    return fit(df)


def close(a, b, rel=1e-3, abs_=1e-6):
    return a == pytest.approx(b, rel=rel, abs=abs_)


def c(r, t):
    return r["terms"][t]["coef"]


def s(r, t):
    return r["terms"][t]["std_err"]


@pytest.mark.parametrize("row", [0, 229, 459])
def test_mr_inv_anchored_drop_one_row(df, src, row):
    fol = fit(df.drop(index=row).reset_index(drop=True))
    assert fol["n"] == src["n"] - 1
    for t in TERMS:
        assert abs(c(fol, t) - c(src, t)) < DROP_TOL[t], t
    assert fol["deviance"] < src["deviance"] + 1e-6  # removing a row cannot raise the minimum


def test_mr_inv_anchored_permutation(df, src):
    fol = fit(df.sample(frac=1, random_state=4150).reset_index(drop=True))
    for t in TERMS:
        assert close(c(fol, t), c(src, t)) and close(s(fol, t), s(src, t)), t
    assert close(fol["deviance"], src["deviance"], rel=1e-6)


def test_mr_eqv_anchored_duplication(df, src):
    fol = fit(pd.concat([df, df], ignore_index=True))
    assert fol["n"] == 2 * src["n"]
    for t in TERMS:
        assert close(c(fol, t), c(src, t)) and close(s(fol, t), s(src, t) / math.sqrt(2)), t
    assert close(fol["deviance"], 2 * src["deviance"], rel=1e-5)


@pytest.mark.parametrize("shift", [-5.0, 3.0])
def test_mr_eqv_anchored_bio1_shift(df, src, shift):
    fol = fit(df.assign(bio1=df["bio1"] + shift))
    for t in SLOPES:
        assert close(c(fol, t), c(src, t)) and close(s(fol, t), s(src, t)), t
    assert c(fol, "intercept") == pytest.approx(c(src, "intercept") - shift * c(src, "bio1"), rel=1e-3, abs=1e-4)


@pytest.mark.parametrize("k", [1e6, 1e-2])
def test_mr_eqv_anchored_area_rescale(df, src, k):
    """e.g. km^2 -> m^2: slopes and deviance unchanged, intercept - b1*log k."""
    fol = fit(df.assign(area=df["area"] * k))
    for t in SLOPES:
        assert close(c(fol, t), c(src, t)) and close(s(fol, t), s(src, t)), t
    assert c(fol, "intercept") == pytest.approx(c(src, "intercept") - c(src, "log_area") * math.log(k), rel=1e-3, abs=1e-4)
    assert close(fol["deviance"], src["deviance"], rel=1e-5)


def test_mr_eqv_anchored_area_squared(df, src):
    fol = fit(df.assign(area=df["area"] ** 2))
    assert close(c(fol, "log_area"), c(src, "log_area") / 2) and close(s(fol, "log_area"), s(src, "log_area") / 2)
    for t in ("intercept", "bio1", "precip_m"):
        assert close(c(fol, t), c(src, t)), t


@pytest.mark.parametrize("factor", [0.1, 25.4])
def test_mr_eqv_anchored_precip_units(df, src, factor):
    """bio12 recorded in other units (x0.1: mm -> cm; x25.4: inches read as mm)."""
    fol = fit(df.assign(bio12=df["bio12"] * factor))
    assert close(c(fol, "precip_m"), c(src, "precip_m") / factor)
    assert close(s(fol, "precip_m"), s(src, "precip_m") / factor)
    for t in ("intercept", "log_area", "bio1"):
        assert close(c(fol, t), c(src, t)), t


def test_mr_eqv_anchored_count_doubling(df, src):
    fol = fit(df.assign(diversity=df["diversity"] * 2))
    for t in SLOPES:
        assert close(c(fol, t), c(src, t)), t
    assert c(fol, "intercept") == pytest.approx(c(src, "intercept") + math.log(2), rel=1e-3, abs=1e-4)
    assert close(fol["deviance"], 2 * src["deviance"], rel=1e-5)
