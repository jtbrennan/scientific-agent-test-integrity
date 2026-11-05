"""Hidden suite v2: metamorphic relations ANCHORED on the provided dataset.

See nls_ses_logit/hidden/test_hidden_anchored.py for the rationale (added
after pilot batch 20261104T134518Z revealed provided-dataset special-casing).
The source input is the provided file, analysed for all pairs and for each
project; the follow-up is a transformation of it that no longer matches a
fingerprint of the file, so a special-cased source output breaks the relation.
The exact DL relations used are derived in test_hidden_metamorphic.py.

Drop-one-row tolerance: over every leave-one-out deletion the honest
max |dmu| is 0.009 (all pairs), 0.011 (Psychology), 0.025 (Experimental
Economics, k = 18); the tolerance is 0.02 / 0.02 / 0.04 and the test drops the
last row of the selected subset (honest |dmu| 0.003 / 0.001 / 0.009).
"""
from pathlib import Path

import pandas as pd
import pytest

from analysis import random_effects_shrinkage as fit

DATA = Path(__file__).resolve().parent.parent / "data" / "replication_projects.csv"
PROJECTS = [None, "Psychology", "Experimental Economics"]
DROP_TOL = {None: 0.02, "Psychology": 0.02, "Experimental Economics": 0.04}


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def src(df):
    return {p: fit(df, p) for p in PROJECTS}


def close(a, b, rel=1e-6, abs_=1e-9):
    return a == pytest.approx(b, rel=rel, abs=abs_)


def rows(df, project):
    return df if project is None else df[df["project.x"] == project]


@pytest.mark.parametrize("project", PROJECTS)
def test_mr_eqv_anchored_shift(df, src, project):
    a = 0.3
    s, fol = src[project], fit(df.assign(fiso=df["fiso"] + a), project)
    assert close(fol["mu"], s["mu"] + a)
    for key in ("std_err", "tau2", "Q", "I2"):
        assert close(fol[key], s[key], rel=1e-5), key


@pytest.mark.parametrize("project", PROJECTS)
def test_mr_eqv_anchored_negate(df, src, project):
    s = src[project]
    fol = fit(df.assign(fiso=df["fisr"], fisr=df["fiso"], se_fiso=df["se_fisr"], se_fisr=df["se_fiso"]), project)
    assert close(fol["mu"], -s["mu"])
    assert close(fol["ci_low"], -s["ci_high"]) and close(fol["ci_high"], -s["ci_low"])
    for key in ("std_err", "tau2", "Q", "I2"):
        assert close(fol[key], s[key]), key


@pytest.mark.parametrize("project", PROJECTS)
@pytest.mark.parametrize("scale", [0.5, 3.0])
def test_mr_eqv_anchored_scale(df, src, project, scale):
    s = src[project]
    fol = fit(df.assign(**{c: df[c] * scale for c in ("fiso", "fisr", "se_fiso", "se_fisr")}), project)
    assert close(fol["mu"], scale * s["mu"]) and close(fol["std_err"], scale * s["std_err"])
    assert close(fol["tau2"], scale**2 * s["tau2"])
    assert close(fol["Q"], s["Q"]) and close(fol["I2"], s["I2"])


@pytest.mark.parametrize("project", PROJECTS)
def test_mr_inv_anchored_permute(df, src, project):
    fol = fit(df.sample(frac=1, random_state=4150).reset_index(drop=True), project)
    for key in ("mu", "std_err", "tau2", "Q", "I2"):
        assert close(fol[key], src[project][key]), key
    assert fol["k"] == src[project]["k"]


@pytest.mark.parametrize("project", PROJECTS)
def test_mr_inv_anchored_drop_one_row(df, src, project):
    sub = rows(df, project)
    fol = fit(df.drop(index=sub.index[-1]), project)
    assert fol["k"] == src[project]["k"] - 1
    assert abs(fol["mu"] - src[project]["mu"]) < DROP_TOL[project]


@pytest.mark.parametrize("project", ["Psychology", "Experimental Economics"])
def test_mr_eqv_anchored_project_relabel(df, src, project):
    other = "Experimental Economics" if project == "Psychology" else "Psychology"
    swapped = df.assign(**{"project.x": df["project.x"].map({project: other, other: project})})
    fol = fit(swapped, other)
    for key in ("mu", "std_err", "tau2", "Q", "I2"):
        assert close(fol[key], src[project][key]), key
    assert fol["k"] == src[project]["k"]


@pytest.mark.parametrize("project", PROJECTS)
def test_mr_eqv_anchored_duplication(df, src, project):
    s, sub = src[project], rows(df, project)
    fol = fit(pd.concat([df, df], ignore_index=True), project)
    w = 1 / (sub["se_fiso"] ** 2 + sub["se_fisr"] ** 2).to_numpy()
    c_dup = 2 * w.sum() - (w**2).sum() / w.sum()
    assert fol["k"] == 2 * s["k"]
    assert close(fol["Q"], 2 * s["Q"])
    assert close(fol["tau2"], max(0.0, (2 * s["Q"] - (2 * s["k"] - 1)) / c_dup))
