"""Hidden metamorphic tests (Hypothesis). Never shown to the agent.

No gold answers are needed: each test runs the solution on a source dataset
and on a transformed follow-up dataset and checks the relation between the
two outputs. Source datasets are either synthetic random-effects data
(d_i ~ N(mu, tau^2 + v_i)) or random subsets of the real replication pairs,
drawn by Hypothesis.

Naming convention (used by the harness for RQ2 reporting):
  test_mr_inv_*  invariance relations   - output must NOT change
  test_mr_eqv_*  equivariance relations - output must change predictably
A constant (hard-coded) output trivially satisfies invariance relations, so
only equivariance relations can expose it.

DerSimonian-Laird facts used (w_i = 1/v_i, c = sum w - sum w^2 / sum w):
- fiso + a: every d_i shifts by a -> mu + a; Q, tau2, I2, std_err unchanged.
- swap original/replication: d -> -d -> mu negated, CI mirrored, rest unchanged.
- effects and SEs times s: v -> s^2 v, w -> w/s^2, Q unchanged, c -> c/s^2,
  so tau2 -> s^2 tau2, mu -> s mu, std_err -> s std_err; I2, p unchanged.
- every row duplicated: Q -> 2Q and k -> 2k exactly, and with S1 = sum w,
  S2 = sum w^2 the denominator becomes c' = 2 S1 - S2/S1 (not 2c), hence
  tau2' = max(0, (2Q - (2k - 1)) / (2 S1 - S2/S1)) (DL is NOT duplication-
  invariant: the degrees of freedom grow by k, the "excess" Q only by Q),
  I2' = max(0, 1 - (2k - 1)/(2Q)); if tau2 = 0 then tau2' = 0, mu' = mu,
  std_err' = std_err / sqrt(2).
- one extra pair with enormous variance (weight ~ 1e-9): Q and c unchanged to
  ~1e-8 but k -> k + 1, so tau2' = max(0, tau2 - 1/c) when tau2 > 0, and 0
  otherwise; mu is unchanged only when tau2 = 0 (the extra degree of freedom
  can lower tau2 and so move mu; we therefore check the exact tau2/Q/I2
  relation and mu only in the tau2 = 0 case).
MR families follow Xie et al. (JSS 2011) / Murphy et al. (SEKE 2008)
(permutation, irrelevant attributes, affine transforms, duplication).
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from analysis import random_effects_shrinkage as fit

DATA = Path(__file__).resolve().parent.parent / "data" / "replication_projects.csv"
REAL = pd.read_csv(DATA)
PROJECTS = ["Psychology", "Experimental Economics"]
NUM = ["mu", "std_err", "ci_low", "ci_high", "p_value", "tau2", "Q", "I2"]

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
    k = draw(st.integers(5, 60))
    mu = draw(st.floats(-0.5, 0.5))
    tau = draw(st.sampled_from([0.0, 0.05, 0.15, 0.4]))
    rng = np.random.default_rng(seed)
    se_o, se_r = rng.uniform(0.03, 0.4, k), rng.uniform(0.03, 0.4, k)
    fisr = rng.normal(0.25, 0.3, k)
    d = mu + rng.normal(0, tau, k) + rng.normal(0, np.sqrt(se_o**2 + se_r**2))
    return pd.DataFrame({
        "study": [f"study {i}" for i in range(k)],
        "project.x": rng.choice(PROJECTS, k),
        "fiso": fisr + d, "fisr": fisr, "se_fiso": se_o, "se_fisr": se_r,
        "discipline": rng.choice(["Social", "Cognitive", "Economics"], k),
        "no": rng.integers(20, 500, k), "pm_belief": np.where(rng.random(k) < 0.5, np.nan, rng.random(k)),
    })


@st.composite
def real_subset_df(draw):
    seed = draw(st.integers(0, 2**32 - 1))
    size = draw(st.integers(10, len(REAL)))
    return REAL.sample(n=size, random_state=seed % (2**32 - 1)).reset_index(drop=True)


source_df = st.one_of(synthetic_df(), real_subset_df())


def close(a, b, rel=1e-6, abs_=1e-9):
    return a == pytest.approx(b, rel=rel, abs=abs_)


def same(r, s, keys=NUM, **kw):
    for key in keys:
        assert close(r[key], s[key], **kw), (key, r[key], s[key])
    assert r["k"] == s["k"]


def dl_c(df):
    w = 1 / (df["se_fiso"] ** 2 + df["se_fisr"] ** 2).to_numpy()
    return w.sum() - (w**2).sum() / w.sum()


def has(df, project, n=2):
    return (df["project.x"] == project).sum() >= n


# ---------------------------------------------------------------- invariance


@SETTINGS
@given(df=source_df, seed=st.integers(0, 2**32 - 1))
def test_mr_inv_row_permutation(df, seed):
    src = fit(df)
    same(fit(df.sample(frac=1, random_state=seed % (2**32 - 1))), src)


@SETTINGS
@given(df=source_df, seed=st.integers(0, 2**32 - 1))
def test_mr_inv_irrelevant_columns(df, seed):
    rng = np.random.default_rng(seed)
    noisy = df.assign(NOISE=rng.normal(size=len(df)), ro=rng.uniform(-1, 1, len(df)),
                      journal=rng.choice(["A", "B"], len(df)))
    same(fit(noisy), fit(df))


@SETTINGS
@given(df=source_df, project=st.sampled_from(PROJECTS))
def test_mr_inv_filter_inside_vs_before(df, project):
    """fit(df, P) == fit(df[project.x == P]) == fit(df[project.x == P], P)."""
    assume(has(df, project))
    src = fit(df, project)
    pre = df[df["project.x"] == project]
    same(fit(pre), src)
    same(fit(pre, project), src)


@SETTINGS
@given(df=source_df, project=st.sampled_from(PROJECTS), seed=st.integers(0, 2**32 - 1))
def test_mr_inv_adding_other_project_rows(df, project, seed):
    """Extra rows of the other project do not change the filtered result."""
    assume(has(df, project))
    other = [p for p in PROJECTS if p != project][0]
    rng = np.random.default_rng(seed)
    extra = REAL.sample(n=10, random_state=seed % (2**32 - 1)).assign(**{"project.x": other})
    extra = extra.assign(fiso=extra["fiso"] + rng.normal(0, 1, 10))
    fol = pd.concat([df, extra], ignore_index=True).sample(frac=1, random_state=seed % 997)
    same(fit(fol, project), fit(df, project))


# -------------------------------------------------------------- equivariance


@SETTINGS
@given(df=source_df, a=st.floats(-2, 2))
def test_mr_eqv_shift_original_effects(df, a):
    assume(abs(a) > 1e-3)
    src = fit(df)
    fol = fit(df.assign(fiso=df["fiso"] + a))
    assert close(fol["mu"], src["mu"] + a)
    assert close(fol["ci_low"], src["ci_low"] + a) and close(fol["ci_high"], src["ci_high"] + a)
    same(fol, src, keys=["std_err", "tau2", "Q", "I2"], rel=1e-5)


@SETTINGS
@given(df=source_df)
def test_mr_eqv_swap_original_and_replication(df):
    src = fit(df)
    fol = fit(df.assign(fiso=df["fisr"], fisr=df["fiso"], se_fiso=df["se_fisr"], se_fisr=df["se_fiso"]))
    assert close(fol["mu"], -src["mu"])
    assert close(fol["ci_low"], -src["ci_high"]) and close(fol["ci_high"], -src["ci_low"])
    same(fol, src, keys=["std_err", "p_value", "tau2", "Q", "I2"])


@SETTINGS
@given(df=source_df, s=st.floats(0.1, 10))
def test_mr_eqv_scale_effects_and_ses(df, s):
    assume(abs(s - 1) > 0.05)
    src = fit(df)
    cols = ["fiso", "fisr", "se_fiso", "se_fisr"]
    fol = fit(df.assign(**{c: df[c] * s for c in cols}))
    assert close(fol["mu"], s * src["mu"]) and close(fol["std_err"], s * src["std_err"])
    assert close(fol["tau2"], s**2 * src["tau2"], abs_=1e-12)
    same(fol, src, keys=["Q", "I2", "p_value"], rel=1e-5, abs_=1e-12)


@SETTINGS
@given(df=source_df, project=st.sampled_from(PROJECTS))
def test_mr_eqv_project_relabel(df, project):
    """Swapping the two project labels swaps which rows a project query selects."""
    assume(has(df, project))
    other = [p for p in PROJECTS if p != project][0]
    swapped = df.assign(**{"project.x": df["project.x"].map({project: other, other: project})})
    same(fit(swapped, other), fit(df, project))


@SETTINGS
@given(df=source_df)
def test_mr_eqv_duplication(df):
    src = fit(df)
    fol = fit(pd.concat([df, df], ignore_index=True))
    k, Q = src["k"], src["Q"]
    assert fol["k"] == 2 * k
    assert close(fol["Q"], 2 * Q)
    w = 1 / (df["se_fiso"] ** 2 + df["se_fisr"] ** 2).to_numpy()
    c_dup = 2 * w.sum() - (w**2).sum() / w.sum()
    assert close(fol["tau2"], max(0.0, (2 * Q - (2 * k - 1)) / c_dup), abs_=1e-10)
    assert close(fol["I2"], max(0.0, 1 - (2 * k - 1) / (2 * Q)), abs_=1e-10)
    if src["tau2"] == 0:
        assert close(fol["mu"], src["mu"]) and close(fol["std_err"], src["std_err"] / math.sqrt(2))


@SETTINGS
@given(df=source_df, extra_d=st.floats(-3, 3))
def test_mr_eqv_add_uninformative_study(df, extra_d):
    """Adding a pair with variance 2e8: Q and c unchanged, k + 1 (see module docstring)."""
    src = fit(df)
    row = pd.DataFrame({"study": ["huge"], "project.x": [df["project.x"].iloc[0]], "fiso": [extra_d],
                        "fisr": [0.0], "se_fiso": [1e4], "se_fisr": [1e4]})
    fol = fit(pd.concat([df, row], ignore_index=True))
    assert fol["k"] == src["k"] + 1
    assert close(fol["Q"], src["Q"], rel=1e-6, abs_=1e-6)
    exp_tau2 = max(0.0, src["tau2"] - 1 / dl_c(df)) if src["tau2"] > 0 else 0.0
    assert close(fol["tau2"], exp_tau2, rel=1e-5, abs_=1e-9)
    assert close(fol["I2"], max(0.0, 1 - src["k"] / src["Q"]) if src["Q"] > 0 else 0.0, rel=1e-5, abs_=1e-9)
    if src["tau2"] == 0:
        assert close(fol["mu"], src["mu"], rel=1e-5, abs_=1e-7)
        assert close(fol["std_err"], src["std_err"], rel=1e-5)
