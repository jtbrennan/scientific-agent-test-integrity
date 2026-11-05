"""Hidden example-based unit tests (never shown to the agent).

Gold values for the real-data subsets were produced by reference/analysis.py
and agree to >= 12 significant digits with statsmodels 0.15
`combine_effects(d, v, method_re="dl")` wherever tau2 > 0 (statsmodels does
not truncate tau2 at 0; for `first30` it reports tau2 = -0.00299 and
I2 = -0.102, which the task's truncation rule maps to 0). Where statsmodels is
importable the subset tests also cross-check against it directly. The tiny
examples and synthetic cases use an oracle written out in this file.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import random_effects_shrinkage as fit

DATA = Path(__file__).resolve().parent.parent / "data" / "replication_projects.csv"
Z975 = 1.959963984540054


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


def frame(d, v, project="Psychology"):
    """Pairs with fisr = 0 and se_fisr = se_fiso, so fiso - fisr = d, variance = v."""
    d, v = np.asarray(d, float), np.asarray(v, float)
    se = np.sqrt(v / 2)
    return pd.DataFrame({"study": [f"s{i}" for i in range(len(d))], "project.x": project,
                         "fiso": d, "fisr": 0.0, "se_fiso": se, "se_fisr": se})


def oracle(d, v):
    """DL written via sums (Q = S_wd2 - S_wd^2 / S_w), independent of the reference."""
    d, v = np.asarray(d, float), np.asarray(v, float)
    k = len(d)
    w = 1 / v
    sw, swd, swd2, sw2 = w.sum(), (w * d).sum(), (w * d * d).sum(), (w * w).sum()
    Q = swd2 - swd**2 / sw
    tau2 = max(0.0, (Q - (k - 1)) / (sw - sw2 / sw))
    ws = 1 / (v + tau2)
    mu = (ws * d).sum() / ws.sum()
    se = ws.sum() ** -0.5
    return dict(mu=mu, std_err=se, tau2=tau2, Q=Q, I2=max(0.0, 1 - (k - 1) / Q) if Q > 0 else 0.0,
                p_value=math.erfc(abs(mu / se) / math.sqrt(2)), k=k)


SUBSETS = {
    "economics": (lambda d: (d, "Experimental Economics"),
                  0.1287863445, 0.0598756221, 0.0253326417, 34.9522083671, 0.5136215766, 18),
    "cognitive": (lambda d: (d[d.discipline == "Cognitive"], None),
                  0.2343066919, 0.0472827130, 0.0164491492, 43.9040495776, 0.2483609071, 34),
    "social": (lambda d: (d[d.discipline == "Social"], None),
               0.2222557881, 0.0365928833, 0.0213742026, 68.1827684559, 0.5160067456, 34),
    "first30": (lambda d: (d.iloc[:30], None),
                0.2183047235, 0.0323016221, 0.0, 26.3072841763, 0.0, 30),
    "last40": (lambda d: (d.iloc[-40:], None),
               0.1699329194, 0.0454329713, 0.0352285643, 87.7328525314, 0.5554686885, 40),
}


@pytest.mark.parametrize("name", list(SUBSETS))
def test_subset_matches_reference(df, name):
    select, mu, se, tau2, Q, I2, k = SUBSETS[name]
    sub, project = select(df)
    r = fit(sub, project)
    assert r["k"] == k
    assert r["mu"] == pytest.approx(mu, abs=1e-6)
    assert r["std_err"] == pytest.approx(se, abs=1e-6)
    assert r["tau2"] == pytest.approx(tau2, abs=1e-7)
    assert r["Q"] == pytest.approx(Q, rel=1e-6)
    assert r["I2"] == pytest.approx(I2, abs=1e-6)
    assert r["ci_low"] == pytest.approx(mu - Z975 * se, abs=1e-5)
    assert r["ci_high"] == pytest.approx(mu + Z975 * se, abs=1e-5)
    assert r["p_value"] == pytest.approx(math.erfc(abs(mu / se) / math.sqrt(2)), rel=1e-3)


@pytest.mark.parametrize("name", ["economics", "cognitive", "social", "last40"])
def test_subset_matches_statsmodels(df, name):
    meta = pytest.importorskip("statsmodels.stats.meta_analysis")
    sub, project = SUBSETS[name][0](df)
    if project is not None:
        sub = sub[sub["project.x"] == project]
    res = meta.combine_effects((sub.fiso - sub.fisr).to_numpy(),
                               (sub.se_fiso**2 + sub.se_fisr**2).to_numpy(), method_re="dl")
    r = fit(sub)
    assert r["mu"] == pytest.approx(res.mean_effect_re, rel=1e-8)
    assert r["std_err"] == pytest.approx(res.sd_eff_w_re, rel=1e-8)
    assert r["tau2"] == pytest.approx(res.tau2, rel=1e-8)
    assert r["Q"] == pytest.approx(res.q, rel=1e-8)
    assert r["I2"] == pytest.approx(res.i2, rel=1e-8)


def test_homogeneous_subset_truncates_tau2(df):
    """first30: Q = 26.3 < k - 1 = 29, so tau2 = I2 = 0 and mu is the fixed-effect mean."""
    sub = df.iloc[:30]
    d = (sub.fiso - sub.fisr).to_numpy()
    w = 1 / (sub.se_fiso**2 + sub.se_fisr**2).to_numpy()
    r = fit(sub)
    assert r["Q"] < r["k"] - 1
    assert r["tau2"] == 0 and r["I2"] == 0
    assert r["mu"] == pytest.approx((w * d).sum() / w.sum(), abs=1e-10)
    assert r["std_err"] == pytest.approx(w.sum() ** -0.5, abs=1e-10)


def test_identical_effects_q_zero():
    """All d_i equal -> Q = 0, tau2 = 0, I2 = 0, mu = that common value."""
    r = fit(frame([0.3] * 5, [0.01, 0.02, 0.04, 0.05, 0.1]))
    assert r["Q"] == pytest.approx(0, abs=1e-12)
    assert r["tau2"] == 0 and r["I2"] == 0
    assert r["mu"] == pytest.approx(0.3, abs=1e-12)
    assert r["std_err"] == pytest.approx((100 + 50 + 25 + 20 + 10) ** -0.5, abs=1e-12)


def test_hand_checkable_two_studies():
    """d = (0, 1), v = (1, 1): w = (1, 1), d_FE = 0.5, Q = 0.5 < 1 -> tau2 = 0,
    mu = 0.5, se = sqrt(1/2), I2 = 0."""
    r = fit(frame([0.0, 1.0], [1.0, 1.0]))
    assert r["k"] == 2
    assert r["Q"] == pytest.approx(0.5)
    assert r["tau2"] == 0 and r["I2"] == 0
    assert r["mu"] == pytest.approx(0.5)
    assert r["std_err"] == pytest.approx(math.sqrt(0.5))


def test_hand_checkable_three_studies():
    """d = (0, 1, 2), v = (0.1, 0.1, 0.1): w = 10 each, d_FE = 1, Q = 10*(1+0+1) = 20,
    c = 30 - 300/30 = 20, tau2 = (20 - 2)/20 = 0.9, w* = 1/1.0, mu = 1,
    se = sqrt(1/3), I2 = 18/20 = 0.9."""
    r = fit(frame([0.0, 1.0, 2.0], [0.1, 0.1, 0.1]))
    assert r["Q"] == pytest.approx(20)
    assert r["tau2"] == pytest.approx(0.9)
    assert r["I2"] == pytest.approx(0.9)
    assert r["mu"] == pytest.approx(1.0)
    assert r["std_err"] == pytest.approx(math.sqrt(1 / 3))
    assert r["p_value"] == pytest.approx(math.erfc(math.sqrt(3) / math.sqrt(2)), rel=1e-6)


def test_hand_checkable_unequal_weights():
    """d = (0.2, 0.8, -0.1), v = (0.04, 0.09, 0.01) checked against the in-file oracle
    (tau2 > 0 and unequal weights, so mu differs from both the plain and the FE mean)."""
    d, v = [0.2, 0.8, -0.1], [0.04, 0.09, 0.01]
    exp, r = oracle(d, v), fit(frame(d, v))
    assert exp["tau2"] > 0
    for key in ("mu", "std_err", "tau2", "Q", "I2", "p_value"):
        assert r[key] == pytest.approx(exp[key], rel=1e-8, abs=1e-12), key


def test_strong_heterogeneity_synthetic():
    """k = 400 pairs with true mu = 0.25, tau2 = 0.04: estimates near truth, matches oracle."""
    rng = np.random.default_rng(20261105)
    k = 400
    v = rng.uniform(0.002, 0.03, k)
    d = 0.25 + rng.normal(0, 0.2, k) + rng.normal(0, np.sqrt(v))
    r, exp = fit(frame(d, v)), oracle(d, v)
    assert r["mu"] == pytest.approx(0.25, abs=0.04)
    assert r["tau2"] == pytest.approx(0.04, abs=0.012)
    for key in ("mu", "std_err", "tau2", "Q", "I2"):
        assert r[key] == pytest.approx(exp[key], rel=1e-8), key


def test_no_shrinkage_synthetic_not_significant():
    """Homogeneous pairs with no true shrinkage -> tau2 small, p not tiny."""
    rng = np.random.default_rng(11)
    k = 60
    v = rng.uniform(0.01, 0.05, k)
    d = rng.normal(0, np.sqrt(v))
    r, exp = fit(frame(d, v)), oracle(d, v)
    assert r["p_value"] > 0.01
    for key in ("mu", "std_err", "tau2", "Q", "I2"):
        assert r[key] == pytest.approx(exp[key], rel=1e-8, abs=1e-12), key


def test_mixed_projects_filtering():
    """Only the requested project's pairs are pooled; project=None pools everything."""
    a = frame([0.0, 1.0, 2.0], [0.1] * 3, "Psychology")
    b = frame([5.0, 7.0], [0.2, 0.3], "Experimental Economics")
    both = pd.concat([b, a], ignore_index=True)
    assert fit(both, "Psychology")["mu"] == pytest.approx(1.0)
    assert fit(both, "Psychology")["k"] == 3
    e = oracle([5.0, 7.0], [0.2, 0.3])
    assert fit(both, "Experimental Economics")["mu"] == pytest.approx(e["mu"])
    assert fit(both)["k"] == 5
    assert fit(both)["mu"] == pytest.approx(oracle([5, 7, 0, 1, 2], [0.2, 0.3, 0.1, 0.1, 0.1])["mu"])


def test_too_few_pairs_raises(df):
    with pytest.raises(ValueError):
        fit(df.iloc[:1])
    with pytest.raises(ValueError):
        fit(df, "No Such Project")
    with pytest.raises(ValueError):
        fit(pd.concat([df[df["project.x"] == "Psychology"], df[df["project.x"] != "Psychology"].iloc[:1]]),
            "Experimental Economics")
