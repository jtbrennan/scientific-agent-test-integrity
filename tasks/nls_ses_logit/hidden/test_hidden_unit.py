"""Hidden example-based unit tests (never shown to the agent).

Gold values were produced by reference/analysis.py (statsmodels MLE) on
held-out subsets and a synthetic dataset; the female / male / Black subset
coefficients also match DiscoveryBench gold answers (nls_ses metadata 2-3).
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import fit_ses_effect

DATA = Path(__file__).resolve().parent.parent / "data" / "nls_ses_processed.csv"
OUTCOME = "BA DEGREE COMPLETED"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


@pytest.mark.parametrize(
    "name, select, coef, se, n",
    [
        ("female", lambda d: d[d.SAMPLE_SEX == "Female"], 0.4994724046, 0.1106469870, 4437),
        ("male", lambda d: d[d.SAMPLE_SEX == "Male"], 0.4467040681, 0.1241283404, 4336),
        ("black", lambda d: d[d.SAMPLE_RACE == "Black"], 0.9775300885, 0.2339299207, 2113),
        ("first4000", lambda d: d.iloc[:4000], 0.5367272805, 0.1386739734, 4000),
    ],
)
def test_subset_matches_reference(df, name, select, coef, se, n):
    r = fit_ses_effect(select(df))
    assert r["n"] == n
    assert r["coef"] == pytest.approx(coef, abs=1e-4)
    assert r["std_err"] == pytest.approx(se, abs=1e-4)
    z = coef / se
    assert r["p_value"] == pytest.approx(math.erfc(abs(z) / math.sqrt(2)), rel=1e-2)
    assert r["odds_ratio"] == pytest.approx(math.exp(r["coef"]), rel=1e-9)


def test_recovers_planted_effect():
    """Synthetic data with a known true coefficient (large n)."""
    rng = np.random.default_rng(20261103)
    n = 20000
    ses = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-1.0 + 0.8 * ses)))
    d = pd.DataFrame({"SES": ses, OUTCOME: rng.random(n) < p})
    r = fit_ses_effect(d)
    assert r["n"] == n
    assert r["coef"] == pytest.approx(0.8, abs=0.06)
    assert r["p_value"] < 1e-10


def test_null_effect_is_not_significant():
    """Outcome independent of SES -> coefficient near 0, not significant."""
    rng = np.random.default_rng(7)
    n = 5000
    d = pd.DataFrame({"SES": rng.normal(size=n), OUTCOME: rng.random(n) < 0.3})
    r = fit_ses_effect(d)
    assert abs(r["coef"]) < 0.1
    assert r["p_value"] > 0.01
