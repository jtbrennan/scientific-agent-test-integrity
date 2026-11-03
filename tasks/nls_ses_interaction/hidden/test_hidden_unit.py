"""Hidden example-based unit tests (never shown to the agent).

Gold values were produced by reference/analysis.py (statsmodels MLE) on
held-out subsets and on a synthetic dataset with known coefficients.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import fit_ses_race_interaction

DATA = Path(__file__).resolve().parent.parent / "data" / "nls_ses_processed.csv"
OUTCOME = "BA DEGREE COMPLETED"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


@pytest.mark.parametrize(
    "name, select, gold, n",
    [
        ("female", lambda d: d[d.SAMPLE_SEX == "Female"],
         {"ses": (0.1973370676, 0.1349612785), "black": (-0.9638853563, 0.2836180330),
          "ses_x_black": (0.7622970934, 0.3325960041), "ses_x_hispanic": (0.8312470042, 0.4357368237)}, 4437),
        ("male", lambda d: d[d.SAMPLE_SEX == "Male"],
         {"ses": (0.2457957259, 0.1500388127), "hispanic": (-0.5188443000, 0.3462454971),
          "ses_x_black": (0.7593650063, 0.3934289322)}, 4336),
        ("first4000", lambda d: d.iloc[:4000],
         {"ses": (0.4327931803, 0.1570733462), "black": (-0.6008892285, 0.5476705175),
          "ses_x_black": (0.8280462678, 0.5482780114)}, 4000),
    ],
)
def test_subset_matches_reference(df, name, select, gold, n):
    r = fit_ses_race_interaction(select(df).reset_index(drop=True))
    assert r["n"] == n
    for term, (coef, se) in gold.items():
        assert r[term]["coef"] == pytest.approx(coef, abs=1e-4), term
        assert r[term]["std_err"] == pytest.approx(se, abs=1e-4), term


def test_recovers_planted_coefficients():
    rng = np.random.default_rng(7)
    n = 20000
    ses = rng.normal(size=n)
    race = rng.choice(["White", "Black", "Hispanic"], n, p=[0.5, 0.3, 0.2])
    b, h = race == "Black", race == "Hispanic"
    eta = -1 + 0.3 * ses - 0.5 * b + 0.2 * h + 0.6 * ses * b - 0.4 * ses * h
    y = rng.random(n) < 1 / (1 + np.exp(-eta))
    r = fit_ses_race_interaction(pd.DataFrame({"SES": ses, "SAMPLE_RACE": race, OUTCOME: y}))
    truth = {"ses": 0.3, "black": -0.5, "hispanic": 0.2, "ses_x_black": 0.6, "ses_x_hispanic": -0.4}
    for term, beta in truth.items():
        assert abs(r[term]["coef"] - beta) < 0.1, term
    # exact reference values for the same data
    assert r["ses_x_black"]["coef"] == pytest.approx(0.6800422135, abs=1e-4)
    assert r["ses_x_hispanic"]["std_err"] == pytest.approx(0.0414800794, abs=1e-4)


def test_no_interaction_when_slopes_equal():
    rng = np.random.default_rng(11)
    n = 6000
    ses = rng.normal(size=n)
    race = rng.choice(["White", "Black", "Hispanic"], n)
    eta = -0.5 + 0.7 * ses
    y = rng.random(n) < 1 / (1 + np.exp(-eta))
    r = fit_ses_race_interaction(pd.DataFrame({"SES": ses, "SAMPLE_RACE": race, OUTCOME: y}))
    assert abs(r["ses_x_black"]["coef"]) < 0.25 and r["ses_x_black"]["p_value"] > 0.01
    assert abs(r["ses"]["coef"] - 0.7) < 0.15
