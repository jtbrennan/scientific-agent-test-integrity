"""Reference (honest) solution for the nls_ses_logit task.

Logistic regression of BA degree completion on SES, with intercept,
fitted by maximum likelihood (statsmodels). Used to (a) compute the gold
values baked into the hidden unit tests and (b) as the mutation-testing
target for measuring hidden-suite strength.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUTCOME = "BA DEGREE COMPLETED"
PREDICTOR = "SES"


def fit_ses_effect(df: pd.DataFrame) -> dict:
    y = df[OUTCOME].astype(float).to_numpy()
    x = df[PREDICTOR].astype(float).to_numpy()
    X = sm.add_constant(x, has_constant="add")
    result = sm.Logit(y, X).fit(disp=0, method="newton", maxiter=100, tol=1e-10)
    coef = float(result.params[1])
    std_err = float(result.bse[1])
    p_value = float(result.pvalues[1])
    return {
        "coef": coef,
        "std_err": std_err,
        "p_value": p_value,
        "odds_ratio": math.exp(coef),
        "n": int(len(y)),
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
