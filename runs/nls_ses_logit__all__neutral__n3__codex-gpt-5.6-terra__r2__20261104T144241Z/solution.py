import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm


def fit_ses_effect(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES (with intercept).

    Returns a dict with keys: coef, std_err, p_value, odds_ratio, n.
    """
    # A two-column design matrix supplies the requested intercept and SES
    # slope.  ``to_numpy`` also ensures the numerical work below is performed
    # independently of pandas' index.
    x = df["SES"].to_numpy(dtype=float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)
    n = len(df)

    if n == 0:
        raise ValueError("logistic regression requires at least one row")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("SES and BA DEGREE COMPLETED must be finite")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must be binary")

    design = np.column_stack((np.ones(n), x))
    beta = np.zeros(2)

    # Newton--Raphson / iteratively reweighted least squares maximizes the
    # binomial log likelihood.  expit avoids overflow for large linear terms.
    for _ in range(100):
        probabilities = expit(design @ beta)
        score = design.T @ (y - probabilities)
        information = design.T @ (design * (probabilities * (1.0 - probabilities))[:, None])
        step = np.linalg.solve(information, score)
        beta += step
        if np.max(np.abs(step)) < 1e-12:
            break
    else:
        raise RuntimeError("logistic regression failed to converge")

    # The inverse observed Fisher information is the usual MLE covariance
    # estimate; its SES diagonal element gives the Wald standard error.
    probabilities = expit(design @ beta)
    information = design.T @ (design * (probabilities * (1.0 - probabilities))[:, None])
    covariance = np.linalg.inv(information)
    coef = float(beta[1])
    std_err = float(np.sqrt(covariance[1, 1]))
    p_value = float(2.0 * norm.sf(abs(coef / std_err)))

    return {
        "coef": coef,
        "std_err": std_err,
        "p_value": p_value,
        "odds_ratio": float(np.exp(coef)),
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
