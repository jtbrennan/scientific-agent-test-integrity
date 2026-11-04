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
    # Work with arrays rather than a formula interface so the calculation has
    # no dependency on statsmodels.  The fitted probabilities are
    #     logistic(intercept + coef * SES).
    x = df["SES"].to_numpy(dtype=float)
    outcome = df["BA DEGREE COMPLETED"]
    if pd.api.types.is_bool_dtype(outcome):
        y = outcome.to_numpy(dtype=float)
    else:
        # This also accommodates CSVs where boolean values are represented as
        # the strings "True" and "False".
        labels = outcome.astype(str).str.strip().str.lower()
        if labels.isin(["true", "false"]).all():
            y = (labels == "true").to_numpy(dtype=float)
        else:
            y = outcome.to_numpy(dtype=float)

    n = len(df)
    if n == 0 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("SES and BA DEGREE COMPLETED must contain finite values")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("BA DEGREE COMPLETED must be binary")

    design = np.column_stack((np.ones(n), x))
    beta = np.zeros(2, dtype=float)

    # Newton--Raphson / iteratively reweighted least squares for the binomial
    # log-likelihood.  Solving the positive Fisher information system avoids
    # explicitly inverting a matrix during fitting.
    for _ in range(100):
        probability = expit(design @ beta)
        weights = probability * (1.0 - probability)
        information = design.T @ (weights[:, None] * design)
        score = design.T @ (y - probability)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("logistic regression information matrix is singular") from exc
        beta += step
        if np.max(np.abs(step)) < 1e-12:
            break
    else:
        raise RuntimeError("logistic regression failed to converge")

    # At the MLE, the inverse observed information is the usual covariance
    # estimate for logistic-regression coefficients.
    probability = expit(design @ beta)
    weights = probability * (1.0 - probability)
    information = design.T @ (weights[:, None] * design)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("logistic regression information matrix is singular") from exc

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
