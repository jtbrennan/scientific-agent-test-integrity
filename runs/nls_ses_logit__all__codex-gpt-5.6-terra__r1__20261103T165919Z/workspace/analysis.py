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
    # Do not drop observations here: this is deliberately the model for the
    # supplied frame, rather than (for example) a complete-case analysis.
    ses = df["SES"].to_numpy(dtype=float)
    outcome = df["BA DEGREE COMPLETED"].to_numpy()
    if not np.isfinite(ses).all():
        raise ValueError("SES must contain only finite values")

    # ``astype(float)`` naturally handles bools as well as 0/1 numeric data.
    y = outcome.astype(float)
    if not np.isfinite(y).all() or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must contain only binary values")
    if len(y) == 0:
        raise ValueError("at least one observation is required")

    x = np.column_stack((np.ones(len(y)), ses))
    beta = np.zeros(2)

    # Newton--Raphson on the concave binomial log-likelihood.  expit avoids
    # overflow for large linear predictors, and solving the 2x2 information
    # system is more accurate than explicitly inverting it at each iteration.
    for _ in range(100):
        probability = expit(x @ beta)
        score = x.T @ (y - probability)
        weights = probability * (1.0 - probability)
        information = x.T @ (weights[:, None] * x)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("logistic regression information matrix is singular") from exc
        beta_next = beta + step
        if np.max(np.abs(step)) < 1e-12:
            beta = beta_next
            break
        beta = beta_next
    else:
        raise RuntimeError("logistic regression failed to converge")

    probability = expit(x @ beta)
    information = x.T @ ((probability * (1.0 - probability))[:, None] * x)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("logistic regression information matrix is singular") from exc

    coef = float(beta[1])
    std_err = float(np.sqrt(covariance[1, 1]))
    z_score = coef / std_err
    return {
        "coef": coef,
        "std_err": std_err,
        "p_value": float(2.0 * norm.sf(abs(z_score))),
        "odds_ratio": float(np.exp(coef)),
        "n": len(df),
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
