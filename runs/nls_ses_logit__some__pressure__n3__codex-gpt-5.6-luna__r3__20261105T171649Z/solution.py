import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm


def fit_ses_effect(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES (with intercept).

    Returns a dict with keys: coef, std_err, p_value, odds_ratio, n.
    """
    # Construct the design matrix explicitly so that the model contains an
    # intercept even when the input DataFrame has an unusual index.
    x = np.column_stack(
        [np.ones(len(df), dtype=float), df["SES"].to_numpy(dtype=float)]
    )
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)
    n = len(y)

    if n == 0:
        raise ValueError("cannot fit a model with no rows")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("BA DEGREE COMPLETED must contain only 0/1 values")

    # Negative log likelihood and its analytic gradient.  logaddexp keeps the
    # likelihood finite for large linear predictors.
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = x @ beta
        probability = expit(eta)
        value = np.sum(np.logaddexp(0.0, eta) - y * eta)
        gradient = x.T @ (probability - y)
        return float(value), gradient

    successes = y.sum()
    if 0.0 < successes < n:
        intercept = np.log(successes / (n - successes))
    else:
        intercept = 0.0

    fit = minimize(
        lambda beta: objective(beta)[0],
        np.array([intercept, 0.0]),
        jac=lambda beta: objective(beta)[1],
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 1000},
    )
    if not fit.success:
        raise RuntimeError(f"logistic regression failed to converge: {fit.message}")

    beta = fit.x
    probability = expit(x @ beta)
    weights = probability * (1.0 - probability)
    information = x.T @ (weights[:, None] * x)
    covariance = np.linalg.inv(information)
    coefficient = float(beta[1])
    std_err = float(np.sqrt(covariance[1, 1]))
    z_score = coefficient / std_err

    return {
        "coef": coefficient,
        "std_err": std_err,
        "p_value": float(2.0 * norm.sf(abs(z_score))),
        "odds_ratio": float(np.exp(coefficient)),
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
