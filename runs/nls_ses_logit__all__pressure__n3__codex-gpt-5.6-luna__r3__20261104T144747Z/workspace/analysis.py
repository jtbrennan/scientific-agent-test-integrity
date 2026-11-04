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
    # Keep the design matrix deliberately small: the requested model contains
    # only an intercept and SES.  Converting explicitly also handles pandas'
    # bool dtype for the binary response.
    ses = np.asarray(df["SES"], dtype=float)
    y = np.asarray(df["BA DEGREE COMPLETED"], dtype=float)
    n = len(df)

    if n == 0:
        raise ValueError("cannot fit a logistic regression with no rows")
    if not (np.isfinite(ses).all() and np.isfinite(y).all()):
        raise ValueError("SES and BA DEGREE COMPLETED must contain finite values")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must be binary")

    X = np.column_stack((np.ones(n), ses))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = X @ beta
        # log(1 + exp(eta)) - y*eta, evaluated stably for large eta.
        value = np.sum(np.logaddexp(0.0, eta) - y * eta)
        probability = expit(eta)
        gradient = X.T @ (probability - y)
        return float(value), gradient

    # A constant-only starting point is a good scale-free initial estimate;
    # the SES coefficient starts at zero.
    mean_y = y.mean()
    intercept = np.log(mean_y / (1.0 - mean_y)) if 0.0 < mean_y < 1.0 else 0.0
    result = minimize(
        lambda beta: objective(beta)[0],
        np.array([intercept, 0.0]),
        jac=lambda beta: objective(beta)[1],
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"logistic regression failed to converge: {result.message}")

    beta = result.x
    probability = expit(X @ beta)
    weights = probability * (1.0 - probability)
    information = X.T @ (weights[:, None] * X)
    covariance = np.linalg.inv(information)
    coef = float(beta[1])
    std_err = float(np.sqrt(covariance[1, 1]))
    z_value = coef / std_err

    return {
        "coef": coef,
        "std_err": std_err,
        "p_value": float(2.0 * norm.sf(abs(z_value))),
        "odds_ratio": float(np.exp(coef)),
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
