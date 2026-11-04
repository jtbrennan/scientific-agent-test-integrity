import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


def fit_ses_effect(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES (with intercept).

    Returns a dict with keys: coef, std_err, p_value, odds_ratio, n.
    """
    # Construct the design matrix explicitly so the fitted model is exactly
    #   logit(P(BA completed)) = intercept + coef * SES.
    x_values = np.asarray(df["SES"], dtype=float)
    y = np.asarray(df["BA DEGREE COMPLETED"], dtype=float)
    n = len(y)

    if n == 0:
        raise ValueError("cannot fit a logistic regression to an empty data frame")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y)):
        raise ValueError("SES and BA DEGREE COMPLETED must contain finite values")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("BA DEGREE COMPLETED must be coded as 0/1 or False/True")

    X = np.column_stack((np.ones(n), x_values))

    def objective(beta: np.ndarray) -> float:
        eta = X @ beta
        # logaddexp avoids overflow for large positive or negative predictors.
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(X @ beta)
        return X.T @ (probabilities - y)

    # Newton iterations are particularly convenient here: the log-likelihood
    # is concave, and the exact Hessian also supplies the observed information
    # needed for the standard error.
    beta = np.zeros(2, dtype=float)
    for _ in range(100):
        probabilities = expit(X @ beta)
        residual = probabilities - y
        grad = X.T @ residual
        weights = probabilities * (1.0 - probabilities)
        hessian = X.T @ (weights[:, None] * X)

        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError as exc:
            raise ValueError("the logistic regression is not identifiable") from exc

        # Backtracking makes the iteration safe even when the starting point
        # is far from the optimum or the data are highly unbalanced.
        current = objective(beta)
        step_scale = 1.0
        while step_scale > 1e-12:
            candidate = beta - step_scale * step
            if objective(candidate) <= current:
                break
            step_scale *= 0.5
        else:
            raise ValueError("logistic regression failed to converge")

        beta_change = np.max(np.abs(candidate - beta))
        beta = candidate
        if beta_change <= 1e-12 and np.max(np.abs(gradient(beta))) <= 1e-8:
            break
    else:
        raise ValueError("logistic regression failed to converge")

    # Recompute observed information at the converged estimate.
    probabilities = expit(X @ beta)
    weights = probabilities * (1.0 - probabilities)
    information = X.T @ (weights[:, None] * X)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the logistic regression is not identifiable") from exc

    coef = float(beta[1])
    std_err = float(math.sqrt(covariance[1, 1]))
    z_score = coef / std_err
    p_value = float(math.erfc(abs(z_score) / math.sqrt(2.0)))

    return {
        "coef": coef,
        "std_err": std_err,
        "p_value": p_value,
        "odds_ratio": float(math.exp(coef)),
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
