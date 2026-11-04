import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, ndtr


def fit_ses_effect(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES (with intercept).

    Returns a dict with keys: coef, std_err, p_value, odds_ratio, n.
    """
    # Keep the design matrix deliberately small: the requested model contains
    # only an intercept and SES.  Converting explicitly to float also handles
    # pandas' boolean outcome without relying on its implementation details.
    x = np.asarray(df["SES"], dtype=float)
    y = np.asarray(df["BA DEGREE COMPLETED"], dtype=float)
    n = len(y)

    if n == 0:
        raise ValueError("cannot fit logistic regression to an empty data frame")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("SES and BA DEGREE COMPLETED must contain finite values")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("BA DEGREE COMPLETED must be binary")
    if np.all(y == y[0]):
        raise ValueError("logistic regression has no finite MLE for a constant outcome")

    design = np.column_stack((np.ones(n), x))

    # Newton's method for the log likelihood.  The step-halving safeguard
    # keeps the iteration stable even for data sets with very unbalanced
    # outcomes or unusually scaled predictors.
    proportion = y.mean()
    beta = np.array([np.log(proportion / (1.0 - proportion)), 0.0])

    def log_likelihood(parameters: np.ndarray) -> float:
        eta = design @ parameters
        # log(sigmoid(eta)) and log(1 - sigmoid(eta)), evaluated stably.
        return float(np.sum(y * (-np.logaddexp(0.0, -eta))
                           + (1.0 - y) * (-np.logaddexp(0.0, eta))))

    current_ll = log_likelihood(beta)
    for _ in range(100):
        probabilities = expit(design @ beta)
        gradient = design.T @ (y - probabilities)
        weights = probabilities * (1.0 - probabilities)
        information = (design.T * weights) @ design
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError("logistic regression information matrix is singular") from exc

        if np.max(np.abs(step)) < 1e-12:
            break

        step_size = 1.0
        while step_size > 1e-12:
            candidate = beta + step_size * step
            candidate_ll = log_likelihood(candidate)
            if candidate_ll >= current_ll:
                beta = candidate
                current_ll = candidate_ll
                break
            step_size *= 0.5
        else:
            raise ValueError("logistic regression failed to converge")
    else:
        raise ValueError("logistic regression failed to converge")

    # Recompute the observed information at the converged estimate.
    probabilities = expit(design @ beta)
    weights = probabilities * (1.0 - probabilities)
    information = (design.T * weights) @ design
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("logistic regression information matrix is singular") from exc

    coef = float(beta[1])
    std_err = float(np.sqrt(covariance[1, 1]))
    z_score = coef / std_err
    p_value = float(2.0 * ndtr(-abs(z_score)))

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
