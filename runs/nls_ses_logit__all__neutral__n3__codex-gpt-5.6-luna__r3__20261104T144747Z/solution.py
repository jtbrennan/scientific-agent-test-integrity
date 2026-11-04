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
    # Keep the conversion explicit: pandas booleans convert directly to 0/1,
    # which is the coding required by the Bernoulli log-likelihood.
    x = pd.to_numeric(df["SES"], errors="raise").to_numpy(dtype=float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("SES and BA DEGREE COMPLETED must contain finite values")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("BA DEGREE COMPLETED must be binary")
    if len(y) == 0:
        raise ValueError("cannot fit a model with no rows")

    # Center and scale the predictor while optimizing.  This leaves the
    # fitted model unchanged but makes the intercept and slope parameters
    # similarly sized, which improves optimizer accuracy.
    x_mean = float(np.mean(x))
    x_scale = float(np.std(x))
    if x_scale == 0.0:
        raise ValueError("SES must vary to estimate a coefficient")
    design = np.column_stack((np.ones(len(x)), (x - x_mean) / x_scale))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ beta
        # log(1 + exp(eta)) in a stable form, and the negative log likelihood.
        log_likelihood = np.sum(y * eta - np.logaddexp(0.0, eta))
        gradient = design.T @ (y - expit(eta))
        return -log_likelihood, -gradient

    def hessian(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(design @ beta)
        return design.T @ ((probabilities * (1.0 - probabilities))[:, None] * design)

    beta = np.zeros(2)
    for _ in range(100):
        value, gradient = objective(beta)
        if np.linalg.norm(gradient, ord=np.inf) < 1e-10:
            break
        # Newton direction for minimizing the negative log likelihood.
        direction = np.linalg.solve(hessian(beta), -gradient)
        step = 1.0
        while step > 1e-12 and objective(beta + step * direction)[0] >= value:
            step *= 0.5
        if step <= 1e-12:
            raise RuntimeError("logistic regression failed to converge")
        beta += step * direction
    else:
        raise RuntimeError("logistic regression failed to converge")

    probabilities = expit(design @ beta)
    information = design.T @ ((probabilities * (1.0 - probabilities))[:, None] * design)
    covariance = np.linalg.inv(information)
    # Convert the slope and its standard error back to the original SES scale.
    coefficient = float(beta[1] / x_scale)
    standard_error = float(np.sqrt(covariance[1, 1]) / x_scale)
    z_score = coefficient / standard_error

    return {
        "coef": coefficient,
        "std_err": standard_error,
        "p_value": float(2.0 * norm.sf(abs(z_score))),
        "odds_ratio": float(np.exp(coefficient)),
        "n": int(len(y)),
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
