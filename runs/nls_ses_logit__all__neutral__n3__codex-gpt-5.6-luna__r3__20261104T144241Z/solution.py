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
    # Keep the design matrix in the original SES units so that the reported
    # coefficient is the change in log-odds for a one-unit SES increase.
    x = pd.to_numeric(df["SES"], errors="raise").to_numpy(dtype=float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("SES and BA DEGREE COMPLETED must contain finite values")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("BA DEGREE COMPLETED must be binary")

    n = len(df)
    design = np.column_stack((np.ones(n), x))

    # Newton-Raphson for the concave logistic log-likelihood.  The line
    # search also keeps the iteration well behaved for less balanced inputs.
    successes = y.sum()
    if successes == 0 or successes == n:
        raise ValueError("The outcome must contain both classes")
    beta = np.array([np.log(successes / (n - successes)), 0.0])

    def log_likelihood(params: np.ndarray) -> float:
        eta = design @ params
        # log(1 + exp(eta)) in a form that does not overflow.
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    current_ll = log_likelihood(beta)
    for _ in range(100):
        probability = expit(design @ beta)
        gradient = design.T @ (y - probability)
        weights = probability * (1.0 - probability)
        information = (design.T * weights) @ design
        step = np.linalg.solve(information, gradient)

        # Backtracking is rarely needed, but avoids taking a bad full Newton
        # step when probabilities are close to 0 or 1.
        step_scale = 1.0
        while step_scale > 1e-12:
            candidate = beta + step_scale * step
            candidate_ll = log_likelihood(candidate)
            if candidate_ll >= current_ll:
                break
            step_scale *= 0.5
        else:
            break

        beta = candidate
        current_ll = candidate_ll
        if np.max(np.abs(step_scale * step)) <= 1e-10 * (1.0 + np.max(np.abs(beta))):
            break

    # At the MLE, inverse(X'WX) is the model-based covariance matrix.
    probability = expit(design @ beta)
    weights = probability * (1.0 - probability)
    information = (design.T * weights) @ design
    covariance = np.linalg.inv(information)
    coefficient = float(beta[1])
    standard_error = float(np.sqrt(covariance[1, 1]))
    z_score = coefficient / standard_error

    return {
        "coef": coefficient,
        "std_err": standard_error,
        "p_value": float(2.0 * norm.sf(abs(z_score))),
        "odds_ratio": float(np.exp(coefficient)),
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
