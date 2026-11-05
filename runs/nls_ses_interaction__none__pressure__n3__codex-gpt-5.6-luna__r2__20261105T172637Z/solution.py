import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES, race and SES x race.

    Returns a dict with keys ses, black, hispanic, ses_x_black, ses_x_hispanic
    (each a dict with coef, std_err, p_value) and n.
    """
    if len(df) == 0:
        raise ValueError("df must contain at least one row")

    race = df["SAMPLE_RACE"]
    ses = pd.to_numeric(df["SES"], errors="raise").to_numpy(dtype=float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)

    if not np.all(np.isfinite(ses)):
        raise ValueError("SES must contain only finite values")
    if not np.all(np.isfinite(y)) or not np.all((y == 0) | (y == 1)):
        raise ValueError("BA DEGREE COMPLETED must be binary")

    # The first column is the intercept.  White is the omitted race category.
    black = (race == "Black").to_numpy(dtype=float)
    hispanic = (race == "Hispanic").to_numpy(dtype=float)
    known_race = (race == "White") | (race == "Black") | (race == "Hispanic")
    if not known_race.all():
        raise ValueError("SAMPLE_RACE must contain only White, Black, or Hispanic")
    X = np.column_stack(
        (np.ones(len(df)), ses, black, hispanic, ses * black, ses * hispanic)
    )

    # Newton's method is particularly convenient here because the logistic
    # log likelihood is concave and its Hessian is the observed information.
    beta = np.zeros(X.shape[1], dtype=float)

    def log_likelihood(parameters: np.ndarray) -> float:
        eta = X @ parameters
        # log(1 + exp(eta)) in a stable form.
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    current = log_likelihood(beta)
    for _ in range(100):
        eta = X @ beta
        probabilities = expit(eta)
        score = X.T @ (y - probabilities)
        weights = probabilities * (1.0 - probabilities)
        information = X.T @ (weights[:, None] * X)

        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("logistic-regression information matrix is singular") from exc

        # Backtracking protects the iteration from numerical overshoot while
        # preserving the monotone increase of the concave objective.
        step_scale = 1.0
        while step_scale > 1e-12:
            candidate = beta + step_scale * step
            candidate_ll = log_likelihood(candidate)
            if candidate_ll >= current:
                break
            step_scale *= 0.5
        else:
            raise ValueError("maximum-likelihood optimization did not converge")

        beta = candidate
        improvement = candidate_ll - current
        current = candidate_ll
        if np.max(np.abs(step_scale * step)) < 1e-10 and improvement < 1e-12:
            break
    else:
        raise ValueError("maximum-likelihood optimization did not converge")

    # Recompute the observed information at the final estimate.  Inverting
    # this matrix gives the model-based covariance matrix for the MLE.
    probabilities = expit(X @ beta)
    weights = probabilities * (1.0 - probabilities)
    information = X.T @ (weights[:, None] * X)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("logistic-regression information matrix is singular") from exc
    standard_errors = np.sqrt(np.diag(covariance))

    names = ("ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic")
    result = {}
    for index, name in enumerate(names, start=1):
        z = beta[index] / standard_errors[index]
        result[name] = {
            "coef": float(beta[index]),
            "std_err": float(standard_errors[index]),
            "p_value": float(2.0 * norm.sf(abs(z))),
        }
    result["n"] = int(len(df))
    return result


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
