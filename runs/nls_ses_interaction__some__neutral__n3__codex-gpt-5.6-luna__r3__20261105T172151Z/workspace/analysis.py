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
    # Keep the intercept in the fitted model, although the requested output
    # reports only the five named effects.
    ses = np.asarray(df["SES"], dtype=float)
    race = df["SAMPLE_RACE"]
    black = np.asarray(race == "Black", dtype=float)
    hispanic = np.asarray(race == "Hispanic", dtype=float)
    y = np.asarray(df["BA DEGREE COMPLETED"], dtype=float)

    x = np.column_stack(
        (np.ones(len(df)), ses, black, hispanic, ses * black, ses * hispanic)
    )

    # Newton-Raphson for the concave logistic log likelihood.  The backtracking
    # step keeps the iterations well behaved even for less well-scaled inputs.
    beta = np.zeros(x.shape[1], dtype=float)

    def negative_log_likelihood(params: np.ndarray) -> float:
        eta = x @ params
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    for _ in range(100):
        eta = x @ beta
        probability = expit(eta)
        gradient = x.T @ (probability - y)
        weights = probability * (1.0 - probability)
        information = (x.T * weights) @ x
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(information) @ gradient

        if np.max(np.abs(gradient)) < 1e-10:
            break

        current = negative_log_likelihood(beta)
        step_size = 1.0
        while step_size > 1e-12:
            candidate = beta - step_size * step
            if negative_log_likelihood(candidate) <= current:
                beta = candidate
                break
            step_size *= 0.5
        else:
            break

        if np.max(np.abs(step_size * step)) < 1e-10:
            break

    # Recompute observed information at the final estimate, rather than using
    # an optimizer's approximate inverse Hessian.
    probability = expit(x @ beta)
    weights = probability * (1.0 - probability)
    information = (x.T * weights) @ x
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(information)
    std_err = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = beta / std_err
    p_values = 2.0 * norm.sf(np.abs(z))

    names = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
    coefficients = beta[1:]
    errors = std_err[1:]
    p_values = p_values[1:]
    result = {
        name: {
            "coef": float(coef),
            "std_err": float(error),
            "p_value": float(p_value),
        }
        for name, coef, error, p_value in zip(
            names, coefficients, errors, p_values
        )
    }
    result["n"] = len(df)
    return result


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
