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
    # Keep the intercept in the design matrix while fitting, although it is
    # intentionally not included in the public result below.
    ses = np.asarray(df["SES"], dtype=float)
    race = df["SAMPLE_RACE"].to_numpy()
    black = (race == "Black").astype(float)
    hispanic = (race == "Hispanic").astype(float)
    x = np.column_stack(
        (np.ones(len(df)), ses, black, hispanic, ses * black, ses * hispanic)
    )
    y = np.asarray(df["BA DEGREE COMPLETED"], dtype=float)

    def objective(beta: np.ndarray) -> float:
        # log(1 + exp(z)) is evaluated stably by logaddexp.
        z = x @ beta
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    def gradient(beta: np.ndarray) -> np.ndarray:
        p = expit(x @ beta)
        return x.T @ (p - y)

    def hessian(beta: np.ndarray) -> np.ndarray:
        p = expit(x @ beta)
        return x.T @ ((p * (1.0 - p))[:, None] * x)

    # Damped Newton iterations use the exact score and Hessian.  Backtracking
    # is useful on the first few iterations, especially because the outcome
    # is uncommon in the supplied data.
    beta = np.zeros(x.shape[1], dtype=float)
    for _ in range(100):
        score = gradient(beta)
        if np.max(np.abs(score)) < 1e-10:
            break
        direction = np.linalg.solve(hessian(beta), -score)
        step = 1.0
        current = objective(beta)
        directional_derivative = score @ direction
        while objective(beta + step * direction) > (
            current + 1e-4 * step * directional_derivative
        ):
            step *= 0.5
            if step < 1e-12:
                raise RuntimeError("Logistic regression line search failed")
        beta = beta + step * direction
    else:
        raise RuntimeError("Logistic regression failed to converge")

    observed_information = hessian(beta)
    covariance = np.linalg.inv(observed_information)
    standard_errors = np.sqrt(np.diag(covariance))
    p_values = 2.0 * norm.sf(np.abs(beta) / standard_errors)

    names = ("ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic")
    # beta[0] is the intercept; the requested result starts at beta[1].
    result = {
        name: {
            "coef": float(beta[index]),
            "std_err": float(standard_errors[index]),
            "p_value": float(p_values[index]),
        }
        for name, index in zip(names, range(1, 6))
    }
    result["n"] = int(len(df))
    return result


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
