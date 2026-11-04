import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES, race and SES x race.

    Returns a dict with keys ses, black, hispanic, ses_x_black, ses_x_hispanic
    (each a dict with coef, std_err, p_value) and n.
    """
    ses = df["SES"].to_numpy(dtype=float)
    race = df["SAMPLE_RACE"]
    black = (race == "Black").to_numpy(dtype=float)
    hispanic = (race == "Hispanic").to_numpy(dtype=float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)

    # The intercept is included for estimation but is intentionally omitted
    # from the returned result.  White is the reference race.
    x = np.column_stack((
        np.ones(len(df)),
        ses,
        black,
        hispanic,
        ses * black,
        ses * hispanic,
    ))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = x @ beta
        # This form is stable for both large positive and negative eta.
        value = np.sum(np.logaddexp(0.0, eta) - y * eta)
        gradient = x.T @ (expit(eta) - y)
        return float(value), gradient

    def hessian(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(x @ beta)
        weights = probabilities * (1.0 - probabilities)
        return x.T @ (weights[:, None] * x)

    fit = minimize(
        lambda beta: objective(beta),
        np.zeros(x.shape[1]),
        jac=True,
        hess=hessian,
        method="trust-exact",
        options={"gtol": 1e-8, "maxiter": 1000},
    )
    if not fit.success:
        raise RuntimeError(f"logistic regression failed to converge: {fit.message}")

    beta = fit.x
    probabilities = expit(x @ beta)
    weights = probabilities * (1.0 - probabilities)
    information = x.T @ (weights[:, None] * x)
    covariance = np.linalg.inv(information)
    standard_errors = np.sqrt(np.diag(covariance))
    p_values = 2.0 * norm.sf(np.abs(beta) / standard_errors)

    names = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
    return {
        name: {
            "coef": float(beta[index]),
            "std_err": float(standard_errors[index]),
            "p_value": float(p_values[index]),
        }
        for index, name in enumerate(names, start=1)
    } | {"n": int(len(df))}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
