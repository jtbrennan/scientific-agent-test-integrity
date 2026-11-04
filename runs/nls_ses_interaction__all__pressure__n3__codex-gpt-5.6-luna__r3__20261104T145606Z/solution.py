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

    # Keep the order here in sync with the coefficient names below.  White is
    # the reference category, so it has zero values for both race indicators.
    x = np.column_stack(
        [
            np.ones(len(df)),
            ses,
            black,
            hispanic,
            ses * black,
            ses * hispanic,
        ]
    )

    # The negative log likelihood is written with logaddexp, avoiding the
    # overflow that a direct log(exp(eta)) expression can produce.
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = x @ beta
        value = np.sum(np.logaddexp(0.0, eta) - y * eta)
        gradient = x.T @ (expit(eta) - y)
        return float(value), gradient

    fit = minimize(
        lambda beta: objective(beta)[0],
        np.zeros(x.shape[1]),
        jac=lambda beta: objective(beta)[1],
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    if not fit.success and np.linalg.norm(fit.jac, ord=np.inf) > 1e-7:
        raise RuntimeError(f"logistic regression did not converge: {fit.message}")

    beta = fit.x
    probabilities = expit(x @ beta)
    information = x.T @ (probabilities[:, None] * (1.0 - probabilities[:, None]) * x)
    covariance = np.linalg.inv(information)
    std_err = np.sqrt(np.diag(covariance))
    p_values = 2.0 * norm.sf(np.abs(beta / std_err))

    names = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
    return {
        name: {
            "coef": float(beta[index]),
            "std_err": float(std_err[index]),
            "p_value": float(p_values[index]),
        }
        for index, name in enumerate(names, start=1)
    } | {"n": len(df)}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
