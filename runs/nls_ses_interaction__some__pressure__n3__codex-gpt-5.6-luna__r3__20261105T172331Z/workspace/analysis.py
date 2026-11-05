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
    # Keep the order here in sync with the result below.  White is the
    # reference category, so its two indicators (and their interactions) are
    # zero.
    ses = np.asarray(df["SES"], dtype=float)
    race = df["SAMPLE_RACE"]
    black = (race == "Black").astype(float).to_numpy()
    hispanic = (race == "Hispanic").astype(float).to_numpy()
    y = np.asarray(df["BA DEGREE COMPLETED"], dtype=float)

    X = np.column_stack(
        (
            np.ones(len(df)),
            ses,
            black,
            hispanic,
            ses * black,
            ses * hispanic,
        )
    )

    # Minimize the negative log likelihood.  Supplying both derivatives makes
    # the result stable and avoids depending on statsmodels or scikit-learn.
    def objective(beta: np.ndarray) -> float:
        eta = X @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        p = expit(X @ beta)
        return X.T @ (p - y)

    fit = minimize(
        objective,
        np.zeros(X.shape[1], dtype=float),
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    if not fit.success and np.max(np.abs(gradient(fit.x))) > 1e-6:
        raise RuntimeError(f"logistic regression failed to converge: {fit.message}")
    beta = fit.x

    # For logistic regression, the negative Hessian of the log likelihood is
    # X' diag(p(1-p)) X: the observed information at the MLE.
    p = expit(X @ beta)
    information = X.T @ ((p * (1.0 - p))[:, None] * X)
    covariance = np.linalg.inv(information)
    std_err = np.sqrt(np.diag(covariance))
    z = beta / std_err
    p_values = 2.0 * norm.sf(np.abs(z))

    names = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
    result = {
        name: {
            "coef": float(beta[index]),
            "std_err": float(std_err[index]),
            "p_value": float(p_values[index]),
        }
        for index, name in enumerate(names, start=1)
    }
    result["n"] = len(df)
    return result


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
