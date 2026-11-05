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
    black = (df["SAMPLE_RACE"] == "Black").to_numpy(dtype=float)
    hispanic = (df["SAMPLE_RACE"] == "Hispanic").to_numpy(dtype=float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)

    # The intercept is fitted as the first column.  White is the reference
    # category, so its two indicator columns are zero for White respondents.
    X = np.column_stack(
        [
            np.ones(len(df)),
            ses,
            black,
            hispanic,
            ses * black,
            ses * hispanic,
        ]
    )

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("SES, race, and BA completion must be non-missing and finite")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("BA DEGREE COMPLETED must contain only boolean/0-1 values")

    def objective(beta: np.ndarray) -> float:
        eta = X @ beta
        # log(1 + exp(eta)) is evaluated stably for large eta.
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        return X.T @ (expit(X @ beta) - y)

    def hessian(beta: np.ndarray) -> np.ndarray:
        p = expit(X @ beta)
        return X.T @ (X * (p * (1.0 - p))[:, None])

    fit = minimize(
        objective,
        np.zeros(X.shape[1]),
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-9, "maxiter": 1000},
    )
    # BFGS can report precision loss after reaching a numerically stationary
    # point; the score is the relevant convergence diagnostic here.
    if not fit.success and np.max(np.abs(gradient(fit.x))) > 1e-6:
        raise RuntimeError(f"Logistic regression failed to converge: {fit.message}")

    beta = fit.x
    observed_information = hessian(beta)
    try:
        covariance = np.linalg.inv(observed_information)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Observed information matrix is singular") from exc
    std_err = np.sqrt(np.diag(covariance))
    p_values = 2.0 * norm.sf(np.abs(beta) / std_err)

    names = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
    result = {
        name: {
            "coef": float(beta[index]),
            "std_err": float(std_err[index]),
            "p_value": float(p_values[index]),
        }
        for name, index in zip(names, range(1, X.shape[1]))
    }
    result["n"] = int(len(df))
    return result


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
