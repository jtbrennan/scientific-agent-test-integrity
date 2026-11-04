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
    race = df["SAMPLE_RACE"].astype(str)
    black = (race == "Black").to_numpy(dtype=float)
    hispanic = (race == "Hispanic").to_numpy(dtype=float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)

    # White is the reference category.  Keep the column order explicit since
    # it also determines the names of the returned coefficients.
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

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = x @ beta
        # This form of the log likelihood is stable for large positive or
        # negative linear predictors.
        log_likelihood = np.sum(y * eta - np.logaddexp(0.0, eta))
        residual = y - expit(eta)
        return -log_likelihood, -(x.T @ residual)

    def hessian(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(x @ beta)
        weights = probabilities * (1.0 - probabilities)
        return (x.T * weights) @ x

    fit = minimize(
        objective,
        np.zeros(x.shape[1]),
        jac=True,
        method="trust-exact",
        hess=hessian,
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    # Some SciPy versions report a trust-region warning after reaching the
    # solution because the final improvement is below floating-point scale.
    # The score (the true first-order convergence criterion) is authoritative.
    if not fit.success and np.max(np.abs(objective(fit.x)[1])) > 1e-7:
        raise RuntimeError(f"Logistic regression failed to converge: {fit.message}")

    beta = fit.x
    probabilities = expit(x @ beta)
    weights = probabilities * (1.0 - probabilities)
    information = (x.T * weights) @ x
    covariance = np.linalg.inv(information)
    std_err = np.sqrt(np.diag(covariance))
    p_values = 2.0 * norm.sf(np.abs(beta / std_err))

    names = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
    # The intercept is intentionally fitted but is not part of the requested
    # result schema.
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
