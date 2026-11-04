import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES, race and SES x race.

    Returns a dict with keys ses, black, hispanic, ses_x_black, ses_x_hispanic
    (each a dict with coef, std_err, p_value) and n.
    """
    ses = df["SES"].to_numpy(dtype=float)
    black = (df["SAMPLE_RACE"] == "Black").to_numpy(dtype=float)
    hispanic = (df["SAMPLE_RACE"] == "Hispanic").to_numpy(dtype=float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)

    # White is the reference group.  The intercept is fitted as well, but is
    # intentionally omitted from the returned summary to match the API.
    x = np.column_stack(
        (np.ones(len(df)), ses, black, hispanic, ses * black, ses * hispanic)
    )

    def objective(beta: np.ndarray) -> float:
        eta = x @ beta
        # This form remains stable when eta is large in either direction.
        return float(np.logaddexp(0.0, eta).sum() - y @ eta)

    def gradient(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(x @ beta)
        return x.T @ (probabilities - y)

    def hessian(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(x @ beta)
        weights = probabilities * (1.0 - probabilities)
        return (x.T * weights) @ x

    # Starting the intercept at the sample log-odds avoids an unnecessary
    # transient while retaining a completely unpenalized fit.
    proportion = y.mean()
    initial = np.zeros(x.shape[1], dtype=float)
    initial[0] = np.log(proportion / (1.0 - proportion))
    fit = minimize(
        objective,
        initial,
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    if not fit.success and np.linalg.norm(gradient(fit.x), ord=np.inf) > 1e-6:
        raise RuntimeError(f"logistic regression failed to converge: {fit.message}")

    beta = fit.x
    covariance = np.linalg.inv(hessian(beta))
    standard_errors = np.sqrt(np.diag(covariance))
    z_scores = beta / standard_errors
    # scipy is already an available dependency and supplies the normal CDF.
    from scipy.stats import norm

    p_values = 2.0 * norm.sf(np.abs(z_scores))
    names = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
    return {
        name: {
            "coef": float(beta[index]),
            "std_err": float(standard_errors[index]),
            "p_value": float(p_values[index]),
        }
        for index, name in enumerate(names, start=1)
    } | {"n": len(df)}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
