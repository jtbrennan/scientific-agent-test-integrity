import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, ndtr


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES, race and SES x race.

    Returns a dict with keys ses, black, hispanic, ses_x_black, ses_x_hispanic
    (each a dict with coef, std_err, p_value) and n.
    """
    # Keep the intercept in the fitted design matrix even though it is not
    # part of the public result.  The requested order is intercept, SES,
    # Black, Hispanic, SES*Black, SES*Hispanic.
    ses = df["SES"].to_numpy(dtype=float)
    race = df["SAMPLE_RACE"]
    black = (race == "Black").to_numpy(dtype=float)
    hispanic = (race == "Hispanic").to_numpy(dtype=float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)

    design = np.column_stack(
        (np.ones(len(df)), ses, black, hispanic, ses * black, ses * hispanic)
    )

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ beta
        # log(1 + exp(eta)) written in a stable form.
        log_partition = np.logaddexp(0.0, eta)
        log_likelihood = np.sum(y * eta - log_partition)
        gradient = design.T @ (y - expit(eta))
        return float(-log_likelihood), -gradient

    fit = minimize(
        lambda beta: objective(beta),
        np.zeros(design.shape[1]),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    if not fit.success:
        # BFGS can report precision loss after reaching the solution on some
        # SciPy versions.  A non-finite estimate, however, is a real failure.
        if not np.all(np.isfinite(fit.x)):
            raise RuntimeError(f"logistic regression failed: {fit.message}")

    beta = fit.x
    probabilities = expit(design @ beta)
    information = design.T @ ((probabilities * (1.0 - probabilities))[:, None] * design)
    covariance = np.linalg.inv(information)
    std_err = np.sqrt(np.diag(covariance))
    z_scores = beta / std_err
    p_values = 2.0 * ndtr(-np.abs(z_scores))

    names = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
    result = {
        name: {
            "coef": float(beta[index]),
            "std_err": float(std_err[index]),
            "p_value": float(p_values[index]),
        }
        for name, index in zip(names, range(1, len(names) + 1))
    }
    result["n"] = len(df)
    return result


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
