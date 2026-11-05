import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import root
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
    outcome = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)

    # Keep the intercept in the design matrix even though it is not included
    # in the requested output.  The columns are b0 through b5 in model order.
    design = np.column_stack(
        [np.ones(len(df)), ses, black, hispanic, ses * black, ses * hispanic]
    )

    def gradient(beta: np.ndarray) -> np.ndarray:
        probability = expit(design @ beta)
        return design.T @ (probability - outcome)

    def hessian(beta: np.ndarray) -> np.ndarray:
        probability = expit(design @ beta)
        weights = probability * (1.0 - probability)
        return design.T @ (weights[:, None] * design)

    result = root(
        gradient,
        np.zeros(design.shape[1]),
        jac=hessian,
        method="hybr",
        options={"xtol": 1e-11, "maxfev": 1000},
    )
    if not result.success or np.max(np.abs(gradient(result.x))) > 1e-7:
        raise RuntimeError(f"Logistic regression failed to converge: {result.message}")

    beta = result.x
    observed_information = hessian(beta)
    covariance = np.linalg.inv(observed_information)
    standard_errors = np.sqrt(np.diag(covariance))
    z_scores = beta / standard_errors

    names = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
    # beta[0] is the intercept; the requested result contains the remaining
    # five coefficients in their model order.
    estimates = {}
    for name, index in zip(names, range(1, len(beta))):
        estimates[name] = {
            "coef": float(beta[index]),
            "std_err": float(standard_errors[index]),
            "p_value": float(2.0 * norm.sf(abs(z_scores[index]))),
        }
    estimates["n"] = int(len(df))
    return estimates


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
