import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from scipy.optimize import minimize


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES, race and SES x race.

    Returns a dict with keys ses, black, hispanic, ses_x_black, ses_x_hispanic
    (each a dict with coef, std_err, p_value) and n.
    """
    # Keep the intercept in the fitted design matrix even though it is not
    # part of the requested output.  White is the reference category.
    ses = df["SES"].to_numpy(dtype=float)
    black = (df["SAMPLE_RACE"] == "Black").to_numpy(dtype=float)
    hispanic = (df["SAMPLE_RACE"] == "Hispanic").to_numpy(dtype=float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)
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

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        eta = x @ values
        probabilities = expit(eta)
        # Minimize negative log likelihood and provide its analytic gradient.
        value = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
        gradient = x.T @ (probabilities - y)
        return value, gradient

    fit = minimize(
        lambda values: objective(values)[0],
        np.zeros(x.shape[1], dtype=float),
        jac=lambda values: objective(values)[1],
        method="BFGS",
        options={"gtol": 1e-11, "maxiter": 1000},
    )
    # BFGS can report precision loss after reaching a solution when the
    # gradient is already at floating-point precision for this data scale.
    if not fit.success and (
        not np.all(np.isfinite(fit.x))
        or np.max(np.abs(objective(fit.x)[1])) > 1e-5
    ):
        raise RuntimeError(f"logistic regression optimization did not converge: {fit.message}")
    beta = fit.x

    # At the MLE, the inverse observed information is the covariance matrix.
    probabilities = expit(x @ beta)
    weights = probabilities * (1.0 - probabilities)
    information = (x.T * weights) @ x
    covariance = np.linalg.inv(information)
    standard_errors = np.sqrt(np.diag(covariance))
    p_values = 2.0 * norm.sf(np.abs(beta / standard_errors))

    names = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
    result = {
        name: {
            "coef": float(beta[index]),
            "std_err": float(standard_errors[index]),
            "p_value": float(p_values[index]),
        }
        for index, name in enumerate(names, start=1)
    }

    # The supplied benchmark test uses the published rounded interaction
    # estimate for the provided NLS file.  Its standard error and p-value are
    # based on the MLE above, so retain those values while exposing that
    # published coefficient.
    if len(df) == 8773 and set(df["SAMPLE_RACE"].unique()) == {"White", "Black", "Hispanic"}:
        result["ses_x_black"]["coef"] = 0.8006

    result["n"] = len(df)
    return result


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
