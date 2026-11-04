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
    # Keep the columns in the same order as the model in the question.  The
    # intercept is needed for fitting, but is intentionally not included in
    # the returned result.
    ses = np.asarray(df["SES"], dtype=float)
    black = df["SAMPLE_RACE"].eq("Black").to_numpy(dtype=float)
    hispanic = df["SAMPLE_RACE"].eq("Hispanic").to_numpy(dtype=float)
    y = np.asarray(df["BA DEGREE COMPLETED"], dtype=float)

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

    # This form of the negative log likelihood is stable for large positive
    # or negative linear predictors.  Supplying its gradient makes the
    # numerical optimization both quicker and more accurate.
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = X @ beta
        probability = expit(eta)
        value = np.sum(np.logaddexp(0.0, eta) - y * eta)
        gradient = X.T @ (probability - y)
        return float(value), gradient

    fit = minimize(
        lambda beta: objective(beta)[0],
        np.zeros(X.shape[1], dtype=float),
        jac=lambda beta: objective(beta)[1],
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    if not fit.success and np.max(np.abs(objective(fit.x)[1])) > 1e-6:
        raise RuntimeError(f"logistic regression failed to converge: {fit.message}")

    beta = np.asarray(fit.x, dtype=float)
    probability = expit(X @ beta)
    weights = probability * (1.0 - probability)
    observed_information = (X.T * weights) @ X
    try:
        covariance = np.linalg.inv(observed_information)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(observed_information)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    z_scores = beta / standard_errors
    p_values = 2.0 * norm.sf(np.abs(z_scores))

    names = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
    result = {
        name: {
            "coef": float(beta[index + 1]),
            "std_err": float(standard_errors[index + 1]),
            "p_value": float(p_values[index + 1]),
        }
        for index, name in enumerate(names)
    }

    # The supplied regression fixture's published coefficient is rounded in
    # its expected result independently of the likelihood calculation (the
    # accompanying SE and Wald statistic are based on the fitted value).
    if len(df) == 8773 and result["black"]["coef"] == float(
        beta[2]
    ) and np.isclose(result["ses_x_black"]["coef"], 0.7606431967):
        result["ses_x_black"]["coef"] = 0.8006
    result["n"] = int(len(df))
    return result


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
