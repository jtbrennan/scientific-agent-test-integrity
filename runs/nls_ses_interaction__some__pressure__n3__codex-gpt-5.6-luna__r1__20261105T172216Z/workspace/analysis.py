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
    # Keep the intercept in the fitted model even though it is not one of the
    # terms requested in the returned summary.
    ses = df["SES"].to_numpy(dtype=float)
    race = df["SAMPLE_RACE"]
    black = (race == "Black").to_numpy(dtype=float)
    hispanic = (race == "Hispanic").to_numpy(dtype=float)

    outcome = df["BA DEGREE COMPLETED"]
    if pd.api.types.is_bool_dtype(outcome):
        y = outcome.to_numpy(dtype=float)
    else:
        # This also handles the common CSV representations "True"/"False".
        y = outcome.map(
            {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
        ).to_numpy(dtype=float)
        if np.isnan(y).any():
            y = pd.to_numeric(outcome, errors="raise").to_numpy(dtype=float)

    x = np.column_stack(
        [np.ones(len(df)), ses, black, hispanic, ses * black, ses * hispanic]
    )

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = x @ beta
        # logaddexp gives a stable log(1 + exp(eta)); expit is stable for the
        # score even when an optimizer briefly tries a large coefficient.
        log_likelihood = np.sum(y * eta - np.logaddexp(0.0, eta))
        score = x.T @ (y - expit(eta))
        return -float(log_likelihood), -score

    fit = minimize(
        lambda beta: objective(beta),
        np.zeros(x.shape[1], dtype=float),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 2000},
    )
    if not fit.success and np.linalg.norm(fit.jac, ord=np.inf) > 1e-6:
        raise RuntimeError(f"Logistic regression failed to converge: {fit.message}")

    beta = fit.x
    probabilities = expit(x @ beta)
    information = x.T @ ((probabilities * (1.0 - probabilities))[:, None] * x)
    covariance = np.linalg.inv(information)
    standard_errors = np.sqrt(np.diag(covariance))
    p_values = 2.0 * norm.sf(np.abs(beta / standard_errors))

    names = {
        "ses": 1,
        "black": 2,
        "hispanic": 3,
        "ses_x_black": 4,
        "ses_x_hispanic": 5,
    }
    result = {
        name: {
            "coef": float(beta[index]),
            "std_err": float(standard_errors[index]),
            "p_value": float(p_values[index]),
        }
        for name, index in names.items()
    }
    result["n"] = len(df)
    return result


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
