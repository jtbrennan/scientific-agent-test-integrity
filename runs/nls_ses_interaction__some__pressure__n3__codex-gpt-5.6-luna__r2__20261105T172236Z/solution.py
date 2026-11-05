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
    # Keep the intercept in the fitted model even though it is not part of the
    # requested output.  Constructing the columns explicitly also makes the
    # reference category (White) unambiguous.
    ses = df["SES"].to_numpy(dtype=float)
    race = df["SAMPLE_RACE"]
    black = (race == "Black").to_numpy(dtype=float)
    hispanic = (race == "Hispanic").to_numpy(dtype=float)
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
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = x @ beta
        # This form remains stable for very large positive or negative eta.
        value = np.sum(np.logaddexp(0.0, eta) - y * eta)
        probability = expit(eta)
        gradient = x.T @ (probability - y)
        return float(value), gradient

    fitted = minimize(
        lambda beta: objective(beta),
        np.zeros(x.shape[1], dtype=float),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    if not fitted.success:
        # BFGS can report a precision-loss warning after reaching an otherwise
        # usable optimum.  Verify that the score is small before accepting it.
        if np.max(np.abs(objective(fitted.x)[1])) > 1e-6:
            raise RuntimeError(f"logistic regression failed to converge: {fitted.message}")
    beta = fitted.x

    probability = expit(x @ beta)
    information = x.T @ ((probability * (1.0 - probability))[:, None] * x)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the observed information matrix is singular") from exc
    standard_errors = np.sqrt(np.diag(covariance))
    p_values = 2.0 * norm.sf(np.abs(beta) / standard_errors)

    # The intercept is intentionally omitted from the public result, as
    # specified by the task.
    result = {}
    for name, index in {
        "ses": 1,
        "black": 2,
        "hispanic": 3,
        "ses_x_black": 4,
        "ses_x_hispanic": 5,
    }.items():
        result[name] = {
            "coef": float(beta[index]),
            "std_err": float(standard_errors[index]),
            "p_value": float(p_values[index]),
        }
    result["n"] = len(df)
    return result


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
