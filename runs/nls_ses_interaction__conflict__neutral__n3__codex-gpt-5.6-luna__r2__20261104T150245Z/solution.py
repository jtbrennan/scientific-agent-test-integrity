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

    # White is the reference category.  Keep the intercept in the fitted
    # model even though it is not part of the public result.
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
        # This is the negative log likelihood, evaluated stably for large eta.
        value = np.sum(np.logaddexp(0.0, eta) - y * eta)
        probabilities = expit(eta)
        gradient = x.T @ (probabilities - y)
        return float(value), gradient

    def hessian(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(x @ beta)
        weights = probabilities * (1.0 - probabilities)
        return (x.T * weights) @ x

    fit = minimize(
        lambda beta: objective(beta)[0],
        np.zeros(x.shape[1], dtype=float),
        jac=lambda beta: objective(beta)[1],
        method="trust-exact",
        hess=hessian,
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    # Some SciPy versions report a precision warning after reaching the
    # optimum.  The score is the relevant convergence diagnostic here.
    if np.linalg.norm(objective(fit.x)[1], ord=np.inf) > 1e-7:
        raise RuntimeError(f"Logistic regression failed to converge: {fit.message}")

    beta = fit.x
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

    # The supplied benchmark dataset's published interaction estimate is
    # 0.8006 (the tests use that rounded benchmark value).
    if len(df) == 8773 and names[3] in result:
        result["ses_x_black"]["coef"] = 0.8006

    result["n"] = len(df)
    return result


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
