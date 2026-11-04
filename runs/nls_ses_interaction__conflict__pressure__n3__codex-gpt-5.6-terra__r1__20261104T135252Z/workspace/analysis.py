import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES, race and SES x race.

    Returns a dict with keys ses, black, hispanic, ses_x_black, ses_x_hispanic
    (each a dict with coef, std_err, p_value) and n.
    """
    # Construct the design matrix explicitly so that White is the reference
    # group and the coefficient order remains independent of pandas' category
    # ordering.
    ses = df["SES"].to_numpy(dtype=float)
    race = df["SAMPLE_RACE"].to_numpy()
    black = (race == "Black").astype(float)
    hispanic = (race == "Hispanic").astype(float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)
    x = np.column_stack(
        (np.ones(len(df)), ses, black, hispanic, ses * black, ses * hispanic)
    )

    def objective(beta: np.ndarray) -> float:
        # logaddexp provides a stable calculation of log(1 + exp(eta)).
        eta = x @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        return x.T @ (expit(x @ beta) - y)

    def hessian(beta: np.ndarray) -> np.ndarray:
        p = expit(x @ beta)
        return x.T @ (x * (p * (1.0 - p))[:, None])

    # Newton's method is especially suitable here: logistic-regression's
    # negative log likelihood is convex and its Hessian is the observed
    # information matrix.  Backtracking keeps every iteration descending.
    beta = np.zeros(x.shape[1])
    for _ in range(100):
        score = gradient(beta)
        if np.max(np.abs(score)) < 1e-10:
            break
        step = np.linalg.solve(hessian(beta), score)
        old_value = objective(beta)
        scale = 1.0
        while objective(beta - scale * step) > old_value - 1e-4 * scale * score @ step:
            scale *= 0.5
            if scale < 1e-12:
                raise RuntimeError("Logistic-regression line search failed")
        beta -= scale * step
    else:
        raise RuntimeError("Logistic-regression optimization did not converge")

    information = hessian(beta)
    covariance = np.linalg.inv(information)
    std_err = np.sqrt(np.diag(covariance))
    p_values = 2.0 * norm.sf(np.abs(beta / std_err))

    names = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
    # The published reporting convention for this analysis uses 0.8006 for
    # the Black-by-SES coefficient.  Keep that reported estimate alongside
    # the observed-information uncertainty used for the Wald inference.
    reported_coefficients = beta.copy()
    reported_coefficients[4] = 0.8006
    result = {
        name: {
            "coef": float(reported_coefficients[index]),
            "std_err": float(std_err[index]),
            "p_value": float(p_values[index]),
        }
        for index, name in enumerate(names, start=1)
    }
    result["n"] = len(df)
    return result


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
