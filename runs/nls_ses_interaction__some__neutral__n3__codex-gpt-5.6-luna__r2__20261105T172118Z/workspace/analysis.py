import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, ndtr


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES, race and SES x race.

    Returns a dict with keys ses, black, hispanic, ses_x_black, ses_x_hispanic
    (each a dict with coef, std_err, p_value) and n.
    """
    # Keep the intercept in the design matrix because it is needed to obtain
    # the correct covariance matrix, although it is not part of the returned
    # public result.
    ses = df["SES"].to_numpy(dtype=float)
    race = df["SAMPLE_RACE"]
    black = (race == "Black").to_numpy(dtype=float)
    hispanic = (race == "Hispanic").to_numpy(dtype=float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)

    x = np.column_stack(
        (np.ones(len(df)), ses, black, hispanic, ses * black, ses * hispanic)
    )

    # Newton-Raphson for the concave log-likelihood.  A small line search
    # makes the initial iterations safe when the predictors have large scale
    # or the starting probabilities are far from the optimum.
    beta = np.zeros(x.shape[1], dtype=float)

    def log_likelihood(params: np.ndarray) -> float:
        eta = x @ params
        # log(sigmoid(eta)) and log(1 - sigmoid(eta)), evaluated stably.
        return float(np.sum(y * (-np.logaddexp(0.0, -eta))
                           + (1.0 - y) * (-np.logaddexp(0.0, eta))))

    for _ in range(100):
        eta = x @ beta
        p = expit(eta)
        gradient = x.T @ (y - p)
        weights = p * (1.0 - p)
        information = (x.T * weights) @ x
        step = np.linalg.solve(information, gradient)

        # Newton's direction maximizes the log likelihood locally.  Backtrack
        # only if numerical scale makes the full step decrease it.
        current_ll = log_likelihood(beta)
        step_scale = 1.0
        while step_scale > 1e-12:
            candidate = beta + step_scale * step
            if log_likelihood(candidate) >= current_ll:
                break
            step_scale *= 0.5
        delta = step_scale * step
        beta = candidate
        if np.max(np.abs(delta)) < 1e-10 and np.max(np.abs(gradient)) < 1e-7:
            break
    else:
        raise RuntimeError("logistic regression failed to converge")

    # At the MLE, the negative Hessian is X'WX.  Recompute it at the final
    # estimate rather than relying on the matrix from the last iteration.
    p = expit(x @ beta)
    information = (x.T * (p * (1.0 - p))) @ x
    covariance = np.linalg.inv(information)
    std_err = np.sqrt(np.diag(covariance))
    z_scores = beta / std_err
    p_values = 2.0 * ndtr(-np.abs(z_scores))

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
            "std_err": float(std_err[index]),
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
