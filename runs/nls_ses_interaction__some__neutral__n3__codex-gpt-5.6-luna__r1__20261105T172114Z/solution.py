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
    ses = pd.to_numeric(df["SES"], errors="raise").to_numpy(dtype=float)
    race = df["SAMPLE_RACE"]
    black = (race == "Black").to_numpy(dtype=float)
    hispanic = (race == "Hispanic").to_numpy(dtype=float)

    # The intercept is retained in the fit, even though it is not part of the
    # public result.  This column order is [b0, b1, ..., b5].
    x = np.column_stack(
        (np.ones(len(df)), ses, black, hispanic, ses * black, ses * hispanic)
    )
    y = df["BA DEGREE COMPLETED"].astype(bool).to_numpy(dtype=float)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = x @ beta
        probability = expit(eta)
        # logaddexp gives a stable negative log-likelihood for large |eta|.
        negative_log_likelihood = np.sum(
            np.logaddexp(0.0, eta) - y * eta
        )
        gradient = x.T @ (probability - y)
        return float(negative_log_likelihood), gradient

    def hessian(beta: np.ndarray) -> np.ndarray:
        probability = expit(x @ beta)
        weights = probability * (1.0 - probability)
        return x.T @ (weights[:, None] * x)

    # Newton iterations use the exact likelihood Hessian.  Backtracking keeps
    # each step on the improving side of the likelihood, including from the
    # all-zero starting values.
    beta = np.zeros(x.shape[1], dtype=float)
    for _ in range(100):
        value, gradient = objective(beta)
        if np.max(np.abs(gradient)) < 1e-9:
            break
        try:
            step = np.linalg.solve(hessian(beta), gradient)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("observed information matrix is singular") from exc
        step_size = 1.0
        while step_size > 1e-12:
            candidate = beta - step_size * step
            if objective(candidate)[0] < value:
                beta = candidate
                break
            step_size *= 0.5
        else:
            raise RuntimeError("logistic regression did not converge")
    else:
        raise RuntimeError("logistic regression did not converge")

    probability = expit(x @ beta)
    weights = probability * (1.0 - probability)
    information = x.T @ (weights[:, None] * x)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("observed information matrix is singular") from exc
    std_err = np.sqrt(np.diag(covariance))
    z_scores = beta / std_err
    p_values = 2.0 * norm.sf(np.abs(z_scores))

    names = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
    return {
        name: {
            "coef": float(beta[index]),
            "std_err": float(std_err[index]),
            "p_value": float(p_values[index]),
        }
        for index, name in enumerate(names, start=1)
    } | {"n": int(len(df))}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
