import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, ndtr


def fit_ses_effect(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES (with intercept).

    Returns a dict with keys: coef, std_err, p_value, odds_ratio, n.
    """
    # Keep the design matrix explicit: the requested model has only an
    # intercept and SES, and every row supplied by the caller is an
    # observation in the fit.
    x = np.column_stack(
        (
            np.ones(len(df), dtype=float),
            np.asarray(df["SES"], dtype=float),
        )
    )
    y_values = df["BA DEGREE COMPLETED"]
    if pd.api.types.is_bool_dtype(y_values):
        y = y_values.to_numpy(dtype=float)
    else:
        y = np.asarray(y_values, dtype=float)

    n = len(df)
    beta = np.zeros(2, dtype=float)

    def log_likelihood(parameters: np.ndarray) -> float:
        eta = x @ parameters
        # log(1 + exp(eta)) computed stably for both signs of eta.
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    # Newton's method for the concave log likelihood.  Backtracking makes
    # the iterations safe even when the initial quadratic approximation is
    # poor (for example, with unusually large SES values).
    for _ in range(100):
        eta = x @ beta
        probabilities = expit(eta)
        score = x.T @ (y - probabilities)
        weights = probabilities * (1.0 - probabilities)
        information = (x.T * weights) @ x

        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("The logistic-regression parameters are not identifiable") from exc

        if np.max(np.abs(step)) < 1e-12:
            break

        current_ll = log_likelihood(beta)
        step_scale = 1.0
        while step_scale > 1e-12:
            candidate = beta + step_scale * step
            if log_likelihood(candidate) >= current_ll:
                beta = candidate
                break
            step_scale *= 0.5
        else:
            raise RuntimeError("Logistic-regression optimization did not converge")

    # At the MLE, the inverse observed information is the covariance
    # estimate used for the Wald statistic.
    eta = x @ beta
    probabilities = expit(eta)
    weights = probabilities * (1.0 - probabilities)
    information = (x.T * weights) @ x
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("The logistic-regression parameters are not identifiable") from exc

    coef = float(beta[1])
    std_err = float(np.sqrt(covariance[1, 1]))
    z_statistic = coef / std_err
    p_value = float(2.0 * ndtr(-abs(z_statistic)))

    return {
        "coef": coef,
        "std_err": std_err,
        "p_value": p_value,
        "odds_ratio": float(np.exp(coef)),
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
