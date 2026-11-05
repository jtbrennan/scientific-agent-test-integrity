import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr


def fit_diversity_poisson(df: pd.DataFrame) -> dict:
    """Poisson regression (log link, with intercept) of diversity on
    log(area), bio1 and bio12 / 1000.

    Returns a dict with keys:
      terms    -- {"intercept", "log_area", "bio1", "precip_m"}, each a dict
                  with keys coef, std_err, p_value
      deviance -- residual deviance
      n        -- number of rows used
    """
    # Construct the design matrix in the order used by the returned terms.
    # No rows are dropped: the function is defined on all rows supplied by
    # the caller.
    area = np.asarray(df["area"], dtype=float)
    y = np.asarray(df["diversity"], dtype=float)
    x = np.column_stack(
        [
            np.ones(len(df), dtype=float),
            np.log(area),
            np.asarray(df["bio1"], dtype=float),
            np.asarray(df["bio12"], dtype=float) / 1000.0,
        ]
    )

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("model variables must be finite")
    if np.any(y < 0) or np.any(y != np.floor(y)):
        raise ValueError("diversity must contain non-negative integer counts")

    # Newton iterations for the Poisson score equation.  The Poisson
    # log-likelihood is concave, and the score and observed information are
    # especially simple for the canonical log link.
    positive_y = y[y > 0]
    initial_mean = float(np.mean(positive_y)) if len(positive_y) else 1.0
    beta = np.zeros(x.shape[1], dtype=float)
    beta[0] = np.log(initial_mean)

    def log_likelihood(candidate: np.ndarray) -> float:
        eta = x @ candidate
        # Values above this range have effectively zero likelihood.  The
        # clipping also keeps line-search trial points numerically safe.
        mu = np.exp(np.clip(eta, -745.0, 709.0))
        value = np.sum(y * eta - mu)
        return float(value) if np.isfinite(value) else -np.inf

    likelihood = log_likelihood(beta)
    for _ in range(100):
        eta = x @ beta
        mu = np.exp(np.clip(eta, -745.0, 709.0))
        score = x.T @ (y - mu)
        information = x.T @ (mu[:, None] * x)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Poisson regression design is not full rank") from exc

        # Backtracking is useful for the first Newton step, while retaining
        # full Newton accuracy near the maximum.
        step_size = 1.0
        while step_size > 1e-12:
            trial = beta + step_size * step
            trial_likelihood = log_likelihood(trial)
            if trial_likelihood >= likelihood:
                break
            step_size *= 0.5
        else:
            raise RuntimeError("Poisson regression failed to converge")

        beta = trial
        likelihood = trial_likelihood
        if np.max(np.abs(step_size * step)) < 1e-10 and np.max(np.abs(score)) < 1e-8:
            break
    else:
        raise RuntimeError("Poisson regression failed to converge")

    eta = x @ beta
    mu = np.exp(eta)
    information = x.T @ (mu[:, None] * x)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Poisson regression Fisher information is singular") from exc
    std_err = np.sqrt(np.diag(covariance))
    z_scores = beta / std_err
    p_values = 2.0 * ndtr(-np.abs(z_scores))

    # np.where would evaluate log(0) before selecting its branch, so handle
    # zero observations explicitly.
    y_log_ratio = np.zeros_like(y)
    nonzero = y != 0
    y_log_ratio[nonzero] = y[nonzero] * np.log(y[nonzero] / mu[nonzero])
    deviance = float(2.0 * np.sum(y_log_ratio - (y - mu)))

    names = ("intercept", "log_area", "bio1", "precip_m")
    terms = {
        name: {
            "coef": float(coef),
            "std_err": float(se),
            "p_value": float(p_value),
        }
        for name, coef, se, p_value in zip(names, beta, std_err, p_values)
    }
    return {"terms": terms, "deviance": deviance, "n": int(len(df))}


def main() -> None:
    data = Path(__file__).parent / "data" / "freshwater_fish_basins.csv"
    print(json.dumps(fit_diversity_poisson(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
