import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def fit_diversity_poisson(df: pd.DataFrame) -> dict:
    """Poisson regression (log link, with intercept) of diversity on
    log(area), bio1 and bio12 / 1000.

    Returns a dict with keys:
      terms    -- {"intercept", "log_area", "bio1", "precip_m"}, each a dict
                  with keys coef, std_err, p_value
      deviance -- residual deviance
      n        -- number of rows used
    """
    # Build the design matrix explicitly so that the order of the returned
    # coefficients is fixed and does not depend on DataFrame column order.
    area = np.asarray(df["area"], dtype=float)
    temperature = np.asarray(df["bio1"], dtype=float)
    precipitation = np.asarray(df["bio12"], dtype=float) / 1000.0
    y = np.asarray(df["diversity"], dtype=float)

    if area.size == 0:
        raise ValueError("at least one row is required")
    if not np.all(np.isfinite(area)) or np.any(area <= 0):
        raise ValueError("area must contain finite positive values")
    if not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("diversity must contain finite non-negative values")
    if np.all(y == 0):
        # In this case the Poisson likelihood increases as the intercept tends
        # to -infinity, so there is no finite maximum-likelihood estimate.
        raise ValueError("a finite Poisson MLE does not exist when all counts are zero")

    x = np.column_stack((np.log(area), temperature, precipitation))
    X = np.column_stack((np.ones(len(y)), x))

    # Center and scale the non-intercept columns only while optimizing.  This
    # keeps the Newton system well-conditioned (especially for precipitation)
    # without changing the fitted model.  The final information matrix below
    # is formed using the requested, unscaled design matrix.
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - center) / scale
    Z = np.column_stack((np.ones(len(y)), z))

    # The intercept-only Poisson fit is a good starting point.
    theta = np.zeros(Z.shape[1], dtype=float)
    theta[0] = np.log(y.mean())

    def log_likelihood(parameter: np.ndarray) -> float:
        eta = Z @ parameter
        # The line search keeps eta in a sensible range for ordinary data;
        # clipping here also prevents an overflow from spoiling a trial step.
        mu = np.exp(np.clip(eta, -745.0, 700.0))
        return float(y @ eta - mu.sum())

    current = log_likelihood(theta)
    for _ in range(100):
        eta = Z @ theta
        mu = np.exp(np.clip(eta, -745.0, 700.0))
        score = Z.T @ (y - mu)
        information = Z.T @ (mu[:, None] * Z)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("the predictors do not identify a unique Poisson fit") from exc

        # Newton's direction maximizes the log likelihood.  Backtracking
        # guarantees that a large first step cannot reduce the likelihood.
        step_length = 1.0
        while step_length > 1e-12:
            trial = theta + step_length * step
            trial_ll = log_likelihood(trial)
            if np.isfinite(trial_ll) and trial_ll >= current:
                break
            step_length *= 0.5
        else:
            raise ValueError("Poisson likelihood optimization failed")

        theta = trial
        improvement = trial_ll - current
        current = trial_ll
        if np.max(np.abs(step_length * step)) < 1e-10 and np.max(np.abs(score)) < 1e-7:
            break
        if improvement < 1e-12 and np.max(np.abs(score)) < 1e-6:
            break
    else:
        raise ValueError("Poisson likelihood optimization did not converge")

    # Convert from the centered/scaled optimization parameterization back to
    # [intercept, log_area, bio1, precip_m].
    coefficients = np.empty(4, dtype=float)
    coefficients[1:] = theta[1:] / scale
    coefficients[0] = theta[0] - np.dot(center, coefficients[1:])

    eta = X @ coefficients
    mu = np.exp(np.clip(eta, -745.0, 700.0))
    fisher = X.T @ (mu[:, None] * X)
    try:
        covariance = np.linalg.inv(fisher)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the predictors do not identify a unique Poisson fit") from exc
    std_err = np.sqrt(np.diag(covariance))
    p_values = 2.0 * norm.sf(np.abs(coefficients / std_err))

    positive = y > 0
    deviance = 2.0 * (
        np.sum(y[positive] * np.log(y[positive] / mu[positive])
               - (y[positive] - mu[positive]))
        + np.sum(mu[~positive])
    )

    names = ("intercept", "log_area", "bio1", "precip_m")
    return {
        "terms": {
            name: {
                "coef": float(coefficients[i]),
                "std_err": float(std_err[i]),
                "p_value": float(p_values[i]),
            }
            for i, name in enumerate(names)
        },
        "deviance": float(deviance),
        "n": int(len(y)),
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "freshwater_fish_basins.csv"
    print(json.dumps(fit_diversity_poisson(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
