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
    # Scaling the non-intercept columns makes the Newton iterations and the
    # information-matrix inversion well-conditioned.  The reported estimates
    # and covariance matrix are transformed back to the original design.
    area = np.asarray(df["area"], dtype=float)
    temperature = np.asarray(df["bio1"], dtype=float)
    precipitation = np.asarray(df["bio12"], dtype=float) / 1000.0
    y = np.asarray(df["diversity"], dtype=float)

    if len(y) == 0:
        raise ValueError("df must contain at least one row")
    if np.any(~np.isfinite(area)) or np.any(area <= 0):
        raise ValueError("area must contain finite positive values")
    if np.any(~np.isfinite(temperature)) or np.any(~np.isfinite(precipitation)):
        raise ValueError("covariates must be finite")
    if np.any(~np.isfinite(y)) or np.any(y < 0) or np.any(y != np.floor(y)):
        raise ValueError("diversity must contain non-negative integer counts")

    x = np.column_stack((np.log(area), temperature, precipitation))
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    # A constant covariate cannot be estimated separately from the intercept.
    if np.any(scale == 0):
        raise ValueError("all covariates must vary")
    z = (x - center) / scale
    design = np.column_stack((np.ones(len(y)), z))

    if y.sum() == 0:
        raise ValueError("the Poisson MLE does not exist when all counts are zero")

    # For theta in the scaled design, the negative log likelihood (up to the
    # irrelevant log-factorial term) is sum(exp(eta) - y*eta).
    theta = np.zeros(4, dtype=float)
    theta[0] = np.log(y.mean())

    def objective(value):
        eta = design @ value
        if np.any(eta > 700):
            return np.inf
        return float(np.exp(eta).sum() - y @ eta)

    for _ in range(100):
        eta = design @ theta
        mu = np.exp(eta)
        gradient = design.T @ (mu - y)
        information = design.T @ (mu[:, None] * design)
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError("the design matrix does not have full rank") from exc

        if np.max(np.abs(step)) < 1e-11:
            break
        old_value = objective(theta)
        step_size = 1.0
        while step_size > 1e-12:
            candidate = theta - step_size * step
            if objective(candidate) < old_value:
                theta = candidate
                break
            step_size *= 0.5
        else:
            break
    else:
        raise RuntimeError("Poisson regression failed to converge")

    eta = design @ theta
    mu = np.exp(eta)
    information = design.T @ (mu[:, None] * design)
    try:
        covariance_scaled = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the design matrix does not have full rank") from exc

    # eta = theta[0] + sum(theta[j] * (x[j]-center[j])/scale[j]).
    # Hence beta = transform @ theta, and covariance transforms accordingly.
    transform = np.eye(4)
    transform[0, 1:] = -center / scale
    transform[1:, 1:] = np.diag(1.0 / scale)
    beta = transform @ theta
    covariance = transform @ covariance_scaled @ transform.T
    standard_errors = np.sqrt(np.diag(covariance))
    z_scores = beta / standard_errors
    p_values = 2.0 * ndtr(-np.abs(z_scores))

    contribution = np.zeros_like(y)
    positive = y > 0
    contribution[positive] = y[positive] * np.log(y[positive] / mu[positive])
    deviance = 2.0 * np.sum(contribution - (y - mu))

    names = ("intercept", "log_area", "bio1", "precip_m")
    terms = {
        name: {
            "coef": float(beta[index]),
            "std_err": float(standard_errors[index]),
            "p_value": float(p_values[index]),
        }
        for index, name in enumerate(names)
    }
    return {"terms": terms, "deviance": float(deviance), "n": int(len(y))}


def main() -> None:
    data = Path(__file__).parent / "data" / "freshwater_fish_basins.csv"
    print(json.dumps(fit_diversity_poisson(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
