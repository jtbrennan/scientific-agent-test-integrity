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
    # Construct the design matrix explicitly so that the order of the
    # returned coefficients is unambiguous.
    area = np.asarray(df["area"], dtype=float)
    y = np.asarray(df["diversity"], dtype=float)
    X = np.column_stack(
        [
            np.ones(len(df)),
            np.log(area),
            np.asarray(df["bio1"], dtype=float),
            np.asarray(df["bio12"], dtype=float) / 1000.0,
        ]
    )

    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)):
        raise ValueError("area, climate covariates, and diversity must be finite")
    if np.any(area <= 0):
        raise ValueError("area must be positive")
    if np.any(y < 0) or np.any(y != np.floor(y)):
        raise ValueError("diversity must contain non-negative integer counts")

    # Optimize on centered/scaled covariates.  This is only a numerical
    # reparameterization; estimates and information below are reported in the
    # covariate units requested by the question.
    centers = X[:, 1:].mean(axis=0)
    scales = X[:, 1:].std(axis=0)
    if np.any(scales == 0):
        raise ValueError("covariates must not be constant")
    Z = np.column_stack([np.ones(len(df)), (X[:, 1:] - centers) / scales])

    # Negative log likelihood (the factorial term is omitted because it does
    # not depend on the coefficients) and its analytic gradient.
    def objective(theta: np.ndarray) -> float:
        eta = Z @ theta
        mu = np.exp(eta)
        return float(np.sum(mu - y * eta))

    def gradient(theta: np.ndarray) -> np.ndarray:
        eta = Z @ theta
        mu = np.exp(eta)
        return Z.T @ (mu - y)

    def hessian(theta: np.ndarray) -> np.ndarray:
        eta = Z @ theta
        mu = np.exp(eta)
        return Z.T @ (mu[:, None] * Z)

    mean_y = float(np.mean(y))
    initial = np.zeros(Z.shape[1], dtype=float)
    initial[0] = np.log(mean_y) if mean_y > 0 else -1.0
    # Newton/IRLS iterations with backtracking.  The Poisson log likelihood is
    # concave, so this gives a direct, accurate maximum-likelihood solve.
    theta = initial
    gradient_tolerance = 1e-8 * max(1.0, float(np.sum(y)))
    for _ in range(100):
        grad = gradient(theta)
        if np.max(np.abs(grad)) < gradient_tolerance:
            break
        step = np.linalg.solve(hessian(theta), grad)
        current = objective(theta)
        step_scale = 1.0
        while step_scale > 1e-12:
            candidate = theta - step_scale * step
            if objective(candidate) < current:
                theta = candidate
                break
            step_scale *= 0.5
        else:
            if np.max(np.abs(grad)) < gradient_tolerance * 10:
                break
            raise RuntimeError("Poisson optimization failed: line search stalled")
    else:
        raise RuntimeError("Poisson optimization failed to converge")

    beta = np.empty_like(theta)
    beta[1:] = theta[1:] / scales
    beta[0] = theta[0] - np.dot(beta[1:], centers)
    mu = np.exp(X @ beta)

    # For a Poisson GLM with a log link, the observed and expected Fisher
    # information coincide: I(beta) = X' diag(mu) X.
    information = X.T @ (mu[:, None] * X)
    covariance = np.linalg.inv(information)
    std_err = np.sqrt(np.diag(covariance))
    z_values = beta / std_err
    p_values = 2.0 * norm.sf(np.abs(z_values))

    with np.errstate(divide="ignore", invalid="ignore"):
        terms = y * np.log(y / mu)
    terms[y == 0] = 0.0
    deviance = float(2.0 * np.sum(terms - (y - mu)))

    names = ("intercept", "log_area", "bio1", "precip_m")
    return {
        "terms": {
            name: {
                "coef": float(beta[i]),
                "std_err": float(std_err[i]),
                "p_value": float(p_values[i]),
            }
            for i, name in enumerate(names)
        },
        "deviance": deviance,
        "n": len(df),
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "freshwater_fish_basins.csv"
    print(json.dumps(fit_diversity_poisson(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
